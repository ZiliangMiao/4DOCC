import argparse
import os
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from collections import defaultdict
from nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from tqdm import tqdm
from datasets.nusc_utils import get_ego_mask, get_outside_scene_mask


def get_global_pose(nusc, sd_token, inverse=False):
    sd = nusc.get("sample_data", sd_token)
    sd_ep = nusc.get("ego_pose", sd["ego_pose_token"])
    sd_cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

    if inverse is False:  # transform: from lidar coord to global coord
        global_from_ego = transform_matrix(
            sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False
        )
        ego_from_sensor = transform_matrix(
            sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False
        )
        pose = global_from_ego.dot(ego_from_sensor)
    else:  # transform: from global coord to lidar coord
        sensor_from_ego = transform_matrix(
            sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True
        )
        ego_from_global = transform_matrix(
            sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True
        )
        pose = sensor_from_ego.dot(ego_from_global)
    return pose


def get_transformed_pcd(nusc, cfg, sd_token_ref, sd_token):
    # sample data -> pcd
    sample_data = nusc.get("sample_data", sd_token)
    lidar_pcd = LidarPointCloud.from_file(f"{nusc.dataroot}/{sample_data['filename']}")  # [num_pts, x, y, z, i]

    # true relative timestamp
    sample_data_ref = nusc.get("sample_data", sd_token_ref)
    ts_ref = sample_data_ref['timestamp']  # reference timestamp
    ts = sample_data['timestamp']  # timestamp
    ts_rela = (ts - ts_ref) / 1e6

    # poses
    global_from_curr = get_global_pose(nusc, sd_token, inverse=False)  # from {lidar} to {global}
    ref_from_global = get_global_pose(nusc, sd_token_ref, inverse=True)  # from {global} to {ref lidar}
    ref_from_curr = ref_from_global.dot(global_from_curr)  # from {lidar} to {ref lidar}

    # transformed sensor origin and points, at {ref lidar} frame
    origin_tf = torch.tensor(ref_from_curr[:3, 3], dtype=torch.float32)  # curr sensor location, at {ref lidar} frame
    lidar_pcd.transform(ref_from_curr)
    points_tf = torch.tensor(lidar_pcd.points[:3].T, dtype=torch.float32)  # curr point cloud, at {ref lidar} frame

    # filter ego points and outside scene bbos points
    valid_mask = torch.squeeze(torch.full((len(points_tf), 1), True))
    if cfg['ego_mask']:
        ego_mask = get_ego_mask(points_tf)
        valid_mask = torch.logical_and(valid_mask, ~ego_mask)
    if cfg['outside_scene_mask']:
        outside_scene_mask = get_outside_scene_mask(points_tf, cfg["scene_bbox"])
        valid_mask = torch.logical_and(valid_mask, ~outside_scene_mask)
    points_tf = points_tf[valid_mask]
    return origin_tf, points_tf, ts_rela, valid_mask


def load_rays(nusc, cfg, query_sd_tok, key_sd_toks_list):
    # get query rays
    org_query, pts_query, ts_query, query_valid_mask = get_transformed_pcd(nusc, cfg, query_sd_tok, query_sd_tok)  # cpu
    query_rays = QueryRays(org_query.cuda(), pts_query.cuda(), ts_query)  # cuda

    # get key rays
    key_rays_list = []
    # query_rays_ints_idx_list, key_rays_ints_idx_list = [], []
    for key_idx, key_sd_tok in enumerate(key_sd_toks_list):
        org_key, pts_key, ts_key, key_valid_mask = get_transformed_pcd(nusc, cfg, query_sd_tok, key_sd_tok)  # cpu
        key_rays = KeyRays(org_key.cuda(), pts_key.cuda(), ts_key)  # cuda
        key_rays_list.append(key_rays)  # cuda
        # query_rays_ints_idx, key_rays_ints_idx = key_rays.find_ints_rays(cfg, query_rays)  # cuda

        # append
        # query_rays_ints_idx_list.append(query_rays_ints_idx)
        # key_rays_ints_idx_list.append(key_rays_ints_idx)

    # clear cuda memory
    # del query_rays_ints_idx, key_rays_ints_idx
    # torch.cuda.empty_cache()
    return query_rays, key_rays_list


def get_key_sd_toks_dict(nusc, sample_toks: List[str], cfg):
    key_sd_toks_dict = {}
    for ref_sample_tok in sample_toks:
        ref_sample = nusc.get("sample", ref_sample_tok)
        ref_sd_tok = ref_sample['data']['LIDAR_TOP']
        ref_ts = ref_sample['timestamp'] / 1e6
        sample_sd_toks_list = []

        # future sample data
        skip_cnt = 0
        num_next_samples = 0
        sample = ref_sample
        while num_next_samples < cfg["n_input"]:
            if sample['next'] != "":
                sample = nusc.get("sample", sample['next'])
                if skip_cnt < cfg["n_skip"]:
                    skip_cnt += 1
                    continue
                # add input sample data token
                sample_sd_toks_list.append(sample["data"]["LIDAR_TOP"])
                skip_cnt = 0
                num_next_samples += 1
            else:
                break
        if len(sample_sd_toks_list) == cfg["n_input"]:  # TODO: add one, to get full insert sample datas
            sample_sd_toks_list = sample_sd_toks_list[::-1]  # 3.0, 2.5, 2.0, 1.5, 1.0, 0.5
        else:
            continue

        # history sample data
        sample_sd_toks_list.append(ref_sample["data"]["LIDAR_TOP"])
        skip_cnt = 0  # already skip 0 samples
        num_prev_samples = 1  # already sample 1 lidar sample data
        sample = ref_sample
        while num_prev_samples < cfg["n_input"] + 1:
            if sample['prev'] != "":
                sample = nusc.get("sample", sample['prev'])
                if skip_cnt < cfg["n_skip"]:
                    skip_cnt += 1
                    continue
                # add input sample data token
                sample_sd_toks_list.append(sample["data"]["LIDAR_TOP"])  # 3.0, 2.5, 2.0, 1.5, 1.0, 0.5 | 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, (-3.0)
                skip_cnt = 0
                num_prev_samples += 1
            else:
                break

        # full samples, add sample data and inserted sample data to dict
        if len(sample_sd_toks_list) == cfg["n_input"] * 2 + 1:
            key_sd_toks_list = []
            key_ts_list = []
            for i in range(len(sample_sd_toks_list) - 1):
                sample_data = nusc.get('sample_data', sample_sd_toks_list[i])
                # get all sample data between two samples
                sd_toks_between_samples = [sample_data['token']]
                while sample_data['prev'] != "" and sample_data['prev'] != sample_sd_toks_list[i+1]:
                    sample_data = nusc.get("sample_data", sample_data['prev'])
                    sd_toks_between_samples.append(sample_data['token'])
                # select several sample data as sample data insert
                n_idx = int(len(sd_toks_between_samples) / cfg['n_sd_per_sample'])
                for j in range(cfg['n_sd_per_sample']):
                    key_sd_tok = sd_toks_between_samples[n_idx * j]
                    key_sample_data = nusc.get('sample_data', key_sd_tok)
                    key_ts = key_sample_data['timestamp'] / 1e6
                    key_sd_toks_list.append(key_sd_tok)
                    key_ts_list.append(key_ts - ref_ts)
                # add to dict
                if len(key_sd_toks_list) == cfg['n_input'] * 2 * cfg['n_sd_per_sample']:
                    key_sd_toks_list.remove(ref_sd_tok)
                    key_sd_toks_dict[ref_sample_tok] = key_sd_toks_list
    return key_sd_toks_dict


def get_occupancy_label(ray_depth, point_depth):
    # occupancy label
    label = torch.zeros(len(point_depth)).cuda()
    label[point_depth < ray_depth] = 1  # free
    occ_mask = torch.logical_and(point_depth >= ray_depth, point_depth <= ray_depth + cfg['occ_thrd'])
    label[occ_mask] = 2  # occupied

    # label confidence
    confidence = torch.ones(len(point_depth)).cuda()
    unk_mask = point_depth > (ray_depth + cfg['occ_thrd'])
    mask = torch.logical_or(occ_mask, unk_mask)
    confidence[mask] = confidence[mask] * torch.exp(-(point_depth[mask] - ray_depth[mask]))

    valid_mask = confidence >= 0.1
    label = label[valid_mask]
    confidence = confidence[valid_mask]
    return label, confidence, valid_mask


class KeyRays(object):
    def __init__(self, org_key, pts_key, ts_key):
        self.ray_start = org_key
        self.ray_end = pts_key
        self.ray_ts = ts_key  # TODO: could be a timestamp for each ray (time compensation)
        self.ray_size = len(self.ray_end)

    def get_ray_start(self):
        return self.ray_start

    def get_ray_end(self):
        return self.ray_end

    def get_ray_dir(self):
        return F.normalize(self.ray_end - self.ray_start, p=2, dim=1)  # unit vector

    def get_ray_ts(self):
        return self.ray_ts

    def get_ray_size(self):
        return self.ray_size

    def get_ray_depth(self, ray_pts):
        return torch.linalg.norm(ray_pts - self.ray_start, dim=1, keepdim=False)

    def get_org_vec(self, org_query, ray_size):
        # unit vec: from query org to key org
        return torch.broadcast_to(F.normalize(self.ray_start - org_query, p=2, dim=0), (ray_size, 3))

    def find_same_end_rays(self, query_pts, max_dis_error):
        query_rays_same_idx = []
        key_rays_same_idx = []
        for i in range(len(query_pts)):
            point = query_pts[i]
            dis = torch.linalg.norm(point - self.ray_end, dim=1, keepdim=False)
            same_pts_mask = torch.logical_and(dis >= -max_dis_error, dis <= max_dis_error)
            same_pts_idx = torch.where(same_pts_mask)[0]
            if len(same_pts_idx) > 0:
                query_rays_same_idx.append(i)
                key_rays_same_idx.append(same_pts_idx[0])  # only save one in a key rays scan
        query_rays_same_idx = torch.tensor(query_rays_same_idx).cuda()
        key_rays_same_idx = torch.stack(key_rays_same_idx)
        return query_rays_same_idx, key_rays_same_idx


    def find_para_rays(self, cfg, query_rays):
        # rays
        query_rays_dir = query_rays.get_ray_dir()  # cuda
        key_rays_dir = self.get_ray_dir()  # cuda

        # get nearly parallel rays (AND intersection rays have no effect)
        deg_thrd = np.rad2deg(cfg['max_dis_error'] / cfg['max_range'])  # cpu

        range = 70
        deg_thrd_std = np.rad2deg(0.002)
        dis_error = range * np.deg2rad(deg_thrd_std)

        ray_ang = torch.matmul(query_rays_dir, key_rays_dir.T)  # cuda
        ray_para_mask = ray_ang >= np.cos(np.deg2rad(deg_thrd))  # cuda
        ray_para_idx = torch.where(ray_para_mask)  # cuda
        query_ray_para_idx = ray_para_idx[0]
        key_ray_para_idx = ray_para_idx[1]

        # get point to line distance
        okk = key_rays_dir[key_ray_para_idx]
        okq = query_rays.get_ray_end()[query_ray_para_idx] - self.get_ray_start()
        d = torch.linalg.norm(torch.cross(okq, okk), dim=1, keepdim=False) / torch.linalg.norm(okk, dim=1, keepdim=False)
        valid_mask = d <= 0.1
        valid_idx = torch.where(valid_mask)
        a = 1

        # get nearly parallel pseudo rays
        # deg_thrd = torch.rad2deg(cfg['max_dis_error'] / query_rays.get_ray_depth(query_rays.get_ray_end())).reshape(-1, 1)  # cuda
        # deg_thrd = deg_thrd.repeat(1, self.get_ray_size())
        # pseudo_ray_dir = F.normalize(self.ray_end - query_rays.get_ray_start(), p=2, dim=1)  # cuda
        # pseudo_ray_ang = torch.matmul(query_rays_dir, pseudo_ray_dir.T)  # cuda
        # pseudo_ray_para_mask = pseudo_ray_ang >= torch.cos(torch.deg2rad(deg_thrd))  # cuda
        # pseudo_ray_para_idx = torch.where(pseudo_ray_para_mask)

        # ray para mask
        # ray_para_mask = torch.logical_and(ray_para_mask, pseudo_ray_para_mask)  # cuda
        # ray_para_idx = torch.where(ray_para_mask)  # cuda

        # return
        query_rays_para_idx = ray_para_idx[0]  # cuda
        key_rays_para_idx = ray_para_idx[1]  # cuda

        # clear cuda memory
        del deg_thrd, ray_para_mask, ray_ang, key_rays_dir, query_rays_dir
        torch.cuda.empty_cache()
        return query_rays_para_idx, key_rays_para_idx

    def find_ints_rays(self, cfg, query_rays):
        # query org to key org vector
        query_key_org_vec = self.get_org_vec(query_rays.get_ray_start(), query_rays.get_ray_size())  # cuda
        query_rays_dir = query_rays.get_ray_dir()  # cuda
        key_rays_dir = self.get_ray_dir()  # cuda

        # cal reference plane normal vector: unit vector (query_rays_size, 3)
        ref_plane_norm = torch.cross(query_rays_dir, query_key_org_vec)  # cuda

        # calculate cos of key_rays to reference plane: (query_rays_size, key_rays_size)
        key_rays_to_ref_plane = torch.matmul(ref_plane_norm, key_rays_dir.T)  # cuda

        # get intersection rays
        dvg_ang_v = cfg['dvg_ang_v']  # rad
        ang_thrd = dvg_ang_v / 2
        ray_ints_mask = torch.logical_and(key_rays_to_ref_plane >= np.cos(np.pi/2 + ang_thrd),
                                          key_rays_to_ref_plane <= np.cos(np.pi/2 - ang_thrd))

        # TODO: filter self observation rays that aim to other lidars (no need)
        # query_key_org_vec = self.get_org_vec(query_rays.get_ray_start(), 1)  # cuda
        # self_obs_ang = torch.matmul(query_rays_dir, query_key_org_vec.T)  # cuda
        # self_obs_mask = self_obs_ang >= np.cos(ang_thrd)  # cuda
        # self_obs_idx = torch.where(self_obs_mask)
        # self_obs_mask = self_obs_mask.repeat(1, self.get_ray_size())  # cuda
        # ray_ints_mask = torch.logical_and(ray_ints_mask, ~self_obs_mask)

        # ray index
        ray_ints_idx = torch.where(ray_ints_mask)  # cuda
        query_rays_ints_idx = ray_ints_idx[0]  # cuda
        key_rays_ints_idx = ray_ints_idx[1]  # cuda

        # clear cuda memory
        del ray_ints_mask, key_rays_to_ref_plane, ref_plane_norm, key_rays_dir, query_rays_dir, query_key_org_vec, ray_ints_idx
        torch.cuda.empty_cache()
        return query_rays_ints_idx, key_rays_ints_idx


class QueryRays(object):
    def __init__(self, org_query, pts_query, ts_query):
        self.ray_start = org_query
        self.ray_end = pts_query
        self.ray_size = len(self.ray_end)
        self.ray_ts = ts_query  # TODO: could be a timestamp for each ray (time compensation)
        self.rays_to_ints_pts_dict = dict()

        # for statistics
        self.num_samples_per_ray = []
        self.occ_pct_per_ray = []
        self.free_pct_per_ray = []
        self.unk_pct_per_ray = []

    def get_ray_start(self):
        return self.ray_start

    def get_ray_end(self):
        return self.ray_end

    def get_ray_dir(self):
        return F.normalize(self.ray_end - self.ray_start, p=2, dim=1)  # unit vector

    def get_ray_size(self):
        return self.ray_size

    def get_ray_ts(self):
        return self.ray_ts

    def get_ray_depth(self, ray_pts):
        return torch.linalg.norm(ray_pts - self.ray_start, dim=1, keepdim=False)

    def get_org_vec(self, org_key, ray_size):
        # unit vec: from key org to query org
        return torch.broadcast_to(F.normalize(self.ray_start - org_key, p=2, dim=0), (ray_size, 3))

    def cal_ints_points(self, cfg, key_rays_list: list):
        depth_list = []
        labels_list = []
        confidence_list = []
        query_rays_idx_list = []
        key_rays_idx_list = []
        key_meta_info_list = []
        for key_sensor_idx, key_rays in enumerate(key_rays_list):
            ######################################## intersection points ###############################################
            # query org to key org vector
            query_key_org_vec = key_rays.get_org_vec(self.get_ray_start(), self.get_ray_size())  # cuda
            query_rays_dir = self.get_ray_dir()  # cuda
            key_rays_dir = key_rays.get_ray_dir()  # cuda

            # cal reference plane normal vector: unit vector (query_rays_size, 3)
            ref_plane_norm = torch.cross(query_rays_dir, query_key_org_vec)  # cuda

            # calculate cos of key_rays to reference plane: (query_rays_size, key_rays_size)
            key_rays_to_ref_plane = torch.matmul(ref_plane_norm, key_rays_dir.T)  # cuda

            # get intersection rays
            dvg_ang_v = cfg['dvg_ang_v']  # rad
            ray_ints_mask = torch.logical_and(key_rays_to_ref_plane >= np.cos(np.pi / 2 + dvg_ang_v / 2),
                                              key_rays_to_ref_plane <= np.cos(np.pi / 2 - dvg_ang_v / 2))

            # intersection ray index
            ray_ints_idx = torch.where(ray_ints_mask)  # cuda
            query_rays_ints_idx = ray_ints_idx[0]  # cuda
            key_rays_ints_idx = ray_ints_idx[1]  # cuda

            # common perpendicular line
            ints_query_rays_dir = query_rays_dir[query_rays_ints_idx]  # cuda
            ints_key_rays_dir = key_rays_dir[key_rays_ints_idx]  # cuda
            com_norm = torch.cross(ints_query_rays_dir, ints_key_rays_dir)  # cuda

            # intersection points
            rays_org_vec = torch.broadcast_to(key_rays.get_ray_start() - self.ray_start, (len(query_rays_ints_idx), 3))  # cuda
            q = (torch.sum(torch.cross(rays_org_vec, ints_key_rays_dir) * com_norm, dim=1) / torch.sum(com_norm * com_norm, dim=1))  # cuda
            k = (torch.sum(torch.cross(rays_org_vec, ints_query_rays_dir) * com_norm, dim=1) / torch.sum(com_norm * com_norm, dim=1))  # cuda
            valid_ints_mask = torch.logical_and(torch.logical_and(q >= 0, q <= 2 * cfg['max_range']),
                                                torch.logical_and(k >= 0, k <= 2 * cfg['max_range']))  # cuda
            query_rays_ints_idx = query_rays_ints_idx[valid_ints_mask]  # cuda
            key_rays_ints_idx = key_rays_ints_idx[valid_ints_mask]  # cuda
            q_valid = q[valid_ints_mask]  # cuda
            k_valid = k[valid_ints_mask]  # cuda

            # calculate occupancy label of the ints pts (0: unknown, 1: free, 2:occupied) TODO: with measurement confidence
            key_ray_depth = key_rays.get_ray_depth(key_rays.get_ray_end()[key_rays_ints_idx])  # cuda
            ints_labels, ints_confidence, ints_confident_mask = get_occupancy_label(ray_depth=key_ray_depth, point_depth=k_valid)
            ints_depth = q_valid[ints_confident_mask]
            query_rays_ints_idx = query_rays_ints_idx[ints_confident_mask]
            key_rays_ints_idx = key_rays_ints_idx[ints_confident_mask]

            # clear cuda memory
            del k_valid, q_valid, valid_ints_mask, k, q, com_norm, ints_key_rays_dir, ints_query_rays_dir, key_rays_to_ref_plane, ref_plane_norm, rays_org_vec, query_key_org_vec
            torch.cuda.empty_cache()
            ############################################################################################################

            ########################################## parallel points #################################################
            # TODO: isosceles triangle approximation (point to line distance -> base length)
            dvg_ang_h = cfg['dvg_ang_h']
            ray_ang = torch.matmul(self.get_ray_dir(), key_rays.get_ray_dir().T)  # cuda
            ray_ang = torch.clip(ray_ang, min=-1, max=1)  # TODO: inner product of two unit vector > 1...
            ray_para_mask = ray_ang >= np.cos(dvg_ang_h)  # cuda
            ray_para_mask = torch.logical_and(ray_ints_mask, ray_para_mask)
            ray_para_idx = torch.where(ray_para_mask)
            query_rays_para_idx = ray_para_idx[0]
            key_rays_para_idx = ray_para_idx[1]

            # common perpendicular line
            para_query_rays_dir = query_rays_dir[query_rays_para_idx]  # cuda
            para_key_rays_dir = key_rays_dir[key_rays_para_idx]  # cuda
            para_com_norm = torch.cross(para_query_rays_dir, para_key_rays_dir)  # cuda

            # para rays intersection points
            num_para_rays = len(para_com_norm)
            rays_org_vec = torch.broadcast_to(key_rays.get_ray_start() - self.ray_start, (num_para_rays, 3))  # cuda
            com_norm_dot_prod = torch.sum(para_com_norm * para_com_norm, dim=1)
            q_para = (torch.sum(torch.cross(rays_org_vec, para_key_rays_dir) * para_com_norm, dim=1) / com_norm_dot_prod)  # cuda
            k_para = (torch.sum(torch.cross(rays_org_vec, para_query_rays_dir) * para_com_norm, dim=1) / com_norm_dot_prod)  # cuda
            valid_para_mask = torch.sign(q_para) * torch.sign(k_para) > 0
            q_para = q_para[valid_para_mask]
            k_para = k_para[valid_para_mask]
            query_rays_para_idx = query_rays_para_idx[valid_para_mask]
            key_rays_para_idx = key_rays_para_idx[valid_para_mask]

            # TODO: inverse solution, leg length threshold
            fw_para_mask = torch.logical_and(q_para >= 0, k_para >= 0)  # para rays intersect at forward side
            bw_para_mask = torch.logical_and(q_para < 0, k_para < 0)  # para rays intersect at backward side

            # where the intersection begins (at query ray)
            ang_cos = ray_ang[tuple((query_rays_para_idx, key_rays_para_idx))]
            sin_half_ang = torch.sqrt((1 - ang_cos) / 2)
            tan_half_ang = np.tan(dvg_ang_h / 2)
            x_q_min = torch.zeros(len(q_para)).cuda()
            x_q_min[fw_para_mask] = ((q_para[fw_para_mask] * (sin_half_ang[fw_para_mask] + tan_half_ang) - k_para[fw_para_mask] * tan_half_ang) /
                                   (sin_half_ang[fw_para_mask] + 2 * tan_half_ang))
            x_q_min[bw_para_mask] = ((q_para[bw_para_mask] * (-sin_half_ang[bw_para_mask] + tan_half_ang) - k_para[bw_para_mask] * tan_half_ang) /
                                   (-sin_half_ang[bw_para_mask] + 2 * tan_half_ang))
            x_k_min = k_para - q_para + x_q_min

            valid_para_mask = torch.logical_and(torch.logical_and(x_k_min >= 0, x_k_min <= 2 * cfg['max_range']),
                                                                                x_q_min <= 2 * cfg['max_range'])
            q_para = q_para[valid_para_mask]
            k_para = k_para[valid_para_mask]
            query_rays_para_idx = query_rays_para_idx[valid_para_mask]
            key_rays_para_idx = key_rays_para_idx[valid_para_mask]
            x_q_min = x_q_min[valid_para_mask]
            x_k_min = x_k_min[valid_para_mask]

            # select samples
            query_ray_depth = self.get_ray_depth(self.get_ray_end())[query_rays_para_idx]
            key_ray_depth = key_rays.get_ray_depth(key_rays.get_ray_end())[key_rays_para_idx]
            x_q = torch.cat((query_ray_depth, query_ray_depth / 2, (query_ray_depth + x_q_min) / 2), dim=0)
            x_k = torch.cat((key_ray_depth, key_ray_depth / 2, (key_ray_depth + x_k_min) / 2), dim=0)
            x_k_prime = x_q + (k_para - q_para).repeat(3)
            x_q_prime = x_k + (q_para - k_para).repeat(3)
            x_q = torch.cat((x_q, x_q_prime), dim=0)
            x_k = torch.cat((x_k, x_k_prime), dim=0)
            valid_mask = x_q >= 0

            # save labels
            para_labels, para_confidence, para_confident_mask = get_occupancy_label(ray_depth=key_ray_depth.repeat(6)[valid_mask], point_depth=x_k[valid_mask])
            para_depth = x_q[valid_mask][para_confident_mask]
            query_rays_para_idx = query_rays_para_idx.repeat(6)[valid_mask][para_confident_mask]
            key_rays_para_idx = key_rays_para_idx.repeat(6)[valid_mask][para_confident_mask]

            ######################################## parallel rays #####################################################
            # # TODO: parallel rays (intersection rays contain parallel rays)
            # query_rays_para_idx = query_rays_para_idx_list[key_sensor_idx]  # cuda
            # key_rays_para_idx = key_rays_para_idx_list[key_sensor_idx]  # cuda
            # para_query_rays_dir = self.get_ray_dir()[query_rays_para_idx]  # cuda
            # para_key_rays_pts = key_rays.get_ray_end()[key_rays_para_idx]  # cuda
            # proj_para_depth = torch.sum(para_key_rays_pts * para_query_rays_dir, dim=1)  # cuda
            # proj_para_pts = torch.mul(proj_para_depth.reshape(-1, 1), para_query_rays_dir)  # cuda
            # proj_depth_residual = proj_para_depth - self.get_ray_depth(self.ray_end[query_rays_para_idx])  # cuda
            #
            # # TODO: logic 1, if depth residual almost 0
            # para_same_mask = torch.logical_and(proj_depth_residual >= -cfg['max_dis_error'], proj_depth_residual <= cfg['max_dis_error'])  # cuda
            # query_rays_para_same_idx = query_rays_para_idx[torch.where(para_same_mask)]  # cuda
            # key_rays_para_same_idx = key_rays_para_idx[torch.where(para_same_mask)]  # cuda
            # num_para_same_rays = len(query_rays_para_same_idx)  # cuda
            # ints_pts_para_same = proj_para_pts[para_same_mask]  # cuda
            # ints_labels_para_same = torch.full((num_para_same_rays, 1), 2).cuda()  # cuda
            #
            # # TODO: logic 2, if depth residual far more than 0
            # para_valid_mask = proj_depth_residual > cfg['max_dis_error']  # cuda
            # query_rays_para_valid_idx = query_rays_para_idx[torch.where(para_valid_mask)]  # cuda
            # key_rays_para_valid_idx = key_rays_para_idx[torch.where(para_valid_mask)]  # cuda
            # num_para_valid_rays = len(query_rays_para_valid_idx)  # cuda
            # ints_pts_para_valid_occ = proj_para_pts[para_valid_mask]  # cuda
            # ints_labels_para_valid_occ = torch.full((num_para_valid_rays, 1), 2).cuda()  # cuda
            # ints_pts_para_valid_free = (proj_para_pts[para_valid_mask] + self.ray_end[query_rays_para_valid_idx]) / 2  # cuda
            # ints_labels_para_valid_free = torch.full((num_para_valid_rays, 1), 1).cuda()  # cuda
            #
            # # TODO: logic 3, if depth residual far less than 0 -> unknown label, not used now
            ############################################################################################################

            # dictionary: query ray index -> points list
            depth = torch.cat((ints_depth, para_depth), dim=0).to(torch.float16)  # depth on query ray
            labels = torch.cat((ints_labels, para_labels), dim=0).to(torch.int8)  # occupancy labels
            confidence = torch.cat((ints_confidence, para_confidence), dim=0).to(torch.float16)  # confidence of the labels
            query_rays_idx = torch.cat((query_rays_ints_idx, query_rays_para_idx), dim=0).to(torch.int8)
            key_rays_idx = torch.cat((key_rays_ints_idx, key_rays_para_idx), dim=0).to(torch.int8)

            # append to list
            depth_list.append(depth)
            labels_list.append(labels)
            confidence_list.append(confidence)
            query_rays_idx_list.append(query_rays_idx)
            key_rays_idx_list.append(key_rays_idx)
            key_meta_info_list.append([key_sensor_idx, len(labels)])

            # clear cuda memory
            del ray_ints_mask, ray_para_mask, key_rays_idx, query_rays_idx, confidence, labels, depth
            torch.cuda.empty_cache()

        if len(labels_list) == 0:
            return None
        else:
            depth = torch.cat(depth_list, dim=0)
            labels = torch.cat(labels_list, dim=0)
            confidence = torch.cat(confidence_list, dim=0)
            query_rays_idx = torch.cat(query_rays_idx_list, dim=0)
            key_rays_idx = torch.cat(key_rays_idx_list, dim=0)
            key_meta_info = torch.tensor(key_meta_info_list).to(torch.int8)

            # statistics: for histogram
            self.num_samples_per_ray.append(len(labels))
            num_unk = torch.sum(labels == 0)
            num_free = torch.sum(labels == 1)
            num_occ = torch.sum(labels == 2)

            # statistics: occ percentage per ray
            self.unk_pct_per_ray.append((num_unk / len(labels)) * 100)
            self.free_pct_per_ray.append((num_free / len(labels)) * 100)
            self.occ_pct_per_ray.append((num_occ / len(labels)) * 100)
            return depth, labels, confidence, query_rays_idx, key_rays_idx, key_meta_info


if __name__ == '__main__':
    # mode
    parser = argparse.ArgumentParser()
    parser.add_argument('--autodl', type=bool, default=False)
    args = parser.parse_args()

    # load nusc dataset
    if args.autodl:
        nusc = NuScenes(dataroot="/root/autodl-tmp/Datasets/nuScenes", version="v1.0-trainval")
    else:
        nusc = NuScenes(dataroot="/home/user/Datasets/nuScenes", version="v1.0-trainval")

    with open('configs/ours.yaml', 'r') as f:
        cfg = yaml.safe_load(f)['bg_pretrain']

    # get key sample data tokens
    sample_toks_all = [sample['token'] for sample in nusc.sample]
    key_sd_toks_dict = get_key_sd_toks_dict(nusc, sample_toks_all, cfg)

    #
    num_valid_samples = 0
    for query_sample_idx, query_sample_tok in tqdm(enumerate(key_sd_toks_dict.keys())):
        key_sd_toks_list = key_sd_toks_dict[query_sample_tok]

        # get rays
        query_sample = nusc.get('sample', query_sample_tok)
        query_sd_tok = query_sample['data']['LIDAR_TOP']
        query_rays, key_rays_list = load_rays(nusc, cfg, query_sd_tok, key_sd_toks_list)  # cuda

        # calculate intersection points
        depth, labels, confidence, query_rays_idx, key_rays_idx, key_meta_info = query_rays.cal_ints_points(cfg, key_rays_list)

        if labels is not None:
            num_valid_samples += 1
            labels_folder = os.path.join(nusc.dataroot, 'labels_cuda', nusc.version)
            os.makedirs(labels_folder, exist_ok=True)
            torch.save(depth.clone(), os.path.join(labels_folder, query_sd_tok + "_depth.pt"))
            torch.save(labels.clone(), os.path.join(labels_folder, query_sd_tok + "_labels.pt"))
            torch.save(confidence.clone(), os.path.join(labels_folder, query_sd_tok + "_confidence.pt"))
            torch.save(query_rays_idx.clone(), os.path.join(labels_folder, query_sd_tok + "_query_rays_idx.pt"))

            key_rays_folder = os.path.join(nusc.dataroot, 'labels_key_rays_cuda', nusc.version)
            os.makedirs(key_rays_folder, exist_ok=True)
            torch.save(key_rays_idx.clone(), os.path.join(key_rays_folder, query_sd_tok + "_key_rays_idx.pt"))
            torch.save(key_meta_info.clone(), os.path.join(key_rays_folder, query_sd_tok + "_key_meta_info.pt"))

            # test load
            # a = torch.load(bg_samples_path, map_location=torch.device('cpu'))

            # clear cuda memory
            del depth, labels, confidence, query_rays_idx, key_rays_idx, key_meta_info, query_rays, key_rays_list
            torch.cuda.empty_cache()
        else:
            print(f"Sample data tok {query_sd_tok}, index {query_sample_idx} do not have valid background points")
    print(f"Number of valid samples: {num_valid_samples}")

    # TODO: torch.where(condition) is identical to torch.nonzero(condition, as_tuple=True)
    # TODO: matmul will cost too much gpu memory
    # #################### calculate ray intersection points: solution-1 ####################
    # vis_list = []
    # for vis_point_idx in range(len(ints_rays_idx[0])):
    #     P1 = points_query[ints_rays_idx[0][vis_point_idx]]
    #     D1 = F.normalize(P1 - origin_query, p=2, dim=0)
    #     P2 = points[ints_rays_idx[1][vis_point_idx]]
    #     D2 = F.normalize(P2 - origin, p=2, dim=0)
    #     N = torch.cross(D1, D2)  # common perpendicular line
    #     N_norm = N / torch.linalg.norm(N)
    #     # distance
    #     distance = torch.abs(torch.dot((P2 - P1), N_norm))  # length of the common perpendicular line
    #     # calculate foot drop
    #     t = torch.dot(torch.cross(P2 - P1, D2), N) / torch.dot(N, N)
    #     s = torch.dot(torch.cross(P2 - P1, D1), N) / torch.dot(N, N)
    #     if t > 0:
    #         vis_list.append(vis_point_idx)
    #         Pt = P1 + t * D1
    #         Ps = P2 + s * D2
    # #######################################################################################
    # #################### calculate ray intersection points: solution-2 ####################
    # P1 = points_query[intersection_ray_idx[0][vis_point_idx]]
    # D1 = F.normalize(P1 - origin_query, p=2, dim=0)
    # P2 = points[intersection_ray_idx[1][vis_point_idx]]
    # D2 = F.normalize(P2 - origin, p=2, dim=0)
    # a = torch.dot(D2, D1)
    # b = torch.dot(D1, D1)
    # c = torch.dot(D2, D2)
    # d = torch.dot(P2 - P1, D1)
    # e = torch.dot(P2 - P1, D2)
    # t_prime = (a * e - c * d) / (a * a - b * c)
    # s_prime = b / a * t_prime - d / a
    # Pt_prime = P1 + t_prime * D1
    # Ps_prime = P2 + s_prime * D2
    # #######################################################################################