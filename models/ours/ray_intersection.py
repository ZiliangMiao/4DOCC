import os
import time
from typing import List

import open3d.visualization
import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.geometry_utils import points_in_box
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import argparse
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
    org_query, pts_query, ts_query, query_valid_mask = get_transformed_pcd(nusc, cfg, query_sd_tok, query_sd_tok)
    query_rays = QueryRays(org_query, pts_query, ts_query)

    # get key rays
    key_rays_list = []
    query_rays_ints_idx_list, key_rays_ints_idx_list, query_rays_para_idx_list, key_rays_para_idx_list = [], [], [], []
    for key_idx, key_sd_tok in enumerate(key_sd_toks_list):
        org_key, pts_key, ts_key, key_valid_mask = get_transformed_pcd(nusc, cfg, query_sd_tok, key_sd_tok)
        key_rays = KeyRays(org_key, pts_key, ts_key)
        key_rays_list.append(key_rays)
        query_rays_ints_idx, key_rays_ints_idx, query_rays_para_idx, key_rays_para_idx = key_rays.find_ints_rays(cfg, query_rays)

        # append
        query_rays_ints_idx_list.append(query_rays_ints_idx.cpu())
        key_rays_ints_idx_list.append(key_rays_ints_idx.cpu())
        query_rays_para_idx_list.append(query_rays_para_idx.cpu())
        key_rays_para_idx_list.append(key_rays_para_idx.cpu())

    # clear cuda memory
    if cfg['cuda']:
        del query_rays_ints_idx, key_rays_ints_idx, query_rays_para_idx, key_rays_para_idx,
        torch.cuda.empty_cache()
    return query_rays, key_rays_list, query_rays_ints_idx_list, key_rays_ints_idx_list, query_rays_para_idx_list, key_rays_para_idx_list


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

    def find_ints_rays(self, cfg, query_rays):
        # query org to key org vector
        query_key_org_vec = self.get_org_vec(query_rays.get_ray_start(), query_rays.get_ray_size())
        query_rays_dir = query_rays.get_ray_dir()
        key_rays_dir = self.get_ray_dir()
        if cfg['cuda']:
            query_key_org_vec = query_key_org_vec.cuda()  # cuda: unit vector
            query_rays_dir = query_rays_dir.cuda()  # cuda
            key_rays_dir = key_rays_dir.cuda()  # cuda

        # cal reference plane normal vector: unit vector (rays_size, 3)
        ref_plane_norm = torch.cross(query_rays_dir, query_key_org_vec)  # cuda

        # calculate cos of key_rays to reference plane: (query_rays_size, key_rays_size)
        key_rays_to_ref_plane = torch.matmul(ref_plane_norm, key_rays_dir.T)  # cuda

        # get intersection rays
        deg_thrd = np.rad2deg(cfg['max_dis_error'] / cfg['max_range'])  # cpu
        ray_ints_mask = torch.abs(key_rays_to_ref_plane) <= torch.cos(torch.deg2rad(torch.tensor(90 - deg_thrd)))  # cuda
        ray_ints_idx = torch.where(ray_ints_mask)  # cuda

        # get small angle rays (nearly parallel)
        rays_ang = torch.matmul(query_rays_dir, key_rays_dir.T)  # cuda
        small_ang_mask = rays_ang >= torch.cos(torch.deg2rad(torch.tensor(deg_thrd)))  # cuda
        small_ang_mask = torch.logical_and(small_ang_mask, ray_ints_mask)  # cuda
        small_ang_idx = torch.where(small_ang_mask)  # cuda

        # TODO: filter rays that aim to other lidars (self observation)
        query_key_org_vec = self.get_org_vec(query_rays.get_ray_start(), 1)
        if cfg['cuda']:
            query_key_org_vec = query_key_org_vec.cuda()  # cuda
        self_obs_ang = torch.matmul(query_rays_dir, query_key_org_vec.T)  # query org -> key org
        self_obs_mask = self_obs_ang >= torch.cos(torch.deg2rad(torch.tensor(deg_thrd)))
        self_obs_idx = torch.where(self_obs_mask)

        # return
        query_rays_ints_idx = ray_ints_idx[0]  # cuda
        key_rays_ints_idx = ray_ints_idx[1]  # cuda
        query_rays_para_idx = small_ang_idx[0]  # cuda
        key_rays_para_idx = small_ang_idx[1]  # cuda

        # clear cuda memory
        if cfg['cuda']:
            del small_ang_mask, rays_ang, ray_ints_mask, key_rays_to_ref_plane, ref_plane_norm, key_rays_dir, query_rays_dir, query_key_org_vec, self_obs_ang, self_obs_mask
            torch.cuda.empty_cache()
        return query_rays_ints_idx, key_rays_ints_idx, query_rays_para_idx, key_rays_para_idx


class QueryRays(object):
    def __init__(self, org_query, pts_query, ts_query):
        self.ray_start = org_query
        self.ray_end = pts_query
        self.ray_size = len(self.ray_end)
        self.ray_ts = ts_query  # TODO: could be a timestamp for each ray (time compensation)
        self.rays_to_ints_pts_dict = dict()

        # for statistics
        self.num_bg_samples_per_ray = []
        self.num_occ_percentage_per_ray = []
        self.num_unk_free_occ_per_scan = []

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

    def cal_ints_points(self, cfg, key_rays_list: list, query_rays_ints_idx_list: list, key_rays_ints_idx_list: list, query_rays_para_idx_list: list, key_rays_para_idx_list: list):
        query_ray_to_bg_samples_dict = defaultdict(list)
        query_ray_to_key_rays_dict = defaultdict(list)
        for key_sensor_idx, key_rays in enumerate(key_rays_list):
            # intersection rays
            query_rays_ints_idx = query_rays_ints_idx_list[key_sensor_idx]  # cpu
            key_rays_ints_idx = key_rays_ints_idx_list[key_sensor_idx]  # cpu

            # common perpendicular line
            ints_query_rays = self.get_ray_dir()[query_rays_ints_idx]
            ints_key_rays = key_rays.get_ray_dir()[key_rays_ints_idx]
            if cfg['cuda']:
                ints_query_rays = ints_query_rays.cuda()  # cuda
                ints_key_rays = ints_key_rays.cuda()  # cuda
            com_norm = torch.cross(ints_query_rays, ints_key_rays)  # cuda

            # calculate intersection points
            num_ints_rays = len(query_rays_ints_idx)
            rays_org_vec = torch.broadcast_to(key_rays.get_ray_start() - self.ray_start, (num_ints_rays, 3))
            if cfg['cuda']:
                rays_org_vec = rays_org_vec.cuda()  # cuda
            q = (torch.sum(torch.cross(rays_org_vec, ints_key_rays) * com_norm, dim=1) / torch.sum(com_norm * com_norm, dim=1)).cpu()
            k = (torch.sum(torch.cross(rays_org_vec, ints_query_rays) * com_norm, dim=1) / torch.sum(com_norm * com_norm, dim=1)).cpu()
            query_bg_mask = q >= self.get_ray_depth(self.ray_end[query_rays_ints_idx])  # cpu
            key_same_dir_mask = k >= cfg['occ_thrd']  # cpu
            valid_ints_mask = torch.logical_and(query_bg_mask, key_same_dir_mask)  # cpu
            query_rays_ints_idx = query_rays_ints_idx[valid_ints_mask]  # cpu
            key_rays_ints_idx = key_rays_ints_idx[valid_ints_mask]  # cpu
            q = q[valid_ints_mask]  # cpu
            ints_pts = self.ray_start + q.reshape(-1, 1) * self.get_ray_dir()[query_rays_ints_idx]  # cpu

            # calculate occupancy label of the ints pts (0: unknown, 1: free, 2:occupied)
            bg_labels = torch.zeros(len(ints_pts))  # cpu
            key_ray_depth = key_rays.get_ray_depth(key_rays.get_ray_end()[key_rays_ints_idx])  # cpu
            key_ints_depth = key_rays.get_ray_depth(ints_pts)  # cpu
            key_ints_depth_res = key_ints_depth - key_ray_depth  # cpu
            free_mask = key_ints_depth_res < 0  # cpu
            bg_labels[free_mask] = 1  # cpu
            occ_mask = torch.logical_and(key_ints_depth_res >= 0, key_ints_depth_res <= cfg['occ_thrd'])  # cpu
            bg_labels[occ_mask] = 2  # cpu

            # valid intersection points: used for save or vis
            ints_valid_mask = torch.logical_or(free_mask, occ_mask)  # cpu
            query_rays_ints_valid_idx = query_rays_ints_idx[ints_valid_mask]  # cpu
            key_rays_ints_valid_idx = key_rays_ints_idx[ints_valid_mask]  # cpu
            ints_pts_valid = ints_pts[ints_valid_mask]  # cpu
            ints_labels_valid = bg_labels[ints_valid_mask].reshape(-1, 1)  # cpu
            num_ints_valid_rays = len(ints_pts_valid)  # cpu

            # TODO: parallel rays (intersection rays contain parallel rays)
            query_rays_para_idx = query_rays_para_idx_list[key_sensor_idx]  # cpu
            key_rays_para_idx = key_rays_para_idx_list[key_sensor_idx]  # cpu
            para_query_rays = self.get_ray_dir()[query_rays_para_idx]
            para_key_rays_pts = key_rays.get_ray_end()[key_rays_para_idx]
            if cfg['cuda']:
                para_query_rays = para_query_rays.cuda()  # cuda
                para_key_rays_pts = para_key_rays_pts.cuda()  # cuda
            proj_para_depth = torch.sum(para_key_rays_pts * para_query_rays, dim=1)  # cuda, torch不能对二维tensor求点积, 对应位置元素相乘再相加
            proj_para_pts = torch.mul(proj_para_depth.reshape(-1, 1), para_query_rays).cpu()  # cpu
            proj_depth_residual = proj_para_depth.cpu() - self.get_ray_depth(self.ray_end[query_rays_para_idx])  # cpu

            # TODO: logic 1, if depth residual almost 0
            para_same_mask = torch.logical_and(proj_depth_residual >= -cfg['max_dis_error'], proj_depth_residual <= cfg['max_dis_error'])  # cpu
            query_rays_para_same_idx = query_rays_para_idx[torch.where(para_same_mask)]  # cpu
            key_rays_para_same_idx = key_rays_para_idx[torch.where(para_same_mask)]  # cpu
            num_para_same_rays = len(query_rays_para_same_idx)  # cpu
            ints_pts_para_same = proj_para_pts[para_same_mask]  # cpu
            ints_labels_para_same = torch.full((num_para_same_rays, 1), 2)  # cpu

            # TODO: logic 2, if depth residual far more than 0
            para_valid_mask = proj_depth_residual > cfg['max_dis_error']  # cpu
            query_rays_para_valid_idx = query_rays_para_idx[torch.where(para_valid_mask)]  # cpu
            key_rays_para_valid_idx = key_rays_para_idx[torch.where(para_valid_mask)]  # cpu
            num_para_valid_rays = len(query_rays_para_valid_idx)  # cpu
            ints_pts_para_valid_occ = proj_para_pts[para_valid_mask]  # cpu
            ints_labels_para_valid_occ = torch.full((num_para_valid_rays, 1), 2)  # cpu
            ints_pts_para_valid_free = (proj_para_pts[para_valid_mask] + self.ray_end[query_rays_para_valid_idx]) / 2  # cpu
            ints_labels_para_valid_free = torch.full((num_para_valid_rays, 1), 1)  # cpu

            # TODO: logic 3, if depth residual far less than 0 -> unknown label, not used now

            # save labels
            bg_pts = torch.cat((ints_pts_valid, ints_pts_para_same, ints_pts_para_valid_occ, ints_pts_para_valid_free),dim=0)  # cpu
            bg_ts = torch.full((num_ints_valid_rays + num_para_same_rays + num_para_valid_rays * 2, 1), key_rays.get_ray_ts())  # cpu
            bg_labels = torch.cat((ints_labels_valid, ints_labels_para_same, ints_labels_para_valid_occ, ints_labels_para_valid_free), dim=0)  # cpu
            bg_samples = torch.cat((bg_pts, bg_ts, bg_labels), dim=1)

            # statistics
            num_unk = torch.sum(bg_labels == 0)
            num_free = torch.sum(bg_labels == 1)
            num_occ = torch.sum(bg_labels == 2)

            # dictionary: query ray index -> intersection points list
            query_rays_idx = torch.cat((query_rays_ints_valid_idx, query_rays_para_same_idx, query_rays_para_valid_idx, query_rays_para_valid_idx)).numpy()
            bg_key_rays_idx = torch.cat((key_rays_ints_valid_idx, key_rays_para_same_idx, key_rays_para_valid_idx, key_rays_para_valid_idx)).numpy()
            for query_ray_idx, key_ray_idx, bg_sample in zip(query_rays_idx, bg_key_rays_idx, bg_samples):
                query_ray_to_bg_samples_dict[query_ray_idx].append(bg_sample)
                query_ray_to_key_rays_dict[query_ray_idx].append(torch.tensor([key_sensor_idx, key_ray_idx]))

            # clear cuda memory
            if cfg['cuda']:
                del para_key_rays_pts, para_query_rays, k, q, rays_org_vec, com_norm, ints_key_rays, ints_query_rays
                torch.cuda.empty_cache()

        if len(query_ray_to_bg_samples_dict) == 0:
            return np.empty((0, 2)), np.empty((0, 3))
        else:
            ray_samples_save = []
            bg_samples_save = []
            bg_key_rays_save = []
            for query_ray_idx in query_ray_to_bg_samples_dict.keys():
                # bg samples: [x, y, z, ts, occ_label]
                bg_samples_list = query_ray_to_bg_samples_dict[query_ray_idx]
                bg_samples = torch.stack(bg_samples_list)
                bg_samples_save.append(bg_samples)

                # query ray samples: [query ray idx, number of background points]
                ray_samples_save.append([query_ray_idx, len(bg_samples)])

                # key rays samples: [key sensor idx, key ray idx]
                bg_key_rays_list = query_ray_to_key_rays_dict[query_ray_idx]
                bg_key_rays = torch.stack(bg_key_rays_list)
                bg_key_rays_save.append(bg_key_rays)

                # statistics: for histogram
                self.num_bg_samples_per_ray.append(len(bg_samples))
                num_occ_ray = torch.sum(bg_samples[:, -1] == 2)
                num_free_ray = torch.sum(bg_samples[:, -1] == 1)
                # statistics: occ percentage per ray
                self.num_occ_percentage_per_ray.append((num_occ_ray / (num_occ_ray + num_free_ray)) * 100)

            # statistics: num of unk, free, occ in a scan
            self.num_unk_free_occ_per_scan.append(torch.tensor([num_unk, num_free, num_occ]))

            # save
            ray_samples_save = np.vstack(ray_samples_save)
            bg_samples_save = torch.vstack(bg_samples_save).numpy()
            bg_key_rays_save = torch.vstack(bg_key_rays_save).numpy()
            return ray_samples_save, bg_samples_save, bg_key_rays_save


if __name__ == '__main__':
    # load nusc dataset
    nusc = NuScenes(dataroot="/home/user/Datasets/nuScenes", version="v1.0-trainval")
    with open('../../configs/ours.yaml', 'r') as f:
        cfg = yaml.safe_load(f)['bg_pretrain']

    # get key sample data tokens
    sample_toks_all = [sample['token'] for sample in nusc.sample]
    key_sd_toks_dict = get_key_sd_toks_dict(nusc, sample_toks_all, cfg)

    num_valid_samples = 0
    for query_sample_idx, query_sample_tok in tqdm(enumerate(key_sd_toks_dict.keys())):
        key_sd_toks_list = key_sd_toks_dict[query_sample_tok]

        # get rays
        query_sample = nusc.get('sample', query_sample_tok)
        query_sd_tok = query_sample['data']['LIDAR_TOP']
        query_rays, key_rays_list, query_rays_ints_idx_list, key_rays_ints_idx_list, query_rays_para_idx_list, key_rays_para_idx_list = load_rays(nusc, cfg, query_sd_tok, key_sd_toks_list)

        # calculate intersection points
        ray_samples_save, bg_samples_save, bg_key_rays_save = query_rays.cal_ints_points(cfg, key_rays_list,
                                                                                         query_rays_ints_idx_list,
                                                                                         key_rays_ints_idx_list,
                                                                                         query_rays_para_idx_list,
                                                                                         key_rays_para_idx_list)
        assert np.sum(ray_samples_save[:, -1]) == len(bg_samples_save)
        if bg_samples_save is not None:
            num_valid_samples += 1
            bg_label_dir = os.path.join(nusc.dataroot, 'bg_labels', nusc.version)
            os.makedirs(bg_label_dir, exist_ok=True)
            bg_samples_path = os.path.join(bg_label_dir, query_sd_tok + "_bg_samples.label")
            bg_samples_save.tofile(bg_samples_path)
            ray_samples_path = os.path.join(bg_label_dir, query_sd_tok + "_ray_samples.label")
            ray_samples_save.tofile(ray_samples_path)
            bg_key_rays_path = os.path.join(nusc.dataroot, 'bg_labels_key_rays', nusc.version,
                                            query_sd_tok + "_key_rays.label")
            bg_key_rays_save.tofile(bg_key_rays_path)
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