import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import datasets.kitti_utils as kitti_utils
import matplotlib.pyplot as plt


def load_rays(cfg, path_to_seq, query_scan_idx, key_scans_list):
    # get query rays
    org_query, pts_query, ts_query, query_valid_mask = kitti_utils.get_transformed_pcd(cfg, path_to_seq,
                                                                                       query_scan_idx, query_scan_idx)  # cpu
    # TODO: uniform downsample
    pts_query_ds = pts_query[::cfg['uni_ds']]
    # construct query rays
    query_rays = QueryRays(org_query.cuda(), pts_query_ds.cuda(), ts_query)  # cuda

    # get key rays
    key_rays_list = []
    for key_scan_idx in key_scans_list:
        org_key, pts_key, ts_key, key_valid_mask = kitti_utils.get_transformed_pcd(cfg, path_to_seq,
                                                                                   query_scan_idx, key_scan_idx)  # cpu
        # TODO: uniform downsample
        pts_key_ds = pts_key[::cfg['uni_ds']]
        # construct key rays
        key_rays = KeyRays(org_key.cuda(), pts_key_ds.cuda(), ts_key)  # cuda
        key_rays_list.append(key_rays)  # cuda
    return query_rays, key_rays_list


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
        dvg_ang = cfg['dvg_ang']  # rad
        ang_thrd = dvg_ang / 2
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
        self.num_valid_rays_per_scan = None
        self.num_samples_per_scan = None
        self.occ_pct_per_scan = None
        self.free_pct_per_scan = None
        self.unk_pct_per_scan = None

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

    def cal_ints_points(self, cfg, key_rays_list: list, query_scan_idx):
        depth_list = []
        labels_list = []
        confidence_list = []
        query_rays_idx_list = []
        key_rays_idx_list = []
        key_meta_info_list = []

        for key_sensor_idx, key_rays in enumerate(key_rays_list):
            # query org to key org vector
            key_rays_dir = key_rays.get_ray_dir()  # cuda

            # TODO: time -> space, cuda out of memory
            split_size = cfg['split_size']
            query_rays_dir_split = torch.split(self.get_ray_dir(), int(self.get_ray_size()/split_size), dim=0)

            # loop each split
            dvg_ang = cfg['dvg_ang']  # rad
            query_ray_idx_start = 0
            for query_rays_dir in query_rays_dir_split:
                ######################################## intersection points ###########################################
                # cal reference plane normal vector: unit vector (query_rays_size, 3)
                query_key_org_vec = key_rays.get_org_vec(self.get_ray_start(), len(query_rays_dir))  # cuda
                ref_plane_norm = torch.cross(query_rays_dir, query_key_org_vec)  # cuda
                del query_key_org_vec

                # calculate cos of key_rays to reference plane: (query_rays_size / split_size, key_rays_size)
                key_rays_to_ref_plane = torch.matmul(ref_plane_norm, key_rays_dir.T)  # cuda
                del ref_plane_norm

                # get intersection rays
                ray_ints_mask = torch.logical_and(key_rays_to_ref_plane >= np.cos(np.pi / 2 + dvg_ang / 2),
                                                  key_rays_to_ref_plane <= np.cos(np.pi / 2 - dvg_ang / 2))
                del key_rays_to_ref_plane

                # intersection ray index
                ray_ints_idx = torch.where(ray_ints_mask)  # cuda
                query_rays_ints_idx = ray_ints_idx[0]  # cuda
                key_rays_ints_idx = ray_ints_idx[1]  # cuda
                del ray_ints_idx

                # common perpendicular line
                ints_query_rays_dir = query_rays_dir[query_rays_ints_idx]  # cuda
                ints_key_rays_dir = key_rays_dir[key_rays_ints_idx]  # cuda
                com_norm = torch.cross(ints_query_rays_dir, ints_key_rays_dir)  # cuda

                # intersection points
                rays_org_vec = torch.broadcast_to(key_rays.get_ray_start() - self.ray_start, (len(query_rays_ints_idx), 3))  # cuda
                q = (torch.sum(torch.cross(rays_org_vec, ints_key_rays_dir) * com_norm, dim=1) / torch.sum(com_norm * com_norm, dim=1))  # cuda
                k = (torch.sum(torch.cross(rays_org_vec, ints_query_rays_dir) * com_norm, dim=1) / torch.sum(com_norm * com_norm, dim=1))  # cuda
                valid_ints_mask = torch.logical_and(torch.logical_and(q >= 0, q <= cfg['max_range']),
                                                    torch.logical_and(k >= 0, k <= cfg['max_range']))  # cuda
                query_rays_ints_idx = query_rays_ints_idx[valid_ints_mask]  # cuda
                key_rays_ints_idx = key_rays_ints_idx[valid_ints_mask]  # cuda
                q_valid = q[valid_ints_mask]  # cuda
                k_valid = k[valid_ints_mask]  # cuda

                # calculate occupancy label of the ints pts (0: unknown, 1: free, 2:occupied)
                key_ray_depth = key_rays.get_ray_depth(key_rays.get_ray_end()[key_rays_ints_idx])  # cuda
                ints_labels, ints_confidence, ints_confident_mask = get_occupancy_label(ray_depth=key_ray_depth, point_depth=k_valid)
                ints_depth = q_valid[ints_confident_mask]

                # TODO: split, add start index; times uniform downsample rate
                query_rays_ints_idx = (query_rays_ints_idx[ints_confident_mask] + query_ray_idx_start) * cfg['uni_ds']
                key_rays_ints_idx = key_rays_ints_idx[ints_confident_mask]

                # clear cuda memory
                del k_valid, q_valid, valid_ints_mask, k, q, com_norm, ints_key_rays_dir, ints_query_rays_dir, rays_org_vec, ints_confident_mask, key_ray_depth
                torch.cuda.empty_cache()
                ########################################################################################################

                ########################################## parallel points #############################################
                # TODO: isosceles triangle approximation (point to line distance -> base length)
                dvg_ang = cfg['dvg_ang']
                ray_ang = torch.matmul(query_rays_dir, key_rays.get_ray_dir().T)  # cuda
                ray_ang = torch.clip(ray_ang, min=-1, max=1)  # cuda, inner product of two unit vector > 1...
                ray_para_mask = torch.logical_and(torch.logical_and(ray_ang > -1, ray_ang < 1),
                                                  ray_ang >= np.cos(dvg_ang))
                ray_para_mask = torch.logical_and(ray_ints_mask, ray_para_mask)
                ray_para_idx = torch.where(ray_para_mask)
                query_rays_para_idx = ray_para_idx[0]
                key_rays_para_idx = ray_para_idx[1]

                # common perpendicular line
                para_query_rays_dir = query_rays_dir[query_rays_para_idx]  # cuda
                para_key_rays_dir = key_rays_dir[key_rays_para_idx]  # cuda
                para_com_norm = torch.cross(para_query_rays_dir, para_key_rays_dir)  # cuda

                # para rays intersection points
                rays_org_vec = torch.broadcast_to(key_rays.get_ray_start() - self.ray_start, (len(para_com_norm), 3))  # cuda
                com_norm_dot_prod = torch.sum(para_com_norm * para_com_norm, dim=1)
                q_para = (torch.sum(torch.cross(rays_org_vec, para_key_rays_dir) * para_com_norm, dim=1) / com_norm_dot_prod)  # cuda
                k_para = (torch.sum(torch.cross(rays_org_vec, para_query_rays_dir) * para_com_norm, dim=1) / com_norm_dot_prod)  # cuda

                # update
                valid_para_mask_0 = torch.sign(q_para) * torch.sign(k_para) > 0
                q_para = q_para[valid_para_mask_0]
                k_para = k_para[valid_para_mask_0]
                query_rays_para_idx = query_rays_para_idx[valid_para_mask_0]
                key_rays_para_idx = key_rays_para_idx[valid_para_mask_0]

                # TODO: inverse solution, leg length threshold
                fw_mask = torch.logical_and(q_para >= 0, k_para >= 0)  # para rays intersect at forward side
                bw_mask = torch.logical_and(q_para < 0, k_para < 0)  # para rays intersect at backward side

                # where the intersection begins (at query ray)
                # TODO: numerator and denominator for forward and backward cases
                ang_cos = ray_ang[tuple((query_rays_para_idx, key_rays_para_idx))]
                sin_half_ang = torch.sqrt((1 - ang_cos) / 2)
                tan_half_ang = np.tan(dvg_ang / 2)
                fw_numerator = q_para[fw_mask] * (sin_half_ang[fw_mask] + tan_half_ang) - k_para[fw_mask] * tan_half_ang
                fw_denominator = sin_half_ang[fw_mask] + 2 * tan_half_ang
                bw_numerator = q_para[bw_mask] * (-sin_half_ang[bw_mask] + tan_half_ang) - k_para[bw_mask] * tan_half_ang
                bw_denominator = -sin_half_ang[bw_mask] + 2 * tan_half_ang

                x_q_min = torch.ones(len(q_para)).cuda() * 100
                # TODO: logic 1: d = r_q + r_k
                x_q_min_1 = torch.ones(len(q_para)).cuda() * 100
                x_q_min_1[fw_mask] = fw_numerator / fw_denominator
                x_q_min_1[bw_mask] = bw_numerator / bw_denominator
                x_k_min_1 = k_para - q_para + x_q_min_1
                valid_para_mask_1 = torch.logical_and(torch.logical_and(x_q_min_1 >= 0, x_q_min_1 <= cfg['max_range']),
                                                      torch.logical_and(x_k_min_1 >= 0, x_k_min_1 <= cfg['max_range']))
                # update global x_q_min and x_k_min
                update_mask = torch.logical_and(valid_para_mask_1, x_q_min_1 <= x_q_min)
                x_q_min[update_mask] = x_q_min_1[update_mask]

                # TODO: logic 2: d = r_k
                x_q_min_2 = torch.ones(len(q_para)).cuda() * 100
                x_q_min_2[fw_mask] = fw_numerator / (fw_denominator - tan_half_ang)
                x_q_min_2[bw_mask] = bw_numerator / (bw_denominator - tan_half_ang)
                x_k_min_2 = k_para - q_para + x_q_min_2
                valid_para_mask_2 = torch.logical_and(torch.logical_and(x_q_min_2 >= 0, x_q_min_2 <= cfg['max_range']),
                                                      torch.logical_and(x_k_min_2 >= 0, x_k_min_2 <= cfg['max_range']))
                # update global x_q_min and x_k_min
                update_mask = torch.logical_and(valid_para_mask_2, x_q_min_2 <= x_q_min)
                x_q_min[update_mask] = x_q_min_2[update_mask]

                # TODO: logic 3: d = r_k - r_q
                x_q_min_3 = torch.ones(len(q_para)).cuda() * 100
                x_q_min_3[fw_mask] = fw_numerator / (fw_denominator - 2 * tan_half_ang)
                x_q_min_3[bw_mask] = bw_numerator / (bw_denominator - 2 * tan_half_ang)
                x_k_min_3 = k_para - q_para + x_q_min_3
                valid_para_mask_3 = torch.logical_and(torch.logical_and(x_q_min_3 >= 0, x_q_min_3 <= cfg['max_range']),
                                                      torch.logical_and(x_k_min_3 >= 0, x_k_min_3 <= cfg['max_range']))
                # update global x_q_min and x_k_min
                update_mask = torch.logical_and(valid_para_mask_3, x_q_min_3 <= x_q_min)
                x_q_min[update_mask] = x_q_min_3[update_mask]

                # TODO: logic 4: d = r_q
                x_q_min_4 = torch.ones(len(q_para)).cuda() * 100
                x_q_min_4[fw_mask] = (q_para[fw_mask] * (sin_half_ang[fw_mask])) / (fw_denominator - tan_half_ang)
                x_q_min_4[bw_mask] = (q_para[bw_mask] * -sin_half_ang[bw_mask]) / (bw_denominator - tan_half_ang)
                x_k_min_4 = k_para - q_para + x_q_min_4
                valid_para_mask_4 = torch.logical_and(torch.logical_and(x_q_min_4 >= 0, x_q_min_4 <= cfg['max_range']),
                                                      torch.logical_and(x_k_min_4 >= 0, x_k_min_4 <= cfg['max_range']))
                # update global x_q_min and x_k_min
                update_mask = torch.logical_and(valid_para_mask_4, x_q_min_4 <= x_q_min)
                x_q_min[update_mask] = x_q_min_4[update_mask]

                # TODO: logic 5: d = 0
                x_q_min_5 = q_para
                x_k_min_5 = k_para
                valid_para_mask_5 = torch.logical_and(torch.logical_and(x_q_min_5 >= 0, x_q_min_5 <= cfg['max_range']),
                                                      torch.logical_and(x_k_min_5 >= 0, x_k_min_5 <= cfg['max_range']))
                # update global x_q_min and x_k_min
                update_mask = torch.logical_and(valid_para_mask_5, x_q_min_5 <= x_q_min)
                x_q_min[update_mask] = x_q_min_5[update_mask]
                x_k_min = k_para - q_para + x_q_min

                # TODO: all valid
                valid_para_mask = torch.logical_or(valid_para_mask_1, valid_para_mask_2)
                valid_para_mask = torch.logical_or(valid_para_mask, valid_para_mask_3)
                valid_para_mask = torch.logical_or(valid_para_mask, valid_para_mask_4)
                valid_para_mask = torch.logical_or(valid_para_mask, valid_para_mask_5)

                # update
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
                x_k = torch.cat((x_k_prime, x_k), dim=0)
                valid_sample_mask = torch.logical_and(x_q >= x_q_min.repeat(6), x_k >= x_k_min.repeat(6))

                # save labels
                ray_depth = key_ray_depth.repeat(6)[valid_sample_mask]
                key_sample_depth = x_k[valid_sample_mask]
                para_labels, para_confidence, para_confident_mask = get_occupancy_label(ray_depth=ray_depth, point_depth=key_sample_depth)
                para_depth = x_q[valid_sample_mask][para_confident_mask]
                # TODO: split, add start index; times uniform downsample rate
                query_rays_para_idx = (query_rays_para_idx.repeat(6)[valid_sample_mask][para_confident_mask] + query_ray_idx_start) * cfg['uni_ds']
                key_rays_para_idx = key_rays_para_idx.repeat(6)[valid_sample_mask][para_confident_mask]

                # TODO: update query ray index start (due to query ray split)
                query_ray_idx_start += len(query_rays_dir)

                # dictionary: query ray index -> points list
                depth = torch.cat((ints_depth, para_depth), dim=0)  # depth on query ray
                labels = torch.cat((ints_labels, para_labels), dim=0)  # occupancy labels
                confidence = torch.cat((ints_confidence, para_confidence), dim=0)  # confidence of the labels
                query_rays_idx = torch.cat((query_rays_ints_idx, query_rays_para_idx), dim=0)
                key_rays_idx = torch.cat((key_rays_ints_idx, key_rays_para_idx), dim=0)

                # append to list
                depth_list.append(depth.cpu())
                labels_list.append(labels.cpu())
                confidence_list.append(confidence.cpu())
                query_rays_idx_list.append(query_rays_idx.cpu())
                key_rays_idx_list.append(key_rays_idx.cpu())
                key_meta_info_list.append([key_sensor_idx, len(labels)])

                # clear cuda memory
                del ray_ints_mask, key_rays_idx, query_rays_idx, confidence, labels, depth, ray_ang, key_rays_ints_idx, ints_labels, ints_depth, ints_confidence
                torch.cuda.empty_cache()

        if len(labels_list) == 0:
            return None
        else:
            depth = torch.cat(depth_list, dim=0)
            labels = torch.cat(labels_list, dim=0)
            confidence = torch.cat(confidence_list, dim=0)
            query_rays_idx = torch.cat(query_rays_idx_list, dim=0)
            key_rays_idx = torch.cat(key_rays_idx_list, dim=0)
            key_meta_info = torch.tensor(key_meta_info_list).to(torch.int32)

            # statistics: for histogram
            self.num_valid_rays_per_scan = len(torch.unique(query_rays_idx))
            self.num_samples_per_scan = len(labels)
            num_unk = torch.sum(labels == 0)
            num_free = torch.sum(labels == 1)
            num_occ = torch.sum(labels == 2)

            # statistics: occ percentage per ray
            self.unk_pct_per_scan = (num_unk / len(labels)) * 100
            self.free_pct_per_scan = (num_free / len(labels)) * 100
            self.occ_pct_per_scan = (num_occ / len(labels)) * 100
            return depth, labels, confidence, query_rays_idx, key_rays_idx, key_meta_info


if __name__ == '__main__':
    # mode
    parser = argparse.ArgumentParser()
    parser.add_argument('--autodl', type=bool, help="autodl server", default=False)
    parser.add_argument('--mars', type=bool, help="mars server", default=False)
    parser.add_argument('--hpc', type=bool, help="hpc server", default=False)
    args = parser.parse_args()

    # configs
    with open('configs/dataset.yaml', 'r') as f:
        cfg_dataset = yaml.safe_load(f)['sekitti']
    # load kitti dataset
    if args.autodl:
        cfg_dataset['root'] = "/root/autodl-tmp/" + cfg_dataset['root']
    elif args.mars:
        cfg_dataset['root'] = "/home/miaozl" + cfg_dataset['root']
    elif args.hpc:
        cfg_dataset['root'] = "/lustre1/g/mech_mars" + cfg_dataset['root']
    else:
        cfg_dataset['root'] = "/home/ziliang" + cfg_dataset['root']
    with open('configs/ours.yaml', 'r') as f:
        cfg = yaml.safe_load(f)['moco']

    # get key sample data tokens
    seq_list = cfg_dataset['train'] + cfg_dataset['val']
    for seq_idx in seq_list:
        print(f"current sequence: {seq_idx}")
        seqstr = "{0:02d}".format(int(seq_idx))
        path_to_seq = os.path.join(cfg_dataset['root'], seqstr)
        scan_files, label_files = kitti_utils.load_files(os.path.join(path_to_seq, cfg_dataset['lidar']),
                                                         cfg_dataset['root'], seq_idx)  # load all scans in a seq
        scans_idx_list = [scan_file.split(cfg_dataset['lidar'] + "/")[1].split(".")[0] for scan_file in scan_files]
        key_scans_idx_dict = kitti_utils.get_mutual_scans_dict(scans_idx_list, path_to_seq, cfg)

        # loop query rays
        num_valid_scans = 0
        num_valid_rays_per_scan = []
        num_samples_per_scan = []
        unk_pct_per_scan = []
        free_pct_per_scan = []
        occ_pct_per_scan = []
        for query_scan_idx, key_scans_list in tqdm(zip(key_scans_idx_dict.keys(), key_scans_idx_dict.values())):
            print("query scan index: " + str(query_scan_idx))
            # get rays
            query_rays, key_rays_list = load_rays(cfg, path_to_seq, query_scan_idx, key_scans_list)  # cuda

            # calculate intersection points
            depth, labels, confidence, query_rays_idx, key_rays_idx, key_meta_info = query_rays.cal_ints_points(cfg, key_rays_list, query_scan_idx)
            if labels is not None:
                # statistics
                num_valid_rays_per_scan.append(query_rays.num_valid_rays_per_scan)  # int
                num_samples_per_scan.append(query_rays.num_samples_per_scan)  # int
                unk_pct_per_scan.append(query_rays.unk_pct_per_scan)  # cuda tensor
                free_pct_per_scan.append(query_rays.free_pct_per_scan)  # cuda tensor
                occ_pct_per_scan.append(query_rays.occ_pct_per_scan)  # cuda tensor
                num_valid_scans += 1

                # to numpy format
                depth = depth.cpu().numpy()
                depth_fp16 = depth.astype(np.float16)
                labels = labels.cpu().numpy()
                labels_uint8 = labels.astype(np.uint8)
                confidence = confidence.cpu().numpy()
                confidence_fp16 = confidence.astype(np.float16)
                query_rays_idx = query_rays_idx.cpu().numpy()
                query_rays_idx_uint16 = query_rays_idx.astype(np.uint16)
                key_rays_idx = key_rays_idx.cpu().numpy()
                key_rays_idx_uint16 = key_rays_idx.astype(np.uint16)
                key_meta_info = key_meta_info.cpu().numpy()
                key_meta_info_uint32 = key_meta_info.astype(np.uint32)

                # save labels
                labels_folder = os.path.join(path_to_seq, 'top_labels')
                os.makedirs(labels_folder, exist_ok=True)
                depth_fp16.tofile(os.path.join(labels_folder, str(query_scan_idx).zfill(6) + "_depth.bin"))
                labels_uint8.tofile(os.path.join(labels_folder, str(query_scan_idx).zfill(6) + "_labels.bin"))
                confidence_fp16.tofile(os.path.join(labels_folder, str(query_scan_idx).zfill(6) + "_confidence.bin"))
                query_rays_idx_uint16.tofile(os.path.join(labels_folder, str(query_scan_idx).zfill(6) + "_rays_idx.bin"))
                key_rays_idx_uint16.tofile(os.path.join(labels_folder, str(query_scan_idx).zfill(6) + "_key_rays_idx.bin"))
                key_meta_info_uint32.tofile(os.path.join(labels_folder, str(query_scan_idx).zfill(6) + "_key_meta.bin"))
                print("save top labels: " + os.path.join(labels_folder, str(query_scan_idx).zfill(6) + "_labels.bin"))

                # clear cuda memory
                del depth, labels, confidence, query_rays_idx, key_rays_idx, key_meta_info, query_rays, key_rays_list
                torch.cuda.empty_cache()
            else:
                print(f"Scan index {query_scan_idx} do not have valid temporal overlapping points.")
            print(f"Seq: {seq_idx} has {num_valid_scans} valid scans / {len(key_scans_idx_dict.keys())} all scans.")

        # histogram statistics
        num_valid_rays_per_scan = np.stack(num_valid_rays_per_scan)
        num_samples_per_scan = np.stack(num_samples_per_scan)
        unk_pct_per_scan = torch.stack(unk_pct_per_scan).cpu().numpy()
        free_pct_per_scan = torch.stack(free_pct_per_scan).cpu().numpy()
        occ_pct_per_scan = torch.stack(occ_pct_per_scan).cpu().numpy()
        num_valid_rays_per_scan.tofile("./num_valid_rays_per_scan_int64.bin")
        num_samples_per_scan.tofile("./num_samples_per_scan_int64.bin")
        unk_pct_per_scan.tofile("./unk_pct_per_scan_float32.bin")
        free_pct_per_scan.tofile("./free_pct_per_scan_float32.bin")
        occ_pct_per_scan.tofile("./occ_pct_per_scan_float32.bin")

        # save histgram statistics
        fig, axs = plt.subplots(2, 3, figsize=(30, 20))
        fig.suptitle('kitti top samples histgram statistics')

        axs[0, 0].hist(num_valid_rays_per_scan, bins=50, rwidth=0.8, color='skyblue', alpha=0.9)
        axs[0, 0].set_title('Valid Rays per Scan')
        axs[0, 0].set_xlabel("Number of Valid Rays")
        axs[0, 0].set_ylabel("Frequency (Number of Scans)")
        axs[0, 0].grid(True, alpha=0.3)

        axs[0, 1].hist(num_samples_per_scan, bins=100, rwidth=0.8, color='lightgreen', alpha=0.9)
        axs[0, 1].set_title('TOP Samples per Scan')
        axs[0, 1].set_xlabel("Num of Valid TOP Samples")
        axs[0, 1].set_ylabel("Frequency (Number of Scans)")
        axs[0, 1].grid(True, alpha=0.3)

        axs[1, 0].hist(unk_pct_per_scan, bins=50, rwidth=0.8, color='salmon', alpha=0.9)
        axs[1, 0].set_title('Unknown Samples Pct per Scan')
        axs[1, 0].set_xlabel("Unknown Samples Pct.")
        axs[1, 0].set_ylabel("Frequency (Number of Scans)")
        axs[1, 0].grid(True, alpha=0.3)

        axs[1, 1].hist(free_pct_per_scan, bins=50, rwidth=0.8, color='purple', alpha=0.9)
        axs[1, 1].set_title('Free Samples Pct per Scan')
        axs[1, 1].set_xlabel("Free Samples Pct.")
        axs[1, 1].set_ylabel("Frequency (Number of Scans)")
        axs[1, 1].grid(True, alpha=0.3)

        axs[1, 2].hist(occ_pct_per_scan, bins=50, rwidth=0.8, color='orange', alpha=0.9)
        axs[1, 2].set_title('Occupied Samples Pct per Scan')
        axs[1, 2].set_xlabel("Occupied Samples Pct.")
        axs[1, 2].set_ylabel("Frequency (Number of Scans)")
        axs[1, 2].grid(True, alpha=0.3)

        # axs[1, 2].hist(cyc_wo_spd_list, bins=100, range=(0,2), log=False, color='brown', alpha=0.9)
        # axs[1, 2].set_title('cycle without rider')
        # axs[1, 2].set_xlabel('speed (m/s)')
        # axs[1, 2].set_ylabel('frequency')
        # axs[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'seq_{seq_idx}_top_samples_statistics.png', dpi=1000)
        plt.close()