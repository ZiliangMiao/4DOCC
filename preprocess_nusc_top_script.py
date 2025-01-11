import argparse
import os
import time
from typing import List
import open3d
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from collections import defaultdict
from nuscenes import NuScenes
from tqdm import tqdm
import datasets.nusc_utils as nusc_utils
from nuscenes.utils.geometry_utils import points_in_box
from utils.vis.open3d_vis_utils import occ_color_func, mos_color_func, get_confusion_color
from utils.vis.open3d_vis_utils import draw_box
import matplotlib.pyplot as plt


def draw_while_preprocessing(nusc, cfg, sample_tok, depth, labels, query_rays_idx, key_rays_indices, key_meta_info, vis_q_min, vis_k_min):
    vis_obj_idx = 0  # nonlocal variable: moving object index
    vis_ray_idx = 0  # nonlocal variable: ray index of moving object points which have background samples
    view_init = True
    skip_flag = True
    cam_params = None
    mov_obj_num = 0
    ray_num = 0

    # vis sample data
    sample = nusc.get('sample', sample_tok)
    sd_tok = sample['data']['LIDAR_TOP']

    # open3d vis
    vis = open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='ray intersection points', width=3840, height=2160, left=0, top=0)
    def draw(vis):
        nonlocal vis_obj_idx, vis_ray_idx, cam_params, view_init, mov_obj_num, ray_num
        # get key sample data tokens
        sample_data = nusc.get('sample_data', sd_tok)
        sample_token = sample_data['sample_token']
        sample = nusc.get("sample", sample_token)
        key_sd_toks_list = nusc_utils.get_mutual_sd_toks_dict(nusc, [sample_token], cfg)[sample_token]

        # get query rays and key rays
        query_org, query_pts, query_ts, filter_mask = nusc_utils.get_transformed_pcd(nusc, cfg, sd_tok, sd_tok)
        query_dir = F.normalize(query_pts - query_org, p=2, dim=1)  # unit vector
        key_rays_org_list = []
        key_rays_ts_list = []
        key_rays_pts_list = []
        key_mos_labels_list = []
        for key_sd_tok in key_sd_toks_list:
            key_org, key_pts, key_ts, key_mask = nusc_utils.get_transformed_pcd(nusc, cfg, sd_tok, key_sd_tok)
            key_rays_org_list.append(key_org)
            key_rays_pts_list.append(key_pts)
            key_rays_ts_list.append(key_ts)
            key_mos_labels_file = os.path.join(nusc.dataroot, 'mos_labels', nusc.version, key_sd_tok + "_mos.label")
            key_mos_labels = np.fromfile(key_mos_labels_file, dtype=np.uint8)[key_mask]
            key_mos_labels_list.append(key_mos_labels)

        # load gt and pred mos labels
        gt_mos_labels_file = os.path.join(nusc.dataroot, 'mos_labels', nusc.version, sd_tok + "_mos.label")
        gt_mos_labels = np.fromfile(gt_mos_labels_file, dtype=np.uint8)[filter_mask]

        # load gt bg samples
        query_ray_idx_unq = np.unique(query_rays_idx)
        key_sensors_indices = np.concatenate([np.ones(meta[1], dtype=np.uint32) * meta[0] for meta in key_meta_info])

        # vis nusc sample, moving object bboxes
        anns_toks = sample['anns']
        _, boxes, _ = nusc.get_sample_data(sd_tok, selected_anntokens=anns_toks, use_flat_vehicle_coordinates=False)
        obj_boxes_list = []
        obj_ray_indices_list = []
        for ann_tok, box in zip(anns_toks, boxes):
            ann = nusc.get('sample_annotation', ann_tok)
            if ann['num_lidar_pts'] == 0: continue
            obj_pts_mask = points_in_box(box, query_pts[:, :3].T)
            if np.sum(obj_pts_mask) == 0: continue
            gt_obj_labels = gt_mos_labels[obj_pts_mask]
            mov_pts_mask = gt_obj_labels == 2
            if np.sum(mov_pts_mask) == 0:
                continue
            else:
                obj_pts_mask = points_in_box(box, query_pts[:, :3].T)
                obj_pts_idx_list = np.where(obj_pts_mask)[0].tolist()
                ray_idx_list = list(set(obj_pts_idx_list) & set(query_ray_idx_unq))  # only a part of rays have bg samples
                if len(ray_idx_list) == 0: continue
                obj_boxes_list.append(box)
                obj_ray_indices_list.append(ray_idx_list)

        ################################################################################################################
        # draw static geometries: query lidar orgs, key lidar orgs
        vis.clear_geometries()
        axis_size = 2.0
        axis_query = open3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size * 2, origin=query_org)
        vis.add_geometry(axis_query)
        for key_rays_org, key_rays_ts in zip(key_rays_org_list, key_rays_ts_list):
            axis_key = open3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size + key_rays_ts * 0.1, origin=key_rays_org)
            vis.add_geometry(axis_key)

        # draw query pcd
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(query_pts)
        pcd.colors = open3d.utility.Vector3dVector(np.array(mos_color_func(gt_mos_labels)).T)  # static color
        pcd_down = pcd.voxel_down_sample(voxel_size=0.10)  # point cloud downsample
        vis.add_geometry(pcd_down)
        ################################################################################################################

        if len(obj_ray_indices_list) == 0:
            print(f"Sample have no moving objects.")
            return None
        else:
            mov_obj_num = len(obj_ray_indices_list)
            ray_num = len(obj_ray_indices_list[vis_obj_idx])
            print(f"Sample: {sd_tok}, Moving object index: {vis_obj_idx} / {mov_obj_num - 1}, Ray index: {vis_ray_idx} / {ray_num - 1}")
            print(f"Query ray index: {obj_ray_indices_list[vis_obj_idx][vis_ray_idx]}")

        # draw moving object bboxes
        box = obj_boxes_list[vis_obj_idx]
        draw_box(vis, [box])

        # draw query ray point [vis_ray_idx]
        obj_ray_idx = obj_ray_indices_list[vis_obj_idx][vis_ray_idx]
        obj_point_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=200)
        obj_point_sphere = obj_point_sphere.translate(query_pts[obj_ray_idx], relative=False)
        obj_point_sphere.paint_uniform_color(mos_color_func(gt_mos_labels[obj_ray_idx]))
        vis.add_geometry(obj_point_sphere)

        # draw query ray
        query_lineset = open3d.geometry.LineSet()
        vertex_query = np.vstack((query_org, query_pts[obj_ray_idx]))
        lines_query = [[0, 1]]
        color_query = np.tile((234 / 255, 51 / 255, 35 / 255), (1, 1))
        query_lineset.points = open3d.utility.Vector3dVector(vertex_query)
        query_lineset.lines = open3d.utility.Vector2iVector(lines_query)
        query_lineset.colors = open3d.utility.Vector3dVector(color_query)
        vis.add_geometry(query_lineset)

        # get ray samples
        samples_mask = query_rays_idx == obj_ray_idx
        sample_depth = depth[samples_mask]
        sample_pts = query_org + np.broadcast_to(query_dir[obj_ray_idx], (len(sample_depth), 3)) * sample_depth.reshape(-1, 1)

        # voxel down sampling for vis
        sample_pcd = open3d.geometry.PointCloud()
        sample_pcd.points = open3d.utility.Vector3dVector(sample_pts)
        _, _, idx_list = sample_pcd.voxel_down_sample_and_trace(voxel_size=0.05, min_bound=query_org, max_bound=query_pts[obj_ray_idx] * 2)
        down_idx = np.stack([i[0] for i in idx_list])

        # key rays
        key_sensors_indices = key_sensors_indices[samples_mask][down_idx]
        sample_key_rays_indices = key_rays_indices[samples_mask][down_idx]
        key_pts_list = []
        key_mos_list = []
        for i in range(len(sample_key_rays_indices)):
            key_sensor_idx = key_sensors_indices[i]
            key_ray_idx = sample_key_rays_indices[i]
            key_pts_list.append(key_rays_pts_list[key_sensor_idx][key_ray_idx])
            key_mos_list.append(key_mos_labels_list[key_sensor_idx][key_ray_idx])
        key_pts = np.stack(key_pts_list)
        key_mos = np.array(key_mos_list)

        sample_pts = sample_pts[down_idx]
        gt_labels = labels[samples_mask][down_idx]

        # draw bg points and rays to sample points
        lineset_key = open3d.geometry.LineSet()
        vertex_key = np.vstack((np.stack(key_rays_org_list), key_pts, sample_pts))
        lines_key = []
        lineset_corr = open3d.geometry.LineSet()  # sample point to corresponding key point
        vertex_corr = np.vstack((key_pts, sample_pts))
        lines_corr = []
        for i in range(len(sample_pts)):
            # TODO: vis q min and k min
            q_min = vis_q_min[samples_mask][down_idx][i]
            q_min_point = query_org.cpu().numpy() + q_min.reshape(-1, 1) * query_dir[query_rays_idx][samples_mask][down_idx][i].cpu().numpy()
            q_min_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.2, resolution=200)
            q_min_sphere = q_min_sphere.translate(q_min_point.ravel(), relative=False)
            q_min_sphere.paint_uniform_color([1, 1, 1])
            vis.add_geometry(q_min_sphere)
            k_min = vis_k_min[samples_mask][down_idx][i]
            a = key_rays_pts_list[key_sensors_indices[i]][sample_key_rays_indices[i]]
            b = key_rays_org_list[key_sensors_indices[i]]
            key_dir = F.normalize(a - b, p=2, dim=0).cpu().numpy()
            k_min_point = key_rays_org_list[key_sensors_indices[i]].cpu().numpy() + k_min.reshape(-1, 1) * key_dir
            k_min_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.2, resolution=200)
            k_min_sphere = k_min_sphere.translate(k_min_point.ravel(), relative=False)
            k_min_sphere.paint_uniform_color([1, 1, 1])
            vis.add_geometry(k_min_sphere)

            # sample point
            sample_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=200)
            sample_sphere = sample_sphere.translate(sample_pts[i], relative=False)
            sample_sphere.paint_uniform_color(occ_color_func(gt_labels[i]))
            vis.add_geometry(sample_sphere)

            # corresponding key point
            key_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=200)
            key_sphere = key_sphere.translate(key_pts[i], relative=False)
            key_sphere.paint_uniform_color(mos_color_func(key_mos[i]))
            vis.add_geometry(key_sphere)

            # key rays org to bg point
            lines_key.append([key_sensors_indices[i], i + len(key_rays_org_list)])
            # lines_key.append([key_sensors_indices[i], i + len(key_rays_org_list) + len(sample_pts)])
            lines_corr.append([i, i + len(key_pts)])

        color_key = np.tile((1, 1, 1), (len(lines_key), 1))
        lineset_key.points = open3d.utility.Vector3dVector(vertex_key)
        lineset_key.lines = open3d.utility.Vector2iVector(lines_key)
        lineset_key.colors = open3d.utility.Vector3dVector(color_key)
        vis.add_geometry(lineset_key)
        color_corr = np.tile((0.114, 0.875, 0), (len(lines_key), 1))
        lineset_corr.points = open3d.utility.Vector3dVector(vertex_corr)
        lineset_corr.lines = open3d.utility.Vector2iVector(lines_corr)
        lineset_corr.colors = open3d.utility.Vector3dVector(color_corr)
        vis.add_geometry(lineset_corr)

        # open3d view option
        vis.get_render_option().point_size = 7.0
        vis.get_render_option().background_color = (0 / 255, 0 / 255, 0 / 255)
        view_ctrl = vis.get_view_control()
        if view_init:
            view_ctrl.set_front((0.0, 0.0, 1.0))
            view_ctrl.set_lookat((0.33837080335515901, -2.223431055221385, 2.6541285514831543))
            view_ctrl.set_up((0.0, 1.0, 0.0))
            view_ctrl.set_zoom((0.19999999999999959))
            cam_params = view_ctrl.convert_to_pinhole_camera_parameters()
            view_init = False
        view_ctrl.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        while True:
            cam_params = view_ctrl.convert_to_pinhole_camera_parameters()
            # print(cam_params.extrinsic)
            vis.poll_events()
            vis.update_renderer()

    def render_next_obj(vis):
        nonlocal vis_obj_idx, vis_ray_idx, skip_flag, mov_obj_num
        vis_ray_idx = 0  # if vis prev or next obj, set the ray idx to 0
        if skip_flag:
            skip_flag = False
            return None
        else:
            skip_flag = True
            vis_obj_idx += 1
            if vis_obj_idx >= mov_obj_num:
                vis_obj_idx = mov_obj_num - 1
            draw(vis)

    def render_prev_obj(vis):
        nonlocal vis_obj_idx, vis_ray_idx, skip_flag
        vis_ray_idx = 0  # if vis prev or next obj, set the ray idx to 0
        if skip_flag:
            skip_flag = False
            return None
        else:
            skip_flag = True
            vis_obj_idx -= 1
            if vis_obj_idx <= 0:
                vis_obj_idx = 0
            draw(vis)

    def render_next_ray(vis):
        nonlocal vis_obj_idx, vis_ray_idx, skip_flag, ray_num
        if skip_flag:
            skip_flag = False
            return None
        else:
            skip_flag = True
            vis_ray_idx += 1
            if vis_ray_idx >= ray_num:
                vis_ray_idx = ray_num - 1
            draw(vis)

    def render_prev_ray(vis):
        nonlocal vis_ray_idx, skip_flag
        if skip_flag:
            skip_flag = False
            return None
        else:
            skip_flag = True
            vis_ray_idx -= 1
            if vis_ray_idx <= 0:
                vis_ray_idx = 0
            draw(vis)

    # render keyboard control
    vis.register_key_callback(ord('D'), render_next_obj)
    vis.register_key_callback(ord('A'), render_prev_obj)
    vis.register_key_callback(ord('C'), render_next_ray)
    vis.register_key_callback(ord('Z'), render_prev_ray)
    vis.run()


def load_rays(nusc, cfg, query_sd_tok, key_sd_toks_list):
    # get query rays
    org_query, pts_query, ts_query, query_valid_mask = nusc_utils.get_transformed_pcd(nusc, cfg, query_sd_tok, query_sd_tok)  # cpu
    query_rays = QueryRays(org_query.cuda(), pts_query.cuda(), ts_query)  # cuda

    # get key rays
    key_rays_list = []
    # query_rays_ints_idx_list, key_rays_ints_idx_list = [], []
    for key_idx, key_sd_tok in enumerate(key_sd_toks_list):
        org_key, pts_key, ts_key, key_valid_mask = nusc_utils.get_transformed_pcd(nusc, cfg, query_sd_tok, key_sd_tok)  # cpu
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

    def get_mutual_obs_samples(self, cfg, key_rays_list):
        """
        Generate mutual observation samples for pre-training.

        Args:
            cfg:
            key_rays_list:

        Returns:

        """
        asd = 1

    def get_ints_rays(self):
        asd = 1

    def get_para_rays(self):
        asd = 1

    def cal_ints_points(self, cfg, key_rays_list: list, query_sample_tok):
        depth_list = []
        labels_list = []
        confidence_list = []
        query_rays_idx_list = []
        key_rays_idx_list = []
        key_meta_info_list = []

        # # TODO: for vis while preprocessing
        # vis_depth_list = []
        # vis_labels_list = []
        # vis_query_rays_idx_list = []
        # vis_key_rays_idx_list = []
        # vis_key_meta_info_list = []
        # vis_q_min = []
        # vis_k_min = []
        # ###################################

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
            dvg_ang = cfg['dvg_ang']  # rad
            ray_ints_mask = torch.logical_and(key_rays_to_ref_plane >= np.cos(np.pi / 2 + dvg_ang / 2),
                                              key_rays_to_ref_plane <= np.cos(np.pi / 2 - dvg_ang / 2))

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
            valid_ints_mask = torch.logical_and(torch.logical_and(q >= 0, q <= cfg['max_range']),
                                                torch.logical_and(k >= 0, k <= cfg['max_range']))  # cuda
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
            del k_valid, q_valid, valid_ints_mask, k, q, com_norm, ints_key_rays_dir, ints_query_rays_dir, key_rays_to_ref_plane, ref_plane_norm, rays_org_vec, query_key_org_vec, ints_confident_mask, key_ray_depth
            torch.cuda.empty_cache()
            ############################################################################################################

            ########################################## parallel points #################################################
            # TODO: isosceles triangle approximation (point to line distance -> base length)
            dvg_ang = cfg['dvg_ang']
            ray_ang = torch.matmul(self.get_ray_dir(), key_rays.get_ray_dir().T)  # cuda
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

            # # check whether the intersection point is the same point
            # q_pts = self.get_ray_start() + q_para.reshape(-1, 1) * self.get_ray_dir()[query_rays_para_idx]
            # k_pts = key_rays.get_ray_start() + k_para.reshape(-1, 1) * key_rays.get_ray_dir()[key_rays_para_idx]
            # same_pts = torch.where(q_pts == k_pts)
            # ########################################################

            # update
            valid_para_mask_0 = torch.sign(q_para) * torch.sign(k_para) > 0
            q_para = q_para[valid_para_mask_0]
            k_para = k_para[valid_para_mask_0]
            query_rays_para_idx = query_rays_para_idx[valid_para_mask_0]
            key_rays_para_idx = key_rays_para_idx[valid_para_mask_0]

            # TODO: inverse solution, leg length threshold
            fw_mask = torch.logical_and(q_para >= 0, k_para >= 0)  # para rays intersect at forward side
            bw_mask = torch.logical_and(q_para < 0, k_para < 0)  # para rays intersect at backward side
            # fw_idx = torch.where(fw_mask)
            # bw_idx = torch.where(bw_mask)

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

            # TODO: debug
            # if len(torch.where(query_rays_para_idx[valid_para_mask_1] == 596)[0]) != 0:
            #     a = torch.where(query_rays_para_idx[valid_para_mask_1] == 596)[0]
            #     q_min = x_q_min_1[valid_para_mask_1][a]
            #     k_min = x_k_min_1[valid_para_mask_1][a]
            #
            #     q_min_glb = x_q_min[valid_para_mask_1][a]
            #     k_min_glb = x_k_min[valid_para_mask_1][a]
            #
            #     d = (q_para[valid_para_mask_1][a] - q_min) * sin_half_ang[valid_para_mask_1][a]
            #     r_q = q_min * tan_half_ang
            #     r_k = k_min * tan_half_ang
            #     x = 1
            #
            # if len(torch.where(query_rays_para_idx[valid_para_mask_2] == 541)[0]) != 0:
            #     b = torch.where(query_rays_para_idx[valid_para_mask_2] == 541)[0]
            #
            #     q_min = x_q_min_2[valid_para_mask_2][b]
            #     k_min = x_k_min_2[valid_para_mask_2][b]
            #
            #     q_min_glb = x_q_min[valid_para_mask_2][b]
            #     k_min_glb = x_k_min[valid_para_mask_2][b]
            #
            #     q_para = q_para[valid_para_mask_2][b]
            #     sin_half_ang = sin_half_ang[valid_para_mask_2][b]
            #     d = (q_para - q_min) * sin_half_ang
            #     r_q = q_min * tan_half_ang
            #     r_k = k_min * tan_half_ang
            #     x = 1
            # if len(torch.where(query_rays_para_idx[valid_para_mask_3] == 596)[0]) != 0:
            #     c = torch.where(query_rays_para_idx[valid_para_mask_3] == 596)[0]
            # if len(torch.where(query_rays_para_idx[valid_para_mask_4] == 596)[0]) != 0:
            #     d = torch.where(query_rays_para_idx[valid_para_mask_4] == 596)[0]
            # if len(torch.where(query_rays_para_idx[valid_para_mask_5] == 596)[0]) != 0:
            #     e = torch.where(query_rays_para_idx[valid_para_mask_5] == 596)[0]
            ##############

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
            query_rays_para_idx = query_rays_para_idx.repeat(6)[valid_sample_mask][para_confident_mask]
            key_rays_para_idx = key_rays_para_idx.repeat(6)[valid_sample_mask][para_confident_mask]

            # # TODO: for vis while preprocessing
            # vis_depth_list.append(para_depth)
            # vis_labels_list.append(para_labels)
            # vis_query_rays_idx_list.append(query_rays_para_idx)
            # vis_key_rays_idx_list.append(key_rays_para_idx)
            # vis_key_meta_info_list.append([key_sensor_idx, len(para_labels)])
            # vis_q_min.append(x_q_min.repeat(6)[valid_sample_mask][para_confident_mask])
            # vis_k_min.append(x_k_min.repeat(6)[valid_sample_mask][para_confident_mask])
            # ###################################

            # dictionary: query ray index -> points list
            depth = torch.cat((ints_depth, para_depth), dim=0)  # depth on query ray
            labels = torch.cat((ints_labels, para_labels), dim=0)  # occupancy labels
            confidence = torch.cat((ints_confidence, para_confidence), dim=0)  # confidence of the labels
            query_rays_idx = torch.cat((query_rays_ints_idx, query_rays_para_idx), dim=0)
            key_rays_idx = torch.cat((key_rays_ints_idx, key_rays_para_idx), dim=0)

            # append to list
            depth_list.append(depth)
            labels_list.append(labels)
            confidence_list.append(confidence)
            query_rays_idx_list.append(query_rays_idx)
            key_rays_idx_list.append(key_rays_idx)
            key_meta_info_list.append([key_sensor_idx, len(labels)])

            # clear cuda memory
            del ray_ints_mask, key_rays_idx, query_rays_idx, confidence, labels, depth, ray_ang, key_rays_ints_idx, key_rays_dir, ints_labels, ints_depth, ints_confidence
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

            # # TODO: vis while preprocessing ############################################################################
            # vis_depth = torch.cat(vis_depth_list).cpu().numpy()
            # vis_labels = torch.cat(vis_labels_list).cpu().numpy()
            # vis_query_rays_idx = torch.cat(vis_query_rays_idx_list).cpu().numpy()
            # vis_key_rays_idx = torch.cat(vis_key_rays_idx_list).cpu().numpy()
            # vis_key_meta_info = torch.tensor(vis_key_meta_info_list).cpu().numpy()
            # vis_q_min = torch.cat(vis_q_min).cpu().numpy()
            # vis_k_min = torch.cat(vis_k_min).cpu().numpy()
            # draw_while_preprocessing(nusc, cfg, query_sample_tok, vis_depth, vis_labels, vis_query_rays_idx,
            #                          vis_key_rays_idx, vis_key_meta_info, vis_q_min, vis_k_min)
            # ############################################################################################################
            return depth, labels, confidence, query_rays_idx, key_rays_idx, key_meta_info


if __name__ == '__main__':
    # mode
    parser = argparse.ArgumentParser()
    parser.add_argument('--autodl', type=bool, default=False)
    parser.add_argument('--start', type=int, default=0)
    args = parser.parse_args()

    # load nusc dataset
    if args.autodl:
        nusc = NuScenes(dataroot="/root/autodl-tmp/Datasets/nuScenes", version="v1.0-trainval")
    else:
        nusc = NuScenes(dataroot="/home/user/Datasets/nuScenes", version="v1.0-trainval")

    with open('configs/ours.yaml', 'r') as f:
        cfg = yaml.safe_load(f)['moco']

    # get key sample data tokens
    sample_toks_all = [sample['token'] for sample in nusc.sample]
    key_sd_toks_dict = nusc_utils.get_mutual_sd_toks_dict(nusc, sample_toks_all, cfg)

    # loop query rays
    num_valid_samples = 0
    num_valid_rays_per_scan = []
    num_samples_per_scan = []
    unk_pct_per_scan = []
    free_pct_per_scan = []
    occ_pct_per_scan = []
    for query_sample_idx, query_sample_tok in tqdm(enumerate(key_sd_toks_dict.keys())):
        if query_sample_idx < args.start:
            continue
        key_sd_toks_list = key_sd_toks_dict[query_sample_tok]

        # get rays
        query_sample = nusc.get('sample', query_sample_tok)
        query_sd_tok = query_sample['data']['LIDAR_TOP']
        query_rays, key_rays_list = load_rays(nusc, cfg, query_sd_tok, key_sd_toks_list)  # cuda

        # calculate intersection points
        depth, labels, confidence, query_rays_idx, key_rays_idx, key_meta_info = query_rays.cal_ints_points(cfg, key_rays_list, query_sample_tok)
        if labels is not None:
            # statistics
            num_valid_rays_per_scan.append(query_rays.num_valid_rays_per_scan)  # int
            num_samples_per_scan.append(query_rays.num_samples_per_scan)  # int
            unk_pct_per_scan.append(query_rays.unk_pct_per_scan)  # cuda tensor
            free_pct_per_scan.append(query_rays.free_pct_per_scan)  # cuda tensor
            occ_pct_per_scan.append(query_rays.occ_pct_per_scan)  # cuda tensor
            num_valid_samples += 1

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
            labels_folder = os.path.join(nusc.dataroot, 'labels_cuda', nusc.version)
            os.makedirs(labels_folder, exist_ok=True)
            depth_fp16.tofile(os.path.join(labels_folder, query_sd_tok + "_depth.bin"))
            labels_uint8.tofile(os.path.join(labels_folder, query_sd_tok + "_labels.bin"))
            confidence_fp16.tofile(os.path.join(labels_folder, query_sd_tok + "_confidence.bin"))
            query_rays_idx_uint16.tofile(os.path.join(labels_folder, query_sd_tok + "_rays_idx.bin"))
            key_rays_idx_uint16.tofile(os.path.join(labels_folder, query_sd_tok + "_key_rays_idx.bin"))
            key_meta_info_uint32.tofile(os.path.join(labels_folder, query_sd_tok + "_key_meta.bin"))

            # clear cuda memory
            del depth, labels, confidence, query_rays_idx, key_rays_idx, key_meta_info, query_rays, key_rays_list
            torch.cuda.empty_cache()
        else:
            print(f"Sample data tok {query_sd_tok}, index {query_sample_idx} do not have valid background points")
    print(f"Number of valid samples: {num_valid_samples} / number of all samples {len(sample_toks_all)}")

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

    fig_1 = plt.figure()
    ax_1 = fig_1.gca()
    ax_1.hist(num_valid_rays_per_scan, bins=50, rwidth=0.8, align='left')
    ax_1.set_xlabel("Num of valid rays per scan")
    ax_1.set_ylabel("Frequency (Num of scans)")
    fig_1.savefig("./statistics_valid_rays_per_scan.png", dpi=1000)

    fig_2 = plt.figure()
    ax_2 = fig_2.gca()
    ax_2.hist(num_samples_per_scan, bins=100, rwidth=0.8, align='left')
    ax_2.set_xlabel("Num of valid mutual observation samples")
    ax_2.set_ylabel("Frequency (Num of scans)")
    fig_2.savefig("./statistics_mutual_obs_samples_per_scan.png", dpi=1000)

    fig_3 = plt.figure()
    ax_3 = fig_3.gca()
    ax_3.hist(unk_pct_per_scan, bins=50, rwidth=0.8, align='left')
    ax_3.set_xlabel("Unknown samples pct.")
    ax_3.set_ylabel("Frequency (Num of scans)")
    fig_3.savefig("./statistics_unknown_samples_pct_per_scan.png", dpi=1000)

    fig_4 = plt.figure()
    ax_4 = fig_4.gca()
    ax_4.hist(free_pct_per_scan, bins=50, rwidth=0.8, align='left')
    ax_4.set_xlabel("Free samples pct.")
    ax_4.set_ylabel("Frequency (Num of scans)")
    fig_4.savefig("./statistics_free_samples_pct_per_scan.png", dpi=1000)

    fig_5 = plt.figure()
    ax_5 = fig_5.gca()
    ax_5.hist(occ_pct_per_scan, bins=50, rwidth=0.8, align='left')
    ax_5.set_xlabel("Occupied samples pct.")
    ax_5.set_ylabel("Frequency (Num of scans)")
    fig_5.savefig("./statistics_occupied_samples_pct_per_scan.png", dpi=1000)