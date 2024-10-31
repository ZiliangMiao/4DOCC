"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import logging
import os
import sys
from datetime import datetime

import open3d
import argparse
import warnings
import matplotlib
import numpy as np
import torch
import yaml
from tqdm import tqdm
import multiprocessing
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.data_classes import LidarPointCloud, LidarSegPointCloud
from datasets.nusc_utils import get_outside_scene_mask, get_ego_mask

from open3d_vis_utils import draw_box, get_confusion_color, mos_color_func, get_vis_sd_toks


def render_mos_samples(nusc, sd_toks_list, vis_cfg):
    # visualization cfg
    sample_idx = vis_cfg['mos']['start_sample_idx']
    ours_dir = vis_cfg['mos']['ours_dir']
    baseline_dir = vis_cfg['mos']['baseline_dir']
    point_size = vis_cfg['mos']['point_size']

    # TODO: skip flag 实在没办法解决按一次render两次. 打个补丁吧
    skip_flag = True
    view_init = True
    cam_params = None

    # open3d visualizer
    baseline_vis = open3d.visualization.VisualizerWithKeyCallback()
    ours_vis = open3d.visualization.Visualizer()
    baseline_vis.create_window(window_name='baseline mos results', width=853 * 2, height=1440 * 2, left=0, top=0)
    ours_vis.create_window(window_name='ours mos results', width=853 * 2, height=1440 * 2, left=853 * 2, top=0)

    # python logger
    date = datetime.now().strftime('%m%d')
    good_log_file = f"./vis_good_sd_toks_{date}.txt"
    bad_log_file = f"./vis_bad_sd_toks_{date}.txt"
    formatter = logging.Formatter('%(message)s')
    stream_handler = logging.StreamHandler()

    good_handler = logging.FileHandler(good_log_file)
    good_handler.setFormatter(formatter)
    good_cases_logger = logging.getLogger('good_cases')
    good_cases_logger.setLevel(logging.INFO)
    good_cases_logger.addHandler(good_handler)
    good_cases_logger.addHandler(stream_handler)

    bad_handler = logging.FileHandler(bad_log_file)
    bad_handler.setFormatter(formatter)
    bad_cases_logger = logging.getLogger('bad_cases')
    bad_cases_logger.setLevel(logging.INFO)
    bad_cases_logger.addHandler(bad_handler)
    bad_cases_logger.addHandler(stream_handler)

    def draw_sample(baseline_vis, ours_vis):
        nonlocal sample_idx, cam_params, view_init
        print("Rendering sample: " + str(sample_idx))
        # clear geometry
        baseline_vis.clear_geometries()
        ours_vis.clear_geometries()

        # sample and sample data
        sd_tok = sd_toks_list[sample_idx]

        sample_data = nusc.get('sample_data', sd_tok)
        sample_token = sample_data['sample_token']
        sample = nusc.get("sample", sample_token)

        # points
        pcl_path = os.path.join(nusc.dataroot, sample_data['filename'])
        points = LidarPointCloud.from_file(pcl_path).points.T  # [num_points, 4]
        ego_mask = get_ego_mask(torch.tensor(points))
        outside_scene_mask = get_outside_scene_mask(torch.tensor(points), vis_cfg['mos']['scene_bbox'],
                                                                          mask_z=False, upper_bound=True)
        valid_mask = torch.logical_and(~ego_mask, ~outside_scene_mask)
        points = points[valid_mask]

        # labels
        gt_labels_file = os.path.join(nusc.dataroot, 'mos_labels', nusc.version, sd_tok + "_mos.label")
        gt_labels = np.fromfile(gt_labels_file, dtype=np.uint8)[valid_mask]
        baseline_labels_file = os.path.join(baseline_dir, sd_tok + "_mos_pred.label")
        baseline_labels = np.fromfile(baseline_labels_file, dtype=np.uint8)
        ours_labels_file = os.path.join(ours_dir, sd_tok + "_mos_pred.label")
        ours_labels = np.fromfile(ours_labels_file, dtype=np.uint8)

        # moving object bboxes and test metrics
        _, box_list, _ = nusc.get_sample_data(sd_tok, selected_anntokens=sample['anns'], use_flat_vehicle_coordinates=False)
        mov_box_list = []
        mov_obj_num = 0
        inconsistent_obj_num = 0
        for ann_tok, box in zip(sample['anns'], box_list):
            ann = nusc.get('sample_annotation', ann_tok)
            if ann['num_lidar_pts'] == 0: continue
            obj_pts_mask = points_in_box(box, points[:, :3].T)
            if np.sum(obj_pts_mask) == 0: continue
            gt_obj_labels = gt_labels[obj_pts_mask]
            mov_pts_mask = gt_obj_labels == 2
            if np.sum(mov_pts_mask) == 0:
                continue
            else:
                mov_obj_num += 1
                mov_box_list.append(box)
                if len(np.unique(gt_obj_labels)) != 1:
                    inconsistent_obj_num += 1
        baseline_vis = draw_box(baseline_vis, mov_box_list)
        ours_vis = draw_box(ours_vis, mov_box_list)
        print(f"Number of moving objects: {mov_obj_num}, Number of inconsistent moving object: {inconsistent_obj_num}")

        # origin
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        baseline_vis.add_geometry(axis_pcd)
        ours_vis.add_geometry(axis_pcd)

        # points
        baseline_pts = open3d.geometry.PointCloud()
        baseline_pts.points = open3d.utility.Vector3dVector(points[:, :3])
        baseline_vis.add_geometry(baseline_pts)
        ours_pts = open3d.geometry.PointCloud()
        ours_pts.points = open3d.utility.Vector3dVector(points[:, :3])
        ours_vis.add_geometry(ours_pts)

        # colored by TP TN FP FN
        baseline_colors = get_confusion_color(gt_labels, baseline_labels)
        ours_colors = get_confusion_color(gt_labels, ours_labels)
        baseline_pts.colors = open3d.utility.Vector3dVector(np.array(mos_color_func(baseline_colors)).T)
        ours_pts.colors = open3d.utility.Vector3dVector(np.array(mos_color_func(ours_colors)).T)

        # view settings
        baseline_vis.get_render_option().point_size = point_size
        baseline_vis.get_render_option().background_color = (255 / 255, 255 / 255, 255 / 255)
        ours_vis.get_render_option().point_size = point_size
        ours_vis.get_render_option().background_color = (255 / 255, 255 / 255, 255 / 255)

        # view options
        baseline_view = baseline_vis.get_view_control()
        ours_view = ours_vis.get_view_control()

        # only set once while rendering a new sample
        if view_init:
            baseline_view.set_front((-0.0026307837086555776, -0.10911602561147189, 0.99402553887303879))
            baseline_view.set_lookat((-4.1540102715460518, -3.1544474823842861, 16.565590452781144))
            baseline_view.set_up((-0.99303496278054726, 0.11737280671055968, 0.01025606846326051))
            baseline_view.set_zoom((0.14000000000000001))
            cam_params = baseline_view.convert_to_pinhole_camera_parameters()
            view_init = False

        baseline_view.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        ours_view.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        while True:
            cam_params = baseline_view.convert_to_pinhole_camera_parameters()
            ours_view.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
            # update vis
            baseline_vis.poll_events()
            baseline_vis.update_renderer()
            ours_vis.poll_events()  # TODO: if have two poll event, render_prev or next will be triggered twice
            ours_vis.update_renderer()

    def render_next(vis):
        nonlocal sample_idx, skip_flag
        nonlocal baseline_vis, ours_vis

        if skip_flag:
            skip_flag = False
            return None
        else:
            skip_flag = True
            print("render next trigger")
            sample_idx += 1
            if sample_idx >= len(sd_toks_list):
                sample_idx = len(sd_toks_list) - 1
            draw_sample(baseline_vis, ours_vis)

    def render_prev(vis):
        nonlocal sample_idx, skip_flag
        nonlocal baseline_vis, ours_vis
        if skip_flag:
            skip_flag = False
            return None
        else:
            skip_flag = True
            print("render prev trigger")
            sample_idx -= 1
            if sample_idx <= 0:
                sample_idx = 0
            draw_sample(baseline_vis, ours_vis)

    def log_good_cases(vis):
        nonlocal sample_idx, good_cases_logger
        good_cases_logger.info(f"good cases: {sample_idx}, {sd_toks_list[sample_idx]}")

    def log_bad_cases(vis):
        nonlocal sample_idx, bad_cases_logger
        bad_cases_logger.info(f"bad cases: {sample_idx}, {sd_toks_list[sample_idx]}")

    baseline_vis.register_key_callback(ord('D'), render_next)
    baseline_vis.register_key_callback(ord('A'), render_prev)
    baseline_vis.register_key_callback(ord('G'), log_good_cases)
    baseline_vis.register_key_callback(ord('B'), log_bad_cases)

    baseline_vis.run()
    ours_vis.run()


if __name__ == '__main__':
    open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)  # TODO: ignore stupid open3d warnings
    warnings.filterwarnings("ignore")

    with open('vis_cfg.yaml', 'r') as f:
        vis_cfg = yaml.safe_load(f)

    nusc = NuScenes(version='v1.0-trainval', dataroot=vis_cfg['mos']['dataset_root'], verbose=True)
    if not hasattr(nusc, "lidarseg") or len(getattr(nusc, 'lidarseg')) == 0:
        raise RuntimeError(f"No nuscenes-lidarseg annotations found in {nusc.version}")

    # render mos samples
    sd_toks_list = get_vis_sd_toks(nusc, vis_cfg['mos']['source'], 'val', vis_cfg['mos']['ours_dir'], label_suffix='_mos_pred.label')
    render_mos_samples(nusc, sd_toks_list, vis_cfg)