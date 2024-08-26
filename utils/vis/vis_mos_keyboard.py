"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import os
import open3d
import argparse
import warnings
import matplotlib
import numpy as np
import torch
from tqdm import tqdm
import multiprocessing
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.data_classes import LidarPointCloud, LidarSegPointCloud
from nuscenes.utils.splits import create_splits_logs
from datasets.nusc_utils import get_outside_scene_mask, get_ego_mask


classname_to_color = {  # RGB.
        "noise": (255, 255, 255),  # White: noise

        "animal": (100, 149, 237),  # Cornflowerblue: movable people or animals or stuff
        "human.pedestrian.adult": (100, 149, 237),
        "human.pedestrian.child": (100, 149, 237),
        "human.pedestrian.construction_worker": (100, 149, 237),
        "human.pedestrian.personal_mobility": (100, 149, 237),
        "human.pedestrian.police_officer": (100, 149, 237),
        "human.pedestrian.stroller": (100, 149, 237),
        "human.pedestrian.wheelchair": (100, 149, 237),
        "movable_object.barrier": (100, 149, 237),
        "movable_object.debris": (100, 149, 237),
        "movable_object.pushable_pullable": (100, 149, 237),
        "movable_object.trafficcone": (100, 149, 237),

        "static_object.bicycle_rack": (0, 207, 191),  # nuTonomy green: static stuff

        "vehicle.bicycle": (255, 127, 80),  # Coral: movable vehicles
        "vehicle.bus.bendy": (255, 127, 80),
        "vehicle.bus.rigid": (255, 127, 80),
        "vehicle.car": (255, 127, 80),
        "vehicle.construction": (255, 127, 80),
        "vehicle.emergency.ambulance": (255, 127, 80),
        "vehicle.emergency.police": (255, 127, 80),
        "vehicle.motorcycle": (255, 127, 80),
        "vehicle.trailer": (255, 127, 80),
        "vehicle.truck": (255, 127, 80),

        "flat.driveable_surface": (0, 207, 191),  # nuTonomy green: static stuff
        "flat.other": (0, 207, 191),
        "flat.sidewalk": (0, 207, 191),
        "flat.terrain": (0, 207, 191),
        "static.manmade": (0, 207, 191),
        "static.other": (0, 207, 191),
        "static.vegetation": (0, 207, 191),

        "vehicle.ego": (255, 127, 80)  # Coral: movable vehicles
    }


mos_colormap = {
        0: (255/256, 255/256, 255/256),  # unknown: white
        1: (25/256, 80/256, 25/256),    # static: green
        2: (200/256, 20/256, 20/256)     # moving: red
    }

confusion_colormap = {
        0: (255/256, 255/256, 255/256),  # unknown: white
        1: (200/256, 20/256, 20/256),     # tp: mov -> mov
        2: (25/256, 80/256, 25/256),    # tn: sta -> sta
        3: (132/256, 13/256, 209/256), # fp: mov -> sta
        4: (255/256, 232/256, 82/256), # fn: sta -> mov
    }


check_colormap = {
        0: (255/256, 20/256, 20/256),     # moving: red
        1: (255/256, 255/256, 255/256),  # unknown: white
    }


lidarseg_colormap = {  # RGB.
        0: (0, 0, 0),  # Black.
        1: (70, 130, 180),  # Steelblue
        2: (0, 0, 230),  # Blue
        3: (135, 206, 235),  # Skyblue,
        4: (100, 149, 237),  # Cornflowerblue
        5: (219, 112, 147),  # Palevioletred
        6: (0, 0, 128),  # Navy,
        7: (240, 128, 128),  # Lightcoral
        8: (138, 43, 226),  # Blueviolet
        9: (112, 128, 144),  # Slategrey
        10: (210, 105, 30),  # Chocolate
        11: (105, 105, 105),  # Dimgrey
        12: (47, 79, 79),  # Darkslategrey
        13: (188, 143, 143),  # Rosybrown
        14: (220, 20, 60),  # Crimson
        15: (255, 127, 80),  # Coral
        16: (255, 69, 0),  # Orangered
        17: (255, 158, 0),  # Orange
        18: (233, 150, 70),  # Darksalmon
        19: (255, 83, 0),
        20: (255, 215, 0),  # Gold
        21: (255, 61, 99),  # Red
        22: (255, 140, 0),  # Darkorange
        23: (255, 99, 71),  # Tomato
        24: (0, 207, 191),  # nuTonomy green
        25: (175, 0, 75),
        26: (75, 0, 75),
        27: (112, 180, 60),
        28: (222, 184, 135),  # Burlywood
        29: (255, 228, 196),  # Bisque
        30: (0, 175, 0),  # Green
        31: (255, 240, 245)
    }


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def translate_nusc_boxes_to_open3d_instance(gt_boxes):
    """
                 4-------- 6
               /|         /|
              5 -------- 3 .
              | |        | |
              . 7 -------- 1
              |/         |/
              2 -------- 0
        """
    center = gt_boxes[0].center
    w = gt_boxes[0].wlh[0]
    l = gt_boxes[0].wlh[1]
    h = gt_boxes[0].wlh[2]
    lwh = np.array([l, w, h])
    rot = gt_boxes[0].rotation_matrix

    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)
    return line_set, box3d


def draw_box(vis, boxes):
    for box in boxes:
        line_set, box3d = translate_nusc_boxes_to_open3d_instance(box)

        class_name = box[0].name
        color = classname_to_color[class_name]
        color = np.array([color[0]/256, color[1]/256, color[2]/256])
        line_set.paint_uniform_color(color)

        vis.add_geometry(line_set)
    return vis


def render_mos_samples(nusc, sd_toks_list, baseline_dir, ours_dir):
    # vis sample index
    sample_idx = 0

    # open3d visualizer
    gt_vis = open3d.visualization.VisualizerWithKeyCallback()
    baseline_vis = open3d.visualization.Visualizer()
    ours_vis = open3d.visualization.Visualizer()
    gt_vis.create_window(window_name='ground truth mos labels', width=853 * 2, height=1440 * 2, left=0, top=0)
    baseline_vis.create_window(window_name='baseline mos preds', width=853 * 2, height=1440 * 2, left=853 * 2, top=0)
    ours_vis.create_window(window_name='ours mos preds', width=853 * 2, height=1440 * 2, left=853 * 4, top=0)

    def draw_sample(gt_vis, baseline_vis, ours_vis):
        nonlocal sample_idx
        print("Rendering sample: " + str(sample_idx))
        # clear geometry
        gt_vis.clear_geometries()
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
        outside_scene_mask = get_outside_scene_mask(torch.tensor(points), [-70.0, -70.0, -4.5, 70.0, 70.0, 4.5])
        valid_mask = torch.logical_and(~ego_mask, ~outside_scene_mask)
        filter_points = points[valid_mask]

        # bboxes
        boxes = []
        for ann_token in sample['anns']:  # bbox
            _, box, _ = nusc.get_sample_data(sd_tok, selected_anntokens=[ann_token], use_flat_vehicle_coordinates=False)
            boxes.append(box)

        # labels
        gt_labels_file = os.path.join(nusc.dataroot, 'mos_labels', nusc.version, sd_tok + "_mos.label")
        gt_labels = np.fromfile(gt_labels_file, dtype=np.uint8)
        baseline_labels_file = os.path.join(baseline_dir, sd_tok + "_mos_pred.label")
        baseline_labels = np.fromfile(baseline_labels_file, dtype=np.uint8)
        ours_labels_file = os.path.join(ours_dir, sd_tok + "_mos_pred.label")
        ours_labels = np.fromfile(ours_labels_file, dtype=np.uint8)

        # origin
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        gt_vis.add_geometry(axis_pcd)
        baseline_vis.add_geometry(axis_pcd)
        ours_vis.add_geometry(axis_pcd)

        # points
        gt_pts = open3d.geometry.PointCloud()
        gt_pts.points = open3d.utility.Vector3dVector(filter_points[:, :3])
        gt_vis.add_geometry(gt_pts)
        baseline_pts = open3d.geometry.PointCloud()
        baseline_pts.points = open3d.utility.Vector3dVector(filter_points[:, :3])
        baseline_vis.add_geometry(baseline_pts)
        ours_pts = open3d.geometry.PointCloud()
        ours_pts.points = open3d.utility.Vector3dVector(filter_points[:, :3])
        ours_vis.add_geometry(ours_pts)

        # TODO: colored by predicted labels
        # vfunc = np.vectorize(mos_colormap.get)
        # gt_pts.colors = open3d.utility.Vector3dVector(np.array(vfunc(gt_labels[valid_mask])).T)
        # baseline_pts.colors = open3d.utility.Vector3dVector(np.array(vfunc(baseline_labels)).T)
        # ours_pts.colors = open3d.utility.Vector3dVector(np.array(vfunc(ours_labels)).T)

        # TODO: colored by TP TN, FP FN
        gt_labels = gt_labels[valid_mask]
        unk_mask = gt_labels == 0
        minus_labels = gt_labels - baseline_labels
        true_mask = np.logical_and(~unk_mask, minus_labels == 0)
        tp_mask = np.logical_and(true_mask, gt_labels == 2)
        tn_mask = np.logical_and(true_mask, gt_labels == 1)
        fp_mask = np.logical_and(~unk_mask, minus_labels == 1)
        fn_mask = np.logical_and(~unk_mask, minus_labels == -1)

        colors = np.zeros_like(gt_labels)
        colors[tp_mask] = 1
        colors[tn_mask] = 2
        colors[fp_mask] = 3
        colors[fn_mask] = 4

        vfunc = np.vectorize(confusion_colormap.get)
        gt_pts.colors = open3d.utility.Vector3dVector(np.array(vfunc(gt_labels)).T)
        baseline_pts.colors = open3d.utility.Vector3dVector(np.array(vfunc(colors)).T)
        ours_pts.colors = open3d.utility.Vector3dVector(np.array(vfunc(colors)).T)

        # bbox
        gt_vis = draw_box(gt_vis, boxes)
        baseline_vis = draw_box(baseline_vis, boxes)
        ours_vis = draw_box(ours_vis, boxes)

        # view settings
        gt_vis.get_render_option().point_size = 9.0
        gt_vis.get_render_option().background_color = np.zeros(3)
        baseline_vis.get_render_option().point_size = 9.0
        baseline_vis.get_render_option().background_color = np.zeros(3)
        ours_vis.get_render_option().point_size = 9.0
        ours_vis.get_render_option().background_color = np.zeros(3)

        # view options
        gt_view = gt_vis.get_view_control()
        baseline_view = baseline_vis.get_view_control()
        ours_view = ours_vis.get_view_control()

        gt_view.set_front(( -0.0026307837086555776, -0.10911602561147189, 0.99402553887303879 ))
        gt_view.set_lookat((-4.1540102715460518, -3.1544474823842861, 16.565590452781144))
        gt_view.set_up((-0.99303496278054726, 0.11737280671055968, 0.01025606846326051))
        gt_view.set_zoom((0.36000000000000004))

        cam_params = gt_view.convert_to_pinhole_camera_parameters()
        baseline_view.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        ours_view.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)


        # following
        gt_vis.poll_events()
        baseline_vis.poll_events()
        ours_vis.poll_events()

        # update vis
        gt_vis.update_renderer()
        baseline_vis.update_renderer()
        ours_vis.update_renderer()

    def render_next(vis):
        nonlocal sample_idx
        nonlocal gt_vis, baseline_vis, ours_vis
        sample_idx += 1
        if sample_idx >= len(sd_toks_list):
            sample_idx = len(sd_toks_list) - 1
        draw_sample(gt_vis, baseline_vis, ours_vis)

    def render_prev(vis):
        nonlocal sample_idx
        nonlocal gt_vis, baseline_vis, ours_vis
        sample_idx -= 1
        if sample_idx < 0:
            sample_idx = 0
        draw_sample(gt_vis, baseline_vis, ours_vis)

    gt_vis.register_key_callback(ord('D'), render_next)
    gt_vis.register_key_callback(ord('A'), render_prev)
    gt_vis.run()
    baseline_vis.run()
    ours_vis.run()


def render_mos_samples_with_confusion_color(nusc, sd_toks_list, baseline_dir, ours_dir, pred_sd_toks_list):
    # vis sample index
    sample_idx = 0

    # open3d visualizer
    baseline_vis = open3d.visualization.VisualizerWithKeyCallback()
    ours_vis = open3d.visualization.Visualizer()
    baseline_vis.create_window(window_name='baseline mos results', width=853 * 2, height=1440 * 2, left=0, top=0)
    ours_vis.create_window(window_name='ours mos results', width=853 * 2, height=1440 * 2, left=853 * 2, top=0)

    def get_confusion_color(gt_labels, pred_labels):
        unk_mask = gt_labels == 0
        minus_labels = gt_labels - pred_labels
        true_mask = np.logical_and(~unk_mask, minus_labels == 0)
        tp_mask = np.logical_and(true_mask, gt_labels == 2)
        tn_mask = np.logical_and(true_mask, gt_labels == 1)
        fp_mask = np.logical_and(~unk_mask, minus_labels == 1)
        fn_mask = np.logical_and(~unk_mask, minus_labels == -1)

        colors = np.zeros_like(gt_labels)
        colors[tp_mask] = 1
        colors[tn_mask] = 2
        colors[fp_mask] = 3
        colors[fn_mask] = 4
        return colors

    def draw_sample(baseline_vis, ours_vis):
        nonlocal sample_idx
        print("Rendering sample: " + str(sample_idx))
        # clear geometry
        baseline_vis.clear_geometries()
        ours_vis.clear_geometries()

        # sample and sample data
        sd_tok = sd_toks_list[sample_idx]
        if sd_tok not in pred_sd_toks_list:
            return None

        sample_data = nusc.get('sample_data', sd_tok)
        sample_token = sample_data['sample_token']
        sample = nusc.get("sample", sample_token)

        # points
        pcl_path = os.path.join(nusc.dataroot, sample_data['filename'])
        points = LidarPointCloud.from_file(pcl_path).points.T  # [num_points, 4]
        ego_mask = get_ego_mask(torch.tensor(points))
        outside_scene_mask = get_outside_scene_mask(torch.tensor(points), [-70.0, -70.0, -4.5, 70.0, 70.0, 4.5])
        valid_mask = torch.logical_and(~ego_mask, ~outside_scene_mask)
        points = points[valid_mask]

        # bboxes
        boxes = []
        for ann_token in sample['anns']:  # bbox
            _, box, _ = nusc.get_sample_data(sd_tok, selected_anntokens=[ann_token], use_flat_vehicle_coordinates=False)
            boxes.append(box)

        # labels
        gt_labels_file = os.path.join(nusc.dataroot, 'mos_labels', nusc.version, sd_tok + "_mos.label")
        gt_labels = np.fromfile(gt_labels_file, dtype=np.uint8)[valid_mask]
        baseline_labels_file = os.path.join(baseline_dir, sd_tok + "_mos_pred.label")
        baseline_labels = np.fromfile(baseline_labels_file, dtype=np.uint8)
        ours_labels_file = os.path.join(ours_dir, sd_tok + "_mos_pred.label")
        ours_labels = np.fromfile(ours_labels_file, dtype=np.uint8)

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

        # colored by TP TN, FP FN
        baseline_colors = get_confusion_color(gt_labels, baseline_labels)
        ours_colors = get_confusion_color(gt_labels, ours_labels)
        vfunc = np.vectorize(confusion_colormap.get)
        baseline_pts.colors = open3d.utility.Vector3dVector(np.array(vfunc(baseline_colors)).T)
        ours_pts.colors = open3d.utility.Vector3dVector(np.array(vfunc(ours_colors)).T)

        # bbox
        baseline_vis = draw_box(baseline_vis, boxes)
        ours_vis = draw_box(ours_vis, boxes)

        # view settings
        baseline_vis.get_render_option().point_size = 9.0
        baseline_vis.get_render_option().background_color = np.zeros(3)
        ours_vis.get_render_option().point_size = 9.0
        ours_vis.get_render_option().background_color = np.zeros(3)

        # view options
        baseline_view = baseline_vis.get_view_control()
        ours_view = ours_vis.get_view_control()

        baseline_view.set_front((-0.0026307837086555776, -0.10911602561147189, 0.99402553887303879))
        baseline_view.set_lookat((-4.1540102715460518, -3.1544474823842861, 16.565590452781144))
        baseline_view.set_up((-0.99303496278054726, 0.11737280671055968, 0.01025606846326051))
        baseline_view.set_zoom((0.14000000000000001))

        cam_params = baseline_view.convert_to_pinhole_camera_parameters()
        ours_view.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

        # update vis
        baseline_vis.poll_events()
        ours_vis.poll_events()
        baseline_vis.update_renderer()
        ours_vis.update_renderer()

    def render_next(vis):
        nonlocal sample_idx
        nonlocal baseline_vis, ours_vis
        sample_idx += 1
        if sample_idx >= len(sd_toks_list):
            sample_idx = len(sd_toks_list) - 1
        draw_sample(baseline_vis, ours_vis)

    def render_prev(vis):
        nonlocal sample_idx
        nonlocal baseline_vis, ours_vis
        sample_idx -= 1
        if sample_idx < 0:
            sample_idx = 0
        draw_sample(baseline_vis, ours_vis)

    baseline_vis.register_key_callback(ord('D'), render_next)
    baseline_vis.register_key_callback(ord('A'), render_prev)
    baseline_vis.run()
    ours_vis.run()


def split_to_samples(nusc, split_logs):
    sample_tokens = []  # store the sample tokens
    sample_data_tokens = []
    for sample in nusc.sample:
        sample_data_token = sample['data']['LIDAR_TOP']
        scene = nusc.get('scene', sample['scene_token'])
        log = nusc.get('log', scene['log_token'])
        logfile = log['logfile']
        if logfile in split_logs:
            sample_data_tokens.append(sample_data_token)
            sample_tokens.append(sample['token'])
    return sample_tokens, sample_data_tokens


def split_to_samples_singapore(nusc, split_logs):
    sample_tokens = []  # store the sample tokens
    sample_data_tokens = []
    for sample in nusc.sample:
        sample_data_token = sample['data']['LIDAR_TOP']
        scene = nusc.get('scene', sample['scene_token'])
        log = nusc.get('log', scene['log_token'])
        logfile = log['logfile']
        if logfile in split_logs:
            if log["location"].startswith("singapore"):
                sample_data_tokens.append(sample_data_token)
                sample_tokens.append(sample['token'])
    return sample_tokens, sample_data_tokens


def split_to_samples_boston(nusc, split_logs):
    sample_tokens = []  # store the sample tokens
    sample_data_tokens = []
    for sample in nusc.sample:
        sample_data_token = sample['data']['LIDAR_TOP']
        scene = nusc.get('scene', sample['scene_token'])
        log = nusc.get('log', scene['log_token'])
        logfile = log['logfile']
        if logfile in split_logs:
            if log["location"].startswith("boston"):
                sample_data_tokens.append(sample_data_token)
                sample_tokens.append(sample['token'])
    return sample_tokens, sample_data_tokens


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    baseline_dir = '/home/user/Projects/4DOCC/logs/mos_baseline/mos4d_train/50%nuscenes/vs-0.1_t-9.5_bs-4/version_0/predictions/epoch_99'
    ours_dir = '/home/user/Projects/4DOCC/logs/ours/bg_pretrain(epoch-99)-mos_finetune/100%nuscenes-50%nuscenes/vs-0.1_t-9.5_bs-4/version_0/predictions/epoch_99'
    # ours_dir = '/home/user/Projects/4DOCC/logs/ours/bg_pretrain(epoch-49)-mos_finetune/100%nuscenes-50%nuscenes/vs-0.1_t-9.5_bs-4/version_0/predictions/epoch_49'
    samples_source = 'all'  # 'all', 'predicted', 'given'

    parser = argparse.ArgumentParser(description='Generate nuScenes lidar panaptic gt.')
    parser.add_argument('--root_dir', type=str, default='/home/user/Datasets/nuScenes')
    parser.add_argument('--baseline_dir', type=str, default=baseline_dir)
    parser.add_argument('--ours_dir', type=str, default=ours_dir)
    parser.add_argument('--samples_source', type=str, choices=['all', 'predicted', 'given'], default=samples_source)
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--verbose', type=bool, default=True, help='Whether to print to stdout.')
    args = parser.parse_args()

    print(f'Start rendering... \nArguments: {args}')
    nusc = NuScenes(version=args.version, dataroot=args.root_dir, verbose=args.verbose)
    if not hasattr(nusc, "lidarseg") or len(getattr(nusc, 'lidarseg')) == 0:
        raise RuntimeError(f"No nuscenes-lidarseg annotations found in {nusc.version}")
    warnings.filterwarnings("ignore")

    if samples_source == 'all':
        split = "val"
        split_logs = create_splits_logs(split, nusc)
        sample_tokens, sd_toks_list = split_to_samples(nusc, split_logs)

        mos_pred_file_list = os.listdir(args.ours_dir)
        for i in range(len(mos_pred_file_list)):
            mos_pred_file_list[i] = mos_pred_file_list[i].replace('_mos_pred.label', '')
        pred_sd_toks_list = mos_pred_file_list
        # sd_toks_list = list(set(sd_toks_list).intersection(set(pred_sd_toks_list)))
    elif samples_source == 'predicted':
        mos_pred_file_list = os.listdir(args.ours_dir)
        for i in range(len(mos_pred_file_list)):
            mos_pred_file_list[i] = mos_pred_file_list[i].replace('_mos_pred.label', '')
        sd_toks_list = mos_pred_file_list
    elif samples_source == 'given':
        sd_toks_list = [
            # good
            'ceb1ea63e1ed4723a4733519efba7f0c',
            '2a4fbce161e8415bae3b1060fa7aae1b',
            'c6bc2dbe5b054f39affedc328858ed85',
            'd41de152975643e582db47ccb82af9da',
            'f9dfbfbc874741d0bd2f09435e3ecce5',
            '5ddfabbb76be4c3bbd7b399235c58f4b',
            '08afd415763142b4803ea5210bdc7b3c',
            'ee6b827d19eb4f05a9a2926104d41810',
            'cdcd16821ea048d1bf1e856b15bd87be',
            '170657f8642646a2ad676a7d5490ce02',
            'c9d9d8c615c5488ea284b5d109eea834',
            'b110abb9219e473798f10b1c809656ca',

            # bad
            'c2d48f3c6de24822869a31de381d90c8',
            '95b6b42748234237a58df02029b233f3',
            'c6879ea1c3d845eebd7825e6e454bee1',
            '5866a2059b964ff68d6868d8562c4b58',
            'b0ed3aefd1f54695b9c478deb152a0eb',
        ]
    else:
        sd_toks_list = None

    # render mos samples
    render_mos_samples_with_confusion_color(nusc, sd_toks_list, args.baseline_dir, args.ours_dir, pred_sd_toks_list)
    print('Finished rendering.')