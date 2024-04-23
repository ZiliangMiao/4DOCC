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
from tqdm import tqdm
import multiprocessing
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.data_classes import LidarPointCloud, LidarSegPointCloud
from nuscenes.utils.splits import create_splits_logs

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
        0: (255/255, 255/255, 255/255),  # unknown: white
        1: (25/255, 80/255, 25/255),    # static: green
        2: (255/255, 20/255, 20/255)     # moving: red
    }

check_colormap = {
        0: (255/255, 20/255, 20/255),     # moving: red
        1: (255/255, 255/255, 255/255),  # unknown: white
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
        color = np.array([color[0]/255, color[1]/255, color[2]/255])
        line_set.paint_uniform_color(color)

        vis.add_geometry(line_set)
    return vis

def render_samples(nusc, sample_tokens, show_mos_gt=True, show_mos_pred=False, show_lidarseg=False, show_inconsistent=False, show_inside=False):
    if show_mos_gt:
        show_mos_pred = False
        show_lidarseg = False
        show_inconsistent = False
        show_inside = False
    elif show_mos_pred:
        show_mos_gt = False
        show_lidarseg = False
        show_inconsistent = False
        show_inside = False
    elif show_lidarseg:
        show_mos_gt = False
        show_mos_pred = False
        show_inconsistent = False
        show_inside = False
    elif show_inside:
        show_mos_gt = False
        show_mos_pred = False
        show_lidarseg = False
        show_inconsistent = False
    elif show_inconsistent:
        show_mos_gt = False
        show_mos_pred = False
        show_lidarseg = False
        show_inside = False

    sample_idx = 0
    vis = open3d.visualization.VisualizerWithKeyCallback()

    def draw_sample(vis):
        nonlocal sample_idx
        print("Rendering sample: " + str(sample_idx))
        # clear geometry
        vis.clear_geometries()

        # get points and bboxes
        sample_token = sample_tokens[sample_idx]
        sample = nusc.get("sample", sample_token)
        lidar_tok = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_tok)
        pcl_path = os.path.join(nusc.dataroot, lidar_data['filename'])
        points = LidarPointCloud.from_file(pcl_path).points.T  # [num_points, 4]
        boxes = []
        for ann_token in sample['anns']:  # bbox
            _, box, _ = nusc.get_sample_data(lidar_tok, selected_anntokens=[ann_token], use_flat_vehicle_coordinates=False)
            boxes.append(box)
        # get mos, lidarseg, inconsistent_bbox or inside_bbox labels:
        points_label = None
        if show_mos_gt:
            mos_labels_file = os.path.join(nusc.dataroot, 'mos_labels', nusc.version, lidar_tok + "_mos.label")
            points_label = np.fromfile(mos_labels_file, dtype=np.uint8)
        elif show_mos_pred:
            mos_labels_file = os.path.join(nusc.dataroot, '4dmos_sekitti_pred', nusc.version, lidar_tok + "_mos_pred.label")
            points_label = np.fromfile(mos_labels_file, dtype=np.uint8)
        elif show_lidarseg:
            lidarseg_file = os.path.join(nusc.dataroot, nusc.get('lidarseg', lidar_tok)['filename'])
            points_label = np.fromfile(lidarseg_file, dtype=np.uint8)
        elif show_inconsistent:
            inconsistent_file = os.path.join(nusc.dataroot, 'inconsistent_bbox', nusc.version, lidar_tok + "_inconsistent.bin")
            points_label = np.fromfile(inconsistent_file, dtype=np.uint8)
        elif show_inside:
            inside_file = os.path.join(nusc.dataroot, 'inside_bbox', nusc.version, lidar_tok + "_inside.bin")
            points_label = np.fromfile(inside_file, dtype=np.uint8)

        # draw origin
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

        # draw points
        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(points[:, :3])
        vis.add_geometry(pts)

        # draw points label
        if show_mos_gt or show_mos_pred:
            vfunc = np.vectorize(mos_colormap.get)
            points_color = np.array(vfunc(points_label)).T
        elif show_inconsistent:
            vfunc = np.vectorize(check_colormap.get)
            points_color = np.array(vfunc(points_label)).T
        elif show_inside:
            vfunc = np.vectorize(check_colormap.get)
            points_color = np.array(vfunc(points_label)).T
        else:
            vfunc = np.vectorize(lidarseg_colormap.get)
            points_color = np.array(vfunc(points_label)).T
        pts.colors = open3d.utility.Vector3dVector(points_color)

        # draw bbox
        vis = draw_box(vis, boxes)

        # view settings
        vis.get_render_option().point_size = 3.0
        vis.get_render_option().background_color = np.zeros(3)

        view_ctrl = vis.get_view_control()
        view_ctrl.set_front((0.75263429526187886, -0.13358133681379755, 0.64474618575893383))
        view_ctrl.set_lookat((16.206845402638745, -3.8676194858766819, 15.365323753623207))
        view_ctrl.set_up((-0.64932205862151104, 0.011806106960120792, 0.76042190922274799))
        view_ctrl.set_zoom((0.17999999999999999))

        # update vis
        vis.poll_events()
        vis.update_renderer()


    def render_next(vis):
        nonlocal sample_idx
        sample_idx += 1
        if sample_idx >= len(sample_tokens):
            sample_idx = len(sample_tokens) - 1
        draw_sample(vis)

    def render_prev(vis):
        nonlocal sample_idx
        sample_idx -= 1
        if sample_idx < 0:
            sample_idx = 0
        draw_sample(vis)

    vis.create_window()
    vis.register_key_callback(ord('D'), render_next)
    vis.register_key_callback(ord('A'), render_prev)
    vis.run()

def render_mos_samples(nusc, sample_tokens, pred_dir):
    sample_idx = 0
    vis_gt = open3d.visualization.VisualizerWithKeyCallback()
    vis_pred = open3d.visualization.Visualizer()

    def draw_sample(vis_gt, vis_pred):
        nonlocal sample_idx
        print("Rendering sample: " + str(sample_idx))
        # clear geometry
        vis_gt.clear_geometries()
        vis_pred.clear_geometries()

        # get points and bboxes
        sample_token = sample_tokens[sample_idx]
        sample = nusc.get("sample", sample_token)
        lidar_tok = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_tok)
        pcl_path = os.path.join(nusc.dataroot, lidar_data['filename'])
        points = LidarPointCloud.from_file(pcl_path).points.T  # [num_points, 4]
        boxes = []
        for ann_token in sample['anns']:  # bbox
            _, box, _ = nusc.get_sample_data(lidar_tok, selected_anntokens=[ann_token], use_flat_vehicle_coordinates=False)
            boxes.append(box)

        # get gt and pred mos labels:
        gt_labels_file = os.path.join(nusc.dataroot, 'mos_labels', nusc.version, lidar_tok + "_mos.label")
        gt_labels = np.fromfile(gt_labels_file, dtype=np.uint8)
        pred_labels_file = os.path.join(pred_dir, lidar_tok + "_mos_pred.label")
        pred_labels = np.fromfile(pred_labels_file, dtype=np.uint8)

        # draw origin
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis_gt.add_geometry(axis_pcd)
        vis_pred.add_geometry(axis_pcd)

        # draw points
        pts_gt = open3d.geometry.PointCloud()
        pts_gt.points = open3d.utility.Vector3dVector(points[:, :3])
        pts_pred = open3d.geometry.PointCloud()

        ###################################
        # visualize flipped x-axis for singapore dataset
        # points[:, 0] *= -1
        ###################################

        pts_pred.points = open3d.utility.Vector3dVector(points[:, :3])
        vis_gt.add_geometry(pts_gt)
        vis_pred.add_geometry(pts_pred)

        # draw points label
        vfunc = np.vectorize(mos_colormap.get)
        pts_gt.colors = open3d.utility.Vector3dVector(np.array(vfunc(gt_labels)).T)
        pts_pred.colors = open3d.utility.Vector3dVector(np.array(vfunc(pred_labels)).T)

        # draw bbox
        vis_gt = draw_box(vis_gt, boxes)
        vis_pred = draw_box(vis_pred, boxes)

        # view settings
        vis_gt.get_render_option().point_size = 7.0
        vis_gt.get_render_option().background_color = np.zeros(3)
        vis_pred.get_render_option().point_size = 7.0
        vis_pred.get_render_option().background_color = np.zeros(3)

        view_ctrl_gt = vis_gt.get_view_control()
        view_ctrl_gt.set_front((0.75263429526187886, -0.13358133681379755, 0.64474618575893383))
        view_ctrl_gt.set_lookat((16.206845402638745, -3.8676194858766819, 15.365323753623207))
        view_ctrl_gt.set_up((-0.64932205862151104, 0.011806106960120792, 0.76042190922274799))
        view_ctrl_gt.set_zoom((0.17999999999999999))
        view_ctrl_pred = vis_pred.get_view_control()
        view_ctrl_pred.set_front((0.75263429526187886, -0.13358133681379755, 0.64474618575893383))
        view_ctrl_pred.set_lookat((16.206845402638745, -3.8676194858766819, 15.365323753623207))
        view_ctrl_pred.set_up((-0.64932205862151104, 0.011806106960120792, 0.76042190922274799))
        view_ctrl_pred.set_zoom((0.17999999999999999))

        # update vis
        vis_gt.poll_events()
        vis_gt.update_renderer()
        vis_pred.poll_events()
        vis_pred.update_renderer()

    def render_next(vis):
        nonlocal sample_idx
        nonlocal vis_gt, vis_pred
        sample_idx += 1
        if sample_idx >= len(sample_tokens):
            sample_idx = len(sample_tokens) - 1
        draw_sample(vis_gt, vis_pred)

    def render_prev(vis):
        nonlocal sample_idx
        nonlocal vis_gt, vis_pred
        sample_idx -= 1
        if sample_idx < 0:
            sample_idx = 0
        draw_sample(vis_gt, vis_pred)

    # display resolution * display scale (2560 * 1440 * 175%)
    vis_gt.create_window(window_name='ground truth mos labels', width=2160, height=2300, left=0, top=100, visible=True)
    vis_pred.create_window(window_name='predicted mos labels', width=2160, height=2300, left=2160, top=100, visible=True)
    vis_gt.register_key_callback(ord('D'), render_next)
    vis_gt.register_key_callback(ord('A'), render_prev)
    vis_gt.run()
    vis_pred.run()

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
    parser = argparse.ArgumentParser(description='Generate nuScenes lidar panaptic gt.')
    parser.add_argument('--root_dir', type=str, default='/home/user/Datasets/nuScenes')
    parser.add_argument('--pred_dir', type=str, default="/home/user/Projects/4DOCC/logs/mos4d/100%nuscenes/vs-0.2_t-0.5_bs-8/version_1/results/epoch_34/predictions/mos_pred")

    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--verbose', type=bool, default=True, help='Whether to print to stdout.')
    args = parser.parse_args()

    print(f'Start rendering... \nArguments: {args}')
    nusc = NuScenes(version=args.version, dataroot=args.root_dir, verbose=args.verbose)
    if not hasattr(nusc, "lidarseg") or len(getattr(nusc, 'lidarseg')) == 0:
        raise RuntimeError(f"No nuscenes-lidarseg annotations found in {nusc.version}")

    warnings.filterwarnings("ignore")

    # split train, val, test samples
    split = "val"
    split_logs = create_splits_logs(split, nusc)
    sample_tokens, sample_data_tokens = split_to_samples_singapore(nusc, split_logs)

    # render mos samples
    # render_samples(nusc, sample_tokens, show_mos_gt=False, show_mos_pred=True, show_inconsistent=False, show_inside=False)
    render_mos_samples(nusc, sample_tokens, args.pred_dir)
    print('Finished rendering.')