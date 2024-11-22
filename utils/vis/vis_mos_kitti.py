import logging
import os
from datetime import datetime
import open3d
import warnings
import numpy as np
import torch
import yaml

from datasets.mos4d.kitti import KittiMOSDataset
from open3d_vis_utils import draw_box, get_confusion_color, mos_color_func
from datasets.kitti_utils import get_ego_mask, get_outside_scene_mask, load_mos_labels


def render_mos_samples(vis_set, vis_cfg):
    # visualization cfg
    sample_idx = vis_cfg['mos']['start_sample_idx']
    scan_idx_list = [74, 116, 119]
    point_size = vis_cfg['mos']['point_size']
    baseline_dir = vis_cfg['mos']['baseline_dir']
    occ4d_dir = vis_cfg['mos']['occ4d_dir']
    uno_dir = vis_cfg['mos']['uno_dir']
    ours_dir = vis_cfg['mos']['ours_dir']

    # TODO: skip flag 实在没办法解决按一次render两次. 打个补丁吧
    skip_flag = True
    view_init = True
    cam_params = None

    # open3d visualizer
    baseline_vis = open3d.visualization.VisualizerWithKeyCallback()
    occ4d_vis = open3d.visualization.Visualizer()
    uno_vis = open3d.visualization.Visualizer()
    ours_vis = open3d.visualization.Visualizer()
    baseline_vis.create_window(window_name='baseline mos results', width=2560, height=1360, left=0, top=0)
    occ4d_vis.create_window(window_name='occ4d mos results', width=2560, height=1360, left=2560, top=0)
    uno_vis.create_window(window_name='uno mos results', width=2560, height=1360, left=0, top=1510)
    ours_vis.create_window(window_name='ours mos results', width=2560, height=1360, left=2560, top=1510)

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

    def draw_sample(baseline_vis, ours_vis, occ4d_vis, uno_vis):
        nonlocal sample_idx, cam_params, view_init
        # clear geometry
        baseline_vis.clear_geometries()
        ours_vis.clear_geometries()
        occ4d_vis.clear_geometries()
        uno_vis.clear_geometries()
        scan_idx = scan_idx_list[sample_idx]

        # points
        pcd_file = vis_set.samples_scans[scan_idx][-1]
        pcd = np.fromfile(pcd_file, dtype=np.float32)
        pcd = torch.tensor(pcd.reshape((-1, 4)))[:, :3]
        kitti_scan_idx = pcd_file.split(".")[0][-6:]
        print("Rendering sample: " + str(scan_idx) + "; Scan index: " + str(kitti_scan_idx))

        # mos labels
        gt_labels = load_mos_labels(vis_set.samples_labels[scan_idx][-1]).numpy()

        # ego mask and outside scene mask
        ego_mask = get_ego_mask(torch.tensor(pcd))
        outside_scene_mask = get_outside_scene_mask(torch.tensor(pcd), vis_cfg['mos']['scene_bbox'],
                                                                       mask_z=False, upper_bound=True)
        valid_mask = torch.logical_and(~ego_mask, ~outside_scene_mask)
        pcd = pcd[valid_mask]
        gt_labels = gt_labels[valid_mask]

        # predicted labels
        baseline_labels_file = os.path.join(baseline_dir, kitti_scan_idx + "_mos_pred.label") # mos
        baseline_labels = np.fromfile(baseline_labels_file, dtype=np.uint8) # valid mask
        ours_labels_file = os.path.join(ours_dir, kitti_scan_idx + "_mos_pred.label")
        ours_labels = np.fromfile(ours_labels_file, dtype=np.uint8)
        occ4d_labels_file = os.path.join(occ4d_dir, kitti_scan_idx + "_mos_pred.label")
        occ4d_labels = np.fromfile(occ4d_labels_file, dtype=np.uint8)
        uno_labels_file = os.path.join(uno_dir, kitti_scan_idx + "_mos_pred.label")
        uno_labels = np.fromfile(uno_labels_file, dtype=np.uint8)

        # origin
        # axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        # baseline_vis.add_geometry(axis_pcd)
        # ours_vis.add_geometry(axis_pcd)
        # occ4d_vis.add_geometry(axis_pcd)
        # uno_vis.add_geometry(axis_pcd)

        # points
        baseline_pts = open3d.geometry.PointCloud()
        baseline_pts.points = open3d.utility.Vector3dVector(pcd)
        baseline_vis.add_geometry(baseline_pts)
        ours_pts = open3d.geometry.PointCloud()
        ours_pts.points = open3d.utility.Vector3dVector(pcd)
        ours_vis.add_geometry(ours_pts)
        occ4d_pts = open3d.geometry.PointCloud()
        occ4d_pts.points = open3d.utility.Vector3dVector(pcd)
        occ4d_vis.add_geometry(occ4d_pts)
        uno_pts = open3d.geometry.PointCloud()
        uno_pts.points = open3d.utility.Vector3dVector(pcd)
        uno_vis.add_geometry(uno_pts)

        # colored by TP TN FP FN
        baseline_colors = get_confusion_color(gt_labels, baseline_labels)
        ours_colors = get_confusion_color(gt_labels, ours_labels)
        occ4d_colors = get_confusion_color(gt_labels, occ4d_labels)
        uno_colors = get_confusion_color(gt_labels, uno_labels)

        # apply color
        baseline_pts.colors = open3d.utility.Vector3dVector(np.array(mos_color_func(baseline_colors)).T)
        ours_pts.colors = open3d.utility.Vector3dVector(np.array(mos_color_func(ours_colors)).T)
        occ4d_pts.colors = open3d.utility.Vector3dVector(np.array(mos_color_func(occ4d_colors)).T)
        uno_pts.colors = open3d.utility.Vector3dVector(np.array(mos_color_func(uno_colors)).T)

        # view settings
        bg_color = (220 / 255, 220 / 255, 220 / 255)
        baseline_vis.get_render_option().point_size = point_size
        baseline_vis.get_render_option().background_color = bg_color
        ours_vis.get_render_option().point_size = point_size
        ours_vis.get_render_option().background_color = bg_color
        occ4d_vis.get_render_option().point_size = point_size
        occ4d_vis.get_render_option().background_color = bg_color
        uno_vis.get_render_option().point_size = point_size
        uno_vis.get_render_option().background_color = bg_color

        # view options
        baseline_view = baseline_vis.get_view_control()
        ours_view = ours_vis.get_view_control()
        occ4d_view = occ4d_vis.get_view_control()
        uno_view = uno_vis.get_view_control()

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
        occ4d_view.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        uno_view.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        while True:
            cam_params = baseline_view.convert_to_pinhole_camera_parameters()
            ours_view.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
            occ4d_view.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
            uno_view.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

            # update vis
            baseline_vis.poll_events()
            baseline_vis.update_renderer()
            ours_vis.poll_events()  # TODO: if have two poll event, render_prev or next will be triggered twice
            ours_vis.update_renderer()
            occ4d_vis.poll_events()
            occ4d_vis.update_renderer()
            uno_vis.poll_events()
            uno_vis.update_renderer()

    def render_next(vis):
        nonlocal sample_idx, skip_flag
        nonlocal baseline_vis, ours_vis, occ4d_vis, uno_vis

        if skip_flag:
            skip_flag = False
            return None
        else:
            skip_flag = True
            print("render next trigger")
            sample_idx += 1
            if sample_idx >= len(vis_set.samples_labels):
                sample_idx = len(vis_set.samples_labels) - 1
            draw_sample(baseline_vis, ours_vis, occ4d_vis, uno_vis)

    def render_prev(vis):
        nonlocal sample_idx, skip_flag
        nonlocal baseline_vis, ours_vis, occ4d_vis, uno_vis
        if skip_flag:
            skip_flag = False
            return None
        else:
            skip_flag = True
            print("render prev trigger")
            sample_idx -= 1
            if sample_idx <= 0:
                sample_idx = 0
            draw_sample(baseline_vis, ours_vis, occ4d_vis, uno_vis)

    def log_good_cases(vis):
        nonlocal sample_idx, good_cases_logger
        good_cases_logger.info(f"good cases: {sample_idx}")

    def log_bad_cases(vis):
        nonlocal sample_idx, bad_cases_logger
        bad_cases_logger.info(f"bad cases: {sample_idx}")

    baseline_vis.register_key_callback(ord('D'), render_next)
    baseline_vis.register_key_callback(ord('A'), render_prev)
    baseline_vis.register_key_callback(ord('G'), log_good_cases)
    baseline_vis.register_key_callback(ord('B'), log_bad_cases)

    baseline_vis.run()
    ours_vis.run()
    occ4d_vis.run()
    uno_vis.run()


if __name__ == '__main__':
    open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)  # TODO: ignore stupid open3d warnings
    warnings.filterwarnings("ignore")

    with open('vis_cfg.yaml', 'r') as f:
        cfg_vis = yaml.safe_load(f)

    # load semantic kitti dataset
    with open("../../configs/mos4d.yaml", "r") as f:
        cfg_model = yaml.safe_load(f)['train']
    with open("../../configs/dataset.yaml", "r") as f:
        cfg_dataset = yaml.safe_load(f)
    cfg_dataset['sekitti']['root'] = '/home/ziliang' + cfg_dataset['sekitti']['root']
    val_set = KittiMOSDataset(cfg_model, cfg_dataset, split='val')

    # render mos samples
    render_mos_samples(val_set, cfg_vis)