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


def render_mos_samples(vis_dataset, vis_cfg, vis_gt):
    # visualization cfg
    sample_idx = vis_cfg['mos']['start_sample_idx']
    # scan_idx_list = [74, 116, 119]
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
    vis_list = []
    window_configs = []
    baseline_vis = open3d.visualization.VisualizerWithKeyCallback()
    vis_list.append(('baseline', baseline_vis))
    window_configs.append({
        'name': 'baseline mos results',
        'width': 2560,
        'height': 1360,
        'left': 0,
        'top': 0
    })
    if occ4d_dir is not None:
        occ4d_vis = open3d.visualization.Visualizer()
        vis_list.append(('occ4d', occ4d_vis))
        window_configs.append({
            'name': 'occ4d mos results',
            'width': 2560,
            'height': 1360,
            'left': 2560,
            'top': 0
        })
    if uno_dir is not None:
        uno_vis = open3d.visualization.Visualizer()
        vis_list.append(('uno', uno_vis))
        window_configs.append({
            'name': 'uno mos results',
            'width': 2560,
            'height': 1360,
            'left': 0,
            'top': 1510
        })
    if ours_dir is not None:
        ours_vis = open3d.visualization.Visualizer()
        vis_list.append(('ours', ours_vis))
        window_configs.append({
            'name': 'ours mos results',
            'width': 2560,
            'height': 1360,
            'left': 2560,
            'top': 1510
        })

    # create windows
    for (_, vis), config in zip(vis_list, window_configs):
        vis.create_window(
            window_name=config['name'],
            width=config['width'],
            height=config['height'],
            left=config['left'],
            top=config['top']
        )

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

    def draw_sample():
        nonlocal sample_idx, cam_params, view_init
        # clear geometry
        for _, vis in vis_list:
            vis.clear_geometries()
        
        # scan_idx = scan_idx_list[sample_idx]
        scan_idx = sample_idx

        # points
        pcd_file = vis_dataset.samples_scans[scan_idx][-1]
        pcd = np.fromfile(pcd_file, dtype=np.float32)
        pcd = torch.tensor(pcd.reshape((-1, 4)))[:, :3]
        kitti_scan_idx = pcd_file.mode(".")[0][-6:]
        print("Rendering sample: " + str(scan_idx) + "; Scan index: " + str(kitti_scan_idx))

        # mos labels
        gt_labels = load_mos_labels(vis_dataset.samples_labels[scan_idx][-1]).numpy()

        # ego mask and outside scene mask
        ego_mask = get_ego_mask(torch.tensor(pcd))
        outside_scene_mask = get_outside_scene_mask(torch.tensor(pcd), vis_cfg['mos']['scene_bbox'],
                                                                       mask_z=False, upper_bound=True)
        valid_mask = torch.logical_and(~ego_mask, ~outside_scene_mask)
        pcd = pcd[valid_mask]
        gt_labels = gt_labels[valid_mask]

        # todo: uniform downsample
        pcd = pcd[::6]
        gt_labels = gt_labels[::6]

        # origin
        # axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        # baseline_vis.add_geometry(axis_pcd)
        # ours_vis.add_geometry(axis_pcd)
        # occ4d_vis.add_geometry(axis_pcd)
        # uno_vis.add_geometry(axis_pcd)

        # points
        for vis_name, vis in vis_list:
            pts = open3d.geometry.PointCloud()
            pts.points = open3d.utility.Vector3dVector(pcd)
            # colored by TP TN FP FN
            if vis_name == 'baseline':
                if vis_gt:
                    baseline_labels = gt_labels
                else:
                    baseline_labels_file = os.path.join(baseline_dir, kitti_scan_idx + "_mos_pred.label")  # mos
                    baseline_labels = np.fromfile(baseline_labels_file, dtype=np.uint8)  # valid mask
                colors = get_confusion_color(gt_labels, baseline_labels)
            elif vis_name == 'occ4d':
                occ4d_labels_file = os.path.join(occ4d_dir, kitti_scan_idx + "_mos_pred.label")
                occ4d_labels = np.fromfile(occ4d_labels_file, dtype=np.uint8)
                colors = get_confusion_color(gt_labels, occ4d_labels)
            elif vis_name == 'uno':
                uno_labels_file = os.path.join(uno_dir, kitti_scan_idx + "_mos_pred.label")
                uno_labels = np.fromfile(uno_labels_file, dtype=np.uint8)
                colors = get_confusion_color(gt_labels, uno_labels)
            elif vis_name == 'ours':
                ours_labels_file = os.path.join(ours_dir, kitti_scan_idx + "_mos_pred.label")
                ours_labels = np.fromfile(ours_labels_file, dtype=np.uint8)
                colors = get_confusion_color(gt_labels, ours_labels)
            pts.colors = open3d.utility.Vector3dVector(np.array(mos_color_func(colors)).T)
            vis.add_geometry(pts)
            
            # view settings
            vis.get_render_option().point_size = point_size
            vis.get_render_option().background_color = (220/255, 220/255, 220/255)
            
            # view control
            view = vis.get_view_control()
            if view_init and vis_name == 'baseline':
                view.set_front((-0.0026307837086555776, -0.10911602561147189, 0.99402553887303879))
                view.set_lookat((-4.1540102715460518, -3.1544474823842861, 16.565590452781144))
                view.set_up((-0.99303496278054726, 0.11737280671055968, 0.01025606846326051))
                view.set_zoom((0.14000000000000001))
                cam_params = view.convert_to_pinhole_camera_parameters()
                view_init = False
            else:
                view.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

        while True:
            for vis_name, vis in vis_list:
                view = vis.get_view_control()
                if vis_name == 'baseline':
                    cam_params = view.convert_to_pinhole_camera_parameters()
                else:
                    view.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
                # update vis
                vis.poll_events()
                vis.update_renderer()

    def render_next(vis):
        nonlocal sample_idx, skip_flag
        if skip_flag:
            skip_flag = False
            return None
        else:
            skip_flag = True
            print("render next trigger")
            sample_idx += 1
            if sample_idx >= len(vis_dataset.samples_labels):
                sample_idx = len(vis_dataset.samples_labels) - 1
            draw_sample()

    def render_prev(vis):
        nonlocal sample_idx, skip_flag
        if skip_flag:
            skip_flag = False
            return None
        else:
            skip_flag = True
            print("render prev trigger")
            sample_idx -= 1
            if sample_idx <= 0:
                sample_idx = 0
            draw_sample()

    def log_good_cases(vis):
        nonlocal sample_idx, good_cases_logger
        good_cases_logger.info(f"good cases: {sample_idx}")

    def log_bad_cases(vis):
        nonlocal sample_idx, bad_cases_logger
        bad_cases_logger.info(f"bad cases: {sample_idx}")

    for vis_name, vis in vis_list:
        if vis_name == 'baseline':
            vis.register_key_callback(ord('D'), render_next)
            vis.register_key_callback(ord('A'), render_prev)
            vis.register_key_callback(ord('G'), log_good_cases)
            vis.register_key_callback(ord('B'), log_bad_cases)
        vis.run()


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

    # todo: update kitti vis sequence
    vis_dataset = KittiMOSDataset(cfg_model, cfg_dataset, mode='vis')

    # render mos samples
    render_mos_samples(vis_dataset, cfg_vis, vis_gt=True)

    # statistics of different sequences
    # num_train_samples = 0
    # for seq_idx in [8]:
    #     cfg_dataset['sekitti']['vis'] = [seq_idx]
    #     vis_dataset = KittiMOSDataset(cfg_model, cfg_dataset, split='vis')
    #     sta_pts_num = 0
    #     mov_pts_num = 0
    #     for labels in vis_dataset.samples_labels:
    #         gt_label = load_mos_labels(labels[-1]).numpy()
    #         sta_pts_num += np.sum(gt_label == 1)
    #         mov_pts_num += np.sum(gt_label == 2)
    #     print(
    #         f"Sequence {seq_idx}: Static points: {sta_pts_num}, Moving points: {mov_pts_num}, Moving points percentage: "
    #         f"{mov_pts_num / (sta_pts_num + mov_pts_num) * 100:.3f}%")
    #     num_train_samples += len(vis_dataset.samples_scans)
    # for seq_idx in [8]:
    #     cfg_dataset['sekitti']['vis'] = [seq_idx]
    #     vis_dataset = KittiMOSDataset(cfg_model, cfg_dataset, split='vis')
    #     num_samples = len(vis_dataset.samples_scans)
    #     print(f"Sequence {seq_idx}: Samples: {num_samples}, Samples percentage: {num_samples / num_train_samples * 100:.3f}%")