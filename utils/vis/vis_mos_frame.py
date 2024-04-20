import argparse
import os

from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes
import open3d
import numpy as np

mos_colormap = {
        0: (255/255, 255/255, 255/255),  # unknown: white
        1: (25/255, 80/255, 25/255),    # static: green
        2: (255/255, 20/255, 20/255)     # moving: red
    }

check_colormap = {
        0: (255/255, 20/255, 20/255),     # moving: red
        1: (255/255, 255/255, 255/255),  # unknown: white
    }


def draw(vis, points, labels):
    # draw origin
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)
    # draw points
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points)
    vis.add_geometry(pts)
    # draw points label
    vfunc = np.vectorize(mos_colormap.get)
    points_color = np.array(vfunc(labels)).T
    pts.colors = open3d.utility.Vector3dVector(points_color)

def render_mos_pointcloud(points, gt_labels, pred_labels):
    # render gt labels
    vis_gt = open3d.visualization.Visualizer()
    vis_gt.create_window()
    draw(vis_gt, points, gt_labels)

    # render pred labels
    vis_pred = open3d.visualization.Visualizer()
    vis_pred.create_window()
    draw(vis_pred, points, pred_labels)

    # view settings
    vis_gt.get_render_option().point_size = 3.0
    vis_gt.get_render_option().background_color = np.zeros(3)
    view_ctrl_gt = vis_gt.get_view_control()
    view_ctrl_gt.set_front((0.75263429526187886, -0.13358133681379755, 0.64474618575893383))
    view_ctrl_gt.set_lookat((16.206845402638745, -3.8676194858766819, 15.365323753623207))
    view_ctrl_gt.set_up((-0.64932205862151104, 0.011806106960120792, 0.76042190922274799))
    view_ctrl_gt.set_zoom((0.19999999999999998))

    vis_pred.get_render_option().point_size = 3.0
    vis_pred.get_render_option().background_color = np.zeros(3)
    view_ctrl_pred = vis_pred.get_view_control()
    view_ctrl_pred.set_front((0.75263429526187886, -0.13358133681379755, 0.64474618575893383))
    view_ctrl_pred.set_lookat((16.206845402638745, -3.8676194858766819, 15.365323753623207))
    view_ctrl_pred.set_up((-0.64932205862151104, 0.011806106960120792, 0.76042190922274799))
    view_ctrl_pred.set_zoom((0.19999999999999998))

    # run vis
    vis_gt.run()
    vis_pred.run()

def render_mos_comparison(points, gt_labels, pred_labels):
    # open3d vis (for gt labels and pred labels)
    vis_gt = open3d.visualization.Visualizer()
    vis_gt.create_window(window_name='ground truth label', width=2400, height=2000, left=0, top=150)
    vis_pred = open3d.visualization.Visualizer()
    vis_pred.create_window(window_name='predicted label', width=2400, height=2000, left=2400, top=150)
    # vis render option
    vis_gt.get_render_option().point_size = 3.0
    vis_gt.get_render_option().background_color = np.zeros(3)
    vis_pred.get_render_option().point_size = 3.0
    vis_pred.get_render_option().background_color = np.zeros(3)

    # vis view control
    ctrl_gt = vis_gt.get_view_control()
    ctrl_pred = vis_pred.get_view_control()
    # sync camera parameters of two vis
    ctrl_gt.set_front((0.75263429526187886, -0.13358133681379755, 0.64474618575893383))
    ctrl_gt.set_lookat((16.206845402638745, -3.8676194858766819, 15.365323753623207))
    ctrl_gt.set_up((-0.64932205862151104, 0.011806106960120792, 0.76042190922274799))
    ctrl_gt.set_zoom((0.19999999999999998))
    cam_params_gt = ctrl_gt.convert_to_pinhole_camera_parameters()
    ctrl_pred.convert_from_pinhole_camera_parameters(cam_params_gt)

    # draw point cloud
    draw(vis_gt, points, gt_labels)
    draw(vis_pred, points, pred_labels)
    vis_gt.poll_events()
    vis_gt.update_renderer()
    vis_pred.poll_events()
    vis_pred.update_renderer()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dataset", type=str, default="10%NUSC")
    parser.add_argument("--model-name", type=str,
                        default="4docc_100%nuscenes_vs-0.2_t-3.0_bs-1_epo-60_vs-0.2_t-0.5_bs-8_epo-120")
    parser.add_argument("--model-version", type=str, default="version_0")
    parser.add_argument("--test-epoch", type=str, default="epoch_84")
    args = parser.parse_args()

    # load nusc dataset
    nusc = NuScenes(dataroot="/home/user/Datasets/nuScenes", version="v1.0-trainval")

    # get nuscenes validation scenes
    nusc_val_scene_names = create_splits_scenes()["val"]
    nusc_val_scenes = []
    for scene in nusc.scene:
        if scene['name'] in nusc_val_scene_names:
            nusc_val_scenes.append(scene)

    # data root
    nusc_dataroot = nusc.dataroot
    mos_gt_dir = os.path.join(nusc_dataroot, "mos_labels", nusc.version)
    mos_pred_dir = os.path.join("../../../logs/train", args.model_dataset, args.model_name, args.model_version, "results",
                             args.test_epoch, "predictions")
    # loop validation scenes
    for scene in nusc_val_scenes:
        sample = nusc.get('sample', scene['first_sample_token'])
        sample_data_token = sample['data']['LIDAR_TOP']
        sample_data = nusc.get("sample_data", sample_data_token)

        # get pcd
        pcd_file = os.path.join(nusc_dataroot, sample_data['filename'])
        pcd = LidarPointCloud.from_file(pcd_file).points.T[:, 0:-1]  # [num_pts, 4]
        # get gt mos label
        mos_gt_file = os.path.join(mos_gt_dir, f"{sample_data_token}_mos.label")
        mos_gt_label = np.fromfile(mos_gt_file, dtype=np.uint8)
        # get pred mos label
        mos_pred_file = os.path.join(mos_pred_dir, f"{sample_data_token}_mos_pred.label")
        mos_pred_label = np.fromfile(mos_pred_file, dtype=np.uint8)
        # render mos
        render_mos_comparison(pcd, mos_gt_label, mos_pred_label)

        while sample['next'] != "":
            sample_token = sample['next']
            sample = nusc.get('sample', sample_token)
            sample_data_token = sample['data']['LIDAR_TOP']
            sample_data = nusc.get("sample_data", sample_data_token)

            # get pcd
            pcd_file = os.path.join(nusc_dataroot, sample_data['filename'])
            pcd = LidarPointCloud.from_file(pcd_file).points.T[:, 0:-1]  # [num_pts, 4]
            # get gt mos label
            mos_gt_file = os.path.join(mos_gt_dir, f"{sample_data_token}_mos.label")
            mos_gt_label = np.fromfile(mos_gt_file, dtype=np.uint8)
            # get pred mos label
            mos_pred_file = os.path.join(mos_pred_dir, f"{sample_data_token}_mos_pred.label")
            mos_pred_label = np.fromfile(mos_pred_file, dtype=np.uint8)
            # render mos
            render_mos_comparison(pcd, mos_gt_label, mos_pred_label)

