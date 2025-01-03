(window_name='occ4d mos results', width=2560, height=1360, left=2560, top=0)
    uno_vis.create_window(window_name='uno mos results', width=2560, height=1360, left=0, top=1510)
    ours_vis.create_window(window_name='ours mos results'import json
import multiprocessing
import os
import os.path
import subprocess
import threading
import warnings

import numpy as np
import open3d as o3d
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
import argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

seq_idx = 0
o3d_voxel_size = 0.2
o3d_point_size = 2.0

o3d_background_color = (0.859, 0.882, 1.000)  # blue
# o3d_background_color = (0.71, 0.702, 0.706)  # gray

mos_colormap = {
        np.uint8(0): (0.439, 0.439, 0.439),    # unknown: gray
        np.uint8(1): (0.071, 0.369, 0.055),    # static: green-dark
        np.uint8(2): (0, 0, 0)     # moving: black
}

# gray colormap: https://colorswall.com/palette/24454
gray_colormap = {
    0: (211/255, 211/255, 211/255),  # Pinball
    1: (190/255, 190/255, 190/255),
    2: (169/255, 169/255, 169/255),
    3: (148/255, 148/255, 148/255),
    4: (127/255, 127/255, 127/255),
    5: (106/255, 106/255, 106/255),
    6: (84/255,  84/255,  84/255),
    7: (63/255,  63/255,  63/255),
    8: (42/255,  42/255,  42/255),
    9: (21/255,  21/255,  21/255)     # Matt Black
}

# magma colormap: https://waldyrious.net/viridis-palette-generator/
magma_colormap = {
    0: (252/255, 253/255, 191/255),  #fcfdbf
    1: (254/255, 202/255, 141/255),  #feca8d
    2: (253/255, 150/255, 104/255),  #fd9668
    3: (241/255, 96/255,  93/255),   #f1605d
    4: (241/255, 96/255,  93/255),   #cd4071
    5: (158/255, 47/255,  127/255),  #9e2f7f
    6: (114/255, 31/255,  129/255),  #721f81
    7: (68/255,  15/255,  118/255),  #440f76
    8: (24/255,  15/255,  61/255),   #180f3d
    9: (0/255,   0/255,   4/255)     #000004
}

coolwarm_colormap = {
    100: o3d_background_color,  # set nan to backgroud color
    -6: (59/255, 76/255, 192/255),  # blue
    -5: (83/255, 111/255, 220/255),
    -4: (111/255, 145/255, 242/255),
    -3: (140/255, 174/255, 252/255),
    -2: (169/255, 197/255, 252/255),
    -1: (197/255, 213/255, 241/255),
    0: (221/255, 221/255, 221/255), # middle point
    1: (238/255, 205/255, 188/255),
    2: (246/255, 182/255, 155/255),
    3: (243/255, 152/255, 122/255),
    4: (230/255, 116/255, 91/255),
    5: (208/255, 73/255, 62/255),
    6: (180/255, 4/255, 38/255)  # red
}

def run_fuse_kitti_shell():
    # os.system
    print(os.system("/home/mars/MOS_Projects/nvblox/nvblox/script/run_fuse_kitti_frames.sh"))

    # subprocess
    # shell_command_list = [
    #     "cd /home/mars/MOS_Projects/nvblox/nvblox/build",
    #
    #     "make -j64  && \\"
    #     "./executables/fuse_kitti \\"
    #     "/home/mars/MOS_Projects/nvblox/nvblox/tests/data/kitti \\"
    #     "--tsdf_integrator_max_integration_distance_m 30.0 \\"
    #     "--tsdf_integrator_truncation_distance_vox 300 \\"
    #     "--num_frames 2 \\"
    #     "--voxel_size 0.2 \\"
    #     "--esdf_frame_subsampling \\"
    #     "--esdf_mode 0 \\"
    #     "--mesh_output_path \\"
    #     "/home/mars/MOS_Projects/nvblox/nvblox/outputs/kitti/kitti_seq_01_sync_mesh_10.ply \\"
    #     "--esdf_output_path \\"
    #     "/home/mars/MOS_Projects/nvblox/nvblox/outputs/kitti/kitti_seq_01_sync_esdf_10.ply \\"
    # ]
    # print(subprocess.call(shell_command_list, shell=True))

def read_esdf(esdf_file):
    # read esdf from .ply file
    plydata = PlyData.read(esdf_file)
    # esdf = plydata.elements[0].data
    # esdf_data = plydata['vertex'].data
    # esdf_array = np.array([[x, y, z, i] for x, y, z, i in esdf_data])
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    i = plydata['vertex']['intensity']
    esdf_max = np.max(i)
    esdf_min = np.min(i)
    print('ESDF distance range: ', str(esdf_min), " ", str(esdf_max))
    # get the code from: https://blog.csdn.net/phy12321/article/details/107373073
    # NOTE!!! do not use np.fromfile to read the .ply file
    esdf = np.array([x, y, z, i]).T
    return esdf

def transfer_to_colormap_key(res):
    nan_idx = np.argwhere(np.isnan(res))
    valid_color_idx = np.squeeze(np.argwhere(np.invert(np.isnan(res))))
    neg_6_idx = np.argwhere(res <= -1.00)  # (-infinite, -1.00] m
    neg_5_idx = np.argwhere((-1.00 < res) & (res <= -0.50))  # (-1.00, -0.50] m
    neg_4_idx = np.argwhere((-0.50 < res) & (res <= -0.20))  # (-0.50, -0.20] m
    neg_3_idx = np.argwhere((-0.20 < res) & (res <= -0.10))  # (-0.20, -0.10] m
    neg_2_idx = np.argwhere((-0.10 < res) & (res <= -0.05))  # (-0.10, -0.05] m
    neg_1_idx = np.argwhere((-0.05 < res) & (res < 0.00))  # (-0.05, -0.00) m
    mid_0_idx = np.argwhere(res == 0)  # [0] m
    pos_1_idx = np.argwhere((0.00 < res) & (res < 0.05))  # (0.00, 0.05) m
    pos_2_idx = np.argwhere((0.05 <= res) & (res < 0.10))  # [0.05, 0.10) m
    pos_3_idx = np.argwhere((0.10 <= res) & (res < 0.20))  # [0.10, 0.20) m
    pos_4_idx = np.argwhere((0.20 <= res) & (res < 0.50))  # [0.20, 0.50) m
    pos_5_idx = np.argwhere((0.50 <= res) & (res < 1.00))  # [0.50, 1.00) m
    pos_6_idx = np.argwhere(1.00 <= res)  # [1.00, infinite) m

    res[nan_idx] = 100
    res[neg_6_idx] = -6
    res[neg_5_idx] = -5
    res[neg_4_idx] = -4
    res[neg_3_idx] = -3
    res[neg_2_idx] = -2
    res[neg_1_idx] = -1
    res[mid_0_idx] = 0
    res[pos_1_idx] = 1
    res[pos_2_idx] = 2
    res[pos_3_idx] = 3
    res[pos_4_idx] = 4
    res[pos_5_idx] = 5
    res[pos_6_idx] = 6
    return res, valid_color_idx

def get_pcl_in_world_frame(nusc, sample_data_token):
    sample_data = nusc.get('sample_data', sample_data_token)
    pcl_path = os.path.join(nusc.dataroot, sample_data['filename'])
    points_l = LidarPointCloud.from_file(pcl_path).points.T  # [num_points, 4]
    points_l = points_l[:, :3]  # without intensity

    # filter points with distance > max distacne
    valid_points_idx = filter_far_points(points_l)
    # points_l = points_l[valid_points_idx]

    # odom pose: from vehicle to world
    pose_token = sample_data['ego_pose_token']
    pose = nusc.get('ego_pose', pose_token)
    T_v_2_w = transform_matrix(pose['translation'], Quaternion(pose['rotation']))

    # calib pose: from lidar to vehicle
    calib_token = sample_data['calibrated_sensor_token']
    calib = nusc.get('calibrated_sensor', calib_token)
    T_l_2_v = transform_matrix(calib['translation'], Quaternion(calib['rotation']))

    # transform point cloud from lidar frame to world frame
    T_l_2_w = T_v_2_w @ T_l_2_v
    points_l_homo = np.hstack([points_l, np.ones((points_l.shape[0], 1))]).T
    points_w = (T_l_2_w @ points_l_homo).T[:, :3]
    return points_w, valid_points_idx

def filter_far_points(points):
    max_dis = 35
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    points_dist = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    valid_points_idx = np.squeeze(np.argwhere(points_dist <= max_dis))
    return valid_points_idx

def open3d_vis_esdf(esdf):
    esdf_points = esdf[:, :3]
    esdf_dis = esdf[:, -1]

    esdf_dis, valid_color_idx = transfer_to_colormap_key(esdf_dis)
    color_func = np.vectorize(coolwarm_colormap.get)
    esdf_color = np.array(color_func(esdf_dis)).T

    # Open3D Vis
    esdf_pcd = o3d.geometry.PointCloud()
    esdf_pcd.points = o3d.utility.Vector3dVector(esdf_points)
    esdf_pcd.colors = o3d.utility.Vector3dVector(esdf_color)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(esdf_pcd, voxel_size=o3d_voxel_size)

    vis = o3d.visualization.Visualizer()  # visualizer
    vis.create_window()
    vis.add_geometry(voxel_grid)

    opt = vis.get_render_option()  # render option
    opt.background_color = np.asarray(o3d_background_color)
    ctrl = vis.get_view_control()  # view control

    vis.run()  # run
    vis.destroy_window()

def open3d_vis_esdf_mos(esdf, sample_data_token, nusc):
    esdf_points = esdf[:, :3]
    esdf_dis = esdf[:, -1]

    esdf_dis, valid_color_idx = transfer_to_colormap_key(esdf_dis)
    color_func = np.vectorize(coolwarm_colormap.get)
    esdf_color = np.array(color_func(esdf_dis)).T

    # point cloud with mos label
    points_w, valid_points_idx = get_pcl_in_world_frame(nusc, sample_data_token)
    points_color = np.full_like(points_w, 0.3)

    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(points_w)
    pts.colors = o3d.utility.Vector3dVector(points_color)

    # Open3D Vis
    esdf_pcd = o3d.geometry.PointCloud()
    esdf_pcd.points = o3d.utility.Vector3dVector(esdf_points)
    esdf_pcd.colors = o3d.utility.Vector3dVector(esdf_color)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(esdf_pcd, voxel_size=o3d_voxel_size)

    vis = o3d.visualization.Visualizer()  # visualizer
    vis.create_window()
    vis.add_geometry(voxel_grid)
    vis.add_geometry(pts)

    opt = vis.get_render_option()  # render option
    opt.point_size = o3d_point_size
    opt.background_color = np.asarray(o3d_background_color)
    ctrl = vis.get_view_control()  # view control

    vis.run()  # run
    vis.destroy_window()

def open3d_vis_esdfres_point(esdf_res, idx_tup, nusc):  # esdf residual + point cloud (not colored by mos labels)
    # get sample data token from index tuple
    (seq_idx, curr_frame_idx, next_frame_idx) = idx_tup
    sample_data_tok_curr = sd_idx_to_tok_dict[(seq_idx, curr_frame_idx)]
    sample_data_tok_next = sd_idx_to_tok_dict[(seq_idx, next_frame_idx)]
    sample_data_tok_list = [sample_data_tok_curr, sample_data_tok_next]
    gray_value_list = [0.6, 0.1]

    # get color of esdf residual
    residual = esdf_res[:, -1]
    nan_idx = np.argwhere(np.isnan(residual))  # process the newly observed voxels (np.nan)
    neg_6_idx = np.argwhere(residual <= -1.00)                # (-infinite, -1.00] m
    neg_5_idx = np.argwhere((-1.00 < residual) & (residual <= -0.50))  # (-1.00, -0.50] m
    neg_4_idx = np.argwhere((-0.50 < residual) & (residual <= -0.20))  # (-0.50, -0.20] m
    neg_3_idx = np.argwhere((-0.20 < residual) & (residual <= -0.10))  # (-0.20, -0.10] m
    neg_2_idx = np.argwhere((-0.10 < residual) & (residual <= -0.05))  # (-0.10, -0.05] m
    neg_1_idx = np.argwhere((-0.05 < residual) & (residual <   0.00))  # (-0.05, -0.00) m
    mid_0_idx = np.argwhere(residual == 0)                    # [0] m
    pos_1_idx = np.argwhere((0.00  < residual) & (residual < 0.05))    # (0.00, 0.05) m
    pos_2_idx = np.argwhere((0.05 <= residual) & (residual < 0.10))    # [0.05, 0.10) m
    pos_3_idx = np.argwhere((0.10 <= residual) & (residual < 0.20))    # [0.10, 0.20) m
    pos_4_idx = np.argwhere((0.20 <= residual) & (residual < 0.50))    # [0.20, 0.50) m
    pos_5_idx = np.argwhere((0.50 <= residual) & (residual < 1.00))    # [0.50, 1.00) m
    pos_6_idx = np.argwhere(1.00 <= residual)                 # [1.00, infinite) m

    residual[nan_idx] = 100
    residual[neg_6_idx] = -6
    residual[neg_5_idx] = -5
    residual[neg_4_idx] = -4
    residual[neg_3_idx] = -3
    residual[neg_2_idx] = -2
    residual[neg_1_idx] = -1
    residual[mid_0_idx] = 0
    residual[pos_1_idx] = 1
    residual[pos_2_idx] = 2
    residual[pos_3_idx] = 3
    residual[pos_4_idx] = 4
    residual[pos_5_idx] = 5
    residual[pos_6_idx] = 6

    color_func = np.vectorize(coolwarm_colormap.get)
    esdf_res_color = np.array(color_func(residual)).T

    # create Open3D Vis
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # integrate esdf voxel grid to open3d vis
    esdf_pcd = o3d.geometry.PointCloud()
    esdf_pcd.points = o3d.utility.Vector3dVector(esdf_res[:, :3])
    esdf_pcd.colors = o3d.utility.Vector3dVector(esdf_res_color)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(esdf_pcd, voxel_size=o3d_voxel_size)
    vis.add_geometry(voxel_grid)
    print("Vis ESDF Residual: seq %s, current frame %s, next frame %s\n" % (seq_idx, curr_frame_idx, next_frame_idx))

    # whether curr or next is keyframe:
    for sample_data_tok, gray_value in zip(sample_data_tok_list, gray_value_list):
        (seq_idx, frame_idx) = sd_dict[sample_data_tok]
        print("Vis Point Cloud: seq %s, frame %s\n" % (seq_idx, frame_idx))
        points_w, valid_points_idx = get_pcl_in_world_frame(nusc, sample_data_tok)

        # integrate point cloud to open3d vis
        pts = o3d.geometry.PointCloud()
        pts.points = o3d.utility.Vector3dVector(points_w)
        points_color = np.full_like(points_w, gray_value)
        pts.colors = o3d.utility.Vector3dVector(points_color)
        vis.add_geometry(pts)

    # open3d origin
    # axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    # open3d axis
    # vis.add_geometry(axis_pcd)

    # Vis Settings (render options & view control)
    opt = vis.get_render_option()
    opt.point_size = o3d_point_size
    opt.background_color = np.asarray(o3d_background_color)
    ctrl = vis.get_view_control()
    # ctrl.set_front((-0.020588885231781512, 0.10962245064836584, 0.99376003950589564))
    # ctrl.set_lookat((0.066336316011035693, 3.437049303689776, 0.11533577801642712))
    # ctrl.set_up((0.40551647562275533, 0.90945418190674621, -0.091921047703072634))
    # ctrl.set_zoom((0.45999999999999974))

    # Vis Run
    vis.run()
    vis.destroy_window()

def open3d_vis_esdfres_mos(esdf_res, idx_tup, nusc):  # esdf residual + point cloud (colored by mos labels)
    # get sample data token from index tuple
    (seq_idx, curr_frame_idx, next_frame_idx) = idx_tup
    sample_data_tok_curr = sd_idx_to_tok_dict[(seq_idx, curr_frame_idx)]
    sample_data_tok_next = sd_idx_to_tok_dict[(seq_idx, next_frame_idx)]
    sample_data_tok_list = [sample_data_tok_curr, sample_data_tok_next]

    # get color of esdf residual
    residual = esdf_res[:, -1]
    nan_idx = np.argwhere(np.isnan(residual))  # process the newly observed voxels (np.nan)
    neg_6_idx = np.argwhere(residual <= -1.00)                # (-infinite, -1.00] m
    neg_5_idx = np.argwhere((-1.00 < residual) & (residual <= -0.50))  # (-1.00, -0.50] m
    neg_4_idx = np.argwhere((-0.50 < residual) & (residual <= -0.20))  # (-0.50, -0.20] m
    neg_3_idx = np.argwhere((-0.20 < residual) & (residual <= -0.10))  # (-0.20, -0.10] m
    neg_2_idx = np.argwhere((-0.10 < residual) & (residual <= -0.05))  # (-0.10, -0.05] m
    neg_1_idx = np.argwhere((-0.05 < residual) & (residual <   0.00))  # (-0.05, -0.00) m
    mid_0_idx = np.argwhere(residual == 0)                    # [0] m
    pos_1_idx = np.argwhere((0.00  < residual) & (residual < 0.05))    # (0.00, 0.05) m
    pos_2_idx = np.argwhere((0.05 <= residual) & (residual < 0.10))    # [0.05, 0.10) m
    pos_3_idx = np.argwhere((0.10 <= residual) & (residual < 0.20))    # [0.10, 0.20) m
    pos_4_idx = np.argwhere((0.20 <= residual) & (residual < 0.50))    # [0.20, 0.50) m
    pos_5_idx = np.argwhere((0.50 <= residual) & (residual < 1.00))    # [0.50, 1.00) m
    pos_6_idx = np.argwhere(1.00 <= residual)                 # [1.00, infinite) m

    residual[nan_idx] = 100
    residual[neg_6_idx] = -6
    residual[neg_5_idx] = -5
    residual[neg_4_idx] = -4
    residual[neg_3_idx] = -3
    residual[neg_2_idx] = -2
    residual[neg_1_idx] = -1
    residual[mid_0_idx] = 0
    residual[pos_1_idx] = 1
    residual[pos_2_idx] = 2
    residual[pos_3_idx] = 3
    residual[pos_4_idx] = 4
    residual[pos_5_idx] = 5
    residual[pos_6_idx] = 6

    color_func = np.vectorize(coolwarm_colormap.get)
    esdf_res_color = np.array(color_func(residual)).T

    # create Open3D Vis
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # integrate esdf voxel grid to open3d vis
    esdf_pcd = o3d.geometry.PointCloud()
    esdf_pcd.points = o3d.utility.Vector3dVector(esdf_res[:, :3])
    esdf_pcd.colors = o3d.utility.Vector3dVector(esdf_res_color)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(esdf_pcd, voxel_size=o3d_voxel_size)
    vis.add_geometry(voxel_grid)
    print("Vis ESDF Residual: seq %s, current frame %s, next frame %s\n" % (seq_idx, curr_frame_idx, next_frame_idx))

    # whether curr or next is keyframe:
    for sample_data_tok in sample_data_tok_list:
        sample_data = nusc.get('sample_data', sample_data_tok)
        if sample_data['is_key_frame']:
            (seq_idx, frame_idx) = sd_dict[sample_data_tok]
            print("Vis Point Cloud Colored by MOS Label: seq %s, frame %s\n" % (seq_idx, frame_idx))
            points_w, valid_points_idx = get_pcl_in_world_frame(nusc, sample_data_tok)
            mos_labels_file = os.path.join(nusc.dataroot, 'mos_labels', nusc.version, sample_data_tok + "_mos.label")
            points_label = np.fromfile(mos_labels_file, dtype=np.uint8)

            vfunc = np.vectorize(mos_colormap.get)
            # only vis moving points
            mov_idx = np.argwhere(points_label == 2).ravel()
            points_mov = points_w[mov_idx]
            colors_mov = np.array(vfunc(points_label[mov_idx])).T
            o3d_pts_mov = o3d.geometry.PointCloud()
            o3d_pts_mov.points = o3d.utility.Vector3dVector(points_mov)
            o3d_pts_mov.colors = o3d.utility.Vector3dVector(colors_mov)

            unk_idx = np.argwhere(points_label == 0).ravel()
            points_unk = points_w[unk_idx]
            colors_unk = np.array(vfunc(points_label[unk_idx])).T
            o3d_pts_unk = o3d.geometry.PointCloud()
            o3d_pts_unk.points = o3d.utility.Vector3dVector(points_unk)
            o3d_pts_unk.colors = o3d.utility.Vector3dVector(colors_unk)

            sta_idx = np.argwhere(points_label == 1).ravel()
            points_sta = points_w[sta_idx]
            colors_sta = np.array(vfunc(points_label[sta_idx])).T
            o3d_pts_sta = o3d.geometry.PointCloud()
            o3d_pts_sta.points = o3d.utility.Vector3dVector(points_sta)
            o3d_pts_sta.colors = o3d.utility.Vector3dVector(colors_sta)

            vis.add_geometry(o3d_pts_mov)
            vis.add_geometry(o3d_pts_unk)
            vis.add_geometry(o3d_pts_sta)

            # vis_points = np.concatenate((points_w[mov_idx], points_w[unknown_idx]), axis=0)
            # vis_labels = np.concatenate((points_label[mov_idx], points_label[unknown_idx]), axis=0)

    # open3d origin
    # axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    # Vis Settings (render options & view control)
    opt = vis.get_render_option()
    opt.point_size = o3d_point_size
    opt.background_color = np.asarray(o3d_background_color)
    ctrl = vis.get_view_control()
    # ctrl.set_front((-0.020588885231781512, 0.10962245064836584, 0.99376003950589564))
    # ctrl.set_lookat((0.066336316011035693, 3.437049303689776, 0.11533577801642712))
    # ctrl.set_up((0.40551647562275533, 0.90945418190674621, -0.091921047703072634))
    # ctrl.set_zoom((0.45999999999999974))

    # Vis Run
    vis.run()
    vis.destroy_window()

def open3d_vis_esdfres_keyframes(esdf_res, idx_tup, nusc):
    (seq_idx, curr_frame_idx, next_frame_idx) = idx_tup
    sample_data_tok_curr = sd_idx_to_tok_dict[(seq_idx, curr_frame_idx)]
    sample_data_tok_next = sd_idx_to_tok_dict[(seq_idx, next_frame_idx)]
    sample_data_tok_list = [sample_data_tok_curr, sample_data_tok_next]

    res = esdf_res[:, -1]
    res_color_keys, valid_color_idx = transfer_to_colormap_key(res)
    color_func = np.vectorize(coolwarm_colormap.get)
    esdf_res_color = np.array(color_func(res_color_keys)).T[valid_color_idx]

    esdf_pcd = o3d.geometry.PointCloud()
    esdf_pcd.points = o3d.utility.Vector3dVector(esdf_res[:, :3][valid_color_idx])
    esdf_pcd.colors = o3d.utility.Vector3dVector(esdf_res_color)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(esdf_pcd, voxel_size=o3d_voxel_size)
    print("Vis ESDF Residual: seq %s, current frame %s, next frame %s\n" % (seq_idx, curr_frame_idx, next_frame_idx))

    # search three adjacent keyframes, point cloud colored by mos label
    keyframe_token_list = list(kf_dict.keys())
    keyframe_curr_tok = None
    keyframe_prev_tok = None
    keyframe_next_tok = None
    for sample_data_tok in sample_data_tok_list:
        sample_data = nusc.get('sample_data', sample_data_tok)
        if sample_data['is_key_frame']:
            keyframe_curr_tok = sample_data_tok
            keyframe_list_idx = keyframe_token_list.index(sample_data_tok)
            keyframe_prev_tok = keyframe_token_list[keyframe_list_idx - 1]
            keyframe_next_tok = keyframe_token_list[keyframe_list_idx + 1]
            (_, frame_idx_curr) = sd_dict[keyframe_curr_tok]
            (_, frame_idx_prev) = sd_dict[keyframe_prev_tok]
            (_, frame_idx_next) = sd_dict[keyframe_next_tok]
            print("Vis Keyframe Moving Point Clouds with Grayscale Color: frame %s, frame %s, frame %s\n" % (frame_idx_prev, frame_idx_curr, frame_idx_next))
            break

    assert keyframe_curr_tok != None, "current or next sample data is not keyframe, cannot vis three keyframes"

    # get points of keyframes (world frame)
    points_kf_prev, valid_pts_idx_prev = get_pcl_in_world_frame(nusc, keyframe_prev_tok)
    points_kf_curr, valid_pts_idx_curr = get_pcl_in_world_frame(nusc, keyframe_curr_tok)
    points_kf_next, valid_pts_idx_next = get_pcl_in_world_frame(nusc, keyframe_next_tok)

    # load mos labels
    mos_labels_kf_prev_file = os.path.join(nusc.dataroot, 'mos_labels', nusc.version, keyframe_prev_tok + "_mos.label")
    mos_labels_kf_curr_file = os.path.join(nusc.dataroot, 'mos_labels', nusc.version, keyframe_curr_tok + "_mos.label")
    mos_labels_kf_next_file = os.path.join(nusc.dataroot, 'mos_labels', nusc.version, keyframe_next_tok + "_mos.label")
    points_label_kf_prev = np.fromfile(mos_labels_kf_prev_file, dtype=np.uint8)[valid_pts_idx_prev]
    points_label_kf_curr = np.fromfile(mos_labels_kf_curr_file, dtype=np.uint8)[valid_pts_idx_curr]
    points_label_kf_next = np.fromfile(mos_labels_kf_next_file, dtype=np.uint8)[valid_pts_idx_next]

    # only vis moving points
    mov_idx_kf_prev = np.argwhere(points_label_kf_prev == 2).ravel()
    mov_idx_kf_curr = np.argwhere(points_label_kf_curr == 2).ravel()
    mov_idx_kf_next = np.argwhere(points_label_kf_next == 2).ravel()

    # points color
    kf_prev_colormap = {
        # 2: (0.576, 0.878, 0.463)
        0: (0.6, 0.6, 0.6),
        2: (0.6, 0.6, 0.6)
    }
    kf_curr_colormap = {
        # 2: (0.369, 0.71, 0.239)
        0: (0.3, 0.3, 0.3),
        2: (0.3, 0.3, 0.3)
    }
    kf_next_colormap = {
        # 2: (0.114, 0.38, 0.012)
        0: (0.0, 0.0, 0.0),
        2: (0.0, 0.0, 0.0)
    }
    prev_color_value_func = np.vectorize(kf_prev_colormap.get)
    curr_color_value_func = np.vectorize(kf_curr_colormap.get)
    next_color_value_func = np.vectorize(kf_next_colormap.get)
    points_color_kf_prev = np.array(prev_color_value_func(points_label_kf_prev[mov_idx_kf_prev])).T
    points_color_kf_curr = np.array(curr_color_value_func(points_label_kf_curr[mov_idx_kf_curr])).T
    points_color_kf_next = np.array(next_color_value_func(points_label_kf_next[mov_idx_kf_next])).T

    # open3d geometries (point cloud)
    o3d_pcd_kf_prev = o3d.geometry.PointCloud()
    o3d_pcd_kf_prev.points = o3d.utility.Vector3dVector(points_kf_prev[mov_idx_kf_prev])
    o3d_pcd_kf_prev.colors = o3d.utility.Vector3dVector(points_color_kf_prev)
    o3d_pcd_kf_curr = o3d.geometry.PointCloud()
    o3d_pcd_kf_curr.points = o3d.utility.Vector3dVector(points_kf_curr[mov_idx_kf_curr])
    o3d_pcd_kf_curr.colors = o3d.utility.Vector3dVector(points_color_kf_curr)
    o3d_pcd_kf_next = o3d.geometry.PointCloud()
    o3d_pcd_kf_next.points = o3d.utility.Vector3dVector(points_kf_next[mov_idx_kf_next])
    o3d_pcd_kf_next.colors = o3d.utility.Vector3dVector(points_color_kf_next)

    # open3d visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(voxel_grid)
    vis.add_geometry(o3d_pcd_kf_prev)
    vis.add_geometry(o3d_pcd_kf_curr)
    vis.add_geometry(o3d_pcd_kf_next)

    # open3d render options
    opt = vis.get_render_option()
    opt.point_size = o3d_point_size
    opt.background_color = np.asarray(o3d_background_color)
    # open3d view control
    ctrl = vis.get_view_control()

    vis.run()
    vis.destroy_window()

def esdf_histogram(esdf):
    esdf_dis = esdf[:, -1].reshape((-1, 1))
    # n, bins, patches = plt.hist(x=esdf_dis, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    n, bins, patches = plt.hist(x=esdf_dis, bins=100, color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('ESDF Distance')
    plt.ylabel('Frequency')
    plt.title('ESDF Distance Histogram')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

def open3d_compare_resolution(esdf_1, esdf_2):
    esdf_1_points = esdf_1[:, :3]
    esdf_2_points = esdf_2[:, :3]
    esdf_1_dis = esdf_1[:, -1]
    esdf_2_dis = esdf_2[:, -1]

    # get esdf color
    color_func = np.vectorize(coolwarm_colormap.get)
    esdf_1_colors_key, valid_color_idx_1 = transfer_to_colormap_key(esdf_1_dis)
    esdf_2_colors_key, valid_color_idx_2 = transfer_to_colormap_key(esdf_2_dis)
    esdf_1_colors = np.array(color_func(esdf_1_colors_key)).T
    esdf_2_colors = np.array(color_func(esdf_2_colors_key)).T

    # Open3D Vis
    esdf_1_pcd = o3d.geometry.PointCloud()
    esdf_1_pcd.points = o3d.utility.Vector3dVector(esdf_1_points)
    esdf_1_pcd.colors = o3d.utility.Vector3dVector(esdf_1_colors)
    esdf_2_pcd = o3d.geometry.PointCloud()
    esdf_2_pcd.points = o3d.utility.Vector3dVector(esdf_2_points)
    esdf_2_pcd.colors = o3d.utility.Vector3dVector(esdf_2_colors)

    esdf_1_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(esdf_1_pcd, voxel_size=0.2)
    esdf_2_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(esdf_2_pcd, voxel_size=0.5)

    vis_1 = o3d.visualization.Visualizer()  # visualizer
    vis_1.create_window()
    vis_1.add_geometry(esdf_1_voxel_grid)

    vis_2 = o3d.visualization.Visualizer()  # visualizer
    vis_2.create_window()
    vis_2.add_geometry(esdf_2_voxel_grid)

    opt_1 = vis_1.get_render_option()  # render option
    opt_2 = vis_2.get_render_option()  # render option
    opt_1.background_color = np.asarray(o3d_background_color)
    opt_2.background_color = np.asarray(o3d_background_color)

    vis_1.run()  # run
    vis_2.run()
    vis_1.destroy_window()
    vis_2.destroy_window()

def cal_esdf_res(curr_frame_idx):
    print("No ESDF residual file or directory! Start computing ESDF residual:")
    surface_test = True
    if surface_test:
        esdf_curr_file = os.path.join(esdf_dir, str(seq_idx).zfill(4), "3d_resolution-0.2", "global_esdf_" + str(curr_frame_idx) + ".bin")
        esdf_next_file = os.path.join(esdf_dir, str(seq_idx).zfill(4), "3d_resolution-0.2", "global_esdf_" + str(curr_frame_idx + 1) + ".bin")
        resolution_dir = os.path.join(esdf_res_dir, str(seq_idx).zfill(4), "3d_resolution-0.2")
        os.makedirs(resolution_dir, exist_ok=True)
    else:
        esdf_curr_file = os.path.join(esdf_dir, str(seq_idx).zfill(4), "resolution-" + str(o3d_voxel_size), "global_esdf_" + str(curr_frame_idx) + ".bin")
        esdf_next_file = os.path.join(esdf_dir, str(seq_idx).zfill(4), "resolution-" + str(o3d_voxel_size), "global_esdf_" + str(curr_frame_idx + 1) + ".bin")
        resolution_dir = os.path.join(esdf_res_dir, str(seq_idx).zfill(4), "resolution-" + str(o3d_voxel_size))
        os.makedirs(resolution_dir, exist_ok=True)

    esdf_curr = np.fromfile(esdf_curr_file, dtype=np.float32).reshape((-1, 4))
    esdf_next = np.fromfile(esdf_next_file, dtype=np.float32).reshape((-1, 4))

    esdf_next_voxels = esdf_next[:, :3]
    esdf_curr_voxels = esdf_curr[:, :3]
    esdf_residual = np.zeros_like(esdf_next)  # dtype=np.float32

    num_nan = 0
    warnings.filterwarnings("ignore")
    esdf_curr_voxels_list = esdf_curr_voxels.tolist()
    for next_voxel_idx, next_voxel in enumerate(tqdm(esdf_next_voxels)):
        try:
            row_idx = esdf_curr_voxels_list.index(next_voxel.tolist())
            residual = esdf_next[next_voxel_idx, -1] - esdf_curr[row_idx, -1]
            esdf_residual[next_voxel_idx] = np.append(next_voxel, residual)
        except:
            row_idx = -1
            residual = np.nan
            esdf_residual[next_voxel_idx] = np.append(next_voxel, residual)
            num_nan += 1
        # pad and isin function:
        # isin = np.isin(esdf_curr_voxels, esdf_next_voxels)  # esdf_curr_voxels is subset of esdf_next_voxels
        # pad_width = ((0, esdf_next_voxels.shape[0] - esdf_curr_voxels.shape[0]), (0, 0))  # two axis
        # esdf_curr_padded = np.pad(esdf_curr_voxels, pad_width, 'constant')

        # row_idx = np.where((esdf_curr_voxels == next_voxel).all(axis=1))[0]
        # if row_idx:  # not empty
        #     residual = esdf_next[next_voxel_idx, -1] - esdf_curr[row_idx, -1]
        #     esdf_residual[next_voxel_idx] = np.append(next_voxel, residual)
        # else:  # empty
        #     residual = np.nan
        #     esdf_residual[next_voxel_idx] = np.append(next_voxel, residual)
        #     num_nan += 1

    print("Num of current voxels: ", str(esdf_next.shape[0]), "; Num of history voxels: ", str(esdf_curr.shape[0]),
          "; Num of voxels with NaN residual: ", str(num_nan))

    residual_file = os.path.join(resolution_dir, "global_esdf_res_" + str(curr_frame_idx + 1) + "-" + str(curr_frame_idx) + ".bin")
    esdf_residual.tofile(residual_file)
    print("Save ESDF Residual to .bin file: ")
    return esdf_residual

if __name__ == '__main__':
    # Switch
    vis_esdf_histogram = False
    vis_esdf = False
    vis_esdf_res = True
    vis_compare_resolution = False

    # args
    parser = argparse.ArgumentParser(description='Generate nuScenes lidar panaptic gt.')
    parser.add_argument('--root_dir', type=str, default='/home/mars/MOS_Projects/nuScenes_MOS_Labeling/mini_data')
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--verbose', type=bool, default=True, help='Whether to print to stdout.')
    args = parser.parse_args()
    esdf_dir = os.path.join(args.root_dir, "esdf")
    esdf_res_dir = os.path.join(args.root_dir, "esdf_res")

    if vis_esdf or vis_esdf_res or vis_compare_resolution:
        nusc = NuScenes(version=args.version, dataroot=args.root_dir, verbose=args.verbose)
        # load sample data dict & keyframe dict
        sd_dict_file = os.path.join(nusc.dataroot, "tok_to_idx_dict", str(seq_idx).zfill(4) + "_sd_dict.txt")
        kf_dict_file = os.path.join(nusc.dataroot, "tok_to_idx_dict", str(seq_idx).zfill(4) + "_kf_dict.txt")
        with open(sd_dict_file, "r") as fp:
            sd_dict = json.load(fp)
        with open(kf_dict_file, "r") as fp:
            kf_dict = json.load(fp)
        sd_idx_to_tok_dict = {(v[0], v[1]): k for k, v in sd_dict.items()}
        kf_idx_to_tok_dict = {(v[0], v[1]): k for k, v in kf_dict.items()}
    else:
        nusc = None

    # Loop Calculate Esdf Residual (Multi-processing)
    if True:
        pool = multiprocessing.Pool(processes=16)
        for curr_frame_idx in tqdm(range(4, 57)):
            pool.apply_async(func=cal_esdf_res, args=(curr_frame_idx,))
        pool.close()
        pool.join()

    # Visualization Only
    curr_frame_idx = 10
    next_frame_idx = 11
    idx_tup = (seq_idx, curr_frame_idx, next_frame_idx)
    try:  # load stored esdf residual file
        # residual_file = os.path.join(esdf_res_dir, str(seq_idx).zfill(4), "resolution-" + str(o3d_voxel_size), "global_esdf_res_" + str(next_frame_idx) + "-" + str(curr_frame_idx) + ".bin")
        residual_file = os.path.join(esdf_res_dir, str(seq_idx).zfill(4), "surface_test", "global_esdf_res_" + str(next_frame_idx) + "-" + str(curr_frame_idx) + ".bin")
        esdf_residual = np.fromfile(residual_file, dtype=np.float32).reshape((-1, 4))
        # not_nan_idx = np.argwhere(np.invert(np.isnan(esdf_residual[:, -1])))
    except IOError:
        esdf_residual = cal_esdf_res(curr_frame_idx)

    # Vis
    if vis_esdf_res:
        # open3d_vis_esdfres_point(esdf_residual, idx_tup, nusc)
        # open3d_vis_esdfres_mos(esdf_residual, idx_tup, nusc)
        open3d_vis_esdfres_keyframes(esdf_residual, idx_tup, nusc)
    if vis_esdf:
        open3d_vis_esdf_mos(esdf_curr, sample_data_token, nusc)
    if vis_esdf_histogram:
        esdf_histogram(esdf_curr)
    if vis_compare_resolution:
        esdf_1_file = read_esdf(os.path.join(esdf_dir, str(seq_idx).zfill(4), "resolution-0.1",
                                             "global_esdf_res_" + str(curr_frame_idx) + ".bin"))
        esdf_2_file = read_esdf(os.path.join(esdf_dir, str(seq_idx).zfill(4), "resolution-0.2",
                                             "global_esdf_res_" + str(curr_frame_idx) + ".bin"))
        esdf_1 = np.fromfile(esdf_1_file, dtype=np.float32).reshape((-1, 4))
        esdf_2 = np.fromfile(esdf_2_file, dtype=np.float32).reshape((-1, 4))
        open3d_compare_resolution(esdf_1, esdf_2)