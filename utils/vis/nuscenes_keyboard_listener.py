import argparse

import torch
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes
import numpy as np
import open3d as o3d
import os, time, sys, glob
import json
from enum import Enum
import _thread as thread
from pynput import keyboard
import pickle

mos_colormap = {
        0: (255/255, 255/255, 255/255),  # unknown: white
        1: (25/255, 80/255, 25/255),    # static: greenregister_key_callback
        2: (255/255, 20/255, 20/255)     # moving: red
    }
vfunc = np.vectorize(mos_colormap.get)


def color_intensity(intensity):
    min_intensity = 0
    max_intensity = 0.05 * 255
    CMAP='coolwarm'
    cmap = plt.get_cmap(CMAP)
    cmap = cmap(np.arange(256))[:, :3]
    n_colors = cmap.shape[0] - 1
    intensity = np.clip(intensity, 0, max_intensity)
    color_idx = np.floor((intensity - min_intensity) / (max_intensity - min_intensity) * n_colors).astype(int)
    color = cmap[color_idx]
    return color


class ThreadStatus(Enum):
    Init = 0
    Running = 1
    Close = 2


class PlayStatus(Enum):
    init = 4
    start = 1
    stop = 0
    end = 2


def get_ego_mask(pcd):
    # nuscenes car: length (4.084 m), width (1.730 m), height (1.562 m)
    # https://forum.nuscenes.org/t/dimensions-of-the-ego-vehicle-used-to-gather-data/550
    ego_mask = torch.logical_and(
        torch.logical_and(-0.865 <= pcd[:, 0], pcd[:, 0] <= 0.865),
        torch.logical_and(-1.5 <= pcd[:, 1], pcd[:, 1] <= 2.5),
    )
    return ego_mask


def get_outside_scene_mask(pcd, scene_bbox):
    inside_scene_mask = torch.logical_and(
        torch.logical_and(scene_bbox[0] <= pcd[:, 0], pcd[:, 0] <= scene_bbox[3]),
        torch.logical_and(scene_bbox[1] <= pcd[:, 1], pcd[:, 1] <= scene_bbox[4]),
        # torch.logical_and(scene_bbox[2] <= points[:, 2], points[:, 2] <= scene_bbox[5])
    )
    return ~inside_scene_mask


def showPointcloud3d(threadName, mark):
    global pointcloud_data, pred_mos_label, g_thread_status, current_id, pcd_file, files_num

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='model1 result', width=1200,
                      height=1800, left=0, top=150, visible=True)

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='model2 result', width=1200,
                       height=1800, left=1200, top=150, visible=True)

    # 添加控件--点云
    point_cloud = o3d.geometry.PointCloud()
    vis.add_geometry(point_cloud)
    vis1.add_geometry(point_cloud)

    labelset_list = []
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0]))
    vis1.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0]))

    render_option = vis.get_render_option()
    render_option.point_size = 2
    render_option.background_color = np.asarray([0, 0, 0])  # 颜色 0为黑；1为白
    ctr = vis.get_view_control()

    render_option1 = vis1.get_render_option()
    render_option1.point_size = 2
    render_option1.background_color = np.asarray([0, 0, 0])  # 颜色 0为黑；1为白
    ctr1 = vis1.get_view_control()

    to_reset_view_point = True

    while (g_thread_status == ThreadStatus.Init):
        time.sleep(0.1)
    while (g_thread_status == ThreadStatus.Running):
        point_cloud.points = o3d.utility.Vector3dVector(pointcloud_data)
        points_color = np.array(vfunc(pred_mos_label)).T
        point_cloud.colors = o3d.utility.Vector3dVector(points_color)
        vis.update_geometry(point_cloud)

        # 获取当前视图的相机参数
        camera_parameters = ctr.convert_to_pinhole_camera_parameters()
        # 赋值给另外一个相机
        ctr1.convert_from_pinhole_camera_parameters(camera_parameters)

        vis1.update_geometry(point_cloud)

        if to_reset_view_point:
            vis.reset_view_point(True)
            vis1.reset_view_point(True)
            to_reset_view_point = False
        vis.poll_events()
        vis.update_renderer()
        vis1.poll_events()
        vis1.update_renderer()

        sys.stdout.write("\r")  # 清空终端并清空缓冲区
        sys.stdout.write("{},      {}/{}s".format(pcd_file, current_id / 10, files_num / 10))  # 往缓冲区里写数据
        sys.stdout.flush()  # 将缓冲区里的数据刷新到终端，但是不会清空缓冲区


def play_data(threadName, mark):
    global nusc_dataset, pointcloud_data, pred_mos_label,\
        start_id, end_id, label_list, label_dir,\
        g_thread_status, play_status, data_thread_status, current_id, files_num, pcd_file
    while (True):
        while (data_thread_status == ThreadStatus.Init):
            if play_status == PlayStatus.start:
                id = 0
                for id in range(start_id, end_id):
                    current_id = id
                    label_file = label_list[id]
                    label = np.fromfile(os.path.join(label_dir, label_file), dtype=np.uint8)

                    sample_data_token = label_file.partition("_mos_pred.label")[0]
                    lidar_data = nusc_dataset.get('sample_data', sample_data_token)
                    pcd_file = os.path.join(nusc_dataset.dataroot, lidar_data['filename'])
                    pcd = LidarPointCloud.from_file(pcd_file).points.T[:, 0:-1]  # [num_pts, 4]

                    # filter
                    ego_mask = get_ego_mask(torch.tensor(pcd))
                    outside_scene_mask = get_outside_scene_mask(torch.tensor(pcd), [-70.0, -70.0, -4.5, 70.0, 70.0, 4.5])
                    valid_mask = torch.logical_and(~ego_mask, ~outside_scene_mask)
                    pcd = pcd[valid_mask]


                    assert len(pcd) == len(label)
                    (pointcloud_data, pred_mos_label) = (pcd, label)

                    g_thread_status = ThreadStatus.Running
                    if (play_status == PlayStatus.stop):
                        start_id = id
                        break
                    time.sleep(0.01)
                if (start_id == end_id - 1) or (id == end_id - 1):
                    data_thread_status = ThreadStatus.Close
            time.sleep(1)
        time.sleep(1)

def mogo_vis(label_dir, nusc):
    global nusc_dataset, pointcloud_data, pred_mos_label, start_id, end_id, label_list,\
        g_thread_status, play_status, data_thread_status, before_gt_number, before_label_number, current_id, files_num, pcd_file
    # nusc dataset
    nusc_dataset = nusc

    label_list = os.listdir(label_dir)

    files_num = len(label_list)
    start_id = 0
    before_gt_number = 0
    before_label_number = 0
    end_id = len(label_list)
    g_thread_status = ThreadStatus.Init  # Init Running Close
    data_thread_status = ThreadStatus.Init
    play_status = PlayStatus.start
    thread.start_new_thread(play_data, ("playdata", 1,))
    thread.start_new_thread(listen_keyboard, ("listener",))

    showPointcloud3d("show3d", 0)

    g_thread_status = ThreadStatus.Close

def listen_keyboard(thread_name):
    def on_press(key):
        global pointcloud_data, pred_mos_label, start_id, end_id, \
            label_list, label_dir, g_thread_status, play_status, current_id, data_thread_status, pcd_file

        if key == keyboard.Key.space:  # 暂停与播放
            if play_status == PlayStatus.start:
                play_status = PlayStatus.stop
            else:
                play_status = PlayStatus.start

        elif key == keyboard.Key.left:  # 回退
            if play_status == PlayStatus.start:
                play_status = PlayStatus.stop

            current_id = current_id - 1
            if current_id < 0:
                current_id = 0
            start_id = current_id
            label_file = label_list[current_id]
            label = np.fromfile(os.path.join(label_dir, label_file), dtype=np.uint8)

            sample_data_token = label_file.partition("_mos_pred.label")[0]
            lidar_data = nusc_dataset.get('sample_data', sample_data_token)
            pcd_file = os.path.join(nusc_dataset.dataroot, lidar_data['filename'])
            pcd = LidarPointCloud.from_file(pcd_file).points.T[:, 0:-1]  # [num_pts, 4]

            assert len(pcd) == len(label)
            (pointcloud_data, pred_mos_label) = (pcd, label)

            if data_thread_status == ThreadStatus.Close:
                data_thread_status = ThreadStatus.Init
            # print(data_thread_status,start_id,end_id,play_status)

        elif key == keyboard.Key.right:  # 快进
            if play_status == PlayStatus.start:
                play_status = PlayStatus.stop

            current_id = current_id + 1
            if current_id >= end_id:
                current_id = end_id - 1
            start_id = current_id

            label_file = label_list[current_id]
            label = np.fromfile(os.path.join(label_dir, label_file), dtype=np.uint8)

            sample_data_token = label_file.partition("_mos_pred.label")[0]
            lidar_data = nusc_dataset.get('sample_data', sample_data_token)
            pcd_file = os.path.join(nusc_dataset.dataroot, lidar_data['filename'])
            pcd = LidarPointCloud.from_file(pcd_file).points.T[:, 0:-1]  # [num_pts, 4]
            assert len(pcd) == len(label)
            (pointcloud_data, pred_mos_label) = (pcd, label)

            if data_thread_status == ThreadStatus.Close:
                data_thread_status = ThreadStatus.Init
        else:
            pass

    def on_release(key):
        global g_thread_status, ThreadStatus
        if key == keyboard.Key.esc:
            g_thread_status = ThreadStatus.Close
            return False

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == '__main__':
    label_dir = '/home/user/Projects/4DOCC/logs/ours/bg_pretrain(epoch-99)-mos_finetune/100%nuscenes-50%nuscenes/vs-0.1_t-9.5_bs-4/version_0/predictions/epoch_99'

    # load nusc dataset
    nusc = NuScenes(dataroot="/home/user/Datasets/nuScenes", version="v1.0-trainval")
    mogo_vis(label_dir, nusc)