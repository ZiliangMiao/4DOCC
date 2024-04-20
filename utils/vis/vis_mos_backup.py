import argparse
from nuscenes import NuScenes
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
        1: (25/255, 80/255, 25/255),    # static: green
        2: (255/255, 20/255, 20/255)     # moving: red
    }
vfunc = np.vectorize(mos_colormap.get)

class ThreadStatus(Enum):
    Init = 0
    Running = 1
    Close = 2


class PlayStatus(Enum):
    init = 4
    start = 1
    stop = 0
    end = 2

def showPointcloud3d(threadName, mark):
    global pointcloud_data, pred_mos_label, g_thread_status, current_id, file_name, files_num

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
        sys.stdout.write("{},      {}/{}s".format(file_name, current_id / 10, files_num / 10))  # 往缓冲区里写数据
        sys.stdout.flush()  # 将缓冲区里的数据刷新到终端，但是不会清空缓冲区


def play_data(threadName, mark):
    global pointcloud_data, pred_mos_label,\
        start_id, end_id, pcd_list, label_list, \
        g_thread_status, play_status, data_thread_status, \
        file_name, current_id, files_num
    while (True):
        while (data_thread_status == ThreadStatus.Init):
            if play_status == PlayStatus.start:
                id = 0
                for id in range(start_id, end_id):
                    current_id = id
                    pcd_file = pcd_list[id]
                    file_name = os.path.basename(pcd_file)
                    pcd = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 3)
                    label_file = label_list[id]
                    label = np.fromfile(label_file, dtype=np.uint8)
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

def get_file_list(pcd_dir, label_dir):
    pcd_list = glob.glob(os.path.join(pcd_dir, "*.bin"))
    pcd_list.sort()
    label_list = glob.glob(os.path.join(label_dir, "*.label"))
    label_list.sort()
    assert len(pcd_list) == len(label_list)
    return pcd_list, label_list

def mogo_vis(args, nusc):
    global pointcloud_data, pred_mos_label, start_id, end_id, pcd_list, label_list, \
        g_thread_status, play_status, data_thread_status, before_gt_number, before_label_number, file_name, current_id, files_num

    pcd_dir = os.path.join(args.model_dir, "results", args.expt_name, "visualization", "valid_points")
    label_dir = os.path.join(args.model_dir, "results", args.expt_name, "visualization", "pred_mos_labels")
    pcd_list, label_list = get_file_list(pcd_dir, label_dir)

    files_num = len(pcd_list)
    start_id = 0
    before_gt_number = 0
    before_label_number = 0
    end_id = len(pcd_list)
    g_thread_status = ThreadStatus.Init  # Init Running Close
    data_thread_status = ThreadStatus.Init
    play_status = PlayStatus.start
    thread.start_new_thread(play_data, ("playdata", 1,))
    thread.start_new_thread(listen_keyboard, ("listener",))

    showPointcloud3d("show3d", 0)

    g_thread_status = ThreadStatus.Close

def listen_keyboard(thread_name):
    def on_press(key):
        global pointcloud_data, pred_mos_label, start_id, end_id, pcd_list, \
            label_list, g_thread_status, play_status, file_name, current_id, data_thread_status

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
            pcd_file = pcd_list[current_id]
            file_name = os.path.basename(pcd_file)
            pcd = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 3)
            label_file = label_list[current_id]
            label = np.fromfile(label_file, dtype=np.uint8)
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

            pcd_file = pcd_list[current_id]
            pcd = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 3)
            label_file = label_list[current_id]
            label = np.fromfile(label_file, dtype=np.uint8)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="models/nusc/1s_forecasting")
    parser.add_argument("--expt-name", type=str, default="nusc_1s_mos")
    args = parser.parse_args()

    # load nusc dataset
    nusc = NuScenes(dataroot="/home/user/Datasets/nuScenes", version="v1.0-trainval")
    mogo_vis(args, nusc)