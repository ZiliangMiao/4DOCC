#-*- coding: UTF-8 -*-
import numpy as np
import open3d as o3d
import os,time,sys,glob
import json
from enum import Enum
import _thread as thread
from pynput import keyboard
import pickle

label_name = {'unknown':0,'Vehicle':3,'Pedestrian':1,'Cyclist':2,'Sign':4}
box_colormap = [
    [1, 1, 1], #白色
    [1, 1, 0], #黄色 Pedestrian
    [0, 1, 0], #绿色 Cyclist
    [0, 1, 1], #蓝色 Vehicle
    [0, 1, 1], #蓝色 Sign
]

class ThreadStatus(Enum):
    Init = 0
    Running = 1
    Close = 2

class PlayStatus(Enum):
    init=4
    start=1
    stop=0
    end=2

def generate_line(box_size,box_center,box_rotation,type_name):
    yaw = box_rotation
    dir = np.array([np.math.cos(yaw), np.math.sin(yaw), 0])
    ortho_dir = np.array([-dir[1], dir[0], 0])

    width = box_size[1]
    height = box_size[0]
    deep = box_size[2]
    center = box_center[0], box_center[1], box_center[2]
    center = np.array(center)
    # 计算八个点
    points = np.array([[0.0, 0.0, 0.0] for i in range(8)])
    z_dir = np.array([0.0, 0.0, 1.0])

    points[0] = center + dir * (height * 0.5) + ortho_dir * (width * 0.5) - z_dir * (deep * 0.5);
    points[1] = center - dir * (height * 0.5) + ortho_dir * (width * 0.5) - z_dir * (deep * 0.5);
    points[2] = center - dir * (height * 0.5) - ortho_dir * (width * 0.5) - z_dir * (deep * 0.5);
    points[3] = center + dir * (height * 0.5) - ortho_dir * (width * 0.5) - z_dir * (deep * 0.5);
    points[4] = center + dir * (height * 0.5) + ortho_dir * (width * 0.5) + z_dir * (deep * 0.5);
    points[5] = center - dir * (height * 0.5) + ortho_dir * (width * 0.5) + z_dir * (deep * 0.5);
    points[6] = center - dir * (height * 0.5) - ortho_dir * (width * 0.5) + z_dir * (deep * 0.5);
    points[7] = center + dir * (height * 0.5) - ortho_dir * (width * 0.5) + z_dir * (deep * 0.5);

    points = [[points[point_id, 0], points[point_id, 1], points[point_id, 2]] for point_id in range(8)]
    points = np.array(points)
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    line_color = [box_colormap[label_name[type_name]] for i in range(len(lines))]

    return points,lines,line_color

def draw_box(vis,vis1, gt_boxes, ref_labels, lineset_list):
    global before_gt_number
    i=0
    update_number = max(before_gt_number,gt_boxes.shape[0])
    before_gt_number=gt_boxes.shape[0]

    for id in range(update_number):
        if id<gt_boxes.shape[0]:
            box_size = gt_boxes[id,3:6]
            box_center = gt_boxes[id,:3]
            box_rotation = gt_boxes[id,6]
            box_name = ref_labels[id]
        else:
            box_size=[0,0,0]
            box_center=[0,0,0]
            box_rotation=0.0
            box_name = 'unknown'

        points,lines,color=generate_line(box_size,box_center,box_rotation,box_name)

        lineset_list[id].lines = o3d.utility.Vector2iVector(lines)
        lineset_list[id].colors = o3d.utility.Vector3dVector(color)  # 线条颜色
        lineset_list[id].points = o3d.utility.Vector3dVector(points)
        vis.update_geometry(lineset_list[id])
        vis1.update_geometry(lineset_list[id])

        i = i + 1

    return vis,vis1

def showPointcloud3d(threadName,mark):
    global pointcloud_data,g_thread_status,g_gt_boxes,g_gt_name,file_name,current_id,files_num

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='model1 result', width=1400,
                        height=1800, left=300, top=150, visible=True)

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='model2 result', width=1400,
                        height=1800, left=300, top=150, visible=True)

    # 添加控件--点云
    point_cloud = o3d.geometry.PointCloud()
    vis.add_geometry(point_cloud)
    vis1.add_geometry(point_cloud)
    # 添加控件--box
    lineset_list=[]
    for i in range(100):
        lineset = o3d.geometry.LineSet()
        lineset_list.append(lineset)
        vis.add_geometry(lineset)
        vis1.add_geometry(lineset)

    labelset_list = []
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0]))
    vis1.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0]))

    render_option = vis.get_render_option()
    render_option.point_size = 2
    render_option.background_color = np.asarray([0, 0, 0])  # 颜色 0为黑；1为白
    ctr=vis.get_view_control()

    render_option1 = vis1.get_render_option()
    render_option1.point_size = 2
    render_option1.background_color = np.asarray([0, 0, 0])  # 颜色 0为黑；1为白
    ctr1=vis1.get_view_control()

    to_reset_view_point = True

    while(g_thread_status==ThreadStatus.Init):
        time.sleep(0.1)
    while(g_thread_status==ThreadStatus.Running):
        point_cloud.points = o3d.utility.Vector3dVector(pointcloud_data[:,:3])
        # point_cloud.colors = o3d.utility.Vector3dVector(np.array([[1,1,1] for i in pointcloud_data[:,-1]]))
        point_cloud.paint_uniform_color([1, 1, 1])
        vis.update_geometry(point_cloud)

        #获取当前视图的相机参数
        camera_parameters=ctr.convert_to_pinhole_camera_parameters()
        #赋值给另外一个相机
        ctr1.convert_from_pinhole_camera_parameters(camera_parameters)

        vis1.update_geometry(point_cloud)

        if g_gt_boxes.shape[0]>len(lineset_list):
            for i in range(g_gt_boxes.shape[0]-len(lineset_list)):
                lineset = o3d.geometry.LineSet()
                lineset_list.append(lineset)
                vis.add_geometry(lineset)
                vis1.add_geometry(lineset)
        vis,vis1 = draw_box(vis,vis1,g_gt_boxes, g_gt_name, lineset_list)

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

def play_data(threadName,mark):
    global pointcloud_data,g_gt_boxes,g_gt_name,\
        start_id, end_id, pcd_list,label_list, \
        g_thread_status,play_status,data_thread_status, \
        file_name, current_id, files_num
    while(True):
        while(data_thread_status==ThreadStatus.Init):
            if play_status==PlayStatus.start:
                id=0
                for id in range(start_id,end_id):
                    file_path=pcd_list[id]
                    current_id=id
                    file_name=os.path.basename(file_path)
                    pointcloud_data=np.load(file_path)
                    g_gt_boxes,g_gt_name = label_list[id]['boxes'],label_list[id]['name']
                    g_thread_status = ThreadStatus.Running
                    if(play_status==PlayStatus.stop):
                        start_id=id
                        break
                    time.sleep(0.01)
                if (start_id==end_id-1) or (id==end_id-1):
                    data_thread_status=ThreadStatus.Close

            time.sleep(1)
        time.sleep(1)

def get_file_list(file_dir,label_file):

    files_list=os.listdir(file_dir)
    files_list=glob.glob(os.path.join(file_dir,"*.npy"))
    files_list.sort()
    file_length=len(files_list)

    label_list =[]
    with open(label_file,"rb") as user_file:
        parsed_json = pickle.load(user_file)

        for i in range(file_length):
            frame_info = parsed_json[i]['annos']
            dimensions=frame_info['dimensions']
            location = frame_info['location']
            heading_angles=frame_info['heading_angles']
            obj_name = frame_info['name']

            gt_boxes = np.concatenate([location,dimensions,heading_angles.reshape(-1,1)],axis=1)

            label_info_i={}
            label_info_i['boxes']=gt_boxes
            label_info_i['name']=obj_name
            label_list.append(label_info_i)

    return files_list,label_list

def mogo_vis(root_dir,label_file):
    global pointcloud_data,start_id,end_id,pcd_list,label_list,\
        g_thread_status,play_status,data_thread_status,before_gt_number,before_label_number,file_name,current_id,files_num

    pcd_list,label_list=get_file_list(root_dir,label_file)

    files_num=len(pcd_list)
    start_id=0
    before_gt_number=0
    before_label_number=0
    end_id=len(pcd_list)
    g_thread_status=ThreadStatus.Init # Init Running Close
    data_thread_status =ThreadStatus.Init
    play_status=PlayStatus.start
    thread.start_new_thread(play_data,("playdata",1,))
    thread.start_new_thread(listen_keyboard,("listener",))

    showPointcloud3d("show3d",0)
    
    g_thread_status=ThreadStatus.Close

def listen_keyboard(thread_name):

    def on_press(key):
        global pointcloud_data,g_gt_boxes,g_gt_name,start_id,end_id,pcd_list,\
            label_list,g_thread_status,play_status,file_name,current_id,data_thread_status

        if key ==  keyboard.Key.space: #暂停与播放
            if play_status == PlayStatus.start:
                play_status = PlayStatus.stop
            else:
                play_status = PlayStatus.start

        elif key == keyboard.Key.left: #回退
            if play_status == PlayStatus.start:
                play_status = PlayStatus.stop

            current_id=current_id-1
            if current_id<0:
                current_id=0
            start_id=current_id
            file_path= pcd_list[current_id]
            file_name =os.path.basename(file_path)
            pointcloud_data = np.load(file_path)
            
            g_gt_boxes, g_gt_name = label_list[current_id]['boxes'],label_list[current_id]['name']
            if data_thread_status==ThreadStatus.Close:
                data_thread_status=ThreadStatus.Init
            # print(data_thread_status,start_id,end_id,play_status)

        elif key == keyboard.Key.right: #快进
            if play_status == PlayStatus.start:
                play_status = PlayStatus.stop

            current_id=current_id+1
            if current_id>=end_id:
                current_id=end_id-1
            start_id=current_id
            file_path= pcd_list[current_id]
            file_name=os.path.basename(file_path)
            pointcloud_data = np.load(file_path)
            
            g_gt_boxes, g_gt_name = label_list[current_id]['boxes'],label_list[current_id]['name']

            if data_thread_status==ThreadStatus.Close:
                data_thread_status=ThreadStatus.Init
        else:
            pass

    def on_release(key):
        global g_thread_status,ThreadStatus
        if key == keyboard.Key.esc:
            g_thread_status=ThreadStatus.Close
            return False

    with keyboard.Listener(on_press=on_press,on_release=on_release) as listener:
        listener.join()
    
if __name__ == '__main__':

    root_dir = "/home/user/Downloads/open3d_multi_vis/segment-9747453753779078631_940_000_960_000_with_camera_labels/"
    label_file = "/home/user/Downloads/open3d_multi_vis/segment-9747453753779078631_940_000_960_000_with_camera_labels/segment-9747453753779078631_940_000_960_000_with_camera_labels.pkl"

    mogo_vis(root_dir,label_file)