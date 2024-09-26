import os.path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

import open3d as o3d

def pcd_split(pcd_ele_angle, bin_edges):
    pcd_beams_list = []
    for i in range(len(bin_edges)-2):
        row_idx = np.where((pcd_ele_angle[:, -1] >= bin_edges[i]) & (pcd_ele_angle[:, -1] < bin_edges[i+1]))
        pcd_beams_list.append(pcd_ele_angle[row_idx][:, 0:3])
    return pcd_beams_list

def beam_random_mask(pcd_beams_list, num):
    # num of beams to be kept
    indices = np.arange(0, 31, step=1)
    np.random.seed(13)
    np.random.shuffle(indices)
    kept_ind = indices[0:num].tolist()
    pcd_beams_kept_list = [pcd_beams_list[idx] for idx in kept_ind]
    pcd_beams_kept = np.concatenate([pcd_beams_list[idx] for idx in kept_ind])
    return pcd_beams_kept_list

def hex_to_rgb(hex):
    r = int(hex[1:3], 16) / 255
    g = int(hex[3:5], 16) / 255
    b = int(hex[5:7], 16) / 255
    rgb = [r, g, b]
    return rgb

def read_scene_point_cloud(scene_idx):
    dataroot = '/home/mars/catkin_ws/src/nuscenes2bag/mini_data'
    nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)
    nusc.list_scenes()

    scene = nusc.scene[scene_idx]
    # sample: annotated keyframe of a scene at a given timestamp
    first_sample_token = scene['first_sample_token']
    # nusc.render_sample(first_sample_token)
    test_sample = nusc.get('sample', first_sample_token)
    # sensor sample data
    sensor = 'LIDAR_TOP'
    lidar_top_data = nusc.get('sample_data', test_sample['data'][sensor])
    # nusc.render_sample_data(lidar_top_data['token'])
    # test = nusc.get('sample_data', lidar_top_data['token'])
    pcd_bin_file = os.path.join(nusc.dataroot, nusc.get('sample_data', lidar_top_data['token'])['filename'])
    point_cloud = LidarPointCloud.from_file(pcd_bin_file)
    bin_pcd = point_cloud.points.T
    bin_pcd = bin_pcd.reshape((-1, 4))[:, 0:3]

    # sample annotation
    test_annotation_token = test_sample['anns'][0]
    test_annotation_metadata = nusc.get('sample_annotation', test_annotation_token)
    return bin_pcd

def read_label_file(scene_idx=0):
    label_file = '/home/mars/catkin_ws/src/nuscenes2bag/mini_data/000000.label'
    label = np.fromfile(label_file, dtype=np.uint32).reshape((-1)) & 0xFFFF
    return label


if __name__ == '__main__':
    # test num_labels and num_points
    bin_pcd = read_scene_point_cloud(0)
    num_points = bin_pcd.shape[0]
    label = read_label_file()
    num_labels = label.shape[0]

    label_first = label[0]
    label_last = label[-1]


    # 浅粉, 红, 紫罗兰, 深紫, 蓝, 道奇蓝, 钢蓝, 深青, 春绿
    # 森林绿, 金, 橙, 巧克力, 橙红, 珊瑚色, 暗灰, 深红, 黑
    rgb_hex_list = ['#DC143C', '#C71585', '#4B0082', '#0000CD', '#1E90FF', '#4682B4', '#008B8B', '#00FF7F',
                    '#228B22', '#FFD700', '#FFA500', '#D2691E', '#FF4500', '#F08080', '#696969', '#8B0000', '#000000', '#FFB6C1']
    rgb_list = []
    for rgb_hex in rgb_hex_list:
        rgb = hex_to_rgb(rgb_hex)
        rgb_list.append(rgb)

    # start_color = (1.0, 0, 0.0)
    # end_color = (0.0, 0.0, 1.0)
    # steps = 16
    # for i in range(steps + 1):
    #     color = [start + i * (end - start) / steps for start, end in zip(start_color, end_color)]
    #     rgb_list.append(color)

    elevation_angle = np.degrees(np.arctan(bin_pcd[:, 2] / np.sqrt(np.power(bin_pcd[:, 0], 2) + np.power(bin_pcd[:, 1], 2)))).reshape(-1, 1)
    pcd_ele_angle = np.concatenate((bin_pcd, elevation_angle), axis=1)

    hist, bin_edges = np.histogram(elevation_angle, bins=32, range=(-31, 11), density=False)

    num_beam_kept = 24
    pcd_beams_list = pcd_split(pcd_ele_angle, bin_edges)
    pcd_beams_kept_list = beam_random_mask(pcd_beams_list, num_beam_kept)

    o3d_pcd_beam_list = []

    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bin_pcd))
    o3d_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # yellow
    o3d_pcd_beam_list.append(o3d_pcd)
    for i in range(len(pcd_beams_kept_list)):
        o3d_pcd_beam = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_beams_kept_list[i]))
        o3d_pcd_beam.paint_uniform_color(rgb_list[5])
        o3d_pcd_beam_list.append(o3d_pcd_beam)

    # visualization
    o3d.visualization.draw_geometries(o3d_pcd_beam_list, zoom=0.0940000000000000283,  # 0 -> max
                                                    front=[0.29678372412974874, 0.9079246722868356, 0.2959598123808696],
                                                    lookat=[1.1758518052639837, -1.4746038186455057, -2.0579947713357569],
                                                    up=[-0.12965352436494462, -0.26874320434342858, 0.9544459407106172])