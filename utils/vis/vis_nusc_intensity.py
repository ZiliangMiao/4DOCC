import os
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import open3d
import numpy as np
import matplotlib.pyplot as plt

my_cmap = plt.cm.get_cmap('tab10')
my_cmap = my_cmap(np.arange(5))[:,:3]
color_first_return = my_cmap[0]
color_second_return = my_cmap[1]

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


def draw(vis, points, intensity):
    # draw origin
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)
    # draw points
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points)

    # intensity as color
    # intensity = intensity.repeat(3, axis=1) / 255
    pts.colors = open3d.utility.Vector3dVector(color_intensity(intensity))
    vis.add_geometry(pts)


if __name__ == '__main__':
    # load nusc dataset
    nusc = NuScenes(dataroot="/home/user/Datasets/nuScenes", version="v1.0-trainval")

    # vis_sample_token = "4db861b4498c43aba39097dfb31aa9bb"
    # sample = nusc.get('sample', vis_sample_token)

    vis_sample_data_token = "c3f016190e61437699b809b4b6241805"
    # sample_data_token = sample['data']['LIDAR_TOP']
    sample_data = nusc.get("sample_data", vis_sample_data_token)
    sample = nusc.get('sample', sample_data['sample_token'])

    vis_future = True
    time_flag = 0
    if vis_future:
        while time_flag < 5:
            # sample_data_token = sample_data['next']
            # sample_data = nusc.get("sample_data", sample_data_token)
            sample_token = sample['next']
            sample = nusc.get('sample', sample_token)
            sample_data_token = sample['data']['LIDAR_TOP']
            sample_data = nusc.get("sample_data", sample_data_token)
            time_flag += 1

    # get pcd
    pcd_file = os.path.join(nusc.dataroot, sample_data['filename'])
    pcd = LidarPointCloud.from_file(pcd_file).points.T  # [num_pts, x, y, z, i]
    points = pcd[:, 0:-1]
    intensity = pcd[:, -1]

    # open3d visualization
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    draw(vis, points, intensity)

    # view settings
    vis.get_render_option().point_size = 8.0
    vis.get_render_option().background_color = np.ones(3)
    view_ctrl = vis.get_view_control()
    view_ctrl.set_front((0.045542413135007273, 0.017044255410758744, 0.9988169912267878))
    view_ctrl.set_lookat((-0.40307459913255694, 0.48209966856794939, 3.4133266868446852))
    view_ctrl.set_up((0.013605495475171616, 0.99975111323203458, -0.017680556670610532))
    view_ctrl.set_zoom((0.10000000000000001))

    # run vis
    vis.run()
