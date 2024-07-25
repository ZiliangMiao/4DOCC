import os
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import open3d
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

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


def draw(vis, origin, points, intensity):
    # draw origin
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)
    # draw points
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points)

    # draw ray, create an empty line set, Use lineset.point / lineset.line to access the point / line attributes
    ray_points = np.insert(points, 0, origin, 0)
    lines = np.array([[0, i] for i in range(len(points))])
    colors = np.array([[0, 0, 0]])
    lineset = open3d.geometry.LineSet()
    lineset.lines = open3d.utility.Vector2iVector(lines)
    lineset.colors = open3d.utility.Vector3dVector(colors)
    lineset.points = open3d.utility.Vector3dVector(ray_points)
    vis.add_geometry(lineset)

    # intensity as color
    # intensity = intensity.repeat(3, axis=1) / 255
    pts.colors = open3d.utility.Vector3dVector(color_intensity(intensity))
    vis.add_geometry(pts)

def get_global_pose(nusc, sd_token, inverse=False):
    sd = nusc.get("sample_data", sd_token)
    sd_ep = nusc.get("ego_pose", sd["ego_pose_token"])
    sd_cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

    if inverse is False:  # transform: from lidar coord to global coord
        global_from_ego = transform_matrix(
            sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False
        )
        ego_from_sensor = transform_matrix(
            sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False
        )
        pose = global_from_ego.dot(ego_from_sensor)
    else:  # transform: from global coord to lidar coord
        sensor_from_ego = transform_matrix(
            sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True
        )
        ego_from_global = transform_matrix(
            sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True
        )
        pose = sensor_from_ego.dot(ego_from_global)
    return pose

def get_transformed_pcd(nusc, sd_token_ref, sd_token):
    # sample data -> pcd
    sample_data = nusc.get("sample_data", sd_token)
    lidar_pcd = LidarPointCloud.from_file(f"{nusc.dataroot}/{sample_data['filename']}")  # [num_pts, x, y, z, i]

    # poses
    global_from_curr = get_global_pose(nusc, sd_token, inverse=False)  # from {lidar} to {global}
    ref_from_global = get_global_pose(nusc, sd_token_ref, inverse=True)  # from {global} to {ref lidar}
    ref_from_curr = ref_from_global.dot(global_from_curr)  # from {lidar} to {ref lidar}

    # transformed sensor origin and points, at {ref lidar} frame
    origin_tf = np.array(ref_from_curr[:3, 3], dtype=np.float32)  # curr sensor location, at {ref lidar} frame
    lidar_pcd.transform(ref_from_curr)
    points_tf = np.array(lidar_pcd.points[:3].T, dtype=np.float32)  # curr point cloud, at {ref lidar} frame
    intensity = np.array(lidar_pcd.points[-1, :].T, dtype=np.float32)
    return origin_tf, points_tf, intensity



if __name__ == '__main__':
    # load nusc dataset
    nusc = NuScenes(dataroot="/home/user/Datasets/nuScenes", version="v1.0-trainval")

    # ref lidar frame [t]; lidar frame [t-1]
    sd_token_ref = "c3f016190e61437699b809b4b6241805"
    sample_data_ref = nusc.get("sample_data", sd_token_ref)
    sd_token = sample_data_ref['prev']

    # get pcd
    origin_ref, points_ref, intensity_ref = get_transformed_pcd(nusc, sd_token_ref, sd_token_ref)
    origin, points, intensity = get_transformed_pcd(nusc, sd_token_ref, sd_token)

    # open3d visualization
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    draw(vis, origin, points, intensity)

    # view settings
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().line_width = 5.0
    vis.get_render_option().background_color = np.ones(3)
    view_ctrl = vis.get_view_control()
    view_ctrl.set_front((0.045542413135007273, 0.017044255410758744, 0.9988169912267878))
    view_ctrl.set_lookat((-0.40307459913255694, 0.48209966856794939, 3.4133266868446852))
    view_ctrl.set_up((0.013605495475171616, 0.99975111323203458, -0.017680556670610532))
    view_ctrl.set_zoom((0.10000000000000001))

    # run vis
    vis.run()
