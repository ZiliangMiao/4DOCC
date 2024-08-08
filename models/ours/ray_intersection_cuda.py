import os
import time
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import open3d
import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import argparse


my_cmap = plt.cm.get_cmap('tab10')
my_cmap = my_cmap(np.arange(5))[:,:3]
color_first_return = my_cmap[0]
color_second_return = my_cmap[1]

occ_colormap = {
        0: (102/255, 102/255, 102/255),   # unknown: grey
        1: (95/255, 175/255, 101/255),    # free: green
        2: (223/255, 142/255, 42/255)     # occupied: orange
}
mos_colormap = {
        0: (159/255, 53/255, 190/255),    # unknown: purple
        1: (78/255, 97/255, 162/255),     # static points: blue
        2: (147/255, 49/255, 49/255)    # moving points: red
}
occ_color_func = np.vectorize(occ_colormap.get)
mos_color_func = np.vectorize(mos_colormap.get)

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


# TODO: open3d LineMesh, refer to: https://github.com/isl-org/Open3D/pull/738#issuecomment-564785941
class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    @staticmethod
    def normalized(a, axis=-1, order=2):
        """Normalizes a numpy array of points"""
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis), l2

    @staticmethod
    def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
        """
        Aligns vector a to vector b with axis angle rotation
        """
        if np.array_equal(a, b):
            return None, None
        axis_ = np.cross(a, b)
        axis_ = axis_ / np.linalg.norm(axis_)
        angle = np.arccos(np.dot(a, b))

        return axis_, angle

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = self.normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = self.align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = open3d.geometry.TriangleMesh.create_cylinder(self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                R = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_a)
                cylinder_segment = cylinder_segment.rotate(R=R, center=(0, 0, 0))  # TODO: rotate center is not boolean: https://www.open3d.org/docs/release/python_api/open3d.geometry.MeshBase.html#open3d.geometry.MeshBase.rotate
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


def draw(query_rays, key_rays_list, query_ray_to_ints_pts_tensor_dict, key_sensor_idx_to_ray_idx_dict):
    # create open3d vis
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    # draw lidar origin axis
    org_query = query_rays.get_ray_start().cpu()
    axis_query = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2.5, origin=org_query)
    vis.add_geometry(axis_query)

    # draw query points
    pts_query = query_rays.get_ray_end().cpu()
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pts_query)
    pcd.colors = open3d.utility.Vector3dVector(np.tile(my_cmap[0], (len(pcd.points), 1)))
    vis.add_geometry(pcd)

    # ray_points = np.insert(points, 0, origin, 0)
    # lines = np.array([[0, i] for i in range(len(points))])
    # TODO: automatic view all valid vis points

    # query rays with intersection points
    query_rays_idx = torch.tensor(list(idx for idx in query_ray_to_ints_pts_tensor_dict.keys()))
    lineset_query = open3d.geometry.LineSet()  # query reference lidar rays
    vertex_query = torch.vstack((org_query, pts_query[query_rays_idx]))
    lines_query = [[0, i+1] for i in range(len(vertex_query))]
    lineset_query.points = open3d.utility.Vector3dVector(vertex_query)
    lineset_query.lines = open3d.utility.Vector2iVector(lines_query)
    lineset_query.colors = open3d.utility.Vector3dVector(my_cmap[1].reshape(-1, 3))
    vis.add_geometry(lineset_query)

    # org key axis and key rays with intersection points
    for key_idx, ints_rays_idx in key_sensor_idx_to_ray_idx_dict.items():
        key_rays = key_rays_list[key_idx]
        # axis
        org_key = key_rays.get_ray_start().cpu()
        axis_key = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0 + 0.1 * key_idx, origin=org_key)
        vis.add_geometry(axis_key)

        # rays
        pts_key = key_rays.get_ray_end().cpu()
        lineset_key = open3d.geometry.LineSet()
        vertex_key = torch.vstack((org_key, pts_key[ints_rays_idx]))
        lines_key = [[0, i + 1] for i in range(len(vertex_key))]
        lineset_key.points = open3d.utility.Vector3dVector(vertex_key)
        lineset_key.lines = open3d.utility.Vector2iVector(lines_key)
        lineset_key.colors = open3d.utility.Vector3dVector(my_cmap[2].reshape(-1, 3))
        vis.add_geometry(lineset_key)

        # key ray end points
        pcd_key = open3d.geometry.PointCloud()
        pcd_key.points = open3d.utility.Vector3dVector(pts_key[ints_rays_idx])
        pcd_key.colors = open3d.utility.Vector3dVector(np.tile(my_cmap[3], (len(pcd_key.points), 1)))
        vis.add_geometry(pcd_key)

    # intersection points
    ints_pts_list = []
    for ints_pts in query_ray_to_ints_pts_tensor_dict.values():
        ints_pts_list.append(ints_pts[:, 0:3])
    ints_pts = torch.cat(ints_pts_list, dim=0)
    pcd_ints = open3d.geometry.PointCloud()
    pcd_ints.points = open3d.utility.Vector3dVector(ints_pts)
    pcd_ints.colors = open3d.utility.Vector3dVector(np.tile(my_cmap[4], (len(pcd_ints.points), 1)))
    vis.add_geometry(pcd_ints)

    # view settings
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().line_width = 5.0
    vis.get_render_option().background_color = np.ones(3)
    view_ctrl = vis.get_view_control()
    view_ctrl.set_front((0.045542413135007273, 0.017044255410758744, 0.9988169912267878))
    view_ctrl.set_lookat((-0.40307459913255694, 0.48209966856794939, 3.4133266868446852))
    view_ctrl.set_up((0.013605495475171616, 0.99975111323203458, -0.017680556670610532))
    # view_ctrl.set_zoom((0.10000000000000001))

    # run vis
    vis.run()


def draw_dynamic_rays(query_rays, key_rays_list, rays_to_ints_pts_dict, mos_labels_query, mos_labels_key_list):
    vis_ray_idx = 5374  # nonlocal variable
    num_vis_rays = len(rays_to_ints_pts_dict)
    vis = open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='ray intersection points', width=3840, height=2160, left=0, top=0)

    # draw method
    def draw_ray(vis):
        nonlocal vis_ray_idx
        print("Draw query ray index: " + str(vis_ray_idx))
        vis.clear_geometries()

        # draw static geometries (query lidar orgs, key lidar orgs, and query point cloud)
        axis_size = 2.0
        org_query = query_rays.get_ray_start().cpu()
        axis_query = open3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size * 2, origin=org_query)
        vis.add_geometry(axis_query)
        key_rays_points_list = []
        key_orgs_list = []
        for key_rays in key_rays_list:
            key_org = key_rays.get_ray_start().cpu()
            key_orgs_list.append(key_org)
            axis_key = open3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size + key_rays.get_ray_ts() * 0.1, origin=key_org)
            vis.add_geometry(axis_key)
            # append key rays points
            key_rays_points_list.append(key_rays.get_ray_end().cpu())

        # draw query ray point
        pts_query = query_rays.get_ray_end().cpu()
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pts_query)
        pcd.colors = open3d.utility.Vector3dVector(np.array(mos_color_func(mos_labels_query)).T)  # colored by mos labels
        pcd_down = pcd.voxel_down_sample(voxel_size=0.10)  # TODO: point cloud downsample
        vis.add_geometry(pcd_down)

        if len(rays_to_ints_pts_dict) != 0:
            # current query ray for visualization
            query_ray_idx = list(rays_to_ints_pts_dict.keys())[vis_ray_idx]
            (key_sensor_rays_idx, ints_pts_with_label) = list(rays_to_ints_pts_dict.values())[vis_ray_idx]

            # query ray point
            query_ray_point_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=20)
            query_ray_point = np.asarray(query_rays.get_ray_end()[query_ray_idx].cpu().numpy())
            query_ray_point_sphere = query_ray_point_sphere.translate(query_ray_point, relative=False)
            query_ray_point_sphere.paint_uniform_color(mos_color_func(mos_labels_query[query_ray_idx]))
            vis.add_geometry(query_ray_point_sphere)

            # key rays points
            key_ints_rays_pts_list = []
            for i in range(len(key_sensor_rays_idx)):
                key_sensor_idx = key_sensor_rays_idx[i][0]
                key_ray_idx = key_sensor_rays_idx[i][1]
                key_ray_point = key_rays_points_list[key_sensor_idx][key_ray_idx].cpu().numpy()
                key_ray_point_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=20)
                key_ray_point_sphere = key_ray_point_sphere.translate(key_ray_point, relative=False)
                key_ray_point_sphere.paint_uniform_color(mos_color_func(mos_labels_key_list[key_sensor_idx][key_ray_idx]))
                vis.add_geometry(key_ray_point_sphere)
                key_ints_rays_pts_list.append(key_rays_points_list[key_sensor_idx][key_ray_idx].cpu())

            # intersection points
            for i in range(len(ints_pts_with_label)):
                ints_point_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=20)
                ints_point_sphere = ints_point_sphere.translate(ints_pts_with_label[i, 0:3], relative=False)
                ints_point_sphere.paint_uniform_color(occ_color_func(ints_pts_with_label[i, -1]))
                vis.add_geometry(ints_point_sphere)

            # query ray
            lineset_query = open3d.geometry.LineSet()
            vertex_query = torch.vstack((org_query, pts_query[query_ray_idx]))
            lines_query = [[0, 1]]
            color_query = np.tile((234/255, 51/255, 35/255), (1, 1))
            lineset_query.points = open3d.utility.Vector3dVector(vertex_query)
            lineset_query.lines = open3d.utility.Vector2iVector(lines_query)
            lineset_query.colors = open3d.utility.Vector3dVector(color_query)
            vis.add_geometry(lineset_query)
            # TODO: LineMesh is a user defined class, cannot directly add into vis geometry, can only draw_geometries
            # line_mesh = LineMesh(vertex_query.cpu(), lines_query, [0, 0, 0], radius=0.2)
            # line_mesh_geoms = line_mesh.cylinder_segments
            # open3d.visualization.draw_geometries([lineset_query, *line_mesh_geoms])

            # key rays
            lineset_key = open3d.geometry.LineSet()  # TODO: all key rays that intersect with current query ray (at all key lidars)
            vertex_key = torch.vstack((torch.stack(key_orgs_list, dim=0), torch.stack(key_ints_rays_pts_list, dim=0)))  # (len(key_orgs) + len(key_rays_pts), 3)
            lines_key = []
            for i in range(len(key_sensor_rays_idx)):
                key_sensor_idx = key_sensor_rays_idx[i][0]
                lines_key.append([key_sensor_idx, i + len(key_orgs_list)])
            color_key = np.tile((0, 0, 0), (len(lines_key), 1))
            lineset_key.points = open3d.utility.Vector3dVector(vertex_key)
            lineset_key.lines = open3d.utility.Vector2iVector(lines_key)
            lineset_key.colors = open3d.utility.Vector3dVector(color_key)
            vis.add_geometry(lineset_key)

        # open3d view option
        vis.get_render_option().point_size = 5.0
        vis.get_render_option().background_color = np.ones(3)
        view_ctrl = vis.get_view_control()
        view_ctrl.set_front((0.0, 0.0, 1.0))
        view_ctrl.set_lookat((0.33837080335515901, -2.223431055221385, 2.6541285514831543))
        view_ctrl.set_up((0.0, 1.0, 0.0))
        view_ctrl.set_zoom((0.19999999999999959))

        # update vis
        vis.poll_events()
        vis.update_renderer()

    def render_next_ray(vis):
        nonlocal vis_ray_idx
        vis_ray_idx += 1
        if vis_ray_idx >= num_vis_rays:
            vis_ray_idx = num_vis_rays - 1
        draw_ray(vis)
    def render_prev_ray(vis):
        nonlocal vis_ray_idx
        vis_ray_idx -= 1
        if vis_ray_idx < 0:
            vis_ray_idx = 0
        draw_ray(vis)

    # render keyboard control
    vis.register_key_callback(ord('D'), render_next_ray)
    vis.register_key_callback(ord('A'), render_prev_ray)
    vis.run()


def get_sample_tok_list(sample, direction, num_samples, every_k_samples):
    sd_tok_key_list = []
    tok_cnt = 0
    skip_cnt = 0
    while sample[direction] != '' and tok_cnt < num_samples:
        sample_tok = sample[direction]
        sample = nusc.get("sample", sample_tok)
        if skip_cnt < every_k_samples - 1:
            skip_cnt += 1
            continue
        else:
            sd_tok_key_list.append(sample['data']['LIDAR_TOP'])
            tok_cnt += 1
            skip_cnt = 0
    return sd_tok_key_list, tok_cnt == num_samples


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


def filter_points(points, valid_range):
    # nuscenes car: length (4.084 m), width (1.730 m), height (1.562 m)
    ego_mask = torch.logical_and(
        torch.logical_and(-0.865 <= points[:, 0], points[:, 0] <= 0.865),
        torch.logical_and(-1.5 <= points[:, 1], points[:, 1] <= 2.5),
    )
    inside_scene_bbox_mask = torch.logical_and(
        torch.logical_and(-valid_range <= points[:, 0], points[:, 0] <= valid_range),
        torch.logical_and(-valid_range <= points[:, 1], points[:, 1] <= valid_range),
    )
    mask = torch.logical_and(~ego_mask, inside_scene_bbox_mask)
    return points[mask], mask.cpu().numpy()


def get_transformed_pcd(nusc, sd_token_ref, sd_token, max_range):
    # sample data -> pcd
    sample_data = nusc.get("sample_data", sd_token)
    lidar_pcd = LidarPointCloud.from_file(f"{nusc.dataroot}/{sample_data['filename']}")  # [num_pts, x, y, z, i]

    # true relative timestamp
    sample_data_ref = nusc.get("sample_data", sd_token_ref)
    ts_ref = sample_data_ref['timestamp']  # reference timestamp
    ts = sample_data['timestamp']  # timestamp
    ts_rela = (ts - ts_ref) / 1e6

    # poses
    global_from_curr = get_global_pose(nusc, sd_token, inverse=False)  # from {lidar} to {global}
    ref_from_global = get_global_pose(nusc, sd_token_ref, inverse=True)  # from {global} to {ref lidar}
    ref_from_curr = ref_from_global.dot(global_from_curr)  # from {lidar} to {ref lidar}

    # transformed sensor origin and points, at {ref lidar} frame
    origin_tf = torch.tensor(ref_from_curr[:3, 3], dtype=torch.float32).cuda()  # curr sensor location, at {ref lidar} frame
    # intensity = torch.tensor(lidar_pcd.points[-1, :].T, dtype=torch.float32).cuda()
    lidar_pcd.transform(ref_from_curr)
    points_tf = torch.tensor(lidar_pcd.points[:3].T, dtype=torch.float32).cuda()  # curr point cloud, at {ref lidar} frame
    points_tf, filter_mask = filter_points(points_tf, max_range)
    return origin_tf, points_tf, ts_rela, filter_mask


def load_mos_labels(nusc, sd_tok, filter_mask):
    # mos labels
    mos_labels_file = os.path.join(nusc.dataroot, 'mos_labels', nusc.version, sd_tok + "_mos.label")
    mos_labels = np.fromfile(mos_labels_file, dtype=np.uint8)[filter_mask]
    return mos_labels


class KeyRays(object):
    def __init__(self, org_key, pts_key, ts_key):
        self.ray_start = org_key
        self.ray_end = pts_key
        self.ray_ts = ts_key  # TODO: could be a timestamp for each ray (time compensation)
    def get_ray_start(self):
        return self.ray_start
    def get_ray_end(self):
        return self.ray_end
    def get_ray_dir(self):
        return F.normalize(self.ray_end - self.ray_start, p=2, dim=1)  # unit vector
    def get_ray_ts(self):
        return self.ray_ts
    def get_ray_depth(self, ray_pts):
        return torch.linalg.norm(ray_pts - self.ray_start, dim=1, keepdim=False)
    def get_org_vec(self, org_query, ray_size):
        # unit vec: from query org to key org
        return torch.broadcast_to(F.normalize(self.ray_start - org_query, p=2, dim=0), (ray_size, 3))
    def find_ints(self, query_rays, args):
        # calculate norm vector of reference plane (query_rays_dir; query_org -> key_org)
        query_key_org_vec = self.get_org_vec(query_rays.get_ray_start(), query_rays.get_ray_size())  # query org to key org
        query_ray_dir = query_rays.get_ray_dir()
        # TODO: cross product time cost (25219, 3) * (25219, 3) -> 2.539e-5
        ref_plane_norm = torch.cross(query_ray_dir, query_key_org_vec)  # unit vector: (rays_size * key_rays_size, 3)
        # calculate cos value of key_rays to reference plane
        # TODO: matrix product time cost (25219, 3) * (3, 31744) -> 1.516e-3
        key_rays_to_ref_plane = torch.matmul(ref_plane_norm, self.get_ray_dir().T)  # (query_rays_size, key_rays_size)
        deg_thrd = np.rad2deg(args.max_dis_error / args.max_range)
        ray_ints_mask = torch.abs(key_rays_to_ref_plane) <= torch.cos(torch.deg2rad(torch.tensor(90 - deg_thrd)))
        ray_ints_idx = torch.where(ray_ints_mask)

        # TODO: small angle index
        small_ang_mask = torch.matmul(query_rays.get_ray_dir(), self.get_ray_dir().T) >= torch.cos(torch.deg2rad(torch.tensor(deg_thrd)))
        small_ang_mask = torch.logical_and(small_ang_mask, ray_ints_mask)
        small_ang_idx = torch.where(small_ang_mask)

        # del key_rays_to_ref_plane
        # torch.cuda.empty_cache()
        return ray_ints_idx, small_ang_idx


class QueryRays(object):
    def __init__(self, org_query, pts_query, ts_query):
        self.ray_start = org_query
        self.ray_end = pts_query
        self.ray_size = len(self.ray_end)
        self.ray_ts = ts_query  # TODO: could be a timestamp for each ray (time compensation)
        self.rays_to_ints_pts_dict = dict()
    def get_ray_start(self):
        return self.ray_start
    def get_ray_end(self):
        return self.ray_end
    def get_ray_dir(self):
        return F.normalize(self.ray_end - self.ray_start, p=2, dim=1)  # unit vector
    def get_ray_size(self):
        return self.ray_size
    def get_ray_ts(self):
        return self.ray_ts
    def get_ray_depth(self, ray_pts):
        return torch.linalg.norm(ray_pts - self.ray_start, dim=1, keepdim=False)
    def get_org_vec(self, org_key, ray_size):
        # unit vec: from key org to query org
        return torch.broadcast_to(F.normalize(self.ray_start - org_key, p=2, dim=0), (ray_size, 3))
    def cal_ints_points(self, key_rays_list:list, ray_ints_idx_list:list, ray_parallel_idx_list:list, args, for_vis:bool):
        query_ray_to_bg_samples_dict = defaultdict(list)
        query_ray_to_key_rays_dict = defaultdict(list)
        for key_sensor_idx, key_rays in enumerate(key_rays_list):  # key_rays at different space and time
            # intersection rays index
            query_rays_ints_idx = ray_ints_idx_list[key_sensor_idx][0]
            key_rays_ints_idx = ray_ints_idx_list[key_sensor_idx][1]

            # common perpendicular line
            query_rays_dir = self.get_ray_dir()[query_rays_ints_idx]  # query rays (have intersection points)
            key_rays_dir = key_rays.get_ray_dir()[key_rays_ints_idx]  # key rays (have intersection points)
            com_norm = torch.cross(query_rays_dir, key_rays_dir)  # not unit vector

            # calculate intersection points
            rays_org_vec = torch.broadcast_to(key_rays.get_ray_start() - self.ray_start, (len(query_rays_ints_idx), 3))
            q = torch.sum(torch.cross(rays_org_vec, key_rays_dir) * com_norm, dim=1) / torch.sum(com_norm * com_norm, dim=1)
            k = torch.sum(torch.cross(rays_org_vec, query_rays_dir) * com_norm, dim=1) / torch.sum(com_norm * com_norm, dim=1)
            query_bg_mask = q >= self.get_ray_depth(self.ray_end[query_rays_ints_idx])
            key_same_dir_mask = k >= args.occ_thrd
            valid_ints_mask = torch.logical_and(query_bg_mask, key_same_dir_mask)
            query_rays_ints_idx = query_rays_ints_idx[valid_ints_mask]
            key_rays_ints_idx = key_rays_ints_idx[valid_ints_mask]
            q = q[valid_ints_mask]
            ints_pts = self.ray_start + q.reshape(-1, 1) * self.get_ray_dir()[query_rays_ints_idx]

            # calculate occupancy label of the ints pts
            ints_labels = torch.zeros(len(ints_pts)).cuda()  # 0: unknown, 1: free, 2:occupied
            key_ray_depth = key_rays.get_ray_depth(key_rays.get_ray_end()[key_rays_ints_idx])
            key_ints_depth = key_rays.get_ray_depth(ints_pts)
            key_ints_depth_res = key_ints_depth - key_ray_depth
            free_mask = key_ints_depth_res < 0
            ints_labels[free_mask] = 1
            occ_mask = torch.logical_and(key_ints_depth_res >= 0, key_ints_depth_res <= args.occ_thrd)
            ints_labels[occ_mask] = 2

            # valid intersection points: used for save or vis
            ints_valid_mask = torch.logical_or(free_mask, occ_mask)
            query_rays_ints_valid_idx = query_rays_ints_idx[ints_valid_mask]
            key_rays_ints_valid_idx = key_rays_ints_idx[ints_valid_mask]
            ints_pts_valid = ints_pts[ints_valid_mask]
            ints_labels_valid = ints_labels[ints_valid_mask].reshape(-1, 1)
            num_ints_valid_rays = len(ints_pts_valid)

            # TODO: parallel rays (intersection rays contain parallel rays)
            query_rays_para_idx = ray_parallel_idx_list[key_sensor_idx][0]
            key_rays_para_idx = ray_parallel_idx_list[key_sensor_idx][1]
            proj_para_depth = torch.sum(key_rays.get_ray_end()[key_rays_para_idx] * query_rays.get_ray_dir()[query_rays_para_idx], dim=1)  # torch不能对二维tensor求点积, 对应位置元素相乘再相加
            proj_para_pts = torch.mul(proj_para_depth.reshape(-1, 1), query_rays.get_ray_dir()[query_rays_para_idx])
            proj_depth_residual = proj_para_depth - self.get_ray_depth(self.ray_end[query_rays_para_idx])

            # TODO: logic 1, if depth residual almost 0
            para_same_mask = torch.logical_and(proj_depth_residual >= -args.max_dis_error,
                                                proj_depth_residual <= args.max_dis_error)
            query_rays_para_same_idx = query_rays_para_idx[torch.where(para_same_mask)]
            key_rays_para_same_idx = key_rays_para_idx[torch.where(para_same_mask)]
            num_para_same_rays = len(query_rays_para_same_idx)
            ints_pts_para_same = proj_para_pts[para_same_mask]
            ints_labels_para_same = torch.full((num_para_same_rays, 1), 2).cuda()

            # TODO: logic 2, if depth residual far more than 0
            para_valid_mask = proj_depth_residual > args.max_dis_error
            query_rays_para_valid_idx = query_rays_para_idx[torch.where(para_valid_mask)]
            key_rays_para_valid_idx = key_rays_para_idx[torch.where(para_valid_mask)]
            num_para_valid_rays = len(query_rays_para_valid_idx)
            ints_pts_para_valid_occ = proj_para_pts[para_valid_mask]
            ints_labels_para_valid_occ = torch.full((num_para_valid_rays, 1), 2).cuda()
            ints_pts_para_valid_free = (proj_para_pts[para_valid_mask] + self.ray_end[query_rays_para_valid_idx]) / 2
            ints_labels_para_valid_free = torch.full((num_para_valid_rays, 1), 1).cuda()

            # TODO: logic 3, if depth residual far less than 0 -> unknown label, not used now

            # concat to bg points with labels
            bg_pts = torch.cat((ints_pts_valid, ints_pts_para_same, ints_pts_para_valid_occ, ints_pts_para_valid_free), dim=0).cpu()
            bg_ts = torch.full((num_ints_valid_rays + num_para_same_rays + num_para_valid_rays * 2, 1), key_rays.get_ray_ts())
            bg_labels = torch.cat((ints_labels_valid, ints_labels_para_same, ints_labels_para_valid_occ, ints_labels_para_valid_free), dim=0).cpu()
            bg_samples = torch.cat((bg_pts, bg_ts, bg_labels), dim=1)

            # statistics
            num_unk = torch.sum(ints_labels == 0)
            num_free = torch.sum(bg_labels == 1)
            num_occ = torch.sum(bg_labels == 2)

            # dictionary: query ray index -> intersection points list
            query_rays_idx = torch.cat((query_rays_ints_valid_idx, query_rays_para_same_idx, query_rays_para_valid_idx, query_rays_para_valid_idx)).cpu().numpy()
            key_rays_idx = torch.cat((key_rays_ints_valid_idx, key_rays_para_same_idx, key_rays_para_valid_idx, key_rays_para_valid_idx)).cpu().numpy()
            for query_ray_idx, key_ray_idx, bg_sample in zip(query_rays_idx, key_rays_idx, bg_samples):
                query_ray_to_bg_samples_dict[query_ray_idx].append(bg_sample)
                query_ray_to_key_rays_dict[query_ray_idx].append(torch.tensor([key_sensor_idx, key_ray_idx]))

        if for_vis:
            for query_ray_idx, key_sensor_rays_idx, bg_samples_list in zip(query_ray_to_key_rays_dict.keys(), query_ray_to_key_rays_dict.values(), query_ray_to_bg_samples_dict.values()):
                key_sensor_rays_idx = torch.stack(key_sensor_rays_idx)
                bg_samples = torch.stack(bg_samples_list)
                self.rays_to_ints_pts_dict[query_ray_idx] = (key_sensor_rays_idx, bg_samples)
            return self.rays_to_ints_pts_dict
        else:
            if len(query_ray_to_bg_samples_dict) == 0:
                return np.empty((0, 2)), np.empty((0, 3))
            ray_samples_save = []
            bg_samples_save = []
            for query_ray_idx, bg_samples_list in query_ray_to_bg_samples_dict.items():
                bg_samples = torch.stack(bg_samples_list)
                # save query ray samples: [query ray index, number of background points]
                ray_samples_save.append([query_ray_idx, len(bg_samples)])
                # save background points samples: [x, y, z, ts, occ_label]
                bg_samples_save.append(bg_samples)
                # TODO: statistics, global param, for histogram
                num_bg_samples_per_ray.append(len(bg_samples))
                num_occ_ray = torch.sum(bg_samples[:, -1] == 2)
                num_free_ray = torch.sum(bg_samples[:, -1] == 1)
                # TODO: statistics, occ percentage per ray
                num_occ_percentage_per_ray.append((num_occ_ray / (num_occ_ray + num_free_ray)) * 100)
            # TODO: statistics, num of unk, free, occ in a scan
            num_unk_free_occ_per_scan.append(torch.tensor([num_unk, num_free, num_occ]))
            ray_samples_save = np.vstack(ray_samples_save)
            bg_samples_save = torch.vstack(bg_samples_save).numpy()
            return ray_samples_save, bg_samples_save


def load_rays(nusc, sd_tok_query, args, for_vis:bool):
    sample_data_query = nusc.get("sample_data", sd_tok_query)
    sample_query = nusc.get("sample", sample_data_query['sample_token'])
    # query lidar frame [t]; key lidar frame [t-1]
    sd_tok_key_prev_list, fill_flag_prev = get_sample_tok_list(sample_query, 'prev', num_samples=args.num_key_samples / 2, every_k_samples=args.every_k_samples)
    sd_tok_key_next_list, fill_flag_next = get_sample_tok_list(sample_query, 'next', num_samples=args.num_key_samples / 2, every_k_samples=args.every_k_samples)
    sd_tok_key_list = sd_tok_key_prev_list[::-1] + sd_tok_key_next_list  # -0.5, ..., -0.1, 0.1, ..., 0.5

    # get query rays
    org_query, pts_query, ts_query, filter_mask_query = get_transformed_pcd(nusc, sd_tok_query, sd_tok_query, args.max_range)
    query_rays = QueryRays(org_query, pts_query, ts_query)
    # get key rays
    key_rays_list = []
    ray_ints_idx_list = []
    ray_parallel_idx_list = []
    mos_labels_key_list = []
    for sd_tok_key in sd_tok_key_list:
        org_key, pts_key, ts_key, filter_mask_key = get_transformed_pcd(nusc, sd_tok_query, sd_tok_key, args.max_range)
        key_rays = KeyRays(org_key, pts_key, ts_key)
        key_rays_list.append(key_rays)
        ray_ints_idx, ray_parallel_idx = key_rays.find_ints(query_rays, args)
        ray_ints_idx_list.append(ray_ints_idx)
        ray_parallel_idx_list.append(ray_parallel_idx)
        if for_vis:  # load mos labels
            mos_labels_key = load_mos_labels(nusc, sd_tok_key, filter_mask_key)
            mos_labels_key_list.append(mos_labels_key)

    if for_vis:  # load mos labels
        mos_labels_query = load_mos_labels(nusc, sd_tok_query, filter_mask_query)
        return query_rays, key_rays_list, ray_ints_idx_list, ray_parallel_idx_list, mos_labels_query, mos_labels_key_list
    else:
        return query_rays, key_rays_list, ray_ints_idx_list, ray_parallel_idx_list


if __name__ == '__main__':
    # tf32 core
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    # load nusc dataset
    nusc = NuScenes(dataroot="/home/user/Datasets/nuScenes", version="v1.0-trainval")
    # get query rays and key rays
    open3d_vis = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_range", type=float, default=70, help="max measurement range of lidar")
    parser.add_argument("--max_dis_error", type=float, default=0.05, help="distance error at max measurement range")
    parser.add_argument("--occ_thrd", type=float, default=0.10, help="consider space occupied beyond observed point")
    parser.add_argument("--num_key_samples", type=int, default=10, help="number of key samples")
    parser.add_argument("--every_k_samples", type=int, default=2, help="select a sample every k samples")
    args = parser.parse_args()

    # open3d visualization
    if open3d_vis:
        sd_tok_vis = '0b4506df6e3c4e2ba771c81711c066b1'
        query_rays, key_rays_list, ray_ints_idx_list, ray_parallel_idx_list, mos_labels_query, mos_labels_key_list = load_rays(nusc, sd_tok_vis, args, for_vis=True)
        rays_to_ints_pts_dict = query_rays.cal_ints_points(key_rays_list, ray_ints_idx_list, ray_parallel_idx_list, args, for_vis=True)
        # visualization
        draw_dynamic_rays(query_rays, key_rays_list, rays_to_ints_pts_dict, mos_labels_query, mos_labels_key_list)
        # # TODO: save dict labels
        # bg_label_dir = os.path.join(nusc.dataroot, 'bg_labels', nusc.version)
        # os.makedirs(bg_label_dir, exist_ok=True)
        # bg_label_path = os.path.join(bg_label_dir, sd_tok_query + "_bg.pickle")
        # with open(bg_label_path, 'wb') as file:
        #     pickle.dump(rays_to_ints_pts_dict, file)
        # with open(bg_label_path, 'rb') as file:
        #     load_label = pickle.load(file)
    else:
        num_valid_samples = 0
        num_bg_samples_per_ray = []  # for histgram visualization
        num_unk_free_occ_per_scan = []
        num_occ_percentage_per_ray = []
        for sample_idx, sample_query in tqdm(enumerate(nusc.sample)):
            # get rays
            sd_tok_query = sample_query['data']['LIDAR_TOP']
            query_rays, key_rays_list, ray_ints_idx_list, ray_parallel_idx_list = load_rays(nusc, sd_tok_query, args, for_vis=False)

            # calculate intersection points
            ray_samples_save, bg_samples_save = query_rays.cal_ints_points(key_rays_list, ray_ints_idx_list, ray_parallel_idx_list, args, for_vis=False)
            assert np.sum(ray_samples_save[:, -1]) == len(bg_samples_save)
            if ray_samples_save is not None and bg_samples_save is not None:
                num_valid_samples += 1
                bg_label_dir = os.path.join(nusc.dataroot, 'bg_labels', nusc.version)
                os.makedirs(bg_label_dir, exist_ok=True)
                bg_samples_path = os.path.join(bg_label_dir, sd_tok_query + "_bg_samples.npy")
                bg_samples_save.tofile(bg_samples_path)
                ray_samples_path = os.path.join(bg_label_dir, sd_tok_query + "_ray_samples.npy")
                ray_samples_save.tofile(ray_samples_path)
            else:
                print(f"Sample data tok {sd_tok_query}, index {sample_idx} do not have valid background points")
        print(f"Number of valid samples: {num_valid_samples}")

        # number statistics of backgroudn sample points
        plt.figure()
        plt.hist(np.array(num_bg_samples_per_ray), bins=50, color='skyblue', alpha=1, log=True)
        plt.title('Distribution of Background Samples')
        plt.xlabel('Number of Background Points')
        plt.ylabel('Frequency (Ray Samples)')
        plt.savefig('./background samples distribution.jpg')

        # number statistics of background sample points
        plt.figure()
        plt.hist(np.array(num_occ_percentage_per_ray), bins=100, color='lightsalmon', alpha=1, log=True)
        plt.title('Distribution of Occ Percentage')
        plt.xlabel('Occ Percentage (/Occ + Free)')
        plt.ylabel('Frequency (Ray Samples)')
        plt.savefig('./occ percentage distribution.jpg')

        # ratio of unk, free, occ
        num_unk_free_occ = torch.stack(num_unk_free_occ_per_scan).numpy()
        np.savetxt("./number of unk free occ.txt", num_unk_free_occ, fmt='%d')






    # #################### calculate ray intersection points: solution-1 ####################
    # vis_list = []
    # for vis_point_idx in range(len(ints_rays_idx[0])):
    #     P1 = points_query[ints_rays_idx[0][vis_point_idx]]
    #     D1 = F.normalize(P1 - origin_query, p=2, dim=0)
    #     P2 = points[ints_rays_idx[1][vis_point_idx]]
    #     D2 = F.normalize(P2 - origin, p=2, dim=0)
    #     N = torch.cross(D1, D2)  # common perpendicular line
    #     N_norm = N / torch.linalg.norm(N)
    #     # distance
    #     distance = torch.abs(torch.dot((P2 - P1), N_norm))  # length of the common perpendicular line
    #     # calculate foot drop
    #     t = torch.dot(torch.cross(P2 - P1, D2), N) / torch.dot(N, N)
    #     s = torch.dot(torch.cross(P2 - P1, D1), N) / torch.dot(N, N)
    #     if t > 0:
    #         vis_list.append(vis_point_idx)
    #         Pt = P1 + t * D1
    #         Ps = P2 + s * D2
    # #######################################################################################
    # #################### calculate ray intersection points: solution-2 ####################
    # P1 = points_query[intersection_ray_idx[0][vis_point_idx]]
    # D1 = F.normalize(P1 - origin_query, p=2, dim=0)
    # P2 = points[intersection_ray_idx[1][vis_point_idx]]
    # D2 = F.normalize(P2 - origin, p=2, dim=0)
    # a = torch.dot(D2, D1)
    # b = torch.dot(D1, D1)
    # c = torch.dot(D2, D2)
    # d = torch.dot(P2 - P1, D1)
    # e = torch.dot(P2 - P1, D2)
    # t_prime = (a * e - c * d) / (a * a - b * c)
    # s_prime = b / a * t_prime - d / a
    # Pt_prime = P1 + t_prime * D1
    # Ps_prime = P2 + s_prime * D2
    # #######################################################################################

    # TODO: torch.where(condition) is identical to torch.nonzero(condition, as_tuple=True)
    # TODO: matmul will cost too much cuda memory