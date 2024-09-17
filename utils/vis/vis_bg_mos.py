import argparse
import os
from collections import defaultdict

import yaml
import torch
import torch.nn.functional as F
from utils.vis.open3d_vis_utils import draw_box, get_vis_sd_toks
import open3d
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import points_in_box
from ray_intersection_all_cuda import get_key_sd_toks_dict, get_transformed_pcd

# color utils
from open3d_vis_utils import occ_color_func, mos_color_func, get_confusion_color


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


def draw_mov_obj_background(nusc, cfg, sd_toks_list, pred_bg_dir, pred_mos_dir, conf_color: bool):
    vis_sample_idx = 0  # nonlocal variable: sample index
    vis_obj_idx = 0  # nonlocal variable: moving object index
    vis_ray_idx = 0  # nonlocal variable: ray index of moving object points which have background samples
    view_init = True
    skip_flag = True
    cam_params = None
    mov_obj_num = 0
    ray_num = 0

    # open3d vis
    vis = open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='ray intersection points', width=3840, height=2160, left=0, top=0)

    def draw(vis):
        nonlocal vis_sample_idx, vis_obj_idx, vis_ray_idx, cam_params, view_init, mov_obj_num, ray_num
        # get key sample data tokens
        sd_tok = sd_toks_list[vis_sample_idx]
        sample_data = nusc.get('sample_data', sd_tok)
        sample_token = sample_data['sample_token']
        sample = nusc.get("sample", sample_token)
        key_sd_toks_list = get_key_sd_toks_dict(nusc, [sample_token], cfg)[sample_token]

        # get query rays and key rays
        query_org, query_pts, query_ts, filter_mask = get_transformed_pcd(nusc, cfg, sd_tok, sd_tok)
        query_dir = F.normalize(query_pts - query_org, p=2, dim=1)  # unit vector
        key_rays_org_list = []
        key_rays_ts_list = []
        key_rays_pts_list = []
        key_mos_labels_list = []
        for key_sd_tok in key_sd_toks_list:
            key_org, key_pts, key_ts, key_mask = get_transformed_pcd(nusc, cfg, sd_tok, key_sd_tok)
            key_rays_org_list.append(key_org)
            key_rays_pts_list.append(key_pts)
            key_rays_ts_list.append(key_ts)
            key_mos_labels_file = os.path.join(nusc.dataroot, 'mos_labels', nusc.version, key_sd_tok + "_mos.label")
            key_mos_labels = np.fromfile(key_mos_labels_file, dtype=np.uint8)[key_mask]
            key_mos_labels_list.append(key_mos_labels)

        # load predicted labels
        if conf_color:
            # load predicted bg labels
            pred_bg_labels_file = os.path.join(pred_bg_dir, sd_tok + "_bg_pred.label")
            pred_bg_labels = np.fromfile(pred_bg_labels_file, dtype=np.uint8)
            # load predicted mos labels
            pred_mos_labels_file = os.path.join(pred_mos_dir, sd_tok + "_mos_pred.label")
            pred_mos_labels = np.fromfile(pred_mos_labels_file, dtype=np.uint8)

        # load gt and pred mos labels
        gt_mos_labels_file = os.path.join(nusc.dataroot, 'mos_labels', nusc.version, sd_tok + "_mos.label")
        gt_mos_labels = np.fromfile(gt_mos_labels_file, dtype=np.uint8)[filter_mask]

        # load gt bg samples
        samples_dir = os.path.join(nusc.dataroot, "labels_cuda", nusc.version)
        depth = np.fromfile(os.path.join(samples_dir, sd_tok + "_depth.bin"), dtype=np.float16)
        labels = np.fromfile(os.path.join(samples_dir, sd_tok + "_labels.bin"), dtype=np.uint8)
        query_rays_idx = np.fromfile(os.path.join(samples_dir, sd_tok + "_rays_idx.bin"), dtype=np.uint16)
        query_ray_idx_unq = np.unique(query_rays_idx)
        key_rays_indices = np.fromfile(os.path.join(samples_dir, sd_tok + "_key_rays_idx.bin"), dtype=np.uint16)
        key_meta = np.fromfile(os.path.join(samples_dir, sd_tok + "_key_meta.bin"), dtype=np.uint32).reshape(-1, 2)
        key_sensors_indices = np.concatenate([np.ones(meta[1], dtype=np.uint32) * meta[0] for meta in key_meta])

        # vis nusc sample, moving object bboxes
        anns_toks = sample['anns']
        _, boxes, _ = nusc.get_sample_data(sd_tok, selected_anntokens=anns_toks, use_flat_vehicle_coordinates=False)
        obj_boxes_list = []
        obj_ray_indices_list = []
        for ann_tok, box in zip(anns_toks, boxes):
            ann = nusc.get('sample_annotation', ann_tok)
            if ann['num_lidar_pts'] == 0: continue
            obj_pts_mask = points_in_box(box, query_pts[:, :3].T)
            if np.sum(obj_pts_mask) == 0: continue
            gt_obj_labels = gt_mos_labels[obj_pts_mask]
            mov_pts_mask = gt_obj_labels == 2
            if np.sum(mov_pts_mask) == 0:
                continue
            else:
                obj_pts_mask = points_in_box(box, query_pts[:, :3].T)
                obj_pts_idx_list = np.where(obj_pts_mask)[0].tolist()
                ray_idx_list = list(set(obj_pts_idx_list) & set(query_ray_idx_unq))  # only a part of rays have bg samples
                if len(ray_idx_list) == 0: continue
                obj_boxes_list.append(box)
                obj_ray_indices_list.append(ray_idx_list)

        ################################################################################################################
        # draw static geometries: query lidar orgs, key lidar orgs
        vis.clear_geometries()
        axis_size = 2.0
        axis_query = open3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size * 2, origin=query_org)
        vis.add_geometry(axis_query)
        for key_rays_org, key_rays_ts in zip(key_rays_org_list, key_rays_ts_list):
            axis_key = open3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size + key_rays_ts * 0.1, origin=key_rays_org)
            vis.add_geometry(axis_key)

        # draw query pcd
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(query_pts)
        if conf_color:
            mos_confusion_color_indices = get_confusion_color(gt_mos_labels, pred_mos_labels)
            pcd.colors = open3d.utility.Vector3dVector(np.array(mos_color_func(mos_confusion_color_indices)).T)
        else:
            pcd.colors = open3d.utility.Vector3dVector(np.array(mos_color_func(gt_mos_labels)).T)  # static color
        pcd_down = pcd.voxel_down_sample(voxel_size=0.10)  # point cloud downsample
        vis.add_geometry(pcd_down)
        ################################################################################################################

        if len(obj_ray_indices_list) == 0:
            print(f"Sample idx: {vis_sample_idx}, have no moving objects.")
            return None
        else:
            mov_obj_num = len(obj_ray_indices_list)
            ray_num = len(obj_ray_indices_list[vis_obj_idx])
            print(f"Sample idx: {vis_sample_idx} / {len(sd_toks_list) - 1}, Moving object index: {vis_obj_idx} / {mov_obj_num - 1}, Ray index: {vis_ray_idx} / {ray_num - 1}")

        # draw moving object bboxes
        box = obj_boxes_list[vis_obj_idx]
        draw_box(vis, [box])

        # draw query ray point [vis_ray_idx]
        obj_ray_idx = obj_ray_indices_list[vis_obj_idx][vis_ray_idx]
        obj_point_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=200)
        obj_point_sphere = obj_point_sphere.translate(query_pts[obj_ray_idx], relative=False)
        if conf_color:
            obj_point_sphere.paint_uniform_color(mos_color_func(mos_confusion_color_indices[obj_ray_idx]))
        else:
            obj_point_sphere.paint_uniform_color(mos_color_func(gt_mos_labels[obj_ray_idx]))
        vis.add_geometry(obj_point_sphere)

        # draw query ray
        query_lineset = open3d.geometry.LineSet()
        vertex_query = np.vstack((query_org, query_pts[obj_ray_idx]))
        lines_query = [[0, 1]]
        color_query = np.tile((234 / 255, 51 / 255, 35 / 255), (1, 1))
        query_lineset.points = open3d.utility.Vector3dVector(vertex_query)
        query_lineset.lines = open3d.utility.Vector2iVector(lines_query)
        query_lineset.colors = open3d.utility.Vector3dVector(color_query)
        vis.add_geometry(query_lineset)

        # get ray samples
        samples_mask = query_rays_idx == obj_ray_idx
        depth = depth[samples_mask]
        sample_pts = query_org + np.broadcast_to(query_dir[obj_ray_idx], (len(depth), 3)) * depth.reshape(-1, 1)

        # voxel down sampling for vis
        sample_pcd = open3d.geometry.PointCloud()
        sample_pcd.points = open3d.utility.Vector3dVector(sample_pts)
        _, _, idx_list = sample_pcd.voxel_down_sample_and_trace(voxel_size=0.05, min_bound=query_org, max_bound=query_pts[obj_ray_idx] * 2)
        down_idx = np.stack([i[0] for i in idx_list])

        # key rays
        key_sensors_indices = key_sensors_indices[samples_mask][down_idx]
        key_rays_indices = key_rays_indices[samples_mask][down_idx]
        key_pts_list = []
        key_mos_list = []
        for i in range(len(key_rays_indices)):
            key_sensor_idx = key_sensors_indices[i]
            key_ray_idx = key_rays_indices[i]
            key_pts_list.append(key_rays_pts_list[key_sensor_idx][key_ray_idx])
            key_mos_list.append(key_mos_labels_list[key_sensor_idx][key_ray_idx])
        key_pts = np.stack(key_pts_list)
        key_mos = np.array(key_mos_list)

        sample_pts = sample_pts[down_idx]
        gt_labels = labels[samples_mask][down_idx]
        if conf_color:
            pred_bg_labels = pred_bg_labels[samples_mask][down_idx]

        # draw bg points and rays to sample points
        lineset_key = open3d.geometry.LineSet()
        vertex_key = np.vstack((np.stack(key_rays_org_list), key_pts, sample_pts))
        lines_key = []
        lineset_corr = open3d.geometry.LineSet()  # sample point to corresponding key point
        vertex_corr = np.vstack((key_pts, sample_pts))
        lines_corr = []
        for i in range(len(sample_pts)):
            # sample point
            sample_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=200)
            sample_sphere = sample_sphere.translate(sample_pts[i], relative=False)
            if conf_color:
                bg_confusion_color_indices = get_confusion_color(gt_labels, pred_bg_labels)
                sample_sphere.paint_uniform_color(occ_color_func(bg_confusion_color_indices[i]))
            else:
                sample_sphere.paint_uniform_color(occ_color_func(gt_labels[i]))
            vis.add_geometry(sample_sphere)

            # corresponding key point
            key_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=200)
            key_sphere = key_sphere.translate(key_pts[i], relative=False)
            key_sphere.paint_uniform_color(mos_color_func(key_mos[i]))
            vis.add_geometry(key_sphere)

            # key rays org to bg point
            lines_key.append([key_sensors_indices[i], i + len(key_rays_org_list)])
            # lines_key.append([key_sensors_indices[i], i + len(key_rays_org_list) + len(sample_pts)])
            lines_corr.append([i, i + len(key_pts)])

        color_key = np.tile((1, 1, 1), (len(lines_key), 1))
        lineset_key.points = open3d.utility.Vector3dVector(vertex_key)
        lineset_key.lines = open3d.utility.Vector2iVector(lines_key)
        lineset_key.colors = open3d.utility.Vector3dVector(color_key)
        vis.add_geometry(lineset_key)
        color_corr = np.tile((0.114, 0.875, 0), (len(lines_key), 1))
        lineset_corr.points = open3d.utility.Vector3dVector(vertex_corr)
        lineset_corr.lines = open3d.utility.Vector2iVector(lines_corr)
        lineset_corr.colors = open3d.utility.Vector3dVector(color_corr)
        vis.add_geometry(lineset_corr)

        # open3d view option
        vis.get_render_option().point_size = 7.0
        vis.get_render_option().background_color = (0 / 255, 0 / 255, 0 / 255)
        view_ctrl = vis.get_view_control()
        if view_init:
            view_ctrl.set_front((0.0, 0.0, 1.0))
            view_ctrl.set_lookat((0.33837080335515901, -2.223431055221385, 2.6541285514831543))
            view_ctrl.set_up((0.0, 1.0, 0.0))
            view_ctrl.set_zoom((0.19999999999999959))
            cam_params = view_ctrl.convert_to_pinhole_camera_parameters()
            view_init = False
        view_ctrl.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        while True:
            cam_params = view_ctrl.convert_to_pinhole_camera_parameters()
            # print(cam_params.extrinsic)
            vis.poll_events()
            vis.update_renderer()

    def render_next_sample(vis):
        nonlocal vis_sample_idx, vis_obj_idx, vis_ray_idx, skip_flag, mov_obj_num
        vis_obj_idx = 0
        vis_ray_idx = 0
        if skip_flag:
            skip_flag = False
            return None
        else:
            skip_flag = True
            vis_sample_idx += 1
            if vis_sample_idx >= len(sd_toks_list):
                vis_sample_idx = len(sd_toks_list) - 1
            draw(vis)

    def render_prev_sample(vis):
        nonlocal vis_sample_idx, vis_obj_idx, vis_ray_idx, skip_flag, mov_obj_num
        vis_obj_idx = 0
        vis_ray_idx = 0
        if skip_flag:
            skip_flag = False
            return None
        else:
            skip_flag = True
            vis_sample_idx -= 1
            if vis_sample_idx <= 0:
                vis_sample_idx = 0
            draw(vis)


    def render_next_obj(vis):
        nonlocal vis_obj_idx, vis_ray_idx, skip_flag, mov_obj_num
        vis_ray_idx = 0  # if vis prev or next obj, set the ray idx to 0
        if skip_flag:
            skip_flag = False
            return None
        else:
            skip_flag = True
            vis_obj_idx += 1
            if vis_obj_idx >= mov_obj_num:
                vis_obj_idx = mov_obj_num - 1
            draw(vis)

    def render_prev_obj(vis):
        nonlocal vis_obj_idx, vis_ray_idx, skip_flag
        vis_ray_idx = 0  # if vis prev or next obj, set the ray idx to 0
        if skip_flag:
            skip_flag = False
            return None
        else:
            skip_flag = True
            vis_obj_idx -= 1
            if vis_obj_idx <= 0:
                vis_obj_idx = 0
            draw(vis)

    def render_next_ray(vis):
        nonlocal vis_obj_idx, vis_ray_idx, skip_flag, ray_num
        if skip_flag:
            skip_flag = False
            return None
        else:
            skip_flag = True
            vis_ray_idx += 1
            if vis_ray_idx >= ray_num:
                vis_ray_idx = ray_num - 1
            draw(vis)

    def render_prev_ray(vis):
        nonlocal vis_ray_idx, skip_flag
        if skip_flag:
            skip_flag = False
            return None
        else:
            skip_flag = True
            vis_ray_idx -= 1
            if vis_ray_idx <= 0:
                vis_ray_idx = 0
            draw(vis)

    # render keyboard control
    vis.register_key_callback(ord('E'), render_next_sample)
    vis.register_key_callback(ord('Q'), render_prev_sample)
    vis.register_key_callback(ord('D'), render_next_obj)
    vis.register_key_callback(ord('A'), render_prev_obj)
    vis.register_key_callback(ord('C'), render_next_ray)
    vis.register_key_callback(ord('Z'), render_prev_ray)
    vis.run()


if __name__ == '__main__':
    open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)
    pred_bg_dir = '../../logs/ours/bg_pretrain/100%nuscenes/vs-0.1_t-9.5_bs-4/version_1 (with iou sudden drop, up to 99 epoch)/predictions/epoch_99'
    pred_mos_dir = '../../logs/ours/bg_pretrain(epoch-99)-mos_finetune/100%nuscenes-50%nuscenes/vs-0.1_t-9.5_bs-4/version_0/predictions/epoch_179'

    # parameters
    nusc = NuScenes(dataroot="/home/user/Datasets/nuScenes", version="v1.0-trainval")
    with open('../../configs/ours.yaml', 'r') as f:
        cfg = yaml.safe_load(f)['bg_pretrain']

    # visualization
    source = 'all'
    sd_toks_list = get_vis_sd_toks(nusc, source, split='train', bg_label_dir=os.path.join(nusc.dataroot, 'labels_cuda', nusc.version), label_suffix='_depth.bin')
    draw_mov_obj_background(nusc, cfg, sd_toks_list, pred_bg_dir, pred_mos_dir, conf_color=False)
