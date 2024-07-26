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


def draw_dynamic_rays(query_rays, key_rays_list, rays_to_ints_pts_dict):
    # create open3d vis
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    # view settings
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().line_width = 5.0
    vis.get_render_option().background_color = np.ones(3)
    view_ctrl = vis.get_view_control()
    view_ctrl.set_front((0.045542413135007273, 0.017044255410758744, 0.9988169912267878))
    view_ctrl.set_lookat((-0.40307459913255694, 0.48209966856794939, 3.4133266868446852))
    view_ctrl.set_up((0.013605495475171616, 0.99975111323203458, -0.017680556670610532))
    view_ctrl.set_zoom((0.10000000000000001))

    # draw query and key lidar origin axis
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

    # draw query points
    pts_query = query_rays.get_ray_end().cpu()
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pts_query)
    pcd.colors = open3d.utility.Vector3dVector(np.tile([131/255, 64/255, 38/255], (len(pcd.points), 1)))
    vis.add_geometry(pcd)

    # visualize query ray one by one with poll_events():
    query_ray_pcd = open3d.geometry.PointCloud()  # only current query ray point
    key_rays_pcd = open3d.geometry.PointCloud()  # ray points of key rays that intersect with current query ray
    ints_pcd = open3d.geometry.PointCloud()  # all intersection points on current query ray
    lineset_query = open3d.geometry.LineSet()  # TODO: only current query ray
    lineset_key = open3d.geometry.LineSet()  # TODO: all key rays that intersect with current query ray (at all key lidars)
    vis.add_geometry(query_ray_pcd)
    vis.add_geometry(key_rays_pcd)
    vis.add_geometry(ints_pcd)
    vis.add_geometry(lineset_query)
    vis.add_geometry(lineset_key)
    for query_ray_idx, (key_rays_idx, ints_pts_with_label) in rays_to_ints_pts_dict.items():
        # query ray point: (1, 3)
        query_ray_pcd.points = open3d.utility.Vector3dVector(query_rays.get_ray_end()[query_ray_idx].cpu().reshape(-1, 3))
        query_ray_pcd.colors = open3d.utility.Vector3dVector(np.tile([78/255, 171/255, 144/255], (1, 1)))
        vis.update_geometry(query_ray_pcd)

        # key rays pcd
        key_ints_rays_pts_list = []
        for i in range(len(key_rays_idx)):
            key_sensor_idx = key_rays_idx[i][0]
            key_ray_idx = key_rays_idx[i][1]
            key_ints_rays_pts_list.append(key_rays_points_list[key_sensor_idx][key_ray_idx])
        key_rays_pcd.points = open3d.utility.Vector3dVector(torch.cat(key_rays_points_list))
        key_rays_pcd.colors = open3d.utility.Vector3dVector(np.tile([238/255, 191/255, 109/255], (len(key_rays_idx), 1)))
        vis.update_geometry(key_rays_pcd)

        # intersection pcd
        ints_pcd.points = open3d.utility.Vector3dVector(ints_pts_with_label[:, 0:3].reshape(-1, 3))
        ints_pcd.colors = open3d.utility.Vector3dVector(np.tile([217/255, 79/255, 51/255], (len(ints_pts_with_label), 1)))
        vis.update_geometry(ints_pcd)

        # query ray
        vertex_query = torch.vstack((org_query, pts_query[query_ray_idx]))
        lines_query = [[0, 1]]
        lineset_query.points = open3d.utility.Vector3dVector(vertex_query)
        lineset_query.lines = open3d.utility.Vector2iVector(lines_query)
        lineset_query.colors = open3d.utility.Vector3dVector(np.tile([78/255, 171/255, 144/255], (1, 1)))
        vis.update_geometry(lineset_query)

        # key rays
        vertex_key = torch.vstack((torch.stack(key_orgs_list, dim=0), torch.stack(key_ints_rays_pts_list, dim=0)))  # (len(key_orgs) + len(key_rays_pts), 3)
        lines_key = []
        for i in range(len(key_rays_idx)):
            key_sensor_idx = key_rays_idx[i][0]
            lines_key.append([key_sensor_idx, i + len(key_orgs_list)])
        lineset_key.points = open3d.utility.Vector3dVector(vertex_key)
        lineset_key.lines = open3d.utility.Vector2iVector(lines_key)
        lineset_key.colors = open3d.utility.Vector3dVector(np.tile([217/255, 79/255, 551/255], (len(lines_key), 1)))
        vis.update_geometry(lineset_key)

        # poll events
        vis.poll_events()
        vis.update_renderer()
    # destroy
    vis.destroy_window()

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
    return points[mask]


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
    intensity = torch.tensor(lidar_pcd.points[-1, :].T, dtype=torch.float32).cuda()
    lidar_pcd.transform(ref_from_curr)
    points_tf = torch.tensor(lidar_pcd.points[:3].T, dtype=torch.float32).cuda()  # curr point cloud, at {ref lidar} frame
    points_tf = filter_points(points_tf, max_range)
    return origin_tf, points_tf, ts_rela, intensity


class KeyRays(object):
    def __init__(self, org_key, pts_key, ts_key):
        self.ray_start = org_key
        self.ray_end = pts_key
        self.ray_dir = F.normalize(self.ray_end - self.ray_start, p=2, dim=1)  # unit vector
        self.ray_ts = ts_key  # TODO: could be a timestamp for each ray (time compensation)
        self.ray_ints_flag = None

    def get_query_org_vec(self, org_query, query_ray_size):
        return torch.broadcast_to(F.normalize(self.ray_start - org_query, p=2, dim=0), (query_ray_size, 3))  # unit vector

    def get_ray_start(self):
        return self.ray_start
    def get_ray_end(self):
        return self.ray_end
    def get_ray_dir(self):
        return self.ray_dir
    def get_ray_ts(self):
        return self.ray_ts
    def get_ray_ints_flag(self):
        return self.ray_ints_flag
    def get_ray_depth(self, ray_pts):
        return torch.linalg.norm(ray_pts - self.ray_start, dim=1, keepdim=False)
    def find_ints(self, query_rays, deg_shrd:float):
        # calculate norm vector of reference plane (query_rays_dir; query_org -> key_org)
        query_org_vec = self.get_query_org_vec(query_rays.get_ray_start(), query_rays.get_ray_size())
        query_ray_dir = query_rays.get_ray_dir()
        # TODO: cross product time cost (25219, 3) * (25219, 3) -> 2.539e-5
        ref_plane_norm = torch.cross(query_ray_dir, query_org_vec)  # unit vector: (rays_size * key_rays_size, 3)
        # calculate cos value of key_rays to reference plane
        # TODO: matrix product time cost (25219, 3) * (3, 31744) -> 1.516e-3
        key_rays_to_ref_plane = torch.matmul(ref_plane_norm, self.ray_dir.T)  # (query_rays_size, key_rays_size)
        self.ray_ints_flag = torch.abs(key_rays_to_ref_plane) <= torch.cos(torch.deg2rad(torch.tensor(deg_shrd)))


class QueryRays(object):
    def __init__(self, org_query, pts_query, ts_query):
        self.scene_bbox = [-70.0, -70.0, -4.5, 70.0, 70.0, 4.5]
        self.ray_start = org_query
        self.ray_end = pts_query
        self.ray_dir = F.normalize(self.ray_end - self.ray_start, p=2, dim=1)
        self.ray_size = len(self.ray_end)
        self.ray_ts = ts_query  # TODO: could be a timestamp for each ray (time compensation)
        self.rays_to_ints_pts_dict = dict()
    def get_ray_start(self):
        return self.ray_start
    def get_ray_end(self):
        return self.ray_end
    def get_ray_dir(self):
        return self.ray_dir
    def get_ray_size(self):
        return self.ray_size
    def get_ray_ts(self):
        return self.ray_ts
    def cal_ints_points(self, key_rays_list:list, occ_thrd:float):
        query_ray_to_ints_pts_dict = defaultdict(list)
        query_ray_to_key_rays_dict = defaultdict(list)
        for key_sensor_idx, key_rays in enumerate(key_rays_list):  # key_rays at different space and time
            # intersection rays index
            rays_ints_idx = torch.where(key_rays.get_ray_ints_flag())  # dim-0: key rays; dim-1: query rays

            # common perpendicular line
            query_rays_ints = self.ray_dir[rays_ints_idx[0]]  # query rays (have intersection points)
            key_rays_ints = key_rays.get_ray_dir()[rays_ints_idx[1]]  # key rays (have intersection points)
            com_norm = torch.cross(query_rays_ints, key_rays_ints)
            # vector between query and key ray end points
            ray_end_pts_vec = key_rays.get_ray_end()[rays_ints_idx[1]] - self.ray_end[rays_ints_idx[0]]

            # calculate intersection points on query ray, only background intersections are valid:
            q = torch.sum(torch.cross(ray_end_pts_vec, key_rays_ints) * com_norm, dim=1) / torch.sum(com_norm * com_norm, dim=1)
            bg_ints_idx = torch.where(q >= -occ_thrd)
            query_ints_idx = rays_ints_idx[0][bg_ints_idx].detach().cpu().numpy()
            key_ints_idx = rays_ints_idx[1][bg_ints_idx].detach().cpu().numpy()
            q = q[bg_ints_idx]
            ints_pts = self.ray_end[query_ints_idx] + q.reshape(-1, 1) * self.ray_dir[query_ints_idx]
            num_ints = len(ints_pts)

            # calculate occupancy label of the ints pts
            ray_depth = key_rays.get_ray_depth(key_rays.get_ray_end()[key_ints_idx])
            ints_depth = key_rays.get_ray_depth(ints_pts)
            free_idx = torch.where(ints_depth - ray_depth < 0)
            occ_idx = torch.where(torch.logical_and(ints_depth - ray_depth >= 0, ints_depth - ray_depth <= occ_thrd))
            ints_label = torch.zeros(num_ints).cuda()  # 0: unknown, 1: free, 2:occupied
            ints_label[free_idx] = 1
            ints_label[occ_idx] = 2

            # TODO [code patch]: if ray end points of query ray and key ray are too close, let them intersect in advance.
            ray_end_pts_dis = torch.linalg.norm(ray_end_pts_vec[bg_ints_idx], dim=1, keepdim=False)
            ints_adv_idx = torch.where(torch.logical_and(ints_label == 0, ray_end_pts_dis <= occ_thrd))
            q[ints_adv_idx] = ray_end_pts_dis[ints_adv_idx]  # param of line is depth to known point (when dir is unit)
            ints_pts[ints_adv_idx] = self.ray_end[query_ints_idx][ints_adv_idx] + q.reshape(-1, 1)[ints_adv_idx] * \
                                     self.ray_dir[query_ints_idx][ints_adv_idx]  # update intersection points that ints in advance
            ints_label[ints_adv_idx] = 2  # ints pts that ints in advance should be occupied

            # statistics
            num_unk = torch.sum(ints_label == 0)
            num_free = torch.sum(ints_label == 1)
            num_occ = torch.sum(ints_label == 2)

            # TODO: filter the ints points with label unknown, [x, y, z, ts, occ_label]
            valid_idx = torch.cat(occ_idx + free_idx).cpu()
            ints_pts_ts_label = torch.cat((ints_pts[valid_idx].cpu(),
                                           torch.full((len(valid_idx), 1), key_rays.get_ray_ts()),
                                           ints_label[valid_idx].cpu().reshape(-1, 1)), dim=1)
            # maintain a dictionary, query ray index -> intersection points list
            for query_ray_idx, key_ray_idx, ints_pt in zip(query_ints_idx[valid_idx], key_ints_idx[valid_idx], ints_pts_ts_label):
                query_ray_to_ints_pts_dict[query_ray_idx].append(ints_pt)
                query_ray_to_key_rays_dict[query_ray_idx].append(torch.tensor([key_sensor_idx, key_ray_idx]))

        # concat
        num_valid_ints_pts = []  # for histgram visualization
        for query_ray_idx, key_rays_idx, ints_pts in zip(query_ray_to_key_rays_dict.keys(), query_ray_to_key_rays_dict.values(), query_ray_to_ints_pts_dict.values()):
            key_rays_idx = torch.stack(key_rays_idx)
            ints_pts = torch.stack(ints_pts)
            self.rays_to_ints_pts_dict[query_ray_idx] = (key_rays_idx, ints_pts)
            num_valid_ints_pts.append(len(ints_pts))

        # # number statistics of valid intersection points
        # plt.hist(np.array(num_valid_ints_pts), bins=30, color='skyblue', alpha=1, log=True)
        # plt.title('Distribution of Intersection Points')
        # plt.xlabel('Number of Intersection Points')
        # plt.ylabel('Frequency')
        # plt.show()
        return self.rays_to_ints_pts_dict

    # TODO: torch.where(condition) is identical to torch.nonzero(condition, as_tuple=True)
    # TODO: matmul will cost too much cuda memory
    # def find_ints(self, key_rays_list):
    #     query_org_vec_list = []
    #     key_rays_dir_list = []
    #     for key_rays in key_rays_list:
    #         # query_org_vec
    #         query_org_vec = key_rays.get_query_org_vec(self.ray_start, self.ray_size)
    #         query_org_vec_list.append(query_org_vec)
    #         # key_rays_dir
    #         key_rays_dir = key_rays.get_ray_dir()
    #         key_rays_dir_list.append(key_rays_dir)
    #     # calculate norm vector of reference plane (query_rays_dir; query_org -> key_org)
    #     query_org_vec = torch.cat(query_org_vec_list, dim=0)
    #     query_ray_dir = self.ray_dir.repeat(len(key_rays_list), 1)
    #     ref_plane_norm = torch.cross(query_ray_dir, query_org_vec)  # unit vector: (rays_size * key_rays_size, 3)
    #     # calculate cos value of key_rays to reference plane
    #     key_rays_dir = torch.cat(key_rays_dir_list, dim=0).T
    #     key_rays_to_ref_plane = torch.matmul(ref_plane_norm, key_rays_dir)
    #     ints_rays_idx = torch.where(torch.abs(key_rays_to_ref_plane) <= np.cos(np.deg2rad(89.99999)))
    #     return ints_rays_idx

def get_sample_tok_list(sample, direction, num_sample, every_k_sample):
    sd_tok_key_list = []
    tok_cnt = 0
    skip_cnt = 0
    while sample[direction] != '' and tok_cnt < num_sample:
        sample_tok = sample[direction]
        sample = nusc.get("sample", sample_tok)
        if skip_cnt < every_k_sample - 1:
            skip_cnt += 1
            continue
        else:
            sd_tok_key_list.append(sample['data']['LIDAR_TOP'])
            tok_cnt += 1
            skip_cnt = 0
    return sd_tok_key_list, tok_cnt == num_sample


if __name__ == '__main__':
    # load nusc dataset
    nusc = NuScenes(dataroot="/home/user/Datasets/nuScenes", version="v1.0-trainval")

    # query lidar frame [t]; key lidar frame [t-1]
    sd_tok_query = "c3f016190e61437699b809b4b6241805"
    sd_query = nusc.get("sample_data", sd_tok_query)
    sample_query = nusc.get("sample", sd_query['sample_token'])
    sd_tok_key_prev_list, fill_flag_prev = get_sample_tok_list(sample_query, 'prev', num_sample=1, every_k_sample=3)
    sd_tok_key_next_list, fill_flag_next = get_sample_tok_list(sample_query, 'next', num_sample=1, every_k_sample=3)
    sd_tok_key_list = sd_tok_key_prev_list[::-1] + sd_tok_key_next_list  # -0.5, ..., -0.1, 0.1, ..., 0.5

    # get query rays and key rays
    max_range = 50
    max_dis_error = 0.01  # 1cm at max measurement range
    max_ang_error = np.rad2deg(max_dis_error / max_range)  # max_dis_error = max_range * max_ang_error
    occ_thrd = 0.10  # both for occ label threshold and near intersection threshold

    org_query, pts_query, ts_query, _ = get_transformed_pcd(nusc, sd_tok_query, sd_tok_query, max_range=max_range)
    query_rays = QueryRays(org_query, pts_query, ts_query)
    key_rays_list = []
    for sd_tok_key in sd_tok_key_list:
        org_key, pts_key, ts_key, _ = get_transformed_pcd(nusc, sd_tok_query, sd_tok_key, max_range=max_range)
        key_rays = KeyRays(org_key, pts_key, ts_key)
        key_rays.find_ints(query_rays, 90 - max_ang_error)
        key_rays_list.append(key_rays)

    # find and calculate intersection points for each query rays in key rays
    rays_to_ints_pts_dict = query_rays.cal_ints_points(key_rays_list, occ_thrd)

    # open3d visualization
    draw_dynamic_rays(query_rays, key_rays_list, rays_to_ints_pts_dict)

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
