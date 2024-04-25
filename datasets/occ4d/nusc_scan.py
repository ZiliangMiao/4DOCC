"""
  Data loader
"""
from typing import List

import numpy as np
from pyquaternion import Quaternion
import torch
import time
from torch.utils.data import Dataset
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import create_splits_logs, create_splits_scenes


class nuScenesDataset(Dataset):
    def __init__(self, nusc, nusc_split, cfg):
        """
        Figure out a list of sample data tokens for training.
        """
        super(nuScenesDataset, self).__init__()
        self.cfg = cfg
        self.nusc = nusc
        self.nusc_split = nusc_split
        self.nusc_root = self.nusc.dataroot

        self.pc_range = cfg["data"]["pc_range"]
        self.input_within_pc_range = cfg["data"]["input_within_pc_range"]
        self.voxel_size = cfg["data"]["voxel_size"]
        self.grid_shape = [
            int((self.pc_range[5] - self.pc_range[2]) / self.voxel_size),
            int((self.pc_range[4] - self.pc_range[1]) / self.voxel_size),
            int((self.pc_range[3] - self.pc_range[0]) / self.voxel_size),
        ]

        # number of sweeps (every 1 sweep / 0.05s)
        self.n_input = cfg["data"]["n_input"]
        self.n_output = cfg["data"]["n_output"]
        self.n_skip = cfg["data"]["n_skip"]
        self.ego_mask = cfg["data"]["ego_mask"]

        # get splits
        split_scenes = create_splits_scenes(verbose=True)
        split_scenes = split_scenes[self.nusc_split]

        # list all valid sample data
        self.sample_tokens, self.sample_data_tokens = self._split_scenes_to_samples(split_scenes)

        (self.sample_input_lidar_dict, self.sample_input_reftime_dict, self.sample_output_lidar_dict,
         self.sample_output_reftime_dict, self.sample_xflip_dict, self.valid_sample_tokens) = self._get_sample_lidar_tokens_dict(self.sample_tokens)

        assert len(self.sample_input_lidar_dict) == len(self.sample_xflip_dict)
        self.n_samples = len(self.sample_input_lidar_dict)
        print(
            f"{self.nusc_split}: {self.n_samples} valid samples over {len(split_scenes)} scenes"
        )

    def _split_scenes_to_samples(self, split_scenes: List[str]):
        sample_tokens = []  # store the sample tokens
        sample_data_tokens = []
        for scene in self.nusc.scene:
            if scene['name'] not in split_scenes:
                continue
            sample_token = scene["first_sample_token"]
            sample = self.nusc.get('sample', sample_token)
            sample_data_token = sample['data']['LIDAR_TOP']
            sample_tokens.append(sample_token)
            sample_data_tokens.append(sample_data_token)
            while sample['next'] != "":
                sample_token = sample['next']
                sample = self.nusc.get('sample', sample_token)
                sample_data_token = sample['data']['LIDAR_TOP']
                sample_tokens.append(sample_token)
                sample_data_tokens.append(sample_data_token)
        return sample_tokens, sample_data_tokens

    def _get_sample_lidar_tokens_dict(self, sample_tokens: List[str]):
        sample_input_lidar_dict = {}
        sample_input_reftime_dict = {}
        sample_output_lidar_dict = {}
        sample_output_reftime_dict = {}
        sample_xflip_dict = {}
        valid_sample_tokens = []  # samples that have 10 past scans (sample data token)
        for ref_sample_token in sample_tokens:
            # Get records from DB.
            ref_sample = self.nusc.get("sample", ref_sample_token)
            ref_sample_data_token = ref_sample["data"]["LIDAR_TOP"]
            ref_sample_data = self.nusc.get("sample_data", ref_sample_data_token)
            ref_timestamp = ref_sample_data['timestamp']
            # whether to flip x-axis
            scene = self.nusc.get('scene', ref_sample['scene_token'])
            log = self.nusc.get("log", scene["log_token"])
            xflip_flag = True if log["location"].startswith("singapore") else False

            ############################### INPUT ###############################
            input_sd_tokens = [ref_sample_data_token]  # lidar token of current scan
            input_sd_timestamps = [(ref_sample_data['timestamp'] - ref_timestamp) / 1000000]
            # count
            skip_flag = 0  # already skip 0 lidar sample data
            lidar_sample_idx = 1  # already sample 1 lidar sample data
            sample_data = ref_sample_data
            while lidar_sample_idx < self.n_input:
                if sample_data["prev"] != "":
                    # continue to prev sample data
                    sample_data_prev_token = sample_data["prev"]
                    sample_data_prev = self.nusc.get("sample_data", sample_data_prev_token)
                    # whether to skip
                    if skip_flag < self.n_skip:
                        skip_flag += 1
                        sample_data = sample_data_prev
                        continue
                    # if not skip, sample
                    input_sd_tokens.append(sample_data_prev_token)
                    input_sd_timestamps.append((sample_data_prev['timestamp'] - ref_timestamp) / 1000000)
                    # change flag after append
                    skip_flag = 0
                    lidar_sample_idx += 1
                    # assign sample data prev to sample data for next loop
                    sample_data = sample_data_prev
                else:
                    # input_sd_tokens.append(None)  # padded with zero point clouds
                    lidar_sample_idx += 1
                    continue

            ############################### OUTPUT ###############################
            output_sd_tokens = []  # lidar token of current scan
            output_sd_timestamps = []
            # count
            skip_flag = 0  # already skip 0 lidar sample data
            lidar_sample_idx = 0  # already sample 0 lidar sample data
            sample_data = ref_sample_data
            while lidar_sample_idx < self.n_input:
                if sample_data["next"] != "":
                    # continue to prev sample data
                    sample_data_next_token = sample_data["next"]
                    sample_data_next = self.nusc.get("sample_data", sample_data_next_token)
                    # whether to skip
                    if skip_flag < self.n_skip:
                        skip_flag += 1
                        sample_data = sample_data_next
                        continue
                    # if not skip, sample
                    output_sd_tokens.append(sample_data_next_token)
                    output_sd_timestamps.append((sample_data_next['timestamp'] - ref_timestamp) / 1000000)
                    # change flag after append
                    skip_flag = 0
                    lidar_sample_idx += 1
                    # assign sample data prev to sample data for next loop
                    sample_data = sample_data_next
                else:
                    # input_sd_tokens.append(None)  # padded with zero point clouds
                    lidar_sample_idx += 1
                    continue

            if len(input_sd_tokens) == self.n_input & len(output_sd_tokens) == self.n_output:
                valid_sample_tokens.append(ref_sample_token)
                sample_input_lidar_dict[ref_sample_token] = input_sd_tokens
                sample_input_reftime_dict[ref_sample_token] = input_sd_timestamps
                sample_output_lidar_dict[ref_sample_token] = output_sd_tokens
                sample_output_reftime_dict[ref_sample_token] = output_sd_timestamps
                sample_xflip_dict[ref_sample_token] = xflip_flag
        return (sample_input_lidar_dict, sample_input_reftime_dict, sample_output_lidar_dict,
                sample_output_reftime_dict, sample_xflip_dict, valid_sample_tokens)

    def get_global_pose(self, sd_token, inverse=False):
        sd = self.nusc.get("sample_data", sd_token)
        sd_ep = self.nusc.get("ego_pose", sd["ego_pose_token"])
        sd_cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

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

    def get_grid_mask(self, points):
        # input points are numpy array
        eps = 0.00001
        mask1 = np.logical_and(self.pc_range[0] <= points[:, 0], points[:, 0] < (self.pc_range[3] - eps))
        mask2 = np.logical_and(self.pc_range[1] <= points[:, 1], points[:, 1] < (self.pc_range[4] - eps))
        mask3 = np.logical_and(self.pc_range[2] <= points[:, 2], points[:, 2] < (self.pc_range[5] - eps))
        mask = mask1 & mask2 & mask3
        return mask

    @staticmethod
    def get_ego_mask(points):  # mask the points of ego vehicle: x [-0.8, 0.8], y [-1.5, 2.5]
        # nuscenes car: length (4.084 m), width (1.730 m), height (1.562 m)
        # https://forum.nuscenes.org/t/dimensions-of-the-ego-vehicle-used-to-gather-data/550
        ego_mask = np.logical_and(
            np.logical_and(-0.865 <= points[:, 0], points[:, 0] <= 0.865),
            np.logical_and(-1.5 <= points[:, 1], points[:, 1] <= 2.5),
        )
        return ego_mask

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        ref_sample_token = self.valid_sample_tokens[idx]
        ref_sample = self.nusc.get('sample', ref_sample_token)
        ref_sd_token = ref_sample['data']['LIDAR_TOP']
        ref_from_global = self.get_global_pose(ref_sd_token, inverse=True)  # from global coord to ref lidar coord

        flip_flag = self.sample_xflip_dict[ref_sample_token]
        ########################################## INPUT ##########################################
        input_points_list = []
        input_origin_list = []
        input_sd_tokens = self.sample_input_lidar_dict[ref_sample_token]
        input_reftime_list = self.sample_input_reftime_dict[ref_sample_token]
        assert len(input_sd_tokens) == len(input_reftime_list) == self.n_input
        for input_sd_token, input_reftime in zip(input_sd_tokens, input_reftime_list):
            sample_data = self.nusc.get("sample_data", input_sd_token)
            lidar_pcd = LidarPointCloud.from_file(f"{self.nusc_root}/{sample_data['filename']}")
            if self.ego_mask:
                ego_mask = self.get_ego_mask(lidar_pcd.points.T)
                lidar_pcd.points = lidar_pcd.points[:, np.logical_not(ego_mask)]
            # transform from the current lidar coord to global and then to the reference lidar coord
            global_from_curr = self.get_global_pose(input_sd_token, inverse=False)
            ref_from_curr = ref_from_global.dot(global_from_curr)
            lidar_pcd.transform(ref_from_curr)

            # check if in Singapore (if so flip x)
            if flip_flag:
                ref_from_curr[0, 3] *= -1
                lidar_pcd.points[0] *= -1

            # tf means transformed?
            origin_tf = np.array(ref_from_curr[:3, 3], dtype=np.float32)  # location of lidar at curr lidar coord
            points_tf = np.array(lidar_pcd.points[:3].T, dtype=np.float32)  # transformed point cloud, N * 3(x y z)

            # filter out input points outside pc range
            if self.input_within_pc_range:
                inside_mask = self.get_grid_mask(points_tf)
                points = points_tf[inside_mask]
            else:
                points = points_tf
            tindex = np.full(len(points), input_reftime, dtype=np.float32).reshape(-1, 1)  # relative timestamp
            points_4d = np.hstack((points, tindex))

            # append
            input_origin_list.append(origin_tf)
            input_points_list.append(points_4d)

        input_points_4d_tensor = torch.from_numpy(np.concatenate(input_points_list, axis=0))
        displacement = torch.from_numpy(input_origin_list[0] - input_origin_list[1])  # ego vehicle displacement (x y z)

        ########################################## OUTPUT ##########################################
        output_points_list = []
        output_origin_list = []
        output_tindex_list = []
        output_sd_tokens = self.sample_output_lidar_dict[ref_sample_token]
        output_reftime_list = self.sample_output_reftime_dict[ref_sample_token]
        assert len(output_sd_tokens) == len(output_reftime_list) == self.n_output
        for output_sd_token, output_reftime in zip(output_sd_tokens, output_reftime_list):
            sample_data = self.nusc.get("sample_data", output_sd_token)
            lidar_pcd = LidarPointCloud.from_file(f"{self.nusc_root}/{sample_data['filename']}")
            if self.ego_mask:
                ego_mask = self.get_ego_mask(lidar_pcd.points.T)
                lidar_pcd.points = lidar_pcd.points[:, np.logical_not(ego_mask)]
            # transform from the current lidar coord to global and then to the reference lidar coord
            global_from_curr = self.get_global_pose(output_sd_token, inverse=False)
            ref_from_curr = ref_from_global.dot(global_from_curr)
            lidar_pcd.transform(ref_from_curr)

            # check if in Singapore (if so flip x)
            if flip_flag:
                ref_from_curr[0, 3] *= -1
                lidar_pcd.points[0] *= -1

            # tf means transformed?
            origin_tf = np.array(ref_from_curr[:3, 3], dtype=np.float32)  # location of lidar at curr lidar coord
            points_tf = np.array(lidar_pcd.points[:3].T, dtype=np.float32)  # transformed point cloud, N * 3(x y z)

            # filter out input points outside pc range
            if self.input_within_pc_range:
                inside_mask = self.get_grid_mask(points_tf)
                points = points_tf[inside_mask]
            else:
                points = points_tf
            tindex = np.full(len(points), output_reftime, dtype=np.float32)  # relative timestamp

            # append
            output_origin_list.append(origin_tf)
            output_points_list.append(points)
            output_tindex_list.append(tindex)
        output_origin_tensor = torch.from_numpy(np.stack(output_origin_list))
        output_points_tensor = torch.from_numpy(np.concatenate(output_points_list))
        output_tindex_tensor = torch.from_numpy(np.concatenate(output_tindex_list))
        assert len(output_points_tensor) == len(output_tindex_tensor)

        return (
            (ref_sd_token, displacement),
            input_points_4d_tensor,
            output_origin_tensor,
            output_points_tensor,
            output_tindex_tensor
        )
