"""
  Data loader
"""
import numpy as np
from pyquaternion import Quaternion
import torch
import time
from torch.utils.data import Dataset
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import train, val, test


class MyLidarPointCloud(LidarPointCloud):
    def get_ego_mask(self):  # mask the points of ego vehicle: x [-0.8, 0.8], y [-1.5, 2,5]
        ego_mask = np.logical_and(
            np.logical_and(-0.8 <= self.points[0], self.points[0] <= 0.8),
            np.logical_and(-1.5 <= self.points[1], self.points[1] <= 2.5),
        )
        return ego_mask


class nuScenesDataset(Dataset):
    def __init__(self, nusc, nusc_split, kwargs):
        """
        Figure out a list of sample data tokens for training.
        """
        super(nuScenesDataset, self).__init__()

        self.nusc = nusc
        self.nusc_split = nusc_split
        self.nusc_root = self.nusc.dataroot

        self.pc_range = kwargs["pc_range"]
        self.input_within_pc_range = kwargs["input_within_pc_range"]
        self.voxel_size = kwargs["voxel_size"]
        self.grid_shape = [
            int((self.pc_range[5] - self.pc_range[2]) / self.voxel_size),
            int((self.pc_range[4] - self.pc_range[1]) / self.voxel_size),
            int((self.pc_range[3] - self.pc_range[0]) / self.voxel_size),
        ]

        # number of sweeps (every 1 sweep / 0.05s)
        self.n_input = kwargs["n_input"]
        # number of samples (every 10 sweeps / 0.5s)
        self.n_output = kwargs["n_output"]
        self.ego_mask = kwargs["ego_mask"]
        self.fgbg_label = kwargs["fgbg_label"]

        scenes = self.nusc.scene
        if self.nusc_split == "train":
            split_scenes = train
        elif self.nusc_split == "val":
            split_scenes = val
        else:
            split_scenes = test

        # list all sample data
        self.valid_index = []
        self.flip_flags = []
        self.scene_tokens = []
        self.sample_tokens = []
        self.sample_data_tokens = []
        self.timestamps = []
        for scene in scenes:
            if scene["name"] not in split_scenes:
                continue
            scene_token = scene["token"]
            # location
            log = self.nusc.get("log", scene["log_token"])
            # flip x-axis if in left-hand traffic (singapore)
            flip_flag = True if kwargs["flip"] and log["location"].startswith("singapore") else False
            #
            start_index = len(self.sample_tokens)
            first_sample = self.nusc.get("sample", scene["first_sample_token"])
            sample_token = first_sample["token"]
            i = 0  # add all samples to self.sample_tokens, sample_data to self.sample_data_tokens in a specific scene
            while sample_token != "":
                self.flip_flags.append(flip_flag)
                self.scene_tokens.append(scene_token)
                self.sample_tokens.append(sample_token)
                sample = self.nusc.get("sample", sample_token)
                i += 1
                self.timestamps.append(sample["timestamp"])
                sample_data_token = sample["data"]["LIDAR_TOP"]
                self.sample_data_tokens.append(sample_data_token)
                sample_token = sample["next"]

            #
            end_index = len(self.sample_tokens)
            #
            valid_start_index = start_index + self.n_input  # (self.n_input // 10)
            valid_end_index = end_index - self.n_output
            self.valid_index += list(range(valid_start_index, valid_end_index))

        assert len(self.sample_tokens) == len(self.scene_tokens) == len(self.flip_flags) == len(self.timestamps)

        self.n_samples = len(self.valid_index)
        print(
            f"{self.nusc_split}: {self.n_samples} valid samples over {len(split_scenes)} scenes"
        )

    def __len__(self):
        return self.n_samples

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

    def load_fg_labels(self, sample_data_token):
        lidarseg = self.nusc.get("lidarseg", sample_data_token)
        lidarseg_labels = np.fromfile(
            f"{self.nusc.dataroot}/{lidarseg['filename']}", dtype=np.uint8
        )
        fg_labels = np.logical_and(1 <= lidarseg_labels, lidarseg_labels <= 23)
        return fg_labels

    def get_grid_mask(self, points):
        # input points are numpy array
        eps = 0.00001
        mask1 = np.logical_and(self.pc_range[0] <= points[:, 0], points[:, 0] < (self.pc_range[3] - eps))
        mask2 = np.logical_and(self.pc_range[1] <= points[:, 1], points[:, 1] < (self.pc_range[4] - eps))
        mask3 = np.logical_and(self.pc_range[2] <= points[:, 2], points[:, 2] < (self.pc_range[5] - eps))
        mask = mask1 & mask2 & mask3
        return mask

    def __getitem__(self, idx):
        ref_index = self.valid_index[idx]  # ref index = current index?
        ref_sample_token = self.sample_tokens[ref_index]
        ref_scene_token = self.scene_tokens[ref_index]
        ref_timestamp = self.timestamps[ref_index]
        ref_sd_token = self.sample_data_tokens[ref_index]
        flip_flag = self.flip_flags[ref_index]

        # reference coordinate frame
        ref_from_global = self.get_global_pose(ref_sd_token, inverse=True)  # trans: from global coord to ref lidar coord

        # NOTE: getting input frames
        input_points_4d_list = []
        input_origin_list = []
        for i in range(self.n_input):
            index = ref_index - i
            # if this exists a valid target
            if self.scene_tokens[index] == ref_scene_token:
                curr_sd_token = self.sample_data_tokens[index]  # sample data token
                curr_sd = self.nusc.get("sample_data", curr_sd_token)  # sample data

                # load the current lidar sweep
                curr_lidar_pc = MyLidarPointCloud.from_file(f"{self.nusc_root}/{curr_sd['filename']}")  # current point cloud, 4(x y z i) * N
                if self.ego_mask:
                    ego_mask = curr_lidar_pc.get_ego_mask()
                    curr_lidar_pc.points = curr_lidar_pc.points[:, np.logical_not(ego_mask)]  # mask the points of ego vehicle, N: 34688 -> 24733

                # transform from the current lidar coord to global and then to the reference lidar coord
                global_from_curr = self.get_global_pose(curr_sd_token, inverse=False)
                ref_from_curr = ref_from_global.dot(global_from_curr)
                curr_lidar_pc.transform(ref_from_curr)

                # NOTE: check if we are in Singapore (if so flip x)
                if flip_flag:
                    ref_from_curr[0, 3] *= -1
                    curr_lidar_pc.points[0] *= -1

                # tf means transformed?
                origin_tf = np.array(ref_from_curr[:3, 3], dtype=np.float32)  # location of the lidar sensor (at current lidar coord)
                points_tf = np.array(curr_lidar_pc.points[:3].T, dtype=np.float32)  # transformed point cloud, N * 3(x y z)

                # filter out input points outside pc range
                if self.input_within_pc_range:
                    inside_mask = self.get_grid_mask(points_tf)
                    points = points_tf[inside_mask]
                else:
                    points = points_tf
                tindex = np.full(len(points), i, dtype=np.float32).reshape(-1, 1)  # relative timestamp
                points_4d = np.hstack((points, tindex))

            # input 4d points
            input_points_4d_list.append(points_4d)
            # origin
            input_origin_list.append(origin_tf)
        input_points_4d_tensor = torch.from_numpy(np.concatenate(input_points_4d_list, axis=0))
        displacement = torch.from_numpy(input_origin_list[0] - input_origin_list[1])  # ego vehicle displacement (x y z)

        # NOTE: getting output frames
        output_origin_list = []
        output_points_list = []
        output_tindex_list = []
        if self.fgbg_label:
            output_labels_list = []
        for i in range(self.n_output):
            index = ref_index + i + 1
            # if this exists a valid target
            if self.scene_tokens[index] == ref_scene_token:  # avoid different samples in different scenes
                curr_sd_token = self.sample_data_tokens[index]

                curr_sd = self.nusc.get("sample_data", curr_sd_token)

                # load the current lidar sweep
                curr_lidar_pc = MyLidarPointCloud.from_file(f"{self.nusc_root}/{curr_sd['filename']}")
                if self.ego_mask:
                    ego_mask = curr_lidar_pc.get_ego_mask()
                    curr_lidar_pc.points = curr_lidar_pc.points[:, np.logical_not(ego_mask)]

                # transform from the current lidar coord to global and then to the reference lidar coord
                global_from_curr = self.get_global_pose(curr_sd_token, inverse=False)
                ref_from_curr = ref_from_global.dot(global_from_curr)
                curr_lidar_pc.transform(ref_from_curr)

                # NOTE: check if we are in Singapore (if so flip x)
                if flip_flag:
                    ref_from_curr[0, 3] *= -1
                    curr_lidar_pc.points[0] *= -1

                origin_tf = np.array(ref_from_curr[:3, 3], dtype=np.float32)
                points_tf = np.array(curr_lidar_pc.points[:3].T, dtype=np.float32)
                if self.fgbg_label and self.nusc_split != "test":
                    labels = self.load_fg_labels(curr_sd_token).astype(np.float32)
                    if self.ego_mask:
                        labels = labels[np.logical_not(ego_mask)]
                        assert len(labels) == len(points_tf)
            else:  # filler
                origin_tf = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                points_tf = np.full((0, 3), float("nan"), dtype=np.float32)
                if self.fgbg_label:
                    labels = np.full((len(points_tf),), -1, dtype=np.float32)
                    assert len(labels) == len(points_tf)
            # origin
            output_origin_list.append(origin_tf)

            # points
            output_points_list.append(points_tf)

            # timestamp index
            tindex = np.full(len(points_tf), i, dtype=np.float32)
            output_tindex_list.append(tindex)
            if self.fgbg_label:
                output_labels_list.append(labels)

        output_origin_tensor = torch.from_numpy(np.stack(output_origin_list))
        output_points_tensor = torch.from_numpy(np.concatenate(output_points_list))
        output_tindex_tensor = torch.from_numpy(np.concatenate(output_tindex_list))
        if self.fgbg_label:
            output_labels_tensor = torch.from_numpy(np.concatenate(output_labels_list))
            return (
                    (ref_scene_token, ref_sample_token, ref_sd_token, displacement),
                    input_points_4d_tensor,
                    output_origin_tensor,
                    output_points_tensor,
                    output_tindex_tensor,
                    output_labels_tensor
                )
        else:
            return (
                (ref_scene_token, ref_sample_token, ref_sd_token, displacement),
                input_points_4d_tensor,
                output_origin_tensor,
                output_points_tensor,
                output_tindex_tensor
            )

