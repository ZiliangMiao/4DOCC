"""
  Moving Object Segmentation Data loader
"""
import os.path

import numpy as np
from pyquaternion import Quaternion
import torch
import time
from torch.utils.data import Dataset
import nuscenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
import nuscenes.utils.splits as splits


class MyLidarPointCloud(LidarPointCloud):
    def get_ego_mask(self):  # mask the points of ego vehicle: x [-0.8, 0.8], y [-1.5, 2,5]
        ego_mask = np.logical_and(
            np.logical_and(-0.8 <= self.points[0], self.points[0] <= 0.8),
            np.logical_and(-1.5 <= self.points[1], self.points[1] <= 2.5),
        )
        return ego_mask


class nuScenesMosDataset(Dataset):
    def __init__(self, nusc, nusc_split, kwargs):
        """
        Figure out a list of sample data tokens for mos fine-tuning.
        """
        super(nuScenesMosDataset, self).__init__()
        # dataset root and split
        self.nusc = nusc
        self.nusc_split = nusc_split
        self.nusc_version = kwargs['nusc_version']
        self.nusc_root = self.nusc.dataroot
        # spatial resolution
        self.pc_range = kwargs["pc_range"]
        self.voxel_size = kwargs["voxel_size"]
        self.grid_shape = [
            int((self.pc_range[5] - self.pc_range[2]) / self.voxel_size),
            int((self.pc_range[4] - self.pc_range[1]) / self.voxel_size),
            int((self.pc_range[3] - self.pc_range[0]) / self.voxel_size),
        ]
        # number of input point cloud scans and output (prediction) scans
        # number of sweeps (every 1 sweep / 0.05s), actually only samples have been employed
        self.n_input = kwargs["n_input"]
        self.n_output = kwargs["n_output"]
        self.ego_mask = kwargs["ego_mask"]

        # nusc scenes splits
        scenes = self.nusc.scene
        if self.nusc_split == "train":
            split_scenes = splits.train
        elif self.nusc_split == "val":
            split_scenes = splits.val
        else:
            split_scenes = splits.test

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
            # location: flip x-axis if in left-hand traffic (singapore)
            scene_log = self.nusc.get("log", scene["log_token"])
            flip_flag = True if scene_log["location"].startswith("singapore") else False
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
                sample_token = sample["next"]  # note: not sample_data_next

            end_index = len(self.sample_tokens) - 1
            valid_start_index = start_index + self.n_input - 1
            valid_end_index = end_index - self.n_output
            self.valid_index += list(range(valid_start_index, valid_end_index))

        assert len(self.sample_tokens) == len(self.scene_tokens) == len(self.flip_flags) == len(self.timestamps)

        self.n_samples = len(self.valid_index)
        print(f"{self.nusc_split}: {self.n_samples} valid samples over {len(split_scenes)} scenes")

    def __len__(self):
        return self.n_samples

    def get_global_pose(self, sample_data_token, inverse=False):
        sd = self.nusc.get("sample_data", sample_data_token)
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

    def load_mos_labels(self, sample_data_token):
        mos_labels_dir = os.path.join(self.nusc_root, "mos_labels", self.nusc_version)
        mos_label_file = os.path.join(mos_labels_dir, sample_data_token + "_mos.label")
        mos_label = np.fromfile(mos_label_file, dtype=np.uint8)
        return mos_label

    def __getitem__(self, idx):
        ref_index = self.valid_index[idx]  # ref index = current index?
        ref_sample_token = self.sample_tokens[ref_index]
        ref_sample = self.nusc.get("sample", ref_sample_token)
        ref_scene_token = self.scene_tokens[ref_index]
        ref_timestamp = self.timestamps[ref_index]
        ref_sd_token = self.sample_data_tokens[ref_index]
        flip_flag = self.flip_flags[ref_index]

        # reference coordinate frame
        ref_from_global = self.get_global_pose(ref_sd_token, inverse=True)  # trans: from global coord to ref lidar coord

        # NOTE: getting input frames
        points_list = []
        tindex_list = []
        mos_labels_list = []
        for i in range(self.n_input):
            index = ref_index - i
            # if this exists a valid target
            assert self.scene_tokens[index] == ref_scene_token

            curr_sd_token = self.sample_data_tokens[index]  # sample data token
            curr_sd = self.nusc.get("sample_data", curr_sd_token)  # sample data

            # load the current lidar sweep
            curr_lidar_pc = MyLidarPointCloud.from_file(f"{self.nusc_root}/{curr_sd['filename']}")  # current point cloud, 4(x y z i) * N
            # load mos labels
            mos_labels = self.load_mos_labels(curr_sd_token)
            if self.ego_mask:
                ego_mask = curr_lidar_pc.get_ego_mask()
                curr_lidar_pc.points = curr_lidar_pc.points[:, np.logical_not(ego_mask)]  # mask the points of ego vehicle, N: 34688 -> 24733
                mos_labels = mos_labels[np.logical_not(ego_mask)]

            # transform from the current lidar coord to global and then to the reference lidar coord
            global_from_curr = self.get_global_pose(curr_sd_token, inverse=False)
            ref_from_curr = ref_from_global.dot(global_from_curr)
            curr_lidar_pc.transform(ref_from_curr)

            # NOTE: check if we are in Singapore (if so flip x-axis)
            if flip_flag:
                ref_from_curr[0, 3] *= -1
                curr_lidar_pc.points[0] *= -1

            # data
            points_tf = np.array(curr_lidar_pc.points[:3].T, dtype=np.float32)  # transformed point cloud, N * 3(x y z)
            points_list.append(points_tf)
            tindex = np.full(len(points_tf), i, dtype=np.float32)  # relative timestamp
            tindex_list.append(tindex)
            mos_labels_list.append(mos_labels)

        points_tensor = torch.from_numpy(np.concatenate(points_list))
        tindex_tensor = torch.from_numpy(np.concatenate(tindex_list))
        mos_labels_tensor = torch.from_numpy(np.concatenate(mos_labels_list))

        return ((ref_scene_token, ref_sample_token, ref_sd_token,),
                points_tensor,
                tindex_tensor,
                mos_labels_tensor,)

