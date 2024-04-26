import os
from typing import List
import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, sampler

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_logs, create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

from datasets.mos4d.augmentation import (
    shift_point_cloud,
    rotate_point_cloud,
    jitter_point_cloud,
    random_flip_point_cloud,
    random_scale_point_cloud,
    rotate_perturbation_point_cloud,
)


class NuscSequentialModule(LightningDataModule):
    """A Pytorch Lightning module for Sequential Nusc data; Contains train, valid, test data"""

    def __init__(self, cfg, nusc, mode):
        super(NuscSequentialModule, self).__init__()
        self.cfg = cfg
        self.nusc = nusc
        self.mode = mode

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """Dataloader and iterators for training, validation and test data"""
        if self.mode == "train":
            train_set = NuscSequentialDataset(self.cfg, self.nusc, split="train")
            val_set = NuscSequentialDataset(self.cfg, self.nusc, split="val")
            ########## Generate dataloaders and iterables
            if self.cfg["data"]["sample_level"] == "sequence":
                self.train_loader = DataLoader(
                    dataset=train_set,
                    batch_size=self.cfg["model"]["batch_size"],
                    collate_fn=self.collate_fn,
                    num_workers=self.cfg["model"]["num_workers"],  # num of multi-processing
                    # shuffle=self.cfg["data"]["shuffle"],
                    pin_memory=True,
                    drop_last=False,  # drop the samples left from full batch
                    timeout=0,
                )
            else:
                train_data_pct = self.cfg["data"]["dataset_pct"] / 100
                self.train_loader = DataLoader(
                    dataset=train_set,
                    batch_size=self.cfg["model"]["batch_size"],
                    collate_fn=self.collate_fn,
                    num_workers=self.cfg["model"]["num_workers"],  # num of multi-processing
                    # shuffle=self.cfg["data"]["shuffle"],
                    pin_memory=True,
                    drop_last=False,  # drop the samples left from full batch
                    timeout=0,
                    sampler=sampler.WeightedRandomSampler(weights=torch.ones(len(train_set)),
                                                          num_samples=int(train_data_pct * len(train_set))),
                )
            self.train_iter = iter(self.train_loader)
            self.val_loader = DataLoader(
                dataset=val_set,
                batch_size=self.cfg["model"]["batch_size"],
                collate_fn=self.collate_fn,
                num_workers=self.cfg["model"]["num_workers"],
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                timeout=0,
            )
            self.valid_iter = iter(self.val_loader)
            print("Loaded {:d} training and {:d} validation samples.".format(len(train_set), len(val_set)))
        elif self.mode == "test":  # no test labels, use val set as test set
            test_set = NuscSequentialDataset(self.cfg, self.nusc, split="val")
            ########## Generate dataloaders and iterables
            self.test_loader = DataLoader(
                dataset=test_set,
                batch_size=self.cfg["model"]["batch_size"],
                collate_fn=self.collate_fn,
                shuffle=False,
                num_workers=self.cfg["model"]["num_workers"],
                pin_memory=True,
                drop_last=False,
                timeout=0,
                # sampler=sampler.WeightedRandomSampler(weights=torch.ones(len(test_set)), num_samples=int(0.01 * len(test_set))),
            )
            self.test_iter = iter(self.test_loader)
            print("Loaded {:d} test samples.".format(len(test_set)))
        else:
            raise ValueError("Invalid Nusc Dataset Mode.")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    @staticmethod
    def collate_fn(batch):  # define how to merge a list of samples to from a mini-batch samples
        sample_data_token = [item[0] for item in batch]
        point_cloud = [item[1] for item in batch]
        mos_label = [item[2] for item in batch]
        return [sample_data_token, point_cloud, mos_label]


class NuscSequentialDataset(Dataset):
    def __init__(self, cfg, nusc, split):
        self.cfg = cfg
        self.version = cfg["dataset"]["nuscenes"]["version"]
        self.data_dir = cfg["dataset"]["nuscenes"]["root"]

        self.nusc = nusc
        self.split = split  # "train" "val" "mini_train" "mini_val" "test"

        self.n_input = self.cfg["data"]["n_input"]  # use how many past scans, default = 10
        self.n_skip = self.cfg["data"]["n_skip"]  # number of skip between sample data
        self.n_output = self.cfg["data"]["n_output"]  # should be 1
        self.dt_pred = self.cfg["data"]["time_interval"]  # time resolution used for prediction

        if self.cfg["mode"] == "train" and self.cfg["data"]["sample_level"] == "sequence":
            split_scenes = create_splits_scenes(verbose=True)
            split_scenes = split_scenes[self.split]
            from random import sample
            train_data_pct = self.cfg["data"]["dataset_pct"] / 100
            split_scenes = sample(split_scenes, int(len(split_scenes) * train_data_pct))
            self.sample_tokens, self.sample_data_tokens = self._split_scenes_to_samples(split_scenes)
        else:
            split_logs = create_splits_logs(split, self.nusc)
            self.sample_tokens, self.sample_data_tokens = self._split_logs_to_samples(split_logs)

        # sample token: 10 past lidar tokens; ignore the samples that have less than 10 past lidar scans
        self.sample_lidar_tokens_dict, self.valid_sample_data_tokens = self._get_sample_lidar_tokens_dict(
            self.sample_tokens)

        # self.gt_poses = self._load_poses()
        self.transform = self.cfg["data"]["transform"]

    def __len__(self):
        return len(self.sample_lidar_tokens_dict)

    def __getitem__(self, idx):  # define how to load each sample
        # sample
        sample_tokens = list(self.sample_lidar_tokens_dict.keys())
        sample_token = sample_tokens[idx]
        sample = self.nusc.get("sample", sample_token)
        sample_data_token = sample['data']['LIDAR_TOP']
        sample_data = self.nusc.get('sample_data', sample_data_token)

        # reference pose (current timestamp)
        ref_pose_token = sample_data['ego_pose_token']
        ref_pose = self.nusc.get('ego_pose', ref_pose_token)
        ref_pose_mat_inv = transform_matrix(ref_pose['translation'], Quaternion(ref_pose['rotation']), inverse=True)  # from global to ref car

        # calib pose
        calib_token = sample_data['calibrated_sensor_token']
        calib = self.nusc.get('calibrated_sensor', calib_token)
        calib_mat = transform_matrix(calib['translation'], Quaternion(calib['rotation']))  # from lidar to car
        calib_mat_inv = transform_matrix(calib['translation'], Quaternion(calib['rotation']), inverse=True)  # from car to lidar

        # sample data: concat 4d point clouds
        lidar_tokens = self.sample_lidar_tokens_dict[sample_token]
        ref_timestamp = sample_data['timestamp']
        pcds_4d_list = []  # 4D Point Cloud (relative timestamp)
        time_idx = 1
        for lidar_token in lidar_tokens:
            assert lidar_token is not None
            # if lidar_token is None:
            #     lidar_data = self.nusc.get('sample_data', lidar_tokens[0])
            #     lidar_file = os.path.join(self.data_dir, lidar_data['filename'])
            #     points = LidarPointCloud.from_file(lidar_file).points.T  # [num_pts, 4]
            #     points_curr = points[:, :3]
            #     points_curr = np.zeros((points_curr.shape[0], points_curr.shape[1]))  # padded with zeros
            # else:

            # from current scan to previous scans
            lidar_data = self.nusc.get('sample_data', lidar_token)
            lidar_file = os.path.join(self.data_dir, lidar_data['filename'])
            points = LidarPointCloud.from_file(lidar_file).points.T  # [num_pts, 4]
            points_curr = points[:, :3]

            ######################################################
            # test, flip x-axis for singapore dataset when testing
            # sample = self.nusc.get('sample', lidar_data["sample_token"])
            # scene = self.nusc.get('scene', sample['scene_token'])
            # log = self.nusc.get("log", scene["log_token"])
            # if log["location"].startswith("singapore"):
            #     points_curr[:, 0] = points_curr[:, 0] * -1
            ######################################################

            if self.transform:
                # transform point cloud from curr pose to ref pose
                curr_pose_token = lidar_data['ego_pose_token']
                curr_pose = self.nusc.get('ego_pose', curr_pose_token)
                curr_pose_mat = transform_matrix(curr_pose['translation'], Quaternion(curr_pose['rotation']))  # from curr car to global

                # transformation: curr_lidar -> curr_car -> global -> ref_car -> ref_lidar
                trans_mat = calib_mat @ curr_pose_mat @ ref_pose_mat_inv @ calib_mat_inv
                points_curr_homo = np.hstack([points_curr, np.ones((points_curr.shape[0], 1))]).T
                points_ref = torch.from_numpy((trans_mat @ points_curr_homo).T[:, :3])

                # 0, -1, -2, ..., -9
                # timestamp = (lidar_data['timestamp'] - ref_timestamp) / 1000000
                time_idx -= 1
                pcd_ref_time = self.timestamp_tensor(points_ref, (time_idx + self.n_input - 1))
                pcds_4d_list.append(pcd_ref_time)
        assert len(pcds_4d_list) == self.n_input
        assert self.n_output == 1
        pcds_4d = torch.cat(pcds_4d_list, dim=0)  # 4D point cloud: [x y z ref_time]
        pcds_4d = pcds_4d.float()  # point cloud has to be float32, otherwise MikEngine will get RunTimeError: in_feat.scalar_type() == kernel.scalar_type()

        # load labels
        mos_labels_dir = os.path.join(self.data_dir, "mos_labels", self.version)
        mos_label_file = os.path.join(mos_labels_dir, sample_data_token + "_mos.label")
        mos_label = torch.tensor(np.fromfile(mos_label_file, dtype=np.uint8))

        # mask ego vehicle point
        if self.cfg["mode"] == "train" or self.cfg["mode"] == "finetune":
            if self.cfg["data"]["augmentation"]:  # will not change the mapping from point to label
                pcds_4d = self.augment_data(pcds_4d)
            if self.cfg["data"]["ego_mask"]:
                ego_mask = self.get_ego_mask(pcds_4d)
                time_mask = pcds_4d[:, -1] == (self.n_input - 1)
                pcds_4d = pcds_4d[~ego_mask]
                mos_label = mos_label[~ego_mask[time_mask]]
        return [sample_data_token, pcds_4d, mos_label]  # sample_data_token, past 4d point cloud, sample mos label

    @staticmethod
    def get_ego_mask(points):  # mask the points of ego vehicle: x [-0.8, 0.8], y [-1.5, 2.5]
        # nuscenes car: length (4.084 m), width (1.730 m), height (1.562 m)
        # https://forum.nuscenes.org/t/dimensions-of-the-ego-vehicle-used-to-gather-data/550
        ego_mask = torch.logical_and(
            torch.logical_and(-0.865 <= points[:, 0], points[:, 0] <= 0.865),
            torch.logical_and(-1.5 <= points[:, 1], points[:, 1] <= 2.5),
        )
        return ego_mask

    def augment_data(self, past_point_clouds):
        past_point_clouds = rotate_point_cloud(past_point_clouds)
        past_point_clouds = rotate_perturbation_point_cloud(past_point_clouds)
        past_point_clouds = jitter_point_cloud(past_point_clouds)
        past_point_clouds = shift_point_cloud(past_point_clouds)
        past_point_clouds = random_flip_point_cloud(past_point_clouds)
        past_point_clouds = random_scale_point_cloud(past_point_clouds)
        return past_point_clouds

    @staticmethod
    def timestamp_tensor(tensor, time):
        """Add time as additional column to tensor"""
        n_points = tensor.shape[0]
        time = time * torch.ones((n_points, 1))
        timestamped_tensor = torch.hstack([tensor, time])
        return timestamped_tensor

    def _split_logs_to_samples(self, split_logs: List[str]):
        sample_tokens = []  # store the sample tokens
        sample_data_tokens = []
        for sample in self.nusc.sample:
            sample_data_token = sample['data']['LIDAR_TOP']
            scene = self.nusc.get('scene', sample['scene_token'])
            log = self.nusc.get('log', scene['log_token'])
            logfile = log['logfile']
            if logfile in split_logs:
                sample_data_tokens.append(sample_data_token)
                sample_tokens.append(sample['token'])
        return sample_tokens, sample_data_tokens

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

    def _get_scene_tokens(self, split_logs: List[str]) -> List[str]:
        """
        Convenience function to get the samples in a particular split.
        :param split_logs: A list of the log names in this split.
        :return: The list of samples.
        """
        scene_tokens = []  # store the scene tokens
        for scene in self.nusc.scene:
            log = self.nusc.get('log', scene['log_token'])
            logfile = log['logfile']
            if logfile in split_logs:
                scene_tokens.append(scene['token'])
        return scene_tokens

    def _get_sample_lidar_tokens_dict(self, sample_tokens: List[str]):
        sample_lidar_dict = {}
        valid_sample_data_tokens = []  # samples that have 10 past scans (sample data token)
        for sample_token in sample_tokens:
            # Get records from DB.
            sample = self.nusc.get("sample", sample_token)
            sample_data_token = sample["data"]["LIDAR_TOP"]
            sample_data = self.nusc.get("sample_data", sample_data_token)

            sd_tokens = [sample_data_token]  # lidar token of current scan
            sd_timestamps = [sample_data['timestamp']]
            ##########################
            skip_flag = 0  # already skip 0 lidar sample data
            lidar_sample_idx = 1  # already sample 1 lidar sample data
            ##########################
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
                    sd_tokens.append(sample_data_prev_token)
                    sd_timestamps.append(sample_data_prev['timestamp'])
                    # change flag after append
                    skip_flag = 0
                    lidar_sample_idx += 1
                    # assign sample data prev to sample data for next loop
                    sample_data = sample_data_prev
                else:
                    # sd_tokens.append(None)  # padded with zero point clouds
                    lidar_sample_idx += 1
                    continue

            if len(sd_tokens) == self.n_input:
                sample_lidar_dict[sample_token] = sd_tokens
                valid_sample_data_tokens.append(sd_tokens[0])
        return sample_lidar_dict, valid_sample_data_tokens

    def _get_sample_pose_tokens_dict(self, sample_tokens: List[str]):
        sample_pose_dict = {}
        for sample_token in sample_tokens:
            sample = self.nusc.get("sample", sample_token)
            sample_data_token = sample["data"]["LIDAR_TOP"]
            sample_data = self.nusc.get("sample_data", sample_data_token)
            pose_token = sample_data['ego_pose_token']
            pose_tokens = []
            pose_tokens.append(pose_token)  # ego pose token of current scan

            lidar_prev_idx = 0
            while lidar_prev_idx < self.n_input - 1:
                lidar_prev_idx += 1
                if sample_data["prev"] != "":
                    sample_data_prev_token = sample_data["prev"]
                    sample_data_prev = self.nusc.get("sample_data", sample_data_prev_token)
                    pose_token_prev = sample_data_prev['ego_pose_token']
                    pose_tokens.append(pose_token_prev)
                else:
                    break
            if len(pose_tokens) == self.n_input:
                sample_pose_dict[sample_token] = pose_tokens
        return sample_pose_dict