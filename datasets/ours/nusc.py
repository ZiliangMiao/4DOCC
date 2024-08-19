import os
from typing import List
import torch
import numpy as np
from collections import defaultdict
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from nuscenes.utils.splits import create_splits_logs, create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from random import sample as random_sample
from nuscenes.utils.geometry_utils import transform_matrix

from utils.augmentation import (
    shift_point_cloud,
    rotate_point_cloud,
    jitter_point_cloud,
    random_flip_point_cloud,
    random_scale_point_cloud,
    rotate_perturbation_point_cloud,
)


class NuscSequentialModule(LightningDataModule):
    """A Pytorch Lightning module for Sequential Nusc data; Contains train, valid, test data"""

    def __init__(self, nusc, cfg_model, cfg_dataset, train_flag: bool):
        super(NuscSequentialModule, self).__init__()
        self.nusc = nusc
        self.cfg_model = cfg_model
        self.cfg_dataset = cfg_dataset
        self.train_flag = train_flag
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_iter = None
        self.val_iter = None
        self.test_iter = None

    def prepare_data(self):
        pass

    @staticmethod
    def collate_fn(batch):  # define how to merge a list of samples to from a mini-batch samples
        meta_info = [item[0] for item in batch]  # a list of tuple (sd_tok, num_rays, num_bg_samples)
        pcds_4d = [item[1] for item in batch]
        ray_to_bg_samples_dict = [item[2] for item in batch]
        return [meta_info, pcds_4d, ray_to_bg_samples_dict]

    def setup(self, stage=None):
        """Dataloader and iterators for training, validation and test data"""
        train_set = NuscSequentialDataset(self.nusc, self.cfg_model, self.cfg_dataset, split="train")
        val_set = NuscSequentialDataset(self.nusc, self.cfg_model, self.cfg_dataset, split="val")
        # train_data_pct = self.cfg_model["downsample_pct"] / 100
        train_loader = DataLoader(
                    dataset=train_set,
                    batch_size=self.cfg_model["batch_size"],
                    collate_fn=self.collate_fn,
                    num_workers=self.cfg_model["num_workers"],  # num of multi-processing
                    shuffle=self.cfg_model["shuffle"],
                    pin_memory=True,
                    drop_last=False,  # drop the samples left from full batch
                    timeout=0,
                    # sampler=sampler.WeightedRandomSampler(weights=torch.ones(len(train_set)),
                    #                                       num_samples=int(train_data_pct * len(train_set))),
                )
        val_loader = DataLoader(
                dataset=val_set,
                batch_size=self.cfg_model["batch_size"],
                collate_fn=self.collate_fn,
                num_workers=self.cfg_model["num_workers"],
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                timeout=0,
            )

        if self.train_flag:
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.train_iter = iter(self.train_loader)
            self.val_iter = iter(self.val_loader)
            print("Loaded {:d} training and {:d} validation samples.".format(len(train_set), len(val_set)))
        else:  # test (use validation set)
            self.test_loader = val_loader
            self.test_iter = iter(self.test_loader)
            print("Loaded {:d} test samples.".format(len(val_set)))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


class NuscSequentialDataset(Dataset):
    def __init__(self, nusc, cfg_model, cfg_dataset, split):
        self.nusc = nusc
        self.cfg_model = cfg_model
        self.cfg_dataset = cfg_dataset
        self.split = split  # "train" "val" "mini_train" "mini_val" "test"

        # dataset down-sampling: sequence level or sample level
        if self.cfg_model["downsample_level"] is None:  # for test set and validation set
            split_logs = create_splits_logs(split, self.nusc)
            sample_toks = self._split_logs_to_samples(split_logs)
        elif self.cfg_model["downsample_level"] == "sequence":
            split_scenes = create_splits_scenes(verbose=True)
            split_scenes = split_scenes[self.split]
            train_data_pct = self.cfg_model["downsample_pct"] / 100
            ds_split_scenes = random_sample(split_scenes, int(len(split_scenes) * train_data_pct))
            sample_toks = self._split_scenes_to_samples(ds_split_scenes)
        elif self.cfg_model["downsample_level"] == "sample":
            split_logs = create_splits_logs(split, self.nusc)
            sample_toks = self._split_logs_to_samples(split_logs)
            train_data_pct = self.cfg_model["downsample_pct"] / 100
            sample_toks = random_sample(sample_toks, int(len(sample_toks) * train_data_pct))
        else:
            raise ValueError("Invalid dataset down-sampling strategy!")

        # sample tokens: drop the samples without full sequence length
        self.sample_to_sd_toks_dict = self._get_sd_toks_dict(sample_toks)

    def __len__(self):
        return len(self.sample_to_sd_toks_dict)

    def __getitem__(self, batch_idx):
        # sample
        sample_toks_list = list(self.sample_to_sd_toks_dict.keys())
        sample_tok = sample_toks_list[batch_idx]
        sample = self.nusc.get('sample', sample_tok)
        ref_sd_tok = sample['data']['LIDAR_TOP']
        sample_data = self.nusc.get('sample_data', ref_sd_tok)

        # reference pose (current timestamp)
        ref_pose_token = sample_data['ego_pose_token']
        ref_pose = self.nusc.get('ego_pose', ref_pose_token)
        trans_global_to_ref_car = transform_matrix(ref_pose['translation'], Quaternion(ref_pose['rotation']), inverse=True)  # from global to ref car

        # calib pose
        calib_token = sample_data['calibrated_sensor_token']
        calib = self.nusc.get('calibrated_sensor', calib_token)
        trans_lidar_to_car = transform_matrix(calib['translation'], Quaternion(calib['rotation']))  # from lidar to car
        trans_car_to_lidar = transform_matrix(calib['translation'], Quaternion(calib['rotation']), inverse=True)  # from car to lidar

        # sample data: concat 4d point clouds
        input_sd_toks = self.sample_to_sd_toks_dict[sample_tok]  # sequence: -1, -2 ...
        assert len(input_sd_toks) == self.cfg_model["n_input"], "Invalid input sequence length"
        pcds_4d_list = []  # 4D Point Cloud (relative timestamp)
        time_idx = 1
        for sd_tok in input_sd_toks:
            assert sd_tok is not None
            # from current scan to previous scans
            input_sd = self.nusc.get('sample_data', sd_tok)
            pcd_file = os.path.join(self.cfg_dataset["nuscenes"]["root"], input_sd['filename'])
            pcd = LidarPointCloud.from_file(pcd_file).points.T[:, :3]  # [x, y, z, intensity]

            if self.cfg_model["transform"]:
                # transform point cloud from curr pose to ref pose
                pose_tok = input_sd['ego_pose_token']
                pose = self.nusc.get('ego_pose', pose_tok)
                trans_car_to_global = transform_matrix(pose['translation'], Quaternion(pose['rotation']))  # from car to global

                # transformation: lidar -> car -> global -> ref_car -> ref_lidar
                trans_to_ref = trans_lidar_to_car @ trans_car_to_global @ trans_global_to_ref_car @ trans_car_to_lidar
                pcd_homo = np.hstack([pcd, np.ones((pcd.shape[0], 1))]).T
                pcd_ref = torch.from_numpy((trans_to_ref @ pcd_homo).T[:, :3])

                # add timestamp (0, -1, -2, ...)
                time_idx -= 1
                pcd_4d_ref = self.add_timestamp(pcd_ref, time_idx)
                pcds_4d_list.append(pcd_4d_ref)
        pcds_4d = torch.cat(pcds_4d_list, dim=0).float()  # 4D point cloud: [x y z ref_ts]

        # filter ego vehicle points and out of scene bbox points (make the ray index same to ray_intersection_cuda.py)
        if self.cfg_model['ego_mask']:
            pcds_4d, _ = self.filter_points(pcds_4d, self.cfg_model["scene_bbox"])

        # data augmentation: will not change the order of points
        if self.cfg_model["augmentation"]:
            pcds_4d = self.augment_data(pcds_4d)

        # load labels
        bg_samples_dir = os.path.join(self.cfg_dataset["nuscenes"]["root"], "bg_labels", self.cfg_dataset["nuscenes"]["version"])
        bg_samples_file = os.path.join(bg_samples_dir, ref_sd_tok + "_bg_samples.npy")
        ray_samples_file = os.path.join(bg_samples_dir, ref_sd_tok + "_ray_samples.npy")
        ray_samples = np.fromfile(ray_samples_file, dtype=np.int64).reshape(-1, 2)
        bg_samples = np.fromfile(bg_samples_file, dtype=np.float32).reshape(-1, 5)

        # merge into dict for training
        ray_to_bg_samples_dict = defaultdict()
        num_bg_samples_per_ray_list = []
        for ray_sample in list(ray_samples):
            ray_idx = ray_sample[0]
            num_bg_samples = ray_sample[1]
            num_bg_samples_per_ray_list.append(num_bg_samples)
            # TODO: balanced sampling of occ and free bg points
            ray_to_bg_samples_dict[ray_idx] = bg_samples[0:num_bg_samples, :]
            bg_samples = bg_samples[num_bg_samples:, :]

        # number of samples
        num_rays_all = len(ray_samples)
        num_bg_samples_all = np.sum(ray_samples[:, -1])
        return [(ref_sd_tok, num_rays_all, num_bg_samples_all, num_bg_samples_per_ray_list), pcds_4d, ray_to_bg_samples_dict]

    @staticmethod
    def filter_points(pcd, scene_bbox):
        # nuscenes car: length (4.084 m), width (1.730 m), height (1.562 m)
        ego_mask = torch.logical_and(
            torch.logical_and(-0.865 <= pcd[:, 0], pcd[:, 0] <= 0.865),
            torch.logical_and(-1.5 <= pcd[:, 1], pcd[:, 1] <= 2.5),
        )
        inside_scene_bbox_mask = torch.logical_and(
            torch.logical_and(scene_bbox[0] <= pcd[:, 0], pcd[:, 0] <= scene_bbox[3]),
            torch.logical_and(scene_bbox[1] <= pcd[:, 1], pcd[:, 1] <= scene_bbox[4]),
            # torch.logical_and(scene_bbox[2] <= points[:, 2], points[:, 2] <= scene_bbox[5])
        )
        mask = torch.logical_and(~ego_mask, inside_scene_bbox_mask)
        return pcd[mask], mask.cpu().numpy()

    @staticmethod
    def augment_data(input_pcd):
        input_pcd = rotate_point_cloud(input_pcd)
        input_pcd = rotate_perturbation_point_cloud(input_pcd)
        input_pcd = jitter_point_cloud(input_pcd)
        input_pcd = shift_point_cloud(input_pcd)
        input_pcd = random_flip_point_cloud(input_pcd)
        input_pcd = random_scale_point_cloud(input_pcd)
        return input_pcd

    @staticmethod
    def add_timestamp(tensor, time):
        """Add time as additional column to tensor"""
        n_points = tensor.shape[0]
        time = time * torch.ones((n_points, 1))
        timestamped_tensor = torch.hstack([tensor, time])
        return timestamped_tensor

    def _split_logs_to_samples(self, split_logs: List[str]):
        sample_tokens = []  # store the sample tokens
        # sample_data_tokens = []
        for sample in self.nusc.sample:
            sample_data_token = sample['data']['LIDAR_TOP']
            scene = self.nusc.get('scene', sample['scene_token'])
            log = self.nusc.get('log', scene['log_token'])
            logfile = log['logfile']
            if logfile in split_logs:
                # sample_data_tokens.append(sample_data_token)
                sample_tokens.append(sample['token'])
        return sample_tokens

    def _split_scenes_to_samples(self, split_scenes: List[str]):
        sample_tokens = []  # store the sample tokens
        # sample_data_tokens = []
        for scene in self.nusc.scene:
            if scene['name'] not in split_scenes:
                continue
            sample_token = scene["first_sample_token"]
            sample = self.nusc.get('sample', sample_token)
            # sample_data_token = sample['data']['LIDAR_TOP']
            sample_tokens.append(sample_token)
            # sample_data_tokens.append(sample_data_token)
            while sample['next'] != "":
                sample_token = sample['next']
                sample = self.nusc.get('sample', sample_token)
                # sample_data_token = sample['data']['LIDAR_TOP']
                sample_tokens.append(sample_token)
                # sample_data_tokens.append(sample_data_token)
        return sample_tokens

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

    def _get_sd_toks_dict(self, sample_toks: List[str]):
        sample_to_input_sd_toks_dict = {}
        for sample_tok in sample_toks:
            # Get records from DB.
            sample = self.nusc.get("sample", sample_tok)
            input_sd_toks_list = [sample["data"]["LIDAR_TOP"]]  # lidar token of current scan
            # skip or select sample data
            skip_cnt = 0  # already skip 0 samples
            num_input_samples = 1  # already sample 1 lidar sample data
            while num_input_samples < self.cfg_model["n_input"]:
                if sample["prev"] != "":
                    sample_prev = self.nusc.get("sample", sample["prev"])
                    if skip_cnt < self.cfg_model["n_skip"]:
                        skip_cnt += 1
                        sample = sample_prev
                        continue
                    # add input sample data token
                    input_sd_toks_list.append(sample_prev["data"]["LIDAR_TOP"])
                    skip_cnt = 0
                    num_input_samples += 1
                    # assign sample data prev to sample data for next loop
                    sample = sample_prev
                else:
                    break
            assert len(input_sd_toks_list) == num_input_samples
            if num_input_samples == self.cfg_model["n_input"]:  # valid sample tokens (full sequence length)
                sample_to_input_sd_toks_dict[sample_tok] = input_sd_toks_list
        return sample_to_input_sd_toks_dict

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
            while lidar_prev_idx < self.cfg_model["n_input"] - 1:
                lidar_prev_idx += 1
                if sample_data["prev"] != "":
                    sample_data_prev_token = sample_data["prev"]
                    sample_data_prev = self.nusc.get("sample_data", sample_data_prev_token)
                    pose_token_prev = sample_data_prev['ego_pose_token']
                    pose_tokens.append(pose_token_prev)
                else:
                    break
            if len(pose_tokens) == self.cfg_model["n_input"]:
                sample_pose_dict[sample_token] = pose_tokens
        return sample_pose_dict