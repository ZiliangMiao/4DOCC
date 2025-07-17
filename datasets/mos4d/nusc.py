import os
from typing import List
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_logs, create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from random import sample as random_sample
from nuscenes.utils.geometry_utils import transform_matrix
from datasets.nusc_utils import split_logs_to_samples, split_scenes_to_samples, get_input_sd_toks, get_sample_data_level_seq_input, get_ego_mask, get_outside_scene_mask, add_timestamp
from utils.augmentation import augment_pcds


class NuscMosDataset(Dataset):
    def __init__(self, nusc, cfg_model, cfg_dataset, mode):
        self.nusc = nusc
        self.cfg_model = cfg_model
        self.cfg_dataset = cfg_dataset
        self.mode = mode

        if self.mode == 'train':
            nusc_split = 'train'
            # dataset down-sampling: sequence level or sample level
            if self.cfg_model["downsample_level"] is None:  # for test set and validation set
                split_logs = create_splits_logs(nusc_split, self.nusc)
                sample_toks = split_logs_to_samples(self.nusc, split_logs)
            elif self.cfg_model["downsample_level"] == "sequence":
                split_scenes = create_splits_scenes(verbose=False)
                split_scenes = split_scenes[nusc_split]
                train_data_pct = self.cfg_model["downsample_pct"] / 100
                ds_split_scenes = random_sample(split_scenes, int(len(split_scenes) * train_data_pct))
                sample_toks = split_scenes_to_samples(self.nusc, ds_split_scenes)
            elif self.cfg_model["downsample_level"] == "sample":
                split_logs = create_splits_logs(nusc_split, self.nusc)
                sample_toks = split_logs_to_samples(self.nusc, split_logs)
                train_data_pct = self.cfg_model["downsample_pct"] / 100
                sample_toks = random_sample(sample_toks, int(len(sample_toks) * train_data_pct))
            else:
                raise ValueError("Invalid dataset down-sampling strategy!")
        elif self.mode == 'val':
            nusc_split = 'train'
            # down-sampling a subset for validation using uniform sampling
            split_scenes = create_splits_scenes(verbose=False)
            split_scenes = split_scenes[nusc_split]
            num_scenes = int(len(split_scenes) * 0.1)  # TODO: 10% training set as validation set
            step = len(split_scenes) // num_scenes
            ds_split_scenes = split_scenes[::step][:num_scenes]  # uniform sampling with step
            sample_toks = split_scenes_to_samples(self.nusc, ds_split_scenes)
        elif self.mode == 'test':
            nusc_split = 'val'
            split_logs = create_splits_logs(nusc_split, self.nusc)
            sample_toks = split_logs_to_samples(self.nusc, split_logs)
        else:
            raise ValueError("Invalid dataset mode!")

        # sample tokens: drop the samples without full sequence length
        if cfg_model['time_interval'] == 0.5:  # sample level input
            self.sample_to_sd_toks_dict = get_input_sd_toks(self.nusc, self.cfg_model, sample_toks)
        elif cfg_model['time_interval'] == 0.05:  # sample data level input
            self.sample_to_sd_toks_dict = get_sample_data_level_seq_input(self.nusc, self.cfg_model, sample_toks)

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
                trans_car_to_global = transform_matrix(pose['translation'],
                                                       Quaternion(pose['rotation']))  # from car to global

                # transformation: lidar -> car -> global -> ref_car -> ref_lidar
                trans_to_ref = trans_lidar_to_car @ trans_car_to_global @ trans_global_to_ref_car @ trans_car_to_lidar
                pcd_homo = np.hstack([pcd, np.ones((pcd.shape[0], 1))]).T
                pcd_ref = torch.from_numpy((trans_to_ref @ pcd_homo).T[:, :3])

                # add timestamp (0, -1, -2, ...)
                time_idx -= 1
                pcd_4d_ref = add_timestamp(pcd_ref, time_idx)
                pcds_4d_list.append(pcd_4d_ref)
        pcds_4d = torch.cat(pcds_4d_list, dim=0).float()  # 4D point cloud: [x y z ref_ts]

        # load labels
        mos_labels_dir = os.path.join(self.cfg_dataset["nuscenes"]["root"], "mos_labels", self.cfg_dataset["nuscenes"]["version"])
        mos_label_file = os.path.join(mos_labels_dir, ref_sd_tok + "_mos.label")
        mos_labels = torch.tensor(np.fromfile(mos_label_file, dtype=np.uint8))

        # TODO: make the ray index same to ray_intersection_cuda.py, they should have the same filter params
        # Filter all the outside scene points for both train, val, and test
        ref_time_mask = pcds_4d[:, -1] == 0
        valid_mask = torch.squeeze(torch.full((len(pcds_4d), 1), True))
        if self.cfg_model['outside_scene_mask']:
            outside_scene_mask = get_outside_scene_mask(pcds_4d, self.cfg_model["scene_bbox"],
                                                        self.cfg_model['outside_scene_mask_z'],
                                                        self.cfg_model['outside_scene_mask_ub'])
            valid_mask = torch.logical_and(valid_mask, ~outside_scene_mask)

        # Data augmentation and ego mask for train
        if self.mode == 'train':
            if self.cfg_model['ego_mask']:
                ego_mask = get_ego_mask(pcds_4d)
                valid_mask = torch.logical_and(valid_mask, ~ego_mask)
            pcds_4d = pcds_4d[valid_mask]
            mos_labels = mos_labels[valid_mask[ref_time_mask]]

            if self.cfg_model["augmentation"]:
                pcds_4d = augment_pcds(pcds_4d) # will not change the order of points

        # TODO: test data processing
        if self.mode == 'val' or 'test':
            eval_test = True
            if not eval_test:
                ego_mask = get_ego_mask(pcds_4d)
                valid_mask = torch.logical_and(valid_mask, ~ego_mask)
            pcds_4d = pcds_4d[valid_mask]
            mos_labels = mos_labels[valid_mask[ref_time_mask]]

        return [(ref_sd_tok, valid_mask[ref_time_mask]), pcds_4d, mos_labels]