import os
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

from nuscenes.utils.splits import create_splits_logs, create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from random import sample as random_sample
from nuscenes.utils.geometry_utils import transform_matrix
from utils.augmentation import augment_pcds
from datasets.nusc_utils import split_logs_to_samples, split_scenes_to_samples, get_sd_toks_dict, get_ego_mask, get_outside_scene_mask, add_timestamp


class NuscBgDataset(Dataset):
    def __init__(self, nusc, cfg_model, cfg_dataset, split):
        self.nusc = nusc
        self.cfg_model = cfg_model
        self.cfg_dataset = cfg_dataset
        self.split = split  # "train" "val" "mini_train" "mini_val" "test"

        if split == 'train':
            # dataset down-sampling: sequence level or sample level
            if self.cfg_model["downsample_level"] is None:  # for test set and validation set
                split_logs = create_splits_logs(split, self.nusc)
                sample_toks = split_logs_to_samples(self.nusc, split_logs)
            elif self.cfg_model["downsample_level"] == "sequence":
                split_scenes = create_splits_scenes(verbose=True)
                split_scenes = split_scenes[self.split]
                train_data_pct = self.cfg_model["downsample_pct"] / 100
                ds_split_scenes = random_sample(split_scenes, int(len(split_scenes) * train_data_pct))
                sample_toks = split_scenes_to_samples(self.nusc, ds_split_scenes)
            elif self.cfg_model["downsample_level"] == "sample":
                split_logs = create_splits_logs(split, self.nusc)
                sample_toks = split_logs_to_samples(self.nusc, split_logs)
                train_data_pct = self.cfg_model["downsample_pct"] / 100
                sample_toks = random_sample(sample_toks, int(len(sample_toks) * train_data_pct))
            else:
                raise ValueError("Invalid dataset down-sampling strategy!")
        else:
            split_logs = create_splits_logs(split, self.nusc)
            sample_toks = split_logs_to_samples(self.nusc, split_logs)

        # sample tokens: drop the samples without full sequence length
        self.sample_to_sd_toks_dict = get_sd_toks_dict(self.nusc, self.cfg_model, sample_toks)

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
                pcd_4d_ref = add_timestamp(pcd_ref, time_idx)
                pcds_4d_list.append(pcd_4d_ref)
        pcds_4d = torch.cat(pcds_4d_list, dim=0).float()  # 4D point cloud: [x y z ref_ts]

        # TODO: make the ray index same to ray_intersection_cuda.py, they should have the same filter params
        valid_mask = torch.squeeze(torch.full((len(pcds_4d), 1), True))
        if self.cfg_model['ego_mask']:
            ego_mask = get_ego_mask(pcds_4d)
            valid_mask = torch.logical_and(valid_mask, ~ego_mask)
        if self.cfg_model['outside_scene_mask']:
            outside_scene_mask = get_outside_scene_mask(pcds_4d, self.cfg_model["scene_bbox"])
            valid_mask = torch.logical_and(valid_mask, ~outside_scene_mask)
        pcds_4d = pcds_4d[valid_mask]

        # data augmentation: will not change the order of points
        if self.cfg_model["augmentation"]:
            pcds_4d = augment_pcds(pcds_4d)

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