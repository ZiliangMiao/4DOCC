import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from nuscenes.utils.splits import create_splits_logs, create_splits_scenes
from random import sample as random_sample
from utils.augmentation import augment_pcds
import datasets.nusc_utils as nusc_utils


class NuscUnODataset(Dataset):
    def __init__(self, nusc, cfg_model, cfg_dataset, split):
        self.nusc = nusc
        self.cfg_model = cfg_model
        self.cfg_dataset = cfg_dataset
        self.split = split  # "train" "val" "mini_train" "mini_val" "test"

        if split == 'train':
            # dataset down-sampling: sequence level or sample level
            if self.cfg_model["downsample_level"] is None:  # for test set and validation set
                split_logs = create_splits_logs(split, self.nusc)
                sample_toks = nusc_utils.split_logs_to_samples(self.nusc, split_logs)
            elif self.cfg_model["downsample_level"] == "sequence":
                split_scenes = create_splits_scenes(verbose=True)
                split_scenes = split_scenes[self.split]
                train_data_pct = self.cfg_model["downsample_pct"] / 100
                ds_split_scenes = random_sample(split_scenes, int(len(split_scenes) * train_data_pct))
                sample_toks = nusc_utils.split_scenes_to_samples(self.nusc, ds_split_scenes)
            elif self.cfg_model["downsample_level"] == "sample":
                split_logs = create_splits_logs(split, self.nusc)
                sample_toks = nusc_utils.split_logs_to_samples(self.nusc, split_logs)
                train_data_pct = self.cfg_model["downsample_pct"] / 100
                sample_toks = random_sample(sample_toks, int(len(sample_toks) * train_data_pct))
            else:
                raise ValueError("Invalid dataset down-sampling strategy!")
        else:
            split_logs = create_splits_logs(split, self.nusc)
            sample_toks = nusc_utils.split_logs_to_samples(self.nusc, split_logs)

        # sample tokens: drop the samples without full sequence length
        self.sample_to_sd_toks_dict = nusc_utils.get_sample_level_seq_input(self.nusc, self.cfg_model, sample_toks)

    def __len__(self):
        return len(self.sample_to_sd_toks_dict)

    def __getitem__(self, batch_idx):
        # sample
        sample_toks_list = list(self.sample_to_sd_toks_dict.keys())
        sample_tok = sample_toks_list[batch_idx]
        sample = self.nusc.get('sample', sample_tok)
        ref_sd_tok = sample['data']['LIDAR_TOP']

        # sample data: concat 4d point clouds
        input_sd_toks = self.sample_to_sd_toks_dict[sample_tok]  # sequence: -1, -2 ...
        assert len(input_sd_toks) == self.cfg_model["n_input"], "Invalid input sequence length"
        pcd_4d_list = []  # 4D Point Cloud (relative timestamp)
        time_idx = 1
        for i, sd_tok in enumerate(input_sd_toks):
            lidar_org, pcd, rela_ts, valid_mask = nusc_utils.get_transformed_pcd(self.nusc, self.cfg_model, ref_sd_tok, sd_tok)  # filter ego and outside inside func
            # add timestamp (0, -1, -2, ...) to pcd -> pcd_4d
            time_idx -= 1
            pcd_4d = torch.hstack([pcd, torch.ones(len(pcd)).reshape(-1, 1) * time_idx])
            pcd_4d_list.append(pcd_4d)
        pcds_4d = torch.cat(pcd_4d_list, dim=0).float()  # 4D point cloud: [x y z ref_ts]

        # data augmentation: will not change the order of points
        if self.split == 'train' and self.cfg_model["augmentation"]:
            pcds_4d = augment_pcds(pcds_4d)

        # generate uno labels
        uno_sd_toks = nusc_utils.get_curr_future_sd_toks_dict(self.nusc, [sample_tok], self.cfg_model)[sample_tok]

        # future pcds
        num_cls_samples = self.cfg_model['num_cls_samples']
        num_ray_cls_samples = self.cfg_model['num_ray_cls_samples']
        num_rays_per_scan = int(num_cls_samples / num_ray_cls_samples / len(uno_sd_toks))
        uno_points_list = []
        uno_labels_list = []
        for sd_tok in uno_sd_toks:  # 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
            org, pcd, ts, _ = nusc_utils.get_transformed_pcd(self.nusc, self.cfg_model, ref_sd_tok, sd_tok)

            # balanced sampling of occ points and free points
            ds_ray_idx = random_sample(range(len(pcd)), num_rays_per_scan)
            ray_pts = pcd[ds_ray_idx]
            ray_dir = F.normalize(ray_pts - org, p=2, dim=1)  # unit vector
            ray_depth = torch.linalg.norm(ray_pts - org, dim=1, keepdim=False)

            # uno balanced sampling
            free_depth_scale = torch.rand((num_rays_per_scan, num_ray_cls_samples))
            occ_depth_scale = torch.rand((num_rays_per_scan, num_ray_cls_samples))

            test_a = free_depth_scale * ray_depth
            test_b = free_depth_scale * ray_depth

            free_pts = org + free_depth_scale * ray_depth * ray_dir
            occ_pts = org + occ_depth_scale * (ray_depth + self.cfg_model['occ_thrd']) * ray_dir
            free_labels = torch.zeros(len(free_pts))
            occ_labels = torch.ones(len(occ_pts))
            uno_points = torch.cat((free_pts, occ_pts), dim=0)
            uno_labels = torch.cat((free_labels, occ_labels), dim=0)
            uno_points_list.append(uno_points)
            uno_labels_list.append(uno_labels)
        uno_points = torch.stack(uno_points_list)
        uno_labels = torch.stack(uno_labels_list)
        return [(ref_sd_tok, uno_sd_toks), pcds_4d, (uno_points, uno_labels)]