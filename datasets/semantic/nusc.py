import os
from collections import defaultdict

import torch
import numpy as np
import yaml
from torch.utils.data import Dataset
import torch.nn.functional as F
from nuscenes.utils.splits import create_splits_logs, create_splits_scenes
from random import sample as random_sample
from utils.augmentation import augment_pcds
import datasets.nusc_utils as nusc_utils


class NuscSemanticDataset(Dataset):
    def __init__(self, nusc, cfg_model, cfg_dataset, split):
        self.nusc = nusc
        self.cfg_model = cfg_model
        self.cfg_dataset = cfg_dataset
        self.split = split  # "train" "val" "mini_train" "mini_val" "test"

        # semantic learning map
        with open("configs/semantic_learning_map.yaml", "r") as f:
            cfg_semantic = yaml.safe_load(f)
        self.learning_map = cfg_semantic['learning_map']

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
        self.sample_to_sd_toks_dict = nusc_utils.get_input_sd_toks(self.nusc, self.cfg_model, sample_toks)

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
        ref_pcd = None
        ref_valid_mask = None
        for i, sd_tok in enumerate(input_sd_toks):
            # TODO: filter ego and outside points
            org, pcd, ts, valid_mask = nusc_utils.get_transformed_pcd(self.nusc, self.cfg_model, ref_sd_tok, sd_tok)
            if i == 0:
                ref_pcd = pcd
                ref_valid_mask = valid_mask
            # add timestamp (0, -1, -2, ...) to pcd -> pcd_4d
            time_idx -= 1
            # assert time_idx == round(ts / 0.5), "relative timestamp repeated"  # TODO: has repeated corner cases
            pcd_4d = torch.hstack([pcd, torch.ones(len(pcd)).reshape(-1, 1) * time_idx])
            pcd_4d_list.append(pcd_4d)
        pcds_4d = torch.cat(pcd_4d_list, dim=0).float()  # 4D point cloud: [x y z ref_ts]

        # data augmentation: will not change the order of points
        if self.split == 'train' and self.cfg_model["augmentation"]:
            pcds_4d = augment_pcds(pcds_4d)

        # semantic labels
        lidarseg_labels_filename = os.path.join(self.nusc.dataroot,
                                                self.nusc.get('lidarseg', ref_sd_tok)['filename'])
        semantic_labels = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)
        semantic_labels = torch.tensor(np.vectorize(self.learning_map.__getitem__)(semantic_labels))
        semantic_labels = semantic_labels[ref_valid_mask]
        return [ref_sd_tok, pcds_4d, semantic_labels]