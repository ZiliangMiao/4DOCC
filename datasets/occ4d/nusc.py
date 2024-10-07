import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from nuscenes.utils.splits import create_splits_logs, create_splits_scenes
from random import sample as random_sample
from utils.augmentation import augment_pcds
import datasets.nusc_utils as nusc_utils
from datasets.nusc_utils import get_outside_scene_mask


class NuscOcc4dDataset(Dataset):
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

        # input sample data tokens: drop the samples without full sequence length
        self.sample_to_sd_toks_dict = nusc_utils.get_input_sd_toks(self.nusc, self.cfg_model, sample_toks)

        # occ4d sample data tokens (current and future):
        self.sample_to_occ4d_sd_toks = nusc_utils.get_curr_future_sd_toks_dict(self.nusc, sample_toks, self.cfg_model, get_curr=False)

        # valid sample tokens
        self.valid_sample_toks = list(set(self.sample_to_sd_toks_dict.keys()) & set(self.sample_to_occ4d_sd_toks.keys()))

    def __len__(self):
        return len(self.valid_sample_toks)

    def __getitem__(self, batch_idx):
        # sample
        ref_sample_tok = self.valid_sample_toks[batch_idx]
        ref_sample = self.nusc.get('sample', ref_sample_tok)
        ref_sd_tok = ref_sample['data']['LIDAR_TOP']

        # sample data: concat 4d point clouds
        input_sd_toks = self.sample_to_sd_toks_dict[ref_sample_tok]  # sequence: -1, -2 ...
        assert len(input_sd_toks) == self.cfg_model["n_input"], "Invalid input sequence length"
        pcd_4d_list = []  # 4D Point Cloud (relative timestamp)
        tindex = 1
        for i, sd_tok in enumerate(input_sd_toks):
            org, pcd, ts, valid_mask = nusc_utils.get_transformed_pcd(self.nusc, self.cfg_model, ref_sd_tok, sd_tok)  # filter ego and outside inside func
            # add timestamp (0, -1, -2, ...) to pcd -> pcd_4d
            tindex -= 1
            pcd_4d = torch.hstack([pcd, torch.full((len(pcd), 1), tindex)])
            pcd_4d_list.append(pcd_4d)
        pcds_4d = torch.cat(pcd_4d_list, dim=0).float()  # 4D point cloud: [x y z ref_ts]

        # data augmentation: will not change the order of points
        if self.split == 'train' and self.cfg_model["augmentation"]:
            # TODO: augmentation may cause outside scene bbox
            pcds_4d = augment_pcds(pcds_4d)
            outside_scene_mask = get_outside_scene_mask(pcds_4d, self.cfg_model['scene_bbox'],
                                                        self.cfg_model['outside_scene_mask_z'],
                                                        self.cfg_model['outside_scene_mask_ub'])
            pcds_4d = pcds_4d[~outside_scene_mask]

        # generate uno labels
        occ4d_sd_toks = self.sample_to_occ4d_sd_toks[ref_sample_tok]

        # future pcds
        future_orgs = []
        future_pcds = []
        future_tindex = []
        tindex = -1
        for sd_tok in occ4d_sd_toks:  # 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
            org, pcd, ts, _ = nusc_utils.get_transformed_pcd(self.nusc, self.cfg_model, ref_sd_tok, sd_tok)
            # future timestamp
            tindex += 1
            # append
            future_orgs.append(org)
            future_pcds.append(pcd)
            future_tindex.append(torch.ones(len(pcd)) * tindex)  # has to be 1 dimension [0, 1, 2, 3, 4, 5]
        future_orgs = torch.stack(future_orgs)
        future_pcds = torch.cat(future_pcds, dim=0)
        future_tindex = torch.cat(future_tindex, dim=0)
        return [(ref_sd_tok, occ4d_sd_toks), pcds_4d, (future_orgs, future_pcds, future_tindex)]
