import os
import re
import torch
import numpy as np
from sympy.codegen.ast import continue_
from torch.utils.data import Dataset
import torch.nn.functional as F
from nuscenes.utils.splits import create_splits_logs, create_splits_scenes
from random import sample as random_sample
from utils.augmentation import augment_pcds
import datasets.nusc_utils as nusc_utils
from datasets.nusc_utils import get_outside_scene_mask


class CreatePoints(object):
    def __init__(self, n_non_manifold_pts=None, non_manifold_dist=0.1):
        print(f"Transforms - CreatePoints - non_manifold {n_non_manifold_pts} - non_manifold_dist {non_manifold_dist}")
        self.n_non_manifold_pts = n_non_manifold_pts
        self.non_manifold_dist = non_manifold_dist

    def __call__(self, curr_pcd):
        # non manifold points
        if self.n_non_manifold_pts is None:
            print("No sample points.")
            return None

        # nmp -> non_manifold points
        n_nmp = self.n_non_manifold_pts

        # select the points for the current frame
        n_nmp_out = n_nmp // 3
        n_nmp_out_far = n_nmp // 3
        n_nmp_in = n_nmp - 2 * (n_nmp // 3)
        nmp_choice_in = torch.randperm(curr_pcd.shape[0])[:n_nmp_in]
        nmp_choice_out = torch.randperm(curr_pcd.shape[0])[:n_nmp_out]
        nmp_choice_out_far = torch.randperm(curr_pcd.shape[0])[:n_nmp_out_far]

        # center
        center = torch.zeros((1, 3), dtype=torch.float)

        # in points (todo: q_behind, occupied)
        pos = curr_pcd[nmp_choice_in]
        dirs = F.normalize(pos, dim=1)
        pos_in = pos + self.non_manifold_dist * dirs * torch.rand((pos.shape[0], 1))
        occ_in = torch.ones(pos_in.shape[0], dtype=torch.long)

        # out points (todo: q_front, free)
        pos = curr_pcd[nmp_choice_out]
        dirs = F.normalize(pos, dim=1)
        pos_out = pos - self.non_manifold_dist * dirs * torch.rand((pos.shape[0], 1))
        occ_out = torch.zeros(pos_out.shape[0], dtype=torch.long)

        # out far points (todo: q_sight, free)
        pos = curr_pcd[nmp_choice_out_far]
        dirs = F.normalize(pos, dim=1)
        pos_out_far = (pos - center) * torch.rand((pos.shape[0], 1)) + center
        occ_out_far = torch.zeros(pos_out_far.shape[0], dtype=torch.long)

        also_points = torch.cat([pos_in, pos_out, pos_out_far], dim=0)
        also_labels = torch.cat([occ_in, occ_out, occ_out_far], dim=0)
        return also_points, also_labels


class NuscAlsoDataset(Dataset):
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

        # create also samples
        self.create_points = CreatePoints(n_non_manifold_pts=self.cfg_model["non_manifold_points"],
                                          non_manifold_dist=self.cfg_model["occ_thrd"])

    def __len__(self):
        return len(self.sample_to_sd_toks_dict)

    def __getitem__(self, batch_idx):
        # sample
        sample_toks_list = list(self.sample_to_sd_toks_dict.keys())
        ref_sample_tok = sample_toks_list[batch_idx]
        ref_sample = self.nusc.get('sample', ref_sample_tok)
        ref_sd_tok = ref_sample['data']['LIDAR_TOP']

        # sample data: concat 4d point clouds
        input_sd_toks = self.sample_to_sd_toks_dict[ref_sample_tok]  # sequence: -1, -2 ...
        assert len(input_sd_toks) == self.cfg_model["n_input"], "Invalid input sequence length"
        pcd_4d_list = []  # 4D Point Cloud (relative timestamp)
        time_idx = 1
        for i, sd_tok in enumerate(input_sd_toks):
            org, pcd, ts, valid_mask = nusc_utils.get_transformed_pcd(self.nusc, self.cfg_model, ref_sd_tok, sd_tok)  # filter ego and outside inside func
            # add timestamp (0, -1, -2, ...) to pcd -> pcd_4d
            time_idx -= 1
            pcd_4d = torch.hstack([pcd, torch.full((len(pcd), 1), time_idx)])
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

        # generate also samples
        org, pcd, ts, _ = nusc_utils.get_transformed_pcd(self.nusc, self.cfg_model, ref_sd_tok, ref_sd_tok)

        # generate also samples
        also_pts, also_labels = self.create_points(pcd)

        # shuffle
        shuffle_idx = torch.randperm(len(also_pts))
        also_pts = also_pts[shuffle_idx]
        also_labels = also_labels[shuffle_idx]
        return [ref_sd_tok, pcds_4d, (pcd, also_pts, also_labels)]
