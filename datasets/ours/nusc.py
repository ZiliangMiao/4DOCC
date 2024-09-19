import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from nuscenes.utils.splits import create_splits_logs, create_splits_scenes
from random import sample as random_sample
from utils.augmentation import augment_pcds
from datasets.nusc_utils import split_logs_to_samples, split_scenes_to_samples, get_sample_level_seq_input
from preprocess_rays_mutual_obs_script import get_mutual_sd_toks_dict, get_transformed_pcd


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

        # TODO: temporarily remove samples that have no mutual observation samples ###################################################################
        mutual_obs_folder = os.path.join(self.nusc.dataroot, "mutual_obs_labels", self.nusc.version)
        mutual_obs_sd_tok_list_1 = os.listdir(mutual_obs_folder)
        mutual_obs_sd_tok_list_2 = os.listdir(mutual_obs_folder)
        mutual_obs_sd_tok_list_3 = os.listdir(mutual_obs_folder)
        mutual_obs_sd_tok_list_4 = os.listdir(mutual_obs_folder)
        mutual_obs_sd_tok_list_5 = os.listdir(mutual_obs_folder)
        mutual_obs_sd_tok_list_6 = os.listdir(mutual_obs_folder)
        for i in range(len(mutual_obs_sd_tok_list_1)):
            mutual_obs_sd_tok_list_1[i] = mutual_obs_sd_tok_list_1[i].replace("_depth.bin", '')
            mutual_obs_sd_tok_list_2[i] = mutual_obs_sd_tok_list_2[i].replace("_labels.bin", '')
            mutual_obs_sd_tok_list_3[i] = mutual_obs_sd_tok_list_3[i].replace("_confidence.bin", '')
            mutual_obs_sd_tok_list_4[i] = mutual_obs_sd_tok_list_4[i].replace("_rays_idx.bin", '')
            mutual_obs_sd_tok_list_5[i] = mutual_obs_sd_tok_list_5[i].replace("_key_rays_idx.bin", '')
            mutual_obs_sd_tok_list_6[i] = mutual_obs_sd_tok_list_6[i].replace("_key_meta.bin", '')
        valid_sample_toks = list(set(mutual_obs_sd_tok_list_1) & set(mutual_obs_sd_tok_list_2) & set(mutual_obs_sd_tok_list_3) & set(mutual_obs_sd_tok_list_4) & set(mutual_obs_sd_tok_list_5) & set(mutual_obs_sd_tok_list_6))
        sample_toks = [sample_tok for sample_tok in sample_toks if self.nusc.get('sample', sample_tok)['data']['LIDAR_TOP'] in valid_sample_toks]
        ##############################################################################################################################################

        # sample tokens: drop the samples without full sequence length
        self.sample_to_sd_toks_dict = get_sample_level_seq_input(self.nusc, self.cfg_model, sample_toks)

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
        ref_pts = None
        ref_org = None
        for i, sd_tok in enumerate(input_sd_toks):
            lidar_org, pcd, rela_ts, valid_mask = get_transformed_pcd(self.nusc, self.cfg_model, ref_sd_tok, sd_tok)  # TODO: filter ego and outside inside func
            if i == 0:  # reference sample data token
                ref_org = lidar_org
                ref_pts = pcd
            # add timestamp (0, -1, -2, ...) to pcd -> pcd_4d
            time_idx -= 1
            # assert time_idx == round(rela_ts / 0.5), "relative timestamp repeated"  # TODO: corner cases
            pcd_4d = torch.hstack([pcd, torch.ones(len(pcd)).reshape(-1, 1) * time_idx])
            pcd_4d_list.append(pcd_4d)
        pcds_4d = torch.cat(pcd_4d_list, dim=0).float()  # 4D point cloud: [x y z ref_ts]

        # data augmentation: will not change the order of points
        if self.split == 'train' and self.cfg_model["augmentation"]:
            pcds_4d = augment_pcds(pcds_4d)

        # load labels
        mutual_obs_folder = os.path.join(self.nusc.dataroot, "mutual_obs_labels", self.nusc.version)
        mutual_obs_meta = os.path.join(mutual_obs_folder, ref_sd_tok + "_key_meta.bin")
        mutual_obs_rays_idx = os.path.join(mutual_obs_folder, ref_sd_tok + "_rays_idx.bin")
        mutual_obs_depth = os.path.join(mutual_obs_folder, ref_sd_tok + "_depth.bin")
        mutual_obs_labels = os.path.join(mutual_obs_folder, ref_sd_tok + "_labels.bin")
        mutual_obs_confidence = os.path.join(mutual_obs_folder, ref_sd_tok + "_confidence.bin")
        mutual_obs_meta = np.fromfile(mutual_obs_meta, dtype=np.uint32).reshape(-1, 2).astype(np.int64)
        mutual_obs_rays_idx = torch.from_numpy(np.fromfile(mutual_obs_rays_idx, dtype=np.uint16).astype(np.int64))
        mutual_obs_depth = torch.from_numpy(np.fromfile(mutual_obs_depth, dtype=np.float16).astype(np.float32))
        mutual_obs_labels = torch.from_numpy(np.fromfile(mutual_obs_labels, dtype=np.uint8).astype(np.int64))
        mutual_obs_confidence = torch.from_numpy(np.fromfile(mutual_obs_confidence, dtype=np.float16).astype(np.float32))

        # TODO: balanced sampling
        mutual_unk_idx = torch.where(mutual_obs_labels == 0)[0]
        mutual_free_idx = torch.where(mutual_obs_labels == 1)[0]
        mutual_occ_idx = torch.where(mutual_obs_labels == 2)[0]
        num_unk = len(mutual_unk_idx)
        num_free = len(mutual_free_idx)
        num_occ = len(mutual_occ_idx)
        num_cls_min = np.min((num_unk, num_free, num_occ))
        num_ds_unk = np.min((num_cls_min, self.cfg_model['num_ds_unk_samples']))
        num_ds_free = np.min((num_cls_min, self.cfg_model['num_ds_free_samples']))
        num_ds_occ = np.min((num_cls_min, self.cfg_model['num_ds_occ_samples']))
        ds_mutual_unk_idx = mutual_unk_idx[random_sample(range(num_unk), num_ds_unk)]
        ds_mutual_free_idx = mutual_free_idx[random_sample(range(num_free), num_ds_free)]
        ds_mutual_occ_idx = mutual_occ_idx[random_sample(range(num_occ), num_ds_occ)]
        ds_mutual_sample_indices = torch.cat([ds_mutual_unk_idx, ds_mutual_free_idx, ds_mutual_occ_idx])

        # mutual obs timestamps
        mutual_sd_toks = get_mutual_sd_toks_dict(self.nusc, [sample_tok], self.cfg_model)[sample_tok]
        mutual_sensors_indices = np.concatenate([np.ones(meta[1], dtype=np.int64) * meta[0] for meta in mutual_obs_meta])
        mutual_sensors_timestamps = [(self.nusc.get('sample_data', sd_tok)['timestamp'] - self.nusc.get('sample_data', ref_sd_tok)['timestamp']) / 1e6 for sd_tok in mutual_sd_toks]
        mutual_obs_ts = torch.tensor(mutual_sensors_timestamps)[mutual_sensors_indices]

        # update down-sampled mutual obs samples
        mutual_obs_rays_idx = mutual_obs_rays_idx[ds_mutual_sample_indices]
        mutual_obs_depth = mutual_obs_depth[ds_mutual_sample_indices]
        mutual_obs_ts = mutual_obs_ts[ds_mutual_sample_indices]
        mutual_obs_labels = mutual_obs_labels[ds_mutual_sample_indices]
        mutual_obs_confidence = mutual_obs_confidence[ds_mutual_sample_indices]

        # mutual obs points (down-sampled)
        mutual_rays_dir = F.normalize(ref_pts - ref_org, p=2, dim=1)  # unit vector
        mutual_obs_pts = ref_org + mutual_obs_depth.reshape(-1, 1) * mutual_rays_dir[mutual_obs_rays_idx]
        return [(ref_sd_tok, mutual_sd_toks), pcds_4d, (mutual_obs_rays_idx, mutual_obs_pts, mutual_obs_ts, mutual_obs_labels, mutual_obs_confidence)]