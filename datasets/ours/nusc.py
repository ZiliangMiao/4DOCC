import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from nuscenes.utils.splits import create_splits_logs, create_splits_scenes
from random import sample as random_sample
from utils.augmentation import augment_pcds
import datasets.nusc_utils as nusc_utils


class NuscMopDataset(Dataset):
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
        valid_sd_toks = list(set(mutual_obs_sd_tok_list_1) & set(mutual_obs_sd_tok_list_2) & set(mutual_obs_sd_tok_list_3) & set(mutual_obs_sd_tok_list_4) & set(mutual_obs_sd_tok_list_5) & set(mutual_obs_sd_tok_list_6))
        sample_toks = [sample_tok for sample_tok in sample_toks if self.nusc.get('sample', sample_tok)['data']['LIDAR_TOP'] in valid_sd_toks]
        ##############################################################################################################################################

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
        ref_pts = None
        ref_org = None
        ref_ts = None
        for i, sd_tok in enumerate(input_sd_toks):
            # TODO: filter ego and outside points
            org, pcd, ts, valid_mask = nusc_utils.get_transformed_pcd(self.nusc, self.cfg_model, ref_sd_tok, sd_tok)
            if i == 0:  # reference sample data token
                ref_org = org
                ref_pts = pcd
                ref_ts = ts
            # add timestamp (0, -1, -2, ...) to pcd -> pcd_4d
            time_idx -= 1
            # assert time_idx == round(ts / 0.5), "relative timestamp repeated"  # TODO: has repeated corner cases
            pcd_4d = torch.hstack([pcd, torch.ones(len(pcd)).reshape(-1, 1) * time_idx])
            pcd_4d_list.append(pcd_4d)
        pcds_4d = torch.cat(pcd_4d_list, dim=0).float()  # 4D point cloud: [x y z ref_ts]

        # data augmentation: will not change the order of points
        if self.split == 'train' and self.cfg_model["augmentation"]:
            pcds_4d = augment_pcds(pcds_4d)

        # load mutual observation samples
        mutual_obs_folder = os.path.join(self.nusc.dataroot, "mutual_obs_labels", self.nusc.version)
        mutual_obs_meta = os.path.join(mutual_obs_folder, ref_sd_tok + "_key_meta.bin")
        mutual_sd_toks = nusc_utils.get_mutual_sd_toks_dict(self.nusc, [sample_tok], self.cfg_model)[sample_tok]
        mop_rays_idx = os.path.join(mutual_obs_folder, ref_sd_tok + "_rays_idx.bin")
        mop_depth = os.path.join(mutual_obs_folder, ref_sd_tok + "_depth.bin")
        mop_labels = os.path.join(mutual_obs_folder, ref_sd_tok + "_labels.bin")
        mop_confidence = os.path.join(mutual_obs_folder, ref_sd_tok + "_confidence.bin")
        mutual_obs_meta = np.fromfile(mutual_obs_meta, dtype=np.uint32).reshape(-1, 2).astype(np.int64)
        mop_rays_idx = torch.from_numpy(np.fromfile(mop_rays_idx, dtype=np.uint16).astype(np.int64))
        mop_depth = torch.from_numpy(np.fromfile(mop_depth, dtype=np.float16).astype(np.float32))
        mop_labels = torch.from_numpy(np.fromfile(mop_labels, dtype=np.uint8).astype(np.int64))
        mop_confidence = torch.from_numpy(np.fromfile(mop_confidence, dtype=np.float16).astype(np.float32))
        mutual_sensors_indices = np.concatenate([np.ones(meta[1], dtype=np.int64) * meta[0] for meta in mutual_obs_meta])
        mutual_sensors_timestamps = [(self.nusc.get('sample_data', sd_tok)['timestamp'] -
                                      self.nusc.get('sample_data', ref_sd_tok)['timestamp']) / 1e6 for sd_tok in mutual_sd_toks]
        mop_ts = torch.tensor(mutual_sensors_timestamps)[mutual_sensors_indices]

        # only select background mop samples (unknown occupancy state)
        if self.cfg_model['train_bg_mop_samples']:
            # mask mutual obs points which depth less than ray depth
            rays_depth = torch.linalg.norm(ref_pts - ref_org, dim=1, keepdim=False)[mop_rays_idx]
            bg_mask = mop_depth >= rays_depth
            # update valid samples with bg mask
            mop_rays_idx = mop_rays_idx[bg_mask]
            mop_depth = mop_depth[bg_mask]
            mop_ts = mop_ts[bg_mask]
            mop_labels = mop_labels[bg_mask]
            mop_confidence = mop_confidence[bg_mask]

        # TODO: balanced down-sampling for training (do not balance sampling when testing, but cost a lot of gpu memory)
        mop_unk_idx = torch.where(mop_labels == 0)[0]
        mop_free_idx = torch.where(mop_labels == 1)[0]
        mop_occ_idx = torch.where(mop_labels == 2)[0]

        # downsample occ and free class
        num_mop_free = len(mop_free_idx)
        num_mop_occ = len(mop_occ_idx)
        num_mop_samples_cls = np.min((np.min((num_mop_free, num_mop_occ)), self.cfg_model['max_mop_samples_cls']))
        ds_mop_free_idx = mop_free_idx[random_sample(range(num_mop_free), num_mop_samples_cls)]
        ds_mop_occ_idx = mop_occ_idx[random_sample(range(num_mop_occ), num_mop_samples_cls)]

        # downsample unk (a certain percentage of num_mop_samples_cls)
        num_mop_unk = len(mop_unk_idx)
        num_mop_samples_unk = np.min((num_mop_unk, int(num_mop_samples_cls * self.cfg_model['unk_samples_pct'] / 100)))
        ds_mop_unk_idx = mop_unk_idx[random_sample(range(num_mop_unk), num_mop_samples_unk)]

        # update down-sampled mutual obs samples
        ds_mop_sample_indices = torch.cat([ds_mop_unk_idx, ds_mop_free_idx, ds_mop_occ_idx])
        mop_rays_idx = mop_rays_idx[ds_mop_sample_indices]
        mop_depth = mop_depth[ds_mop_sample_indices]
        mop_ts = mop_ts[ds_mop_sample_indices]
        mop_labels = mop_labels[ds_mop_sample_indices]
        mop_confidence = mop_confidence[ds_mop_sample_indices]

        # rays params
        num_rays = len(ref_pts)
        rays_idx = torch.tensor(range(num_rays)).view(-1, 1)
        rays_dir = F.normalize(ref_pts - ref_org, p=2, dim=1)  # unit vector
        rays_depth = torch.linalg.norm(ref_pts - ref_org, dim=1, keepdim=True)

        # mutual obs points (down-sampled)
        mop_pts = ref_org + mop_depth.view(-1, 1) * rays_dir[mop_rays_idx]
        mop_pts_4d = torch.hstack((mop_pts, mop_ts.reshape(-1, 1)))

        if self.cfg_model['train_co_samples']:
            # TODO: balanced sampling of current observation samples (refer to uno)
            num_co_ray_samples_cls = self.cfg_model['num_co_ray_samples_cls']
            num_co_samples_cls = np.min((num_co_ray_samples_cls * num_rays, num_mop_samples_cls))
            rays_dir_broadcast = rays_dir.repeat(1, num_co_ray_samples_cls)
            rays_depth_broadcast = rays_depth.repeat(1, num_co_ray_samples_cls)  # [num_rays, num_co_ray_samples_cls]

            # TODO: stratified randomization
            co_free_depth_scale, co_occ_depth_scale = [], []
            delta = 1 / num_co_ray_samples_cls
            for i in range(num_co_ray_samples_cls):
                co_free_depth_scale_i = delta * i + torch.rand(num_rays) * delta
                co_occ_depth_scale_i = delta * i + torch.rand(num_rays) * delta
                co_free_depth_scale.append(co_free_depth_scale_i.view(-1, 1))
                co_occ_depth_scale.append(co_occ_depth_scale_i.view(-1, 1))
            co_free_depth_scale = torch.hstack(co_free_depth_scale)
            co_occ_depth_scale = torch.hstack(co_occ_depth_scale)

            # occupancy balanced sampling (free points)
            co_free_pts_depth = (co_free_depth_scale * rays_depth_broadcast).view(-1, 1)  # [ray_1, ... ray_1, ..., ray_n, ... ray_n]
            co_free_pts = ref_org + co_free_pts_depth * rays_dir_broadcast.view(-1, 3)
            co_free_rays_idx = torch.squeeze(rays_idx.repeat(1, num_co_ray_samples_cls).view(-1, 1))

            # occupancy balanced sampling (occupied points)
            occ_thrd = torch.full((num_rays, num_co_ray_samples_cls), self.cfg_model['occ_thrd'])
            co_occ_pts_depth = (rays_depth_broadcast + co_occ_depth_scale * occ_thrd).view(-1, 1)
            co_occ_pts = ref_org + co_occ_pts_depth * rays_dir_broadcast.view(-1, 3)
            co_occ_rays_idx = torch.squeeze(rays_idx.repeat(1, num_co_ray_samples_cls).view(-1, 1))

            # down-sample
            ds_co_idx_cls = random_sample(range(num_co_ray_samples_cls * num_rays), num_co_samples_cls)
            co_free_pts = co_free_pts[ds_co_idx_cls]
            co_free_pts_4d = torch.cat((co_free_pts, torch.full((len(co_free_pts), 1), ref_ts)), dim=1)
            co_occ_pts = co_occ_pts[ds_co_idx_cls]
            co_occ_pts_4d = torch.cat((co_occ_pts, torch.full((len(co_occ_pts), 1), ref_ts)), dim=1)

            # labels
            co_free_rays_idx = co_free_rays_idx[ds_co_idx_cls]
            co_occ_rays_idx = co_occ_rays_idx[ds_co_idx_cls]
            co_free_labels = torch.ones(len(co_free_pts), dtype=torch.int64)
            co_occ_labels = torch.ones(len(co_occ_pts), dtype=torch.int64) * 2
            co_free_confidence = torch.ones(len(co_free_pts), dtype=torch.float32)
            co_occ_confidence = torch.squeeze(torch.exp(-(co_occ_pts_depth[ds_co_idx_cls] - rays_depth_broadcast.view(-1, 1)[ds_co_idx_cls])))

            # concat and append to list
            co_rays_idx = torch.cat((co_free_rays_idx, co_occ_rays_idx), dim=0)
            co_pts_4d = torch.cat((co_free_pts_4d, co_occ_pts_4d), dim=0)
            co_labels = torch.cat((co_free_labels, co_occ_labels), dim=0)
            co_confidence = torch.cat((co_free_confidence, co_occ_confidence), dim=0)

            # shuffle
            shuffle_idx = torch.randperm(len(co_pts_4d))
            co_rays_idx = co_rays_idx[shuffle_idx]
            co_pts_4d = co_pts_4d[shuffle_idx]
            co_labels = co_labels[shuffle_idx]
            co_confidence = co_confidence[shuffle_idx]
            return [(ref_sd_tok, mutual_sd_toks), pcds_4d, (mop_rays_idx, mop_pts_4d, mop_labels, mop_confidence, co_rays_idx, co_pts_4d, co_labels, co_confidence)]
        else:
            return [(ref_sd_tok, mutual_sd_toks), pcds_4d, (mop_rays_idx, mop_pts_4d, mop_labels, mop_confidence)]