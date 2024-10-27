import os
import time
from collections import defaultdict, Counter
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

        # ref rays params
        num_rays = len(ref_pts)
        rays_idx = torch.tensor(range(num_rays)).view(-1, 1)
        rays_dir = F.normalize(ref_pts - ref_org, p=2, dim=1)  # unit vector
        rays_depth = torch.linalg.norm(ref_pts - ref_org, dim=1, keepdim=True)

        # data augmentation: will not change the order of points
        if self.split == 'train' and self.cfg_model["augmentation"]:
            pcds_4d = augment_pcds(pcds_4d)

        # load mutual observation samples
        mutual_obs_folder = os.path.join(self.nusc.dataroot, "mutual_obs_labels", self.nusc.version)
        mutual_obs_meta = os.path.join(mutual_obs_folder, ref_sd_tok + "_key_meta.bin")
        mutual_sd_toks = nusc_utils.get_mutual_sd_toks_dict(self.nusc, [sample_tok], self.cfg_model)[sample_tok]
        mo_rays_idx = os.path.join(mutual_obs_folder, ref_sd_tok + "_rays_idx.bin")
        mo_depth = os.path.join(mutual_obs_folder, ref_sd_tok + "_depth.bin")
        mo_labels = os.path.join(mutual_obs_folder, ref_sd_tok + "_labels.bin")
        mo_confidence = os.path.join(mutual_obs_folder, ref_sd_tok + "_confidence.bin")
        mutual_obs_meta = np.fromfile(mutual_obs_meta, dtype=np.uint32).reshape(-1, 2).astype(np.int64)
        mo_rays_idx = np.fromfile(mo_rays_idx, dtype=np.uint16).astype(np.int64)
        mo_depth = torch.from_numpy(np.fromfile(mo_depth, dtype=np.float16).astype(np.float32))
        mo_labels = torch.from_numpy(np.fromfile(mo_labels, dtype=np.uint8).astype(np.int64))
        mo_confidence = torch.from_numpy(np.fromfile(mo_confidence, dtype=np.float16).astype(np.float32))
        mutual_obs_sensors_indices = np.concatenate([np.ones(meta[1], dtype=np.int64) * meta[0] for meta in mutual_obs_meta])
        mutual_obs_sensors_timestamps = [(self.nusc.get('sample_data', sd_tok)['timestamp'] -
                                      self.nusc.get('sample_data', ref_sd_tok)['timestamp']) / 1e6 for sd_tok in mutual_sd_toks]
        mo_ts = torch.tensor(mutual_obs_sensors_timestamps)[mutual_obs_sensors_indices]

        # only select background mo samples (unknown occupancy state)
        if self.cfg_model['train_bg_mop_samples']:
            # mask mutual obs points which depth less than ray depth
            bg_mask = mo_depth >= torch.squeeze(rays_depth)[mo_rays_idx]
            # update valid bg samples
            mo_rays_idx = mo_rays_idx[bg_mask]
            mo_depth = mo_depth[bg_mask]
            mo_ts = mo_ts[bg_mask]
            mo_labels = mo_labels[bg_mask]
            mo_confidence = mo_confidence[bg_mask]

        ################################################################################################################
        # TODO: ray level sampling
        # mo_dict = defaultdict(list)
        # for mo_ray_idx in torch.unique(mo_rays_idx):
        #     ray_samples_mask = mo_rays_idx == mo_ray_idx
        #     mo_labels_i = mo_labels[ray_samples_mask]
        #     mo_depth_i = mo_depth[ray_samples_mask]
        #     mo_ts_i = mo_ts[ray_samples_mask]
        #     mo_confidence_i = mo_confidence[ray_samples_mask]
        #
        #     # ray: mo cls (free and occ) samples
        #     ray_mo_free_idx = torch.where(mo_labels_i == 1)[0]
        #     ray_mo_occ_idx = torch.where(mo_labels_i == 2)[0]
        #     num_ray_mo_samples_cls = np.min((len(ray_mo_free_idx), len(ray_mo_occ_idx)))
        #
        #     # ray: free and occ down-sample
        #     ds_ray_mo_free_idx = ray_mo_free_idx[random_sample(range(len(ray_mo_free_idx)), num_ray_mo_samples_cls)]
        #     ds_ray_mo_occ_idx = ray_mo_occ_idx[random_sample(range(len(ray_mo_occ_idx)), num_ray_mo_samples_cls)]
        #
        #     # ray: mo unk samples
        #     ray_mo_unk_idx = torch.where(mo_labels_i == 0)[0]
        #     num_ray_mo_samples_unk = np.min((len(ray_mo_unk_idx),
        #                                      int(num_ray_mo_samples_cls * self.cfg_model['unk_samples_pct'] / 100)))
        #     ds_ray_mo_unk_idx = ray_mo_unk_idx[random_sample(range(len(ray_mo_unk_idx)), num_ray_mo_samples_unk)]
        #
        #     # down-sampled mo samples
        #     ds_ray_mo_samples_idx = torch.cat([ds_ray_mo_unk_idx, ds_ray_mo_free_idx, ds_ray_mo_occ_idx])
        #     ds_mo_depth_i = mo_depth_i[ds_ray_mo_samples_idx]
        #     ds_mo_ts_i = mo_ts_i[ds_ray_mo_samples_idx]
        #     ds_mo_labels_i = mo_labels_i[ds_ray_mo_samples_idx]
        #     ds_mo_confidence_i = mo_confidence_i[ds_ray_mo_samples_idx]
        #     mo_dict[mo_ray_idx] = [ds_mo_depth_i, ds_mo_ts_i, ds_mo_labels_i, ds_mo_confidence_i]
        ################################################################################################################

        # TODO: balanced down-sampling for training (still balance sampling when testing, cost a lot of gpu memory)
        mo_unk_idx = torch.where(mo_labels == 0)[0]
        mo_free_idx = torch.where(mo_labels == 1)[0]
        mo_occ_idx = torch.where(mo_labels == 2)[0]

        # downsample occ and free class
        num_mo_free = len(mo_free_idx)
        num_mo_occ = len(mo_occ_idx)
        num_mo_samples_cls = np.min((num_mo_free, num_mo_occ))
        ds_mo_free_idx = mo_free_idx[random_sample(range(num_mo_free), num_mo_samples_cls)]
        ds_mo_occ_idx = mo_occ_idx[random_sample(range(num_mo_occ), num_mo_samples_cls)]

        # down-sample unk (a certain percentage of num_mo_samples_cls)
        num_mo_unk = len(mo_unk_idx)
        num_mo_samples_unk = np.min((num_mo_unk, int(num_mo_samples_cls * self.cfg_model['unk_samples_pct'] / 100)))
        ds_mo_unk_idx = mo_unk_idx[random_sample(range(num_mo_unk), num_mo_samples_unk)]

        # update down-sampled mutual obs samples
        ds_mo_sample_indices = torch.cat([ds_mo_unk_idx, ds_mo_free_idx, ds_mo_occ_idx])
        mo_rays_idx = mo_rays_idx[ds_mo_sample_indices]
        mo_depth = mo_depth[ds_mo_sample_indices]
        mo_ts = mo_ts[ds_mo_sample_indices]
        mo_labels = mo_labels[ds_mo_sample_indices]
        mo_confidence = mo_confidence[ds_mo_sample_indices]

        # mutual obs points (down-sampled)
        mo_pts = ref_org + mo_depth.view(-1, 1) * rays_dir[mo_rays_idx]
        mo_pts_4d = torch.hstack((mo_pts, mo_ts.reshape(-1, 1)))

        # TODO: ray level average
        # t_s = time.perf_counter()
        # count points num on rays
        num_mo_sample_per_ray = Counter(mo_rays_idx)
        # average
        samples_weight_by_ray = np.ones(num_rays)
        samples_weight_by_ray[list(num_mo_sample_per_ray.keys())] = list(num_mo_sample_per_ray.values())
        samples_weight_by_ray = (1. / samples_weight_by_ray).astype(np.float32)
        mo_confidence = mo_confidence * samples_weight_by_ray[mo_rays_idx]
        # t_e = time.perf_counter()
        # print(t_e - t_s)


        if self.cfg_model['train_co_samples']:
            # TODO: balanced sampling of current observation samples (refer to uno)
            num_co_ray_samples_cls = self.cfg_model['num_co_ray_samples_cls']
            num_co_samples_cls = np.min((np.min((num_co_ray_samples_cls * num_rays, num_mo_samples_cls)), self.cfg_model['max_mop_samples_cls']))
            rays_dir_broadcast = rays_dir.repeat(1, num_co_ray_samples_cls)
            rays_depth_broadcast = rays_depth.repeat(1, num_co_ray_samples_cls)  # [num_rays, num_co_ray_samples_cls]

            # stratified randomization
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

            # # TODO: avg. on rays, update confidence of mo-co samples (too slow)
            # mo_rays_idx_unq, mo_inv_idx, mo_cnt = torch.unique(mo_rays_idx, return_inverse=True, return_counts=True)
            # mo_rays_avg_weight = 1 / mo_cnt.float()
            # mo_confidence = mo_confidence * mo_rays_avg_weight[mo_inv_idx]
            #
            # co_rays_idx_unq, co_inv_idx, co_cnt = torch.unique(co_rays_idx, return_inverse=True, return_counts=True)
            # co_rays_avg_weight = 1 / co_cnt.float()
            # co_confidence = co_confidence * co_rays_avg_weight[co_inv_idx]
            #
            # # shuffle
            # shuffle_idx = torch.randperm(len(co_pts_4d))
            # co_rays_idx = co_rays_idx[shuffle_idx]
            # co_pts_4d = co_pts_4d[shuffle_idx]
            # co_labels = co_labels[shuffle_idx]
            # co_confidence = co_confidence[shuffle_idx]
            return [(ref_sd_tok, mutual_sd_toks), pcds_4d, (mo_rays_idx, mo_pts_4d, mo_labels, mo_confidence, co_rays_idx, co_pts_4d, co_labels, co_confidence)]
        else:
            return [(ref_sd_tok, mutual_sd_toks), pcds_4d, (mo_rays_idx, mo_pts_4d, mo_labels, mo_confidence)]