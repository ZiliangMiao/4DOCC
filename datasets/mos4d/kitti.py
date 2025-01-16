import numpy as np
import os
import torch
from random import sample as random_sample
from torch.utils.data import Dataset
from utils.augmentation import augment_pcds
from datasets.kitti_utils import load_files, read_kitti_poses, get_ego_mask, add_timestamp, get_outside_scene_mask, load_mos_labels, transform_point_cloud


class KittiMOSDataset(Dataset):
    """Semantic KITTI Dataset class"""
    def __init__(self, cfg_model, cfg_dataset, split):
        self.cfg_model = cfg_model
        self.cfg_dataset = cfg_dataset['sekitti']
        self.root = self.cfg_dataset['root']
        self.lidar = self.cfg_dataset['lidar']
        self.split = split
        self.sequence = self.cfg_dataset[self.split]  # "train": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]; "val": [8]; "test": [8]

        # get all scans in a kitti sequence
        seq_to_scans = {}  # dict: maps the sequence number to a list of scans file path
        seq_to_labels = {}
        seq_to_poses = {}
        for seq_idx in self.sequence:
            seqstr = "{0:02d}".format(int(seq_idx))
            path_to_seq = os.path.join(self.root, seqstr)
            scan_files, label_files = load_files(os.path.join(path_to_seq, self.lidar), self.root, seq_idx)  # load all scans in a seq
            seq_poses = read_kitti_poses(path_to_seq)

            # get input scans in a sequence
            cnt = 0
            skip_cnt = self.cfg_model['n_skip']
            input_scans = []
            labels = []
            poses = []
            seq_to_scans[seq_idx] = []
            seq_to_labels[seq_idx] = []
            seq_to_poses[seq_idx] = []
            for scan_idx in range(len(scan_files)):
                if cnt < self.cfg_model['n_input']:
                    if skip_cnt == self.cfg_model['n_skip']:
                        input_scans.append(scan_files[scan_idx])
                        labels.append(label_files[scan_idx])
                        poses.append(seq_poses[scan_idx])
                        cnt += 1
                        skip_cnt = 0

                        # add to dict
                        if len(input_scans) == len(labels) == self.cfg_model['n_input']:
                            seq_to_scans[seq_idx].append(input_scans)
                            seq_to_labels[seq_idx].append(labels)
                            seq_to_poses[seq_idx].append(poses)
                            input_scans = []
                            labels = []
                            poses = []
                            cnt = 0
                            skip_cnt = self.cfg_model['n_skip']
                    else:
                        skip_cnt += 1
                        continue
                else:
                    continue

        # merge scans and labels
        self.samples_scans = []
        self.samples_labels = []
        self.samples_poses = []
        for seq_idx in self.sequence:
            self.samples_scans.extend(seq_to_scans[seq_idx])
            self.samples_labels.extend(seq_to_labels[seq_idx])
            self.samples_poses.extend(seq_to_poses[seq_idx])

        # down-sampling
        sample_idx_list = range(len(self.samples_scans))
        if split == 'train':
            # dataset down-sampling: sequence level or sample level
            if self.cfg_model["downsample_level"] == "sequence":
                train_data_pct = self.cfg_model["downsample_pct"] / 100
                ds_samples_idx = random_sample(sample_idx_list, int(len(sample_idx_list) * train_data_pct))
                self.samples_scans = [self.samples_scans[i] for i in ds_samples_idx]
                self.samples_labels = [self.samples_labels[i] for i in ds_samples_idx]
                self.samples_poses = [self.samples_poses[i] for i in ds_samples_idx]

    def __len__(self):
        assert len(self.samples_labels) == len(self.samples_scans)
        return len(self.samples_scans)

    def __getitem__(self, batch_idx):
        scan_files = self.samples_scans[batch_idx]
        label_files = self.samples_labels[batch_idx]
        poses = self.samples_poses[batch_idx]
        ref_scan_idx = scan_files[-1].split(".")[0][-6:]

        # get transformed point clouds
        time_idx = -self.cfg_model['n_input'] + 1
        pcds_4d_list = []  # 4D Point Cloud (relative timestamp)
        for scan_idx, scan_file in enumerate(scan_files):
            pcd = np.fromfile(scan_file, dtype=np.float32)
            pcd = torch.tensor(pcd.reshape((-1, 4)))[:, :3]
            if self.cfg_model["transform"]:
                curr_pose = poses[scan_idx]
                ref_pose = poses[-1] # TODO: last one is reference scan
                pcd_ref = transform_point_cloud(pcd, curr_pose, ref_pose)

                # add timestamp (0, -1, -2, ...)
                pcd_4d_ref = add_timestamp(pcd_ref, time_idx)
                pcds_4d_list.append(pcd_4d_ref)
                time_idx += 1
        pcds_4d = torch.cat(pcds_4d_list, dim=0).float()  # 4D point cloud: [x y z ref_ts]
        mos_labels = load_mos_labels(label_files[-1])

        # ego vehicle mask / outside scene mask
        ref_time_mask = pcds_4d[:, -1] == 0
        valid_mask = torch.squeeze(torch.full((len(pcds_4d), 1), True))
        if self.cfg_model['ego_mask']:
            ego_mask = get_ego_mask(pcds_4d)
            valid_mask = torch.logical_and(valid_mask, ~ego_mask)
        if self.cfg_model['outside_scene_mask']:
            outside_scene_mask = get_outside_scene_mask(pcds_4d, self.cfg_model["scene_bbox"],
                                                        self.cfg_model['outside_scene_mask_z'],
                                                        self.cfg_model['outside_scene_mask_ub'])
            valid_mask = torch.logical_and(valid_mask, ~outside_scene_mask)
        pcds_4d = pcds_4d[valid_mask]
        mos_labels = mos_labels[valid_mask[ref_time_mask]]

        # data augmentation: will not change the order of points
        if self.split == 'train' and self.cfg_model["augmentation"]:
            pcds_4d = augment_pcds(pcds_4d)
        return [ref_scan_idx, pcds_4d, mos_labels]  # [[index of current sequence, current scan, all scans], all scans, all labels]