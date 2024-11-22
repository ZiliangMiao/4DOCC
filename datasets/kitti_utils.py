#!/usr/bin/env python3
# @file      load_files.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved

import os
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class KittiDataloader(LightningDataModule):
    """A Pytorch Lightning module for Sequential KITTI data; Contains train, valid, test data"""

    def __init__(self, cfg_model, train_set, val_set, train_flag: bool):
        super(KittiDataloader, self).__init__()
        self.cfg_model = cfg_model
        self.train_set = train_set
        self.val_set = val_set
        self.train_flag = train_flag
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_iter = None
        self.val_iter = None
        self.test_iter = None

    def prepare_data(self):
        pass

    @staticmethod
    def collate_fn(batch):  # define how to merge a list of samples to from a mini-batch samples
        meta_info = [item[0] for item in batch]  # meta info: (sd_tok, num_rays, num_bg_samples); (sd_toks)
        pcds_4d = [item[1] for item in batch]
        samples = [item[2] for item in batch]  # bg samples; mos labels
        return [meta_info, pcds_4d, samples]

    def setup(self, stage=None):
        """Dataloader and iterators for training, validation and test data"""
        # train_data_pct = self.cfg_model["downsample_pct"] / 100
        train_loader = DataLoader(
            dataset=self.train_set,
            batch_size=self.cfg_model["batch_size"],
            collate_fn=self.collate_fn,
            num_workers=self.cfg_model["num_workers"],  # num of multi-processing
            shuffle=self.cfg_model["shuffle"],
            pin_memory=True,
            drop_last=True,  # drop the samples left from full batch
            timeout=0,
            # sampler=sampler.WeightedRandomSampler(weights=torch.ones(len(train_set)),
            #                                       num_samples=int(train_data_pct * len(train_set))),
        )
        val_loader = DataLoader(
            dataset=self.val_set,
            batch_size=self.cfg_model["batch_size"],
            collate_fn=self.collate_fn,
            num_workers=self.cfg_model["num_workers"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,  # TODO: may be a bug for uno or occ4d test
            timeout=0,
        )

        if self.train_flag:
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.train_iter = iter(self.train_loader)
            self.val_iter = iter(self.val_loader)
            print("Loaded {:d} training and {:d} validation samples.".format(len(self.train_set), len(self.val_set)))
        else:  # test (use validation set)
            self.test_loader = val_loader
            self.test_iter = iter(self.test_loader)
            print("Loaded {:d} test samples.".format(len(self.val_set)))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


def load_poses(pose_path):
    """Load ground truth poses (T_w_cam0) from file.
    Args:
      pose_path: (Complete) filename for the pose file
    Returns:
      A numpy array of size nx4x4 with n poses as 4x4 transformation
      matrices
    """
    # Read and parse the poses
    poses = []
    try:
        if ".txt" in pose_path:
            with open(pose_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=" ")

                    if len(T_w_cam0) == 12:
                        T_w_cam0 = T_w_cam0.reshape(3, 4)
                        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    elif len(T_w_cam0) == 16:
                        T_w_cam0 = T_w_cam0.reshape(4, 4)
                    poses.append(T_w_cam0)
        else:
            poses = np.load(pose_path)["arr_0"]

    except FileNotFoundError:
        print("Ground truth poses are not avaialble.")

    return np.array(poses)


def load_calib(calib_path):
    """Load calibrations (T_cam_velo) from file."""
    # Read and parse the calibrations
    T_cam_velo = []
    try:
        with open(calib_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Tr:" in line:
                    line = line.replace("Tr:", "")
                    T_cam_velo = np.fromstring(line, dtype=float, sep=" ")
                    T_cam_velo = T_cam_velo.reshape(3, 4)
                    T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))

    except FileNotFoundError:
        print("Calibrations are not avaialble.")

    return np.array(T_cam_velo)


def read_kitti_poses(path_to_seq):
    pose_file = os.path.join(path_to_seq, "poses.txt")
    calib_file = os.path.join(path_to_seq, "calib.txt")
    poses = np.array(load_poses(pose_file))
    inv_frame0 = np.linalg.inv(poses[0])

    # load calibrations
    T_cam_velo = load_calib(calib_file)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)

    # convert kitti poses from camera coord to LiDAR coord
    new_poses = []
    for pose in poses:
        new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
    poses = np.array(new_poses)
    return poses


def add_timestamp(tensor, ts):
    """Add time as additional column to tensor"""
    n_points = tensor.shape[0]
    ts = ts * torch.ones((n_points, 1))
    tensor_with_ts = torch.hstack([tensor, ts])
    return tensor_with_ts


def load_files(folder, dataset_root, seq_idx):
    """Load all files path in a folder and sort."""
    # file_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(folder)) for f in fn]
    # file_paths.sort()
    file_paths = []
    label_paths = []
    for root, dirs, files in os.walk(os.path.expanduser(folder)):
        for f in files:
            file_paths.append(os.path.join(root, f))
            scan_idx = f.split(".")[0]
            label_dir = os.path.join(dataset_root, str(seq_idx).zfill(2), "labels")
            assert os.path.exists(label_dir)
            label_file = os.path.join(label_dir, str(scan_idx + ".label"))
            label_paths.append(label_file)
    file_paths.sort()
    label_paths.sort()
    return file_paths, label_paths


def get_ego_mask(pcd):  # mask the points of ego vehicle: x [-0.8, 0.8], y [-1.5, 2.5]
    # kitti car: length (4.084 m), width (1.730 m), height (1.562 m)
    # https://www.cvlibs.net/datasets/kitti/setup.php
    ego_mask = torch.logical_and(
        torch.logical_and(-0.760 - 0.8 <= pcd[:, 0], pcd[:, 0] <= 1.950 + 0.8),
        torch.logical_and(-0.850 - 0.2 <= pcd[:, 1], pcd[:, 1] <= 0.850 + 0.2),
    )
    return ego_mask


def get_outside_scene_mask(pcd, scene_bbox, mask_z: bool, upper_bound: bool):  # TODO: note, preprocessing use both <=
    if upper_bound: # TODO: for mop pre-processing, keep ray index unchanged
        inside_scene_mask = torch.logical_and(scene_bbox[0] <= pcd[:, 0], pcd[:, 0] <= scene_bbox[3])
        inside_scene_mask = torch.logical_and(inside_scene_mask,
                                              torch.logical_and(scene_bbox[1] <= pcd[:, 1], pcd[:, 1] <= scene_bbox[4]))
    else: # TODO: for uno, avoid index out of bound
        inside_scene_mask = torch.logical_and(scene_bbox[0] <= pcd[:, 0], pcd[:, 0] < scene_bbox[3])
        inside_scene_mask = torch.logical_and(inside_scene_mask,
                                              torch.logical_and(scene_bbox[1] <= pcd[:, 1], pcd[:, 1] < scene_bbox[4]))
    if mask_z:
        inside_scene_mask = torch.logical_and(inside_scene_mask,
                                              torch.logical_and(scene_bbox[2] <= pcd[:, 2], pcd[:, 2] < scene_bbox[5]))
    return ~inside_scene_mask


def transform_point_cloud(past_point_clouds, from_pose, to_pose):
    transformation = torch.Tensor(np.linalg.inv(to_pose) @ from_pose)
    NP = past_point_clouds.shape[0]
    xyz1 = torch.hstack([past_point_clouds, torch.ones(NP, 1)]).T
    past_point_clouds = (transformation @ xyz1).T[:, :3]
    return past_point_clouds


def load_mos_labels(filename):
    """Load moving object labels from .label file"""
    semantic_labels = np.fromfile(filename, dtype=np.int32).reshape((-1))
    semantic_labels = semantic_labels & 0xFFFF  # Mask semantics in lower half
    mos_labels = np.ones_like(semantic_labels)
    mos_labels[semantic_labels <= 1] = 0  # Unlabeled (0), outlier (1)
    mos_labels[semantic_labels > 250] = 2  # Moving
    mos_labels = torch.tensor(mos_labels.astype(dtype=np.uint8).reshape(-1))
    return mos_labels
