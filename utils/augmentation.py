#!/usr/bin/env python3
# @file      augmentation.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved

import torch
import numpy as np


def augment_pcds(pcds):
    pcds = rotate_point_cloud(pcds)
    pcds = rotate_perturbation_point_cloud(pcds)
    pcds = jitter_point_cloud(pcds)
    pcds = shift_point_cloud(pcds)
    pcds = random_flip_point_cloud(pcds)
    pcds = random_scale_point_cloud(pcds)
    return pcds


def rotate_point_cloud(pcd):
    """Randomly rotate the point clouds to augument the dataset
    Input:
      Nx4 array, original point cloud
    Return:
      Nx4 array, rotated point cloud
    """
    rotation_angle = np.random.uniform() * 2 * torch.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = torch.Tensor([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])
    pcd[:, :3] = pcd[:, :3] @ rotation_matrix
    return pcd


def rotate_perturbation_point_cloud(pcd, angle_sigma=0.06, angle_clip=0.18):
    """Randomly perturb the point clouds by small rotations
     Input:
      Nx4 array, original point cloud
    Return:
      Nx4 array, rotated point cloud
    """
    angles = torch.clip(angle_sigma * torch.randn(3), -angle_clip, angle_clip)
    Rx = torch.Tensor(
        [
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])],
        ]
    )
    Ry = torch.Tensor(
        [
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])],
        ]
    )
    Rz = torch.Tensor(
        [
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1],
        ]
    )
    rotation_matrix = Rz @ Ry @ Rx
    pcd[:, :3] = pcd[:, :3] @ rotation_matrix
    return pcd


def jitter_point_cloud(pcd, sigma=0.01, clip=0.05):
    """Randomly jitter points. jittering is per point.
      Input:
      Nx4 array, original point cloud
    Return:
      Nx4 array, jittered point cloud
    """
    assert clip > 0
    N, _ = pcd.shape
    jitter = torch.clip(sigma * torch.randn(N, 3), -1 * clip, clip)
    pcd[:, :3] += jitter

    return pcd


def shift_point_cloud(pcd, shift_range=0.1):
    """Randomly shift point cloud. Shift is per point cloud.
    Input:
      Nx4 array, original point cloud
    Return:
      Nx4 array, shifted point cloud
    """
    shifts = np.random.uniform(-shift_range, shift_range, (1, 3))
    pcd[:, :3] += shifts
    return pcd


def random_flip_point_cloud(pcd):
    """Randomly flip the point cloud. Flip is per point cloud.
    Input:
      Nx4 array, original point cloud
    Return:
      Nx4 array, flipped point cloud
    """
    if np.random.random() > 0.5:
        pcd[:, 0] *= -1
    if np.random.random() > 0.5:
        pcd[:, 1] *= -1
    return pcd


def random_scale_point_cloud(pcd, scale_low=0.95, scale_high=1.05):
    """Randomly scale the point cloud. Scale is per point cloud.
    Input:
        BxNx3 array, original batch of point clouds
    Return:
        BxNx3 array, scaled batch of point clouds
    """
    scales = np.random.uniform(scale_low, scale_high, 1)
    pcd[:, :3] *= scales
    return pcd
