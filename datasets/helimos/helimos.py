import copy
import os

import torch
import numpy as np


def read_kitti_point_cloud(filename):
    """Load point clouds from .bin file"""
    point_cloud = np.fromfile(filename, dtype=np.float32)
    point_cloud = torch.tensor(point_cloud.reshape((-1, 4)))
    point_cloud = point_cloud[:, :3]
    return point_cloud


def read_labels(lidarseg_file):
    """Load moving object labels from .label file"""
    if os.path.isfile(lidarseg_file):
        labels = np.fromfile(lidarseg_file, dtype=np.int32)
        # labels = labels.reshape((-1))
        # labels = labels & 0xFFFF  # Mask semantics in lower half
        return torch.tensor(labels)
    else:
        return torch.Tensor(1, 1).long()


if __name__ == "__main__":
    scan_idx = '011662'
    label_folder = '/home/user/Datasets/helimos/Deskewed_LiDAR/Ouster/labels'
    label_file = os.path.join(label_folder, f'{scan_idx}.label')
    pcd_folder = '/home/user/Datasets/helimos/Deskewed_LiDAR/Ouster/velodyne'
    pcd_file = os.path.join(pcd_folder, f'{scan_idx}.bin')

    pcd = read_kitti_point_cloud(pcd_file)
    label = read_labels(label_file)

    mov_idx = torch.where(label == 251)
    sta_idx = torch.where(label == 9)
    unk_idx = torch.where(label == 0)

    a = 1
