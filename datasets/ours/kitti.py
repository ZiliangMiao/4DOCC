import numpy as np
import yaml
import os
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from datasets.kitti_utils import load_poses, load_calib, load_files
from datasets.mos4d.augmentation import (
    shift_point_cloud,
    rotate_point_cloud,
    jitter_point_cloud,
    random_flip_point_cloud,
    random_scale_point_cloud,
    rotate_perturbation_point_cloud,
)


class KittiSequentialModule(LightningDataModule):
    """A Pytorch Lightning module for Sequential KITTI data; Contains train, valid, test data"""

    def __init__(self, cfg):
        super(KittiSequentialModule, self).__init__()
        self.cfg = cfg

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """Dataloader and iterators for training, validation and test data"""

        if self.cfg["MODE"] == "train" or self.cfg["MODE"] == "finetune":
            ########## Point dataset splits
            train_set = KittiSequentialDataset(self.cfg, split="train")
            val_set = KittiSequentialDataset(self.cfg, split="val")

            ########## Generate dataloaders and iterables
            self.train_loader = DataLoader(
                dataset=train_set,
                batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
                collate_fn=self.collate_fn,
                shuffle=self.cfg["DATA"]["SHUFFLE"],
                num_workers=self.cfg["TRAIN"]["NUM_WORKERS"],  # num of multi-processing
                pin_memory=True,
                drop_last=False,  # drop the samples left from full batch
                timeout=0,
            )
            self.train_iter = iter(self.train_loader)
            self.valid_loader = DataLoader(
                dataset=val_set,
                batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
                collate_fn=self.collate_fn,
                shuffle=False,
                num_workers=self.cfg["TRAIN"]["NUM_WORKERS"],
                pin_memory=True,
                drop_last=False,
                timeout=0,
            )
            self.valid_iter = iter(self.valid_loader)
            print("Loaded {:d} training and {:d} validation set.".format(len(train_set), len(val_set)))

        elif self.cfg["MODE"] == "test":
            test_set = KittiSequentialDataset(self.cfg, split="test")
            self.test_loader = DataLoader(
                dataset=test_set,
                batch_size=self.cfg["TEST"]["BATCH_SIZE"],
                collate_fn=self.collate_fn,
                shuffle=False,
                num_workers=self.cfg["TEST"]["NUM_WORKERS"],
                pin_memory=True,
                drop_last=False,
                timeout=0,
            )
            self.test_iter = iter(self.test_loader)
            print("Loaded {:d} testing set.".format(len(test_set)))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader

    @staticmethod
    def collate_fn(batch):  # define how to merge a list of samples to from a mini-batch samples
        meta = [item[0] for item in batch]
        num_curr_pts = [item[1] for item in batch]
        point_cloud = [item[2] for item in batch]
        mos_label = [item[3] for item in batch]
        return [meta, num_curr_pts, point_cloud, mos_label]

class KittiSequentialDataset(Dataset):
    """Semantic KITTI Dataset class"""
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.lidar_name = cfg["DATASET"]["SEKITTI"]["LIDAR_NAME"]
        self.dataset_path = self.cfg["DATASET"]["SEKITTI"]["PATH"]

        # Pose information
        self.transform = self.cfg["DATA"]["TRANSFORM"]
        self.poses = {}
        self.filename_poses = cfg["DATASET"]["SEKITTI"]["POSES"]

        # Semantic information
        self.semantic_config = yaml.safe_load(open(cfg["DATASET"]["SEKITTI"]["SEMANTIC_CONFIG_FILE"]))
        self.n_past_steps = self.cfg["DATA"]["N_PAST_STEPS"]  # use how many past scans, default = 10

        self.split = split
        if self.split == "train":
            self.sequences = self.cfg["DATASET"]["SEKITTI"]["TRAIN"]
        elif self.split == "val":
            self.sequences = self.cfg["DATASET"]["SEKITTI"]["VAL"]
        elif self.split == "test":
            self.sequences = self.cfg["DATASET"]["SEKITTI"]["TEST"]
        else:
            raise Exception("Split must be train/val/test")

        # Check if data and prediction frequency matches
        self.dt_pred = self.cfg["DATA"]["DELTA_T_PRED"]  # time resolution used for prediction
        self.dt_data = self.cfg["DATASET"]["SEKITTI"]["DELTA_T"]  # minimum time resolution of lidar scans
        assert (
            self.dt_pred >= self.dt_data
        ), "DELTA_T_PREDICTION needs to be larger than DELTA_T_DATA!"
        assert np.isclose(
            self.dt_pred / self.dt_data, round(self.dt_pred / self.dt_data), atol=1e-5
        ), "DELTA_T_PREDICTION needs to be a multiple of DELTA_T_DATA!"
        self.skip = round(self.dt_pred / self.dt_data)

        self.dataset_size = 0
        self.filenames = {}  # dict: maps the sequence number to a list of scans file path
        self.idx_mapper = {}  # dict: maps a dataset idx to a seq number and the index of the current scan
        sample_idx = 0  # sample index of idx_mapper (a counter that crosses different sequences)
        for seq_idx in self.sequences:
            seqstr = "{0:04d}".format(int(seq_idx))
            path_to_seq = os.path.join(self.dataset_path, seqstr)
            scan_path = os.path.join(path_to_seq, self.lidar_name)
            self.filenames[seq_idx] = load_files(scan_path)  # load all files path in a folder and sort
            if self.transform:
                if "SEKITTI" in {"SEKITTI", "KITTITRA", "KITTITRA_M", "APOLLO"}:
                    self.poses[seq_idx] = self.read_kitti_poses(path_to_seq)
                    # kitti pose: calib.txt (from lidar to camera), poses.txt (from current cam to previous cam)
                else:
                    self.poses[seq_idx] = self.read_poses(path_to_seq)
                assert len(self.poses[seq_idx]) == len(self.filenames[seq_idx])
            else:
                self.poses[seq_idx] = []

            # num of valid scans of current seq (we need 10 scans for prediction, so 00-08 are not valid), scan seqs begin at 09
            num_valid_scans = max(0, len(self.filenames[seq_idx]) - self.skip * (self.n_past_steps - 1))
            # Add to idx_mapper
            for idx in range(num_valid_scans):  # examp: sample index 00 -> scan index 09
                scan_idx = self.skip * (self.n_past_steps - 1) + idx
                self.idx_mapper[sample_idx] = (seq_idx, scan_idx)  # idx_mapper[sample_idx=0] = (seq_idx=00, scan_idx=09)
                sample_idx += 1
            self.dataset_size += num_valid_scans

    def __len__(self):  # return length of the whole dataset
        return self.dataset_size

    def __getitem__(self, sample_idx):  # define how to load each sample
        seq_idx, scan_idx = self.idx_mapper[sample_idx]
        # Load point clouds
        from_idx = scan_idx - self.skip * (self.n_past_steps - 1)  # included
        to_idx = scan_idx + 1  # not included
        scan_indices = list(range(from_idx, to_idx, self.skip))  # [from_idx, to_idx)
        scan_files = self.filenames[seq_idx][from_idx : to_idx : self.skip]
        if "SEKITTI" in {"SEKITTI", "KITTITRA", "KITTITRA_M", "APOLLO"}:
            scans_list = [self.read_kitti_point_cloud(f) for f in scan_files]
        else:
            scans_list = [self.read_point_cloud(f) for f in scan_files]
        for i, pcd in enumerate(scans_list):
            if self.transform:  # transform to current pose
                from_pose = self.poses[seq_idx][scan_indices[i]]
                to_pose = self.poses[seq_idx][scan_indices[-1]]
                pcd = self.transform_point_cloud(pcd, from_pose, to_pose)
            rela_time_idx = i - self.n_past_steps + 1  # -9, -8 ... 0
            rela_timestamp = round(rela_time_idx * self.dt_pred, 3)  # -0.9, 0.8 ... 0.0 (relative timestamps)
            scans_list[i] = self.timestamp_tensor(pcd, rela_timestamp)
        point_cloud = torch.cat(scans_list, dim=0)  # 4D point cloud: [x y z rela_time]

        # USE CURRENT SCAN LOSS:
        mos_labels_dir = os.path.join(self.dataset_path, str(seq_idx).zfill(4), "mos_labels")
        assert os.path.exists(mos_labels_dir)
        mos_labels_file = os.path.join(mos_labels_dir, str(scan_indices[-1]).zfill(6) + ".label")
        mos_labels = torch.tensor(np.fromfile(mos_labels_file, dtype=np.uint8))
        num_0 = torch.sum(mos_labels == 0)
        num_1 = torch.sum(mos_labels == 1)
        num_2 = torch.sum(mos_labels == 2)
        num_curr_pts = len(mos_labels)

        # USE HISTORY SCANS LOSS
        # mos_labels_dir = os.path.join(self.dataset_path, str(seq_idx).zfill(4), "mos_labels")
        # assert os.path.exists(mos_labels_dir)
        # mos_labels_list = []
        # for i in scan_indices:
        #     mos_labels_file = os.path.join(mos_labels_dir, str(i).zfill(6) + ".label")
        #     scan_mos_labels = torch.Tensor(np.fromfile(mos_labels_file, dtype=np.uint32).astype(np.float32)).long()
        #     scan_mos_labels = scan_mos_labels.reshape(-1, 1)
        #     rela_time_idx = i - self.n_past_steps + 1
        #     rela_timestamp = round(rela_time_idx * self.dt_pred, 3)
        #     mos_labels_list.append(self.timestamp_tensor(scan_mos_labels, rela_timestamp))
        # mos_labels = torch.cat(mos_labels_list, dim=0)

        meta = (seq_idx, scan_idx, scan_indices)
        # mask ego vehicle point
        if self.cfg["MODE"] == "train" or self.cfg["MODE"] == "finetune":
            if self.cfg["TRAIN"]["AUGMENTATION"]:  # will not change the mapping from point to label
                point_cloud = self.augment_data(point_cloud)
            if self.cfg["TRAIN"]["EGO_MASK"]:
                ego_mask = self.get_ego_mask(point_cloud)
                time_mask = point_cloud[:, -1] == 0.0
                point_cloud = point_cloud[~ego_mask]
                mos_labels = mos_labels[~ego_mask[time_mask]]  # all point clouds have mos labels (different from nusc)
                num_curr_pts = len(mos_labels)
        return [meta, num_curr_pts, point_cloud, mos_labels]  # [[index of current sequence, current scan, all scans], all scans, all labels]

    @staticmethod
    def get_ego_mask(points):  # mask the points of ego vehicle: x [-0.8, 0.8], y [-1.5, 2.5]
    # kitti car: length (4.084 m), width (1.730 m), height (1.562 m)
    # https://www.cvlibs.net/datasets/kitti/setup.php
        ego_mask = torch.logical_and(
            torch.logical_and(-0.760 - 0.8 <= points[:, 0], points[:, 0] <= 1.950 + 0.8),
            torch.logical_and(-0.850 - 0.2 <= points[:, 1], points[:, 1] <= 0.850 + 0.2),
        )
        return ego_mask

    def transform_point_cloud(self, past_point_clouds, from_pose, to_pose):
        transformation = torch.Tensor(np.linalg.inv(to_pose) @ from_pose)
        NP = past_point_clouds.shape[0]
        xyz1 = torch.hstack([past_point_clouds, torch.ones(NP, 1)]).T
        past_point_clouds = (transformation @ xyz1).T[:, :3]
        return past_point_clouds

    def augment_data(self, past_point_clouds):
        past_point_clouds = rotate_point_cloud(past_point_clouds)
        past_point_clouds = rotate_perturbation_point_cloud(past_point_clouds)
        past_point_clouds = jitter_point_cloud(past_point_clouds)
        past_point_clouds = shift_point_cloud(past_point_clouds)
        past_point_clouds = random_flip_point_cloud(past_point_clouds)
        past_point_clouds = random_scale_point_cloud(past_point_clouds)
        return past_point_clouds

    def read_kitti_point_cloud(self, filename):
        """Load point clouds from .bin file"""
        point_cloud = np.fromfile(filename, dtype=np.float32)
        point_cloud = torch.tensor(point_cloud.reshape((-1, 4)))
        point_cloud = point_cloud[:, :3]
        return point_cloud

    def read_point_cloud(self, filename):
        """Load point clouds from .bin file"""
        point_cloud = np.fromfile(filename, dtype=np.float32)
        point_cloud = torch.tensor(point_cloud.reshape((-1, 3)))
        return point_cloud

    def read_labels(self, lidarseg_file, save_mos_flag, mos_label_file=None):
        """Load moving object labels from .label file"""
        if os.path.isfile(lidarseg_file):
            labels = np.fromfile(lidarseg_file, dtype=np.uint32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF  # Mask semantics in lower half
            mapped_labels = copy.deepcopy(labels)
            for k, v in self.semantic_config["learning_map"].items():
                mapped_labels[labels == k] = v
            if save_mos_flag:
                assert mos_label_file != None
                mapped_labels.tofile(mos_label_file)
            selected_labels = torch.Tensor(mapped_labels.astype(np.float32)).long()
            selected_labels = selected_labels.reshape((-1, 1))
            return selected_labels
        else:
            return torch.Tensor(1, 1).long()

    @staticmethod
    def timestamp_tensor(tensor, time):
        """Add time as additional column to tensor"""
        n_points = tensor.shape[0]
        time = time * torch.ones((n_points, 1))
        timestamped_tensor = torch.hstack([tensor, time])
        return timestamped_tensor

    def read_kitti_poses(self, path_to_seq):
        pose_file = os.path.join(path_to_seq, self.filename_poses)
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

    def read_poses(self, path_to_seq):
        pose_file = os.path.join(path_to_seq, self.filename_poses)
        poses = np.array(load_poses(pose_file))  # from current vehicle frame to global frame
        return poses