import numpy as np
import os
import torch
from torch.utils.data import Dataset
from datasets.kitti_utils import read_kitti_poses, add_timestamp, transform_point_cloud
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class KittiMOSDataset(Dataset):
    """Semantic KITTI Dataset class"""
    def __init__(self, cfg_model, cfg_dataset, cfg_test, seq_idx):
        self.cfg_test = cfg_test
        self.cfg_model = cfg_model
        self.cfg_dataset = cfg_dataset['sekitti']
        self.root = self.cfg_dataset['root']
        self.lidar = self.cfg_dataset['lidar']

        # get all scans in a kitti sequence
        seqstr = "{0:02d}".format(int(seq_idx))
        path_to_seq = os.path.join(self.root, seqstr)
        scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(os.path.join(path_to_seq, self.lidar))) for f in fn]
        scan_files.sort()
        seq_poses = read_kitti_poses(path_to_seq)

        # get input scans in a sequence
        ref_to_input_seq = {}
        ref_to_input_pose = {}
        input_seq = []
        poses = []
        n_input = self.cfg_model['n_input']
        n_skip = self.cfg_model['n_skip']
        skip_cnt = 0
        for ref_idx in range(len(scan_files)):
            input_seq.append(scan_files[ref_idx])
            poses.append(seq_poses[ref_idx])
            curr_idx = ref_idx
            for i in range(n_input-1):
                while skip_cnt <= n_skip:
                    curr_idx -= 1
                    skip_cnt += 1
                if curr_idx >=0:
                    input_seq.append(scan_files[curr_idx])
                    poses.append(seq_poses[curr_idx])
                    skip_cnt = 0
                else:
                    skip_cnt = 0
                    break
            ref_to_input_seq[ref_idx] = input_seq[::-1]
            ref_to_input_pose[ref_idx] = poses[::-1]
            input_seq = []
            poses = []
        self.input_scans = list(ref_to_input_seq.values())
        self.input_poses = list(ref_to_input_pose.values())

    def __len__(self):
        return len(self.input_scans)

    def __getitem__(self, batch_idx):
        scan_files = self.input_scans[batch_idx]
        poses = self.input_poses[batch_idx]
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
        return [ref_scan_idx, pcds_4d]  # [[index of current sequence, current scan, all scans], all scans]


class KittiDataloader(LightningDataModule):
    """A Pytorch Lightning module for Sequential KITTI data; Contains train, valid, test data"""

    def __init__(self, cfg_model, test_set):
        super(KittiDataloader, self).__init__()
        self.cfg_model = cfg_model
        self.test_set = test_set
        self.test_loader = None
        self.test_iter = None

    def prepare_data(self):
        pass

    @staticmethod
    def collate_fn(batch):  # define how to merge a list of samples to from a mini-batch samples
        meta_info = [item[0] for item in batch]  # meta info: (sd_tok, num_rays, num_bg_samples); (sd_toks)
        pcds_4d = [item[1] for item in batch]
        return [meta_info, pcds_4d]

    def setup(self, stage=None):
        """Dataloader and iterators for training, validation and test data"""
        test_loader = DataLoader(
            dataset=self.test_set,
            batch_size=self.cfg_model["batch_size"],
            collate_fn=self.collate_fn,
            num_workers=self.cfg_model["num_workers"],  # num of multi-processing
            shuffle=False,
            pin_memory=True,
            drop_last=True,  # drop the samples left from full batch
            timeout=0,
        )
        self.test_loader = test_loader
        self.test_iter = iter(self.test_loader)
        print("Loaded {:d} test samples.".format(len(self.test_set)))

    def test_dataloader(self):
        return self.test_loader