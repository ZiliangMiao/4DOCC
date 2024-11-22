import copy
import os


import torch
import numpy as np
from random import sample as random_sample
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from datasets.kitti_utils import load_poses, load_calib
from utils.augmentation import augment_pcds


def get_ref_from_current_transform(poses, ref_scan_idx, curr_scan_idx):
    ref_pose = poses[ref_scan_idx]
    curr_pose = poses[curr_scan_idx]
    return np.linalg.inv(ref_pose) @ curr_pose


def transfrom_points(points, transform):
    assert points.shape[1] >= 3
    pts_homo = np.hstack([points[:, :3], np.ones((points.shape[0], 1))]).T
    pts_transformed = (transform @ pts_homo).T[:, :3]
    if points.shape[1] > 3: # more than [x,y,z]
        pts_transformed = np.hstack([pts_transformed, points[:, 3:]])
    return (transform @ pts_homo).T[:, :3]


def get_ego_mask(pcd):
    # nuscenes car: length (4.084 m), width (1.730 m), height (1.562 m)
    # https://forum.nuscenes.org/t/dimensions-of-the-ego-vehicle-used-to-gather-data/550
    ego_mask = torch.logical_and(
        torch.logical_and(-0.865 <= pcd[:, 0], pcd[:, 0] <= 0.865),
        torch.logical_and(-1.5 <= pcd[:, 1], pcd[:, 1] <= 2.5),
    )
    return ego_mask


def get_outside_scene_mask(pcd, scene_bbox):
    inside_scene_mask = torch.logical_and(
        torch.logical_and(scene_bbox[0] <= pcd[:, 0], pcd[:, 0] <= scene_bbox[3]),
        torch.logical_and(scene_bbox[1] <= pcd[:, 1], pcd[:, 1] <= scene_bbox[4]),
        # torch.logical_and(scene_bbox[2] <= pcd[:, 2], pcd[:, 2] <= scene_bbox[5])
    )
    return ~inside_scene_mask


def filter_ego_and_outside_points(pcd, cfg_model):
    # filter ego points and outside scene bbox points
    valid_mask = torch.squeeze(torch.full((len(pcd), 1), True))
    if cfg_model['ego_mask']:
        ego_mask = get_ego_mask(pcd)
        valid_mask = torch.logical_and(valid_mask, ~ego_mask)
    if cfg_model['outside_scene_mask']:
        outside_scene_mask = get_outside_scene_mask(pcd, cfg_model["scene_bbox"])
        valid_mask = torch.logical_and(valid_mask, ~outside_scene_mask)
    return valid_mask


def add_timestamp(tensor, ts):
    """Add time as additional column to tensor"""
    n_points = tensor.shape[0]
    ts = ts * torch.ones((n_points, 1))
    tensor_with_ts = torch.hstack([tensor, ts])
    return tensor_with_ts


def get_transformed_pcd(poses, ref_scan_name, curr_scan_name, cfg_model):
    # # true relative timestamp
    # ts_ref = sample_data_ref['timestamp']  # reference timestamp
    # ts = sample_data['timestamp']  # timestamp
    # ts_rela = (ts - ts_ref) / 1e6
    ts_rela = 1

    ref_from_curr = get_ref_from_current_transform(poses, ref_scan_name, curr_scan_name)
    scan_pcd = transfrom_points(scan_pcd, ref_from_curr)
    origin_tf = torch.tensor(ref_from_curr[:3, 3], dtype=torch.float32)
    points_tf = torch.tensor(scan_pcd.points[:3].T, dtype=torch.float32)

    valid_mask = filter_ego_and_outside_points(scan_pcd, cfg_model)

    return origin_tf, points_tf, ts_rela, valid_mask


def get_mutual_sd_dict(sample_names, cfg):
    key_sd_toks_dict = {}
    n_inputs = cfg["n_input"]
    n_skip = cfg["n_skip"]

    for sample_idx in range(len(sample_names)):
        # Calculate indices for past and future samples
        ref_sample_name = sample_names[sample_idx]
        past_sample_idx = range(max(0, sample_idx - n_inputs - 1), sample_idx, n_skip + 1)
        future_sample_idx = range(sample_idx + 1, min(len(sample_names), sample_idx + n_inputs + 1), n_skip + 1)

        sample_sd_idx_list = [past_sample_idx, sample_idx, future_sample_idx]
        sample_sd_idx_list = sample_sd_idx_list[::-1]

        if len(sample_sd_idx_list) == cfg["n_input"] * 2 + 1:
            key_sd_toks_list = []
            for i in range(len(sample_sd_idx_list)):
                # get all sample idx between two samples
                sd_idx_between_samples = range(sample_sd_idx_list[i], sample_sd_idx_list[i+1])
                # select several sample data as sample data insert
                n_idx = int(len(sd_idx_between_samples) / cfg['n_sd_per_sample'])
                for j in range(cfg['n_sd_per_sample']):
                    key_sd_tok = sd_idx_between_samples[n_idx * j]
                    key_sd_names = sample_names[key_sd_tok]
                    key_sd_toks_list.append(key_sd_names)
                # add to dict
                if len(key_sd_toks_list) == cfg['n_input'] * 2 * cfg['n_sd_per_sample']:
                    key_sd_toks_list.remove(ref_sample_name)
                    key_sd_toks_dict[ref_sample_name] = key_sd_toks_list


class HeLiMOSDataset(Dataset):
    def __init__(self, cfg_model, cfg_dataset, split):
        self.cfg_model = cfg_model
        self.cfg_dataset = cfg_dataset
        self.split = split

        # information
        self.sample_poses = {}
        self.sample_names = {}
        self.dataset_path = cfg_dataset['PATH']
        self.lidar_names = cfg_dataset['LIDAR_NAMES']
        self.label_map = {
            0: 0,   # unknown
            9: 1,   # static
            251: 2  # moving
        }

        if self.split == 'train':
            self.dataset_idx = os.path.join(self.dataset_path, 'train.txt')
        elif self.split == 'val':
            self.dataset_idx = os.path.join(self.dataset_path, 'val.txt')
        elif self.split == 'test':
            self.dataset_idx = os.path.join(self.dataset_path, 'test.txt')
        else:
            raise Exception('Split must be train/val/test')

        for lidar_name in self.lidar_names:
            assert lidar_name in ['Velodyne', 'Ouster', 'Aeva', 'Avia']
            self.sample_names[lidar_name] = self.load_files()  # load all files path in a folder and sort

            # kitti pose: calib.txt (from lidar to camera), poses.txt (from current cam to previous cam)
            path_to_seq = os.path.join(self.dataset_path, lidar_name)
            self.sample_poses[lidar_name] = self.read_kitti_poses(path_to_seq)


    def load_files(self):
        if self.split == 'train':
            data_idx_file = os.path.join(self.dataset_path, 'train.txt')
        elif self.split == 'val':
            data_idx_file = os.path.join(self.dataset_path, 'val.txt')
        elif self.split == 'test':
            data_idx_file = os.path.join(self.dataset_path, 'test.txt')
        else:
            raise Exception('Split must be train/val/test')
        
        if os.path.isfile(data_idx_file):
            scan_idx = np.fromfile(data_idx_file, dtype=str)
            return scan_idx


    def read_kitti_poses(self, path_to_seq):
        pose_file = os.path.join(path_to_seq, 'poses.txt')
        calib_file = os.path.join(path_to_seq, 'calib.txt')
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


    def read_kitti_point_cloud(self, filenames):
        """Load point clouds from .bin file"""
        if isinstance(filenames, str):
            # If input_data is a string, it's assumed to be a single file path
            point_cloud = np.fromfile(filenames, dtype=np.float32)
            point_cloud = torch.tensor(point_cloud.reshape((-1, 4)))
            point_cloud = point_cloud[:, :3]
            return point_cloud

        elif isinstance(filenames, list):
            # If input_data is a list, it's assumed to be a list of file paths
            point_clouds = []
            for filename in filenames:
                point_cloud = np.fromfile(filename, dtype=np.float32)
                point_cloud = torch.tensor(point_cloud.reshape((-1, 4)))
                point_clouds.append(point_cloud[:, :3])
            return torch.cat(point_clouds, dim=0)

        else:
            raise ValueError("Invalid input type. Expected a string or a list of strings.")


    def read_mos_labels(self, lidarseg_file):
        """Load moving object labels from .label file"""
        if os.path.isfile(lidarseg_file):
            labels = np.fromfile(lidarseg_file, dtype=np.uint32)
            vfunc = np.vectorize(self.label_map.get)
            mapped_labels = np.array(vfunc(labels))
            return torch.tensor(mapped_labels)
        else:
            return torch.Tensor(1, 1).long()
    

    def __len__(self):  # return length of the whole dataset
        return len(self.sample_names[0]) * len(self.lidar_names)
    

    def __getitem__(self, index):
        pcd_4d_list = []
        mos_labels = []
        
        for lidar_name in self.lidar_names:
            scan_path = os.path.join(self.dataset_path, lidar_name, 'velodyne')
            label_path = os.path.join(self.dataset_path, lidar_name, 'labels')
            poses  = self.sample_poses[lidar_name]
            ref_scan_idx = -100 # TBD
            time_idx = 1

            for curr_scan_idx in range(len(self.sample_names[lidar_name])):
                curr_scan_name = self.sample_names[lidar_name][curr_scan_idx]
                ref_scan_name = self.sample_names[lidar_name][ref_scan_idx]

                scan_file = os.path.join(scan_path, f'{curr_scan_name}.bin')
                label_file = os.path.join(label_path, f'{curr_scan_name}.label')
                scan_points = self.read_kitti_point_cloud(scan_file)
                scan_labels = self.read_mos_labels(label_file)

                trans_to_ref = self.get_transform_from_ref_scan(poses, ref_scan_name, curr_scan_name)
                scan_points = self.transfrom_points(scan_points, trans_to_ref)

                # add timestamp (0, -1, -2, ...)
                time_idx -= 1
                pcd_4d_ref = self.add_timestamp(scan_points, time_idx)

                pcd_4d_list.append(pcd_4d_ref)
                mos_labels.append(scan_labels)

        # TODO: make the ray index same to ray_intersection_cuda.py, they should have the same filter params
        ref_time_mask = pcds_4d[:, -1] == 0
        valid_mask = torch.squeeze(torch.full((len(pcds_4d), 1), True))
        if self.cfg_model['ego_mask']:
            ego_mask = get_ego_mask(pcds_4d)
            valid_mask = torch.logical_and(valid_mask, ~ego_mask)
        if self.cfg_model['outside_scene_mask']:
            outside_scene_mask = get_outside_scene_mask(pcds_4d, self.cfg_model["scene_bbox"])
            valid_mask = torch.logical_and(valid_mask, ~outside_scene_mask)
        pcds_4d = pcds_4d[valid_mask]
        mos_labels = mos_labels[valid_mask[ref_time_mask]]

        # data augmentation: will not change the order of points
        if self.split == 'train' and self.cfg_model["augmentation"]:
            pcds_4d = augment_pcds(pcds_4d)
        return [_, pcds_4d, mos_labels] # meta info: (sd_tok, num_rays, num_bg_samples);


def get_sample_level_seq_input(nusc, cfg_model, sample_toks: List[str], dir='prev'):
    input_sample_toks_dict = {}
    n_skip = cfg_model["n_skip"]
    for sample_tok in sample_toks:
        sample = nusc.get("sample", sample_tok)
        sd_toks_list = [sample["data"]["LIDAR_TOP"]]
        # skip or select sample data
        skip_cnt = 0  # already skip 0 samples
        num_input_scans = 1  # already sample 1 lidar sample data
        while num_input_scans < cfg_model["n_input"]:
            if sample[dir] != "":
                sample = nusc.get("sample", sample[dir])
                if skip_cnt < cfg_model["n_skip"]:
                    skip_cnt += 1
                    continue
                # add input sample data token
                sd_toks_list.append(sample["data"]["LIDAR_TOP"])
                skip_cnt = 0
                num_input_scans += 1
            else:
                break
        if num_input_scans == cfg_model["n_input"]:  # valid sample tokens (full sequence length)
            input_sample_toks_dict[sample_tok] = sd_toks_list # drop the samples without full sequence length
    return input_sample_toks_dict


class HeLiMOSBgDataset(Dataset):
    def __init__(self, cfg_model, cfg_dataset, split):
        self.cfg_model = cfg_model
        self.cfg_dataset = cfg_dataset
        self.split = split  # "train" "val" "mini_train" "mini_val" "test"

        # information
        self.sample_poses = {}
        self.sample_names = {}
        self.dataset_path = cfg_dataset['PATH']
        self.lidar_names = cfg_dataset['LIDAR_NAMES']
        self.label_map = {
            0: 0,   # unknown
            9: 1,   # static
            251: 2  # moving
        }

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
            sample_toks = split_logs_to_samples(self.nusc, split_logs) #..

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
        # find valid samples with all the files available
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
    

class HeLiMOSDataloader(LightningDataModule):

    """A Pytorch Lightning module for HeLiMOS data; Contains train, valid, test data"""

    def __init__(self, cfg_model, train_set, val_set, train_flag: bool):
        super(HeLiMOSDataloader, self).__init__()
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

    def setup(self, stage=None):
        """Dataloader and iterators for training, validation and test data"""


        ########## Generate dataloaders and iterables
        train_loader = DataLoader(
            dataset=self.train_set,
            batch_size=self.cfg_model["batch_size"],
            collate_fn=self.collate_fn,
            shuffle=self.cfg_model["shuffle"],
            num_workers=self.cfg_model["num_workers"],  # num of multi-processing
            pin_memory=True,
            drop_last=False,  # drop the samples left from full batch
            timeout=0,
        )

        valid_loader = DataLoader(
            dataset=self.val_set,
            batch_size=self.cfg_model["batch_size"],
            collate_fn=self.collate_fn,
            shuffle=self.cfg_model["shuffle"],
            num_workers=self.cfg_model["num_workers"],  # num of multi-processing
            pin_memory=True,
            drop_last=False,  # drop the samples left from full batch
            timeout=0,
        )

        if self.train_flag:
            self.train_loader = train_loader
            self.val_loader = valid_loader
            self.train_iter = iter(self.train_loader)
            self.val_iter = iter(self.val_loader)
            print("Loaded {:d} training and {:d} validation samples.".format(len(self.train_set), len(self.val_set)))
        else:  # test (use validation set)
            self.test_loader = valid_loader
            self.test_iter = iter(self.test_loader)
            print("Loaded {:d} test samples.".format(len(self.val_set)))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader

    @staticmethod
    def collate_fn(batch):  # define how to merge a list of samples to from a mini-batch samples
        meta_info = [item[0] for item in batch] 
        num_curr_pts = [item[1] for item in batch]
        pcds_4d = [item[2] for item in batch]
        mos_labels = [item[3] for item in batch]
        return [meta_info, num_curr_pts, pcds_4d, mos_labels]

