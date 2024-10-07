import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import List
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix


class NuscDataloader(LightningDataModule):
    """A Pytorch Lightning module for Sequential Nusc data; Contains train, valid, test data"""

    def __init__(self, nusc, cfg_model, train_set, val_set, train_flag: bool):
        super(NuscDataloader, self).__init__()
        self.nusc = nusc
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
                drop_last=False,
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


def split_logs_to_samples(nusc, split_logs: List[str]):
    sample_tokens = []  # store the sample tokens
    # sample_data_tokens = []
    for sample in nusc.sample:
        sample_data_token = sample['data']['LIDAR_TOP']
        scene = nusc.get('scene', sample['scene_token'])
        log = nusc.get('log', scene['log_token'])
        logfile = log['logfile']
        if logfile in split_logs:
            # sample_data_tokens.append(sample_data_token)
            sample_tokens.append(sample['token'])
    return sample_tokens


def split_scenes_to_samples(nusc, split_scenes: List[str]):
    sample_tokens = []  # store the sample tokens
    # sample_data_tokens = []
    for scene in nusc.scene:
        if scene['name'] not in split_scenes:
            continue
        sample_token = scene["first_sample_token"]
        sample = nusc.get('sample', sample_token)
        # sample_data_token = sample['data']['LIDAR_TOP']
        sample_tokens.append(sample_token)
        # sample_data_tokens.append(sample_data_token)
        while sample['next'] != "":
            sample_token = sample['next']
            sample = nusc.get('sample', sample_token)
            # sample_data_token = sample['data']['LIDAR_TOP']
            sample_tokens.append(sample_token)
            # sample_data_tokens.append(sample_data_token)
    return sample_tokens


def get_scene_tokens(nusc, split_logs: List[str]) -> List[str]:
    """
    Convenience function to get the samples in a particular split.
    :param split_logs: A list of the log names in this split.
    :return: The list of samples.
    """
    scene_tokens = []  # store the scene tokens
    for scene in nusc.scene:
        log = nusc.get('log', scene['log_token'])
        logfile = log['logfile']
        if logfile in split_logs:
            scene_tokens.append(scene['token'])
    return scene_tokens


def get_input_sd_toks(nusc, cfg_model, sample_toks: List[str], dir='prev'):
    input_sample_toks_dict = {}
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
            input_sample_toks_dict[sample_tok] = sd_toks_list
    return input_sample_toks_dict


def get_sample_data_level_seq_input(nusc, cfg_model, sample_toks: List[str], dir='prev'):
    input_sample_data_toks_dict = {}
    for sample_tok in sample_toks:
        sample = nusc.get("sample", sample_tok)
        sample_data_tok = sample["data"]["LIDAR_TOP"]
        sample_data = nusc.get("sample_data", sample_data_tok)
        sd_toks_list = [sample_data_tok]  # lidar token of current scan
        # skip or select sample data
        skip_cnt = 0  # already skip 0 samples
        num_input_scans = 1  # already sample 1 lidar sample data
        while num_input_scans < cfg_model["n_input"]:
            if sample_data[dir] != "":
                sample_data = nusc.get("sample_data", sample_data[dir])
                if skip_cnt < cfg_model["n_skip"]:
                    skip_cnt += 1
                    continue
                # add input sample data token
                sd_toks_list.append(sample_data["token"])
                skip_cnt = 0
                num_input_scans += 1
            else:
                break
        if num_input_scans == cfg_model["n_input"]:  # valid sample tokens (full sequence length)
            input_sample_data_toks_dict[sample_tok] = sd_toks_list
    return input_sample_data_toks_dict


def get_ego_mask(pcd):
    # nuscenes car: length (4.084 m), width (1.730 m), height (1.562 m)
    # https://forum.nuscenes.org/t/dimensions-of-the-ego-vehicle-used-to-gather-data/550
    ego_mask = torch.logical_and(
        torch.logical_and(-0.865 <= pcd[:, 0], pcd[:, 0] <= 0.865),
        torch.logical_and(-1.5 <= pcd[:, 1], pcd[:, 1] <= 2.5),
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


def add_timestamp(tensor, ts):
    """Add time as additional column to tensor"""
    n_points = tensor.shape[0]
    ts = ts * torch.ones((n_points, 1))
    tensor_with_ts = torch.hstack([tensor, ts])
    return tensor_with_ts


def get_global_pose(nusc, sd_token, inverse=False):
    sd = nusc.get("sample_data", sd_token)
    sd_ep = nusc.get("ego_pose", sd["ego_pose_token"])
    sd_cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

    if inverse is False:  # transform: from lidar coord to global coord
        global_from_ego = transform_matrix(
            sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False
        )
        ego_from_sensor = transform_matrix(
            sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False
        )
        pose = global_from_ego.dot(ego_from_sensor)
    else:  # transform: from global coord to lidar coord
        sensor_from_ego = transform_matrix(
            sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True
        )
        ego_from_global = transform_matrix(
            sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True
        )
        pose = sensor_from_ego.dot(ego_from_global)
    return pose


def get_transformed_pcd(nusc, cfg, sd_token_ref, sd_token):
    # sample data -> pcd
    sample_data = nusc.get("sample_data", sd_token)
    lidar_pcd = LidarPointCloud.from_file(f"{nusc.dataroot}/{sample_data['filename']}")  # [num_pts, x, y, z, i]

    # true relative timestamp
    sample_data_ref = nusc.get("sample_data", sd_token_ref)
    ts_ref = sample_data_ref['timestamp']  # reference timestamp
    ts = sample_data['timestamp']  # timestamp
    ts_rela = (ts - ts_ref) / 1e6

    # poses
    global_from_curr = get_global_pose(nusc, sd_token, inverse=False)  # from {lidar} to {global}
    ref_from_global = get_global_pose(nusc, sd_token_ref, inverse=True)  # from {global} to {ref lidar}
    ref_from_curr = ref_from_global.dot(global_from_curr)  # from {lidar} to {ref lidar}

    # transformed sensor origin and points, at {ref lidar} frame
    origin_tf = torch.tensor(ref_from_curr[:3, 3], dtype=torch.float32)  # curr sensor location, at {ref lidar} frame
    lidar_pcd.transform(ref_from_curr)
    points_tf = torch.tensor(lidar_pcd.points[:3].T, dtype=torch.float32)  # curr point cloud, at {ref lidar} frame

    # filter ego points and outside scene bbox points
    valid_mask = torch.squeeze(torch.full((len(points_tf), 1), True))
    if cfg['ego_mask']:
        ego_mask = get_ego_mask(points_tf)
        valid_mask = torch.logical_and(valid_mask, ~ego_mask)
    if cfg['outside_scene_mask']:
        outside_scene_mask = get_outside_scene_mask(points_tf, cfg["scene_bbox"], cfg['outside_scene_mask_z'], cfg['outside_scene_mask_ub'])
        valid_mask = torch.logical_and(valid_mask, ~outside_scene_mask)
    points_tf = points_tf[valid_mask]
    return origin_tf, points_tf, ts_rela, valid_mask


def get_mutual_sd_toks_dict(nusc, sample_toks: List[str], cfg):
    key_sd_toks_dict = {}
    for ref_sample_tok in sample_toks:
        ref_sample = nusc.get("sample", ref_sample_tok)
        ref_sd_tok = ref_sample['data']['LIDAR_TOP']
        ref_ts = ref_sample['timestamp'] / 1e6
        sample_sd_toks_list = []

        # future sample data
        skip_cnt = 0
        num_next_samples = 0
        sample = ref_sample
        while num_next_samples < cfg["n_input"]:
            if sample['next'] != "":
                sample = nusc.get("sample", sample['next'])
                if skip_cnt < cfg["n_skip"]:
                    skip_cnt += 1
                    continue
                # add input sample data token
                sample_sd_toks_list.append(sample["data"]["LIDAR_TOP"])
                skip_cnt = 0
                num_next_samples += 1
            else:
                break
        if len(sample_sd_toks_list) == cfg["n_input"]:  # TODO: add one, to get full insert sample datas
            sample_sd_toks_list = sample_sd_toks_list[::-1]  # 3.0, 2.5, 2.0, 1.5, 1.0, 0.5
        else:
            continue

        # history sample data
        sample_sd_toks_list.append(ref_sample["data"]["LIDAR_TOP"])
        skip_cnt = 0  # already skip 0 samples
        num_prev_samples = 1  # already sample 1 lidar sample data
        sample = ref_sample
        while num_prev_samples < cfg["n_input"] + 1:
            if sample['prev'] != "":
                sample = nusc.get("sample", sample['prev'])
                if skip_cnt < cfg["n_skip"]:
                    skip_cnt += 1
                    continue
                # add input sample data token
                sample_sd_toks_list.append(sample["data"]["LIDAR_TOP"])  # 3.0, 2.5, 2.0, 1.5, 1.0, 0.5 | 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, (-3.0)
                skip_cnt = 0
                num_prev_samples += 1
            else:
                break

        # full samples, add sample data and inserted sample data to dict
        if len(sample_sd_toks_list) == cfg["n_input"] * 2 + 1:
            key_sd_toks_list = []
            key_ts_list = []
            for i in range(len(sample_sd_toks_list) - 1):
                sample_data = nusc.get('sample_data', sample_sd_toks_list[i])
                # get all sample data between two samples
                sd_toks_between_samples = [sample_data['token']]
                while sample_data['prev'] != "" and sample_data['prev'] != sample_sd_toks_list[i+1]:
                    sample_data = nusc.get("sample_data", sample_data['prev'])
                    sd_toks_between_samples.append(sample_data['token'])
                # select several sample data as sample data insert
                n_idx = int(len(sd_toks_between_samples) / cfg['n_sd_per_sample'])
                for j in range(cfg['n_sd_per_sample']):
                    key_sd_tok = sd_toks_between_samples[n_idx * j]
                    key_sample_data = nusc.get('sample_data', key_sd_tok)
                    key_ts = key_sample_data['timestamp'] / 1e6
                    key_sd_toks_list.append(key_sd_tok)
                    key_ts_list.append(key_ts - ref_ts)
                # add to dict
                if len(key_sd_toks_list) == cfg['n_input'] * 2 * cfg['n_sd_per_sample']:
                    key_sd_toks_list.remove(ref_sd_tok)
                    key_sd_toks_dict[ref_sample_tok] = key_sd_toks_list
    return key_sd_toks_dict


def get_curr_future_sd_toks_dict(nusc, sample_toks: List[str], cfg, get_curr: bool):
    future_sd_toks_dict = {}
    for ref_sample_tok in sample_toks:
        ref_sample = nusc.get("sample", ref_sample_tok)
        if get_curr:
            sample_sd_toks_list = [ref_sample['data']['LIDAR_TOP']]  # for uno
            full_size = cfg['n_input'] + 1
        else:
            sample_sd_toks_list = []  # for occ4d
            full_size = cfg['n_input']

        # future sample data
        skip_cnt = 0
        num_next_samples = 0
        sample = ref_sample
        while num_next_samples < cfg["n_input"]:
            if sample['next'] != "":
                sample = nusc.get("sample", sample['next'])
                if skip_cnt < cfg["n_skip"]:
                    skip_cnt += 1
                    continue
                # add input sample data token
                sample_sd_toks_list.append(sample["data"]["LIDAR_TOP"])
                skip_cnt = 0
                num_next_samples += 1
            else:
                break
        if len(sample_sd_toks_list) == full_size:
            future_sd_toks_dict[ref_sample_tok] = sample_sd_toks_list
        else:
            continue
    return future_sd_toks_dict