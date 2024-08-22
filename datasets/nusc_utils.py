import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import List


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
                    drop_last=False,  # drop the samples left from full batch
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


def get_sd_toks_dict(nusc, cfg_model, sample_toks: List[str]):
    sample_to_input_sd_toks_dict = {}
    for sample_tok in sample_toks:
        # Get records from DB.
        sample = nusc.get("sample", sample_tok)
        input_sd_toks_list = [sample["data"]["LIDAR_TOP"]]  # lidar token of current scan
        # skip or select sample data
        skip_cnt = 0  # already skip 0 samples
        num_input_samples = 1  # already sample 1 lidar sample data
        while num_input_samples < cfg_model["n_input"]:
            if sample["prev"] != "":
                sample_prev = nusc.get("sample", sample["prev"])
                if skip_cnt < cfg_model["n_skip"]:
                    skip_cnt += 1
                    sample = sample_prev
                    continue
                # add input sample data token
                input_sd_toks_list.append(sample_prev["data"]["LIDAR_TOP"])
                skip_cnt = 0
                num_input_samples += 1
                # assign sample data prev to sample data for next loop
                sample = sample_prev
            else:
                break
        assert len(input_sd_toks_list) == num_input_samples
        if num_input_samples == cfg_model["n_input"]:  # valid sample tokens (full sequence length)
            sample_to_input_sd_toks_dict[sample_tok] = input_sd_toks_list
    return sample_to_input_sd_toks_dict


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
        # torch.logical_and(scene_bbox[2] <= points[:, 2], points[:, 2] <= scene_bbox[5])
    )
    return ~inside_scene_mask


def add_timestamp(tensor, ts):
    """Add time as additional column to tensor"""
    n_points = tensor.shape[0]
    ts = ts * torch.ones((n_points, 1))
    tensor_with_ts = torch.hstack([tensor, ts])
    return tensor_with_ts