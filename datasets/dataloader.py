from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets.mos4d.kitti import KittiMOSDataset
from datasets.mos4d.nusc import NuscMosDataset


def build_dataloader(model_cfg, dataset_cfg, mode, nusc=None):
    if model_cfg['dataset_name'] == 'nuscenes':
        train_set = NuscMosDataset(nusc, model_cfg, dataset_cfg, 'train')
        val_set = NuscMosDataset(nusc, model_cfg, dataset_cfg, 'val')
        test_set = NuscMosDataset(nusc, model_cfg, dataset_cfg, 'test')
        dataloader = Dataloader(model_cfg, train_set, val_set, test_set, mode, nusc=nusc)
    elif model_cfg['dataset_name'] == 'sekitti':
        train_set = KittiMOSDataset(model_cfg, dataset_cfg, mode='train')
        val_set = KittiMOSDataset(model_cfg, dataset_cfg, mode='val')
        test_set = KittiMOSDataset(model_cfg, dataset_cfg, mode='test')
        dataloader = Dataloader(model_cfg, train_set, val_set, test_set, mode)
    else:
        print("Not a supported dataset.")
        return None
    dataloader.setup()
    if mode == 'test':
        return dataloader.test_dataloader()
    elif mode == 'val':
        return dataloader.val_dataloader()
    elif mode in ['train', 'finetune']:
        return dataloader.train_dataloader()
    else:
        return None


class Dataloader(LightningDataModule):
    def __init__(self, cfg_model, train_set, val_set, test_set, mode: str, nusc=None):
        super(Dataloader, self).__init__()
        self.cfg_model = cfg_model
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.mode = mode
        self.nusc = nusc
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def prepare_data(self):
        pass

    @staticmethod
    def collate_fn(batch):  # define how to merge a list of samples to from a mini-batch samples
        meta_info = [item[0] for item in batch]  # meta info: (sd_tok, num_rays, num_bg_samples); (sd_toks)
        pcds_4d = [item[1] for item in batch]
        samples = [item[2] for item in batch]  # bg samples; mos labels
        return [meta_info, pcds_4d, samples]

    def setup(self, stage=None):
        train_loader = DataLoader(
            dataset=self.train_set,
            batch_size=self.cfg_model["batch_size"],
            collate_fn=self.collate_fn,
            num_workers=self.cfg_model["num_workers"],
            shuffle=self.cfg_model["shuffle"],
            pin_memory=True,
            drop_last=True,
            timeout=0,
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
        test_loader = DataLoader(
            dataset=self.test_set,
            batch_size=self.cfg_model["batch_size"],
            collate_fn=self.collate_fn,
            num_workers=self.cfg_model["num_workers"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )

        if self.mode in ["train", "finetune"]:
            self.train_loader = train_loader
            print("Loaded {:d} training samples.".format(len(self.train_loader.dataset)))
        elif self.mode == "val":  # validation, using part of org training set
            self.val_loader = val_loader
            print("Loaded {:d} validation samples.".format(len(self.val_loader.dataset)))
        elif self.mode == "test":  # test, using org validation set
            self.test_loader = test_loader
            print("Loaded {:d} test samples.".format(len(self.test_loader.dataset)))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader