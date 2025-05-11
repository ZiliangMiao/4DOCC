from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class Dataloader(LightningDataModule):
    def __init__(self, cfg_model, train_set, val_set, train_flag: bool, nusc=None):
        super(Dataloader, self).__init__()
        self.cfg_model = cfg_model
        self.train_set = train_set
        self.val_set = val_set
        self.train_flag = train_flag
        self.nusc = nusc
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