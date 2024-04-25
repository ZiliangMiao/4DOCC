import os
import re
import yaml
import torch
from torch.utils.data import DataLoader, sampler
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from datasets.occ4d.common import MinkCollateFn
from models.occ4d.models_pl_2d import MinkOccupancyForecastingNetwork
from utils.deterministic import set_deterministic

def make_mink_dataloaders(cfg):
    batch_size = cfg["model"]["batch_size"]
    num_workers = cfg["model"]["num_workers"]
    shuffle = cfg["data"]["shuffle"]
    data_pct = cfg["data"]["dataset_pct"]
    dataset_name = cfg["data"]["dataset_name"].lower()

    if dataset_name == "nuscenes":
        from datasets.occ4d.nusc_scan import nuScenesDataset
        from nuscenes.nuscenes import NuScenes

        nusc = NuScenes(cfg["dataset"][dataset_name]["version"], cfg["dataset"][dataset_name]["root"])
        train_set = nuScenesDataset(nusc, "train", cfg)
        train_loader = DataLoader(  # 9 parameters
                dataset=train_set,
                batch_size=batch_size,
                collate_fn=MinkCollateFn,
                num_workers=num_workers,
                shuffle=shuffle,
                pin_memory=False,
                drop_last=True,
                timeout=0,
                sampler=sampler.WeightedRandomSampler(weights=torch.ones(len(train_set)),
                                                      num_samples=int(data_pct * len(train_set))),
            )
        val_set = nuScenesDataset(nusc, "val", cfg)
        val_loader = DataLoader(  # 8 parameters, without sampler
                dataset=val_set,
                batch_size=batch_size,
                collate_fn=MinkCollateFn,
                num_workers=num_workers,
                shuffle=shuffle,
                pin_memory=False,
                drop_last=True,
                timeout=0,
            )
        dataloaders = {"train": train_loader, "val": val_loader}
    elif dataset_name == "kitti":
        raise NotImplementedError("KITTI is not supported now, wait for data.kitti_mink.py.")
        # from datasets.kitti import KittiDataset
        # train_set = KittiDataset(cfg["dataset"][dataset_name]["root"], cfg["dataset"][dataset_name]["config"],
        #                          "trainval", cfg),
        # train_loader = DataLoader(  # 9 parameters
        #     dataset=train_set,
        #     batch_size=batch_size,
        #     collate_fn=MinkCollateFn,
        #     num_workers=num_workers,
        #     shuffle=shuffle,
        #     pin_memory=False,
        #     drop_last=True,
        #     timeout=0,
        #     sampler=sampler.WeightedRandomSampler(weights=torch.ones(len(train_set)),
        #                                           num_samples=int(data_pct * len(train_set))),
        # )
        # val_set = KittiDataset(cfg["dataset"][dataset_name]["root"], cfg["dataset"][dataset_name]["config"],
        #                        "test", cfg),
        # val_loader = DataLoader(  # 8 parameters, without sampler
        #     dataset=val_set,
        #     batch_size=batch_size,
        #     collate_fn=MinkCollateFn,
        #     num_workers=num_workers,
        #     shuffle=shuffle,
        #     pin_memory=False,
        #     drop_last=True,
        #     timeout=0,
        # )
        # dataloaders = {"train": train_loader, "val": val_loader}
    elif dataset_name == "argoverse2":
        raise NotImplementedError("Argoverse is not supported now, wait for data.av2_mink.py.")
        # from datasets.av2 import Argoverse2Dataset
        # train_set = Argoverse2Dataset(cfg["dataset"][dataset_name]["root"], "train", dataset_kwargs,
        #                   subsample=cfg["dataset"][dataset_name]["subsample"])
    else:
        raise NotImplementedError("Dataset " + cfg["dataset"]["name"] + "is not supported.")
    return dataloaders


def resume_from_ckpts(ckpt_pth, model, optimizer, scheduler):
    print(f"Resume training from checkpoint {ckpt_pth}")
    checkpoint = torch.load(ckpt_pth)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = 1 + checkpoint["epoch"]
    n_iter = checkpoint["n_iter"]
    return start_epoch, n_iter

def load_pretrained_encoder(ckpt_dir, model):
    if len(os.listdir(ckpt_dir)) > 0:
        pattern = re.compile(r"model_epoch_(\d+).pth")
        epochs = []
        for f in os.listdir(ckpt_dir):
            m = pattern.findall(f)
            if len(m) > 0:
                epochs.append(int(m[0]))
        resume_epoch = max(epochs)
        ckpt_path = f"{ckpt_dir}/model_epoch_{resume_epoch}.pth"
        print(f"Load pretrained encoder from checkpoint {ckpt_path}")

        checkpoint = torch.load(ckpt_path)
        pretrained_dict = checkpoint["model_state_dict"]
        model_dict = model.state_dict()

        # filter out unnecessary keys (generate new dict)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # overwrite finetune model dict
        model_dict.update(pretrained_dict)
        # load the pretrained model dict
        model.load_state_dict(model_dict)
    return model

def pretrain(cfg):
    ## params
    # data params
    voxel_size = cfg["data"]["voxel_size"]
    time_interval = cfg["data"]["time_interval"]
    n_input = cfg["data"]["n_input"]
    n_skip = cfg["data"]["n_skip"]
    time = round(n_input * time_interval + (n_input - 1) * n_skip * time_interval, 2)
    dataset_name = cfg["data"]["dataset_name"].lower()
    dataset_pct = cfg["data"]["dataset_pct"]
    # model params
    batch_size = cfg["model"]["batch_size"]
    num_epoch = cfg["model"]["num_epoch"]
    model_name = f"vs-{voxel_size}_t-{time}_bs-{batch_size}"

    # dataloader
    data_loaders = make_mink_dataloaders(cfg)

    # model
    model = MinkOccupancyForecastingNetwork(cfg)

    # lr monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # checkpoint saver
    checkpoint_saver = ModelCheckpoint(
        monitor="epoch",
        verbose=True,
        save_top_k=num_epoch,
        mode="max",
        filename=model_name + "_{epoch}",
        every_n_epochs=1,
        save_last=True,
    )

    # logger
    logs_dir = f"./logs/occ4d/{dataset_pct}%{dataset_name}/{model_name}"
    os.makedirs(name=logs_dir, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(logs_dir, name=model_name, default_hp_metric=False)

    # trainer
    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=cfg["model"]["num_devices"],
        logger=tb_logger,
        log_every_n_steps=1,
        max_epochs=num_epoch,
        accumulate_grad_batches=cfg["model"]["acc_batches"],  # accumulate batches, default=1
        callbacks=[lr_monitor, checkpoint_saver],
        # check_val_every_n_epoch=5,
        # val_check_interval=100,
    )

    # training
    resume_ckpt_path = cfg["model"]["resume_ckpt"]
    trainer.fit(model, train_dataloaders=data_loaders["train"], ckpt_path=resume_ckpt_path)

if __name__ == "__main__":
    # deterministic
    set_deterministic(666)

    # load pretrain config
    with open("configs/occ4d_train.yaml", "r") as f:
        cfg_pretrain = yaml.safe_load(f)
    pretrain(cfg_pretrain)