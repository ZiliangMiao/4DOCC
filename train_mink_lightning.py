import os
import re
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from data.common import MinkCollateFn
from model_mink_lightning import MinkOccupancyForecastingNetwork

def make_mink_dataloaders(cfg):
    dataset_kwargs = {
        "pc_range": cfg["data"]["pc_range"],
        "input_within_pc_range": cfg["data"]["input_within_pc_range"],
        "voxel_size": cfg["data"]["voxel_size"],
        "n_input": cfg["data"]["n_input"],
        "n_output": cfg["data"]["n_output"],
        "ego_mask": cfg["data"]["ego_mask"],
        "flip": cfg["data"]["flip"],
    }
    data_loader_kwargs = {
        "pin_memory": False,  # NOTE
        "shuffle": cfg["data"]["shuffle"],
        "drop_last": True,
        "batch_size": cfg["model"]["batch_size"],
        "num_workers": cfg["model"]["num_workers"],
    }

    dataset_name = cfg["data"]["dataset_name"].lower()
    if dataset_name == "nuscenes":
        from data.nusc_mink import nuScenesDataset
        from nuscenes.nuscenes import NuScenes

        nusc = NuScenes(cfg["dataset"][dataset_name]["version"], cfg["dataset"][dataset_name]["root"])
        data_loaders = {
            "train": DataLoader(
                nuScenesDataset(nusc, "train", dataset_kwargs),
                collate_fn=MinkCollateFn,
                **data_loader_kwargs,
            ),
            "val": DataLoader(
                nuScenesDataset(nusc, "val", dataset_kwargs),
                collate_fn=MinkCollateFn,
                **data_loader_kwargs,
            ),
        }
    elif dataset_name == "kitti":
        raise NotImplementedError("KITTI is not supported now, wait for data.kitti_mink.py.")
        from data.kitti import KittiDataset
        data_loaders = {
            "train": DataLoader(
                KittiDataset(cfg["dataset"]["name"]["root"], cfg["dataset"]["name"]["config"], "trainval", dataset_kwargs),
                collate_fn=MinkCollateFn,
                **data_loader_kwargs,
            ),
            "val": DataLoader(
                KittiDataset(cfg["dataset"]["name"]["root"], cfg["dataset"]["name"]["config"], "test", dataset_kwargs),
                collate_fn=MinkCollateFn,
                **data_loader_kwargs,
            ),
        }
    elif dataset_name == "argoverse2":
        raise NotImplementedError("Argoverse is not supported now, wait for data.av2_mink.py.")
        from data.av2 import Argoverse2Dataset
        data_loaders = {
            "train": DataLoader(
                Argoverse2Dataset(cfg["dataset"][dataset_name]["root"], "train", dataset_kwargs, subsample=cfg["dataset"][dataset_name]["subsample"]),
                collate_fn=MinkCollateFn,
                **data_loader_kwargs,
            )
        }
    else:
        raise NotImplementedError("Dataset " + cfg["dataset"]["name"] + "is not supported.")
    return data_loaders


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
    # dataloader
    data_loaders = make_mink_dataloaders(cfg)
    # model
    model = MinkOccupancyForecastingNetwork(cfg)
    # params
    voxel_size = cfg["data"]["voxel_size"]
    time_interval = cfg["data"]["time_interval"]
    n_input = cfg["data"]["n_input"]
    time = time_interval * n_input
    batch_size = cfg["model"]["batch_size"]
    num_epoch = cfg["model"]["num_epoch"]
    model_name = f"pretrain_vs-{voxel_size}_t-{time}_bs-{batch_size}_epo-{num_epoch}"

    # pl lr monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # pl checkpoint saver
    checkpoint_saver = ModelCheckpoint(
        monitor="epoch",
        verbose=True,
        save_top_k=cfg["model"]["num_epoch"],
        mode="max",
        filename=model_name + "_{epoch}",
        every_n_epochs=1,
        save_last=True,
    )

    # pl logger
    dataset_name = cfg["data"]["dataset_name"].lower()
    logs_dir = f"./logs/pretrain/{dataset_name}"
    os.makedirs(name=logs_dir, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(logs_dir, name=model_name, default_hp_metric=False)

    # resume training: could be ckpt path or None, resume training and training from scratch respectively
    resume_ckpt_path = cfg["model"]["resume_ckpt"]
    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=cfg["model"]["num_devices"],
        logger=tb_logger,
        log_every_n_steps=1,
        max_epochs=cfg["model"]["num_epoch"],
        accumulate_grad_batches=cfg["model"]["acc_batches"],  # accumulate batches, default=1
        callbacks=[lr_monitor, checkpoint_saver],
        check_val_every_n_epoch=5,
        # val_check_interval=100,
    )

    # pl training
    trainer.fit(model, train_dataloaders=data_loaders["train"], val_dataloaders=data_loaders["val"], ckpt_path=resume_ckpt_path)

if __name__ == "__main__":
    # set random seeds
    np.random.seed(666)
    torch.random.manual_seed(666)

    # load pretrain config
    with open("./configs/occ_pretrain.yaml", "r") as f:
        cfg_pretrain = yaml.safe_load(f)
    if cfg_pretrain["model"]["resume_ckpt"] is not None:
        cfg_pretrain = torch.load(cfg_pretrain["model"]["resume_ckpt"])["hyper_parameters"]
    pretrain(cfg_pretrain)

    # moving object segmentation fine-tuning (finetune at mos project, not 4docc project)
    # with open("./configs/mos_finetune.yaml", "r") as f:
    #     cfg_finetune = yaml.safe_load(f)
    # finetune(cfg_finetune)