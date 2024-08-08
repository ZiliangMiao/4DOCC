# base
import re
import os
import yaml
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# torch
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# dataset
from nuscenes.nuscenes import NuScenes
# lib
from models.ours import models
from datasets.ours import nusc as nusc_dataset
from datasets.ours import kitti as kitti_dataset
from utils.deterministic import set_deterministic

def load_pretrained_encoder(ckpt_path, model):
    # if len(os.listdir(ckpt_dir)) > 0:
    #     pattern = re.compile(r"model_epoch_(\d+).pth")
    #     epochs = []
    #     for f in os.listdir(ckpt_dir):
    #         m = pattern.findall(f)
    #         if len(m) > 0:
    #             epochs.append(int(m[0]))
    #     resume_epoch = max(epochs)
    #     ckpt_path = f"{ckpt_dir}/model_epoch_{resume_epoch}.pth"

    print(f"Load pretrained encoder from checkpoint {ckpt_path}")

    checkpoint = torch.load(ckpt_path)
    pretrained_dict = checkpoint["state_dict"]
    model_dict = model.state_dict()

    # filter out unnecessary keys (generate new dict)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    pretrained_dict.pop('encoder.MinkUNet.final.kernel')
    pretrained_dict.pop('encoder.MinkUNet.final.bias')
    # overwrite finetune model dict
    model_dict.update(pretrained_dict)
    # load the pretrained model dict
    model.load_state_dict(model_dict)
    return model


def main(cfg):
    # data params
    mode = cfg["mode"]
    assert mode != "test"
    dataset_name = cfg["data"]["dataset_name"]
    data_pct = cfg["data"]["dataset_pct"]
    voxel_size = cfg["data"]["voxel_size"]
    time_interval = cfg["data"]["time_interval"]
    n_input = cfg["data"]["n_input"]
    n_skip = cfg["data"]["n_skip"]
    time = round(n_input * time_interval + (n_input - 1) * n_skip * time_interval, 2)
    # model params
    num_epoch = cfg["model"]["num_epoch"]
    batch_size = cfg["model"]["batch_size"]

    if mode == "finetune":  # load pretrained model
        pre_method = cfg["model"]["pretrain_method"]
        pre_dataset = cfg["model"]["pretrain_dataset"]
        pre_model = cfg["model"]["pretrain_model_name"]
        pre_version = cfg["model"]["pretrain_version"]
        pre_epoch = cfg["model"]["pretrain_epoch"]
        if pre_method == "occ4d":
            pre_ckpt_name = f"{pre_model}_epoch={pre_epoch}"
            pre_ckpt = f"./logs/{pre_method}/{pre_dataset}/{pre_model}/{pre_version}/checkpoints/{pre_ckpt_name}.ckpt"
            model_name = f"{pre_method}_{pre_dataset}_{pre_ckpt_name}_vs-{voxel_size}_t-{time}_bs-{batch_size}"
        if dataset_name == "nuscenes":
            pretrain_ckpt = torch.load(pre_ckpt)
            torch.save(pretrain_ckpt, pre_ckpt)
            model = models.MotionPretrainNetwork(cfg)
            model = load_pretrained_encoder(pre_ckpt, model)
        else:
            model = models.MotionPretrainNetwork.load_from_checkpoint(pre_ckpt, hparams=cfg)
    elif mode == "train":  # init network with cfg
        model_name = f"vs-{voxel_size}_t-{time}_bs-{batch_size}"
        if dataset_name == "nuscenes":
            model = models.MotionPretrainNetwork(cfg)
        else:
            model = models.MotionPretrainNetwork(cfg)

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_saver = ModelCheckpoint(
        monitor="epoch",
        verbose=True,
        save_top_k=num_epoch,
        mode="max",
        filename=model_name + "_{epoch}",
        every_n_epochs=1,
        save_last=True,
    )

    # Logger
    log_dir = f"./logs/mos4d/{data_pct}%{dataset_name}"
    os.makedirs(log_dir, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(log_dir, name=model_name, default_hp_metric=False)

    # load data from different datasets
    if dataset_name == "nuscenes":
        nusc = NuScenes(dataroot=cfg["dataset"]["nuscenes"]["root"], version=cfg["dataset"]["nuscenes"]["version"])
        data = nusc_dataset.NuscSequentialModule(cfg, nusc, "train")
        data.setup()
    else:  # KITTI-like datasets
        data = kitti_dataset.KittiSequentialModule(cfg)
        data.setup()

    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    # Setup trainer and fit
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

    # fit
    resume_ckpt_path = cfg["model"]["resume_ckpt"]
    trainer.fit(model, train_dataloader, ckpt_path=resume_ckpt_path)

if __name__ == "__main__":
    # deterministic
    set_deterministic(666)

    # load train config
    with open("configs/ours_train.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # TODO: test bg labels
    # nusc = NuScenes(dataroot=cfg["dataset"]["nuscenes"]["root"], version=cfg["dataset"]["nuscenes"]["version"])
    # nuscenes = nusc_dataset.NuscSequentialDataset(cfg, nusc, "train")
    # (ref_sd_tok, num_rays_all, num_bg_samples_all), pcds_4d, ray_to_bg_samples_dict = nuscenes.__getitem__(0)
    #
    # # number statistics of background sample points
    # num_occ_percentage_per_ray = []
    # for ray_idx, bg_samples in ray_to_bg_samples_dict.items():
    #     num_occ_ray = np.sum(bg_samples[:, -1] == 2)
    #     num_free_ray = np.sum(bg_samples[:, -1] == 1)
    #     num_occ_percentage_per_ray.append((num_occ_ray / (num_occ_ray + num_free_ray)) * 100)
    # plt.figure()
    # plt.hist(np.array(num_occ_percentage_per_ray), bins=10, color='lightsalmon', alpha=1, log=True)
    # plt.title('Distribution of Occ Percentage')
    # plt.xlabel('Occ Percentage (/Occ + Free)')
    # plt.ylabel('Frequency (Ray Samples)')
    # plt.savefig('./occ percentage distribution.jpg')

    # training
    main(cfg)