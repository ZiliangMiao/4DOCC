# base
import re
import os
import click
import yaml
import copy
import numpy as np
from tqdm import tqdm
# torch
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# dataset
from nuscenes.nuscenes import NuScenes
# lib
from models.mos4d import models
from datasets.mos4d import nusc as nusc_dataset
from datasets.mos4d import kitti as kitti_dataset
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
            model = models.MOSNet(cfg)
            model = load_pretrained_encoder(pre_ckpt, model)
        else:
            model = models.MOSNet.load_from_checkpoint(pre_ckpt, hparams=cfg)
    elif mode == "train":  # init network with cfg
        model_name = f"vs-{voxel_size}_t-{time}_bs-{batch_size}"
        if dataset_name == "nuscenes":
            model = models.MOSNet(cfg)
        else:
            model = models.MOSNet(cfg)

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

def create_sekitti_mos_labels(dataset_path):
    semantic_config = yaml.safe_load(open("./config/semantic-kitti-mos.yaml"))
    def save_mos_labels(lidarseg_label_file, mos_label_file):
        """Load moving object labels from .label file"""
        if os.path.isfile(lidarseg_label_file):
            labels = np.fromfile(lidarseg_label_file, dtype=np.uint32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF  # Mask semantics in lower half
            mapped_labels = copy.deepcopy(labels)
            for semantic_label, mos_label in semantic_config["learning_map"].items():
                mapped_labels[labels == semantic_label] = mos_label
            mos_labels = mapped_labels.astype(np.uint8)
            mos_labels.tofile(mos_label_file)
            num_unk = np.sum(mos_labels == 0)
            num_sta = np.sum(mos_labels == 1)
            num_mov = np.sum(mos_labels == 2)
            print(f"num of unknown pts: {num_unk}, num of static pts: {num_sta}, num of moving pts: {num_mov}")
            # Directly load saved mos labels and check if it is correct
            # mos_labels = np.fromfile(mos_label_file, dtype=np.uint32).reshape((-1)) & 0xFFFF
            # check_true = (mos_labels == mapped_labels).all()
            return None
        else:
            return torch.Tensor(1, 1).long()

    seqs_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for seq_idx in tqdm(seqs_list):
        lidarseg_dir = os.path.join(dataset_path, str(seq_idx).zfill(4), "labels")
        mos_labels_dir = os.path.join(dataset_path, str(seq_idx).zfill(4), "mos_labels")
        os.makedirs(mos_labels_dir, exist_ok=True)
        for i, filename in enumerate(os.listdir(lidarseg_dir)):
            lidarseg_label_file = os.path.join(lidarseg_dir, filename)
            mos_label_file = os.path.join(mos_labels_dir, filename)
            save_mos_labels(lidarseg_label_file, mos_label_file)

if __name__ == "__main__":
    # deterministic
    set_deterministic(666)

    # load train config
    with open("configs/mos4d_train.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)

    # create_sekitti_mos_labels("/home/user/Datasets/SeKITTI/sequences")

