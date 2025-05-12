import os
import yaml
import logging
import warnings
import importlib

import torch
import numpy as np
from nuscenes import NuScenes
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor

# ours
import argparse
from utils.deterministic import set_deterministic
from datasets.also.nusc import NuscAlsoDataset
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import Trainer
from models.also.models import AlsoNetwork


def also_pretrain(model_cfg, dataset_cfg, resume_version):
    # model params
    dataset_name = model_cfg['dataset_name']
    assert dataset_name == 'nuscenes'
    downsample_pct = model_cfg['downsample_pct']
    pretrain_dir = f"./logs/forecast_baseline/also/{downsample_pct}%{dataset_name}"
    os.makedirs(pretrain_dir, exist_ok=True)
    quant_size = model_cfg['quant_size']
    batch_size = model_cfg['batch_size']
    time = round(model_cfg['n_input'] * model_cfg['time_interval'] + (model_cfg['n_input'] - 1) * model_cfg['n_skip'] *
                 model_cfg['time_interval'], 2)
    model_params = f"vs-{quant_size}_t-{time}_bs-{batch_size}"

    # dataloader
    from datasets.dataloader import build_dataloader
    nusc = NuScenes(dataroot=dataset_cfg["nuscenes"]["root"], version=dataset_cfg["nuscenes"]["version"])
    train_dataloader = build_dataloader(model_cfg, dataset_cfg, 'train', nusc)

    # pretrain model
    pretrain_model = AlsoNetwork(model_cfg, True, iters_per_epoch=len(train_dataloader))

    # lr_monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # tensorboard logger
    tb_logger = pl_loggers.TensorBoardLogger(pretrain_dir, name=model_params, default_hp_metric=False)

    # checkpoint saver
    checkpoint_saver = ModelCheckpoint(
        monitor="epoch",
        verbose=True,
        save_top_k=model_cfg['num_epoch'],
        mode="max",
        filename="{epoch}",
        every_n_epochs=1,
        save_last=True,
    )

    # trainer
    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=model_cfg["num_devices"],
        logger=tb_logger,
        log_every_n_steps=1,
        max_epochs=model_cfg['num_epoch'],
        accumulate_grad_batches=model_cfg["acc_batches"],  # accumulate batches, default=1
        callbacks=[lr_monitor, checkpoint_saver],
    )

    # training
    if resume_version != -1:  # resume training
        resume_model_cfg = yaml.safe_load(
            open(os.path.join(pretrain_dir, model_params, f'version_{resume_version}', "hparams.yaml")))
        assert set(model_cfg) == set(resume_model_cfg), "resume training: cfg dict keys are not the same."
        assert model_cfg == resume_model_cfg, f"resume training: cfg keys have different values."
        resume_ckpt_path = os.path.join(pretrain_dir, model_params, f'version_{resume_version}', 'checkpoints', 'last.ckpt')
        trainer.fit(pretrain_model, train_dataloader, ckpt_path=resume_ckpt_path)
    else:
        trainer.fit(pretrain_model, train_dataloader)


if __name__ == "__main__":
    # deterministic
    set_deterministic(666)

    # mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['also', 'also_test'], default='also')
    parser.add_argument('--resume_version', type=int, default=-1)  # -1: not resuming
    parser.add_argument('--autodl', type=bool, help="autodl server", default=False)
    parser.add_argument('--mars', type=bool, help="mars server", default=False)
    parser.add_argument('--hpc', type=bool, help="hpc server", default=False)
    args = parser.parse_args()

    # load config
    with open("configs/also.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    with open("configs/dataset.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)

    # dataset root path at different platform
    if args.autodl:
        dataset_cfg['nuscenes']['root'] = '/root/autodl-tmp' + dataset_cfg['nuscenes']['root']
        dataset_cfg['sekitti']['root'] = '/root/autodl-tmp' + dataset_cfg['sekitti']['root']
    elif args.mars:
        dataset_cfg['nuscenes']['root'] = '/home/miaozl' + dataset_cfg['nuscenes']['root']
        dataset_cfg['sekitti']['root'] = '/home/miaozl' + dataset_cfg['sekitti']['root']
    elif args.hpc:
        dataset_cfg['nuscenes']['root'] = '/lustre1/g/mech_mars' + dataset_cfg['nuscenes']['root']
        dataset_cfg['sekitti']['root'] = '/lustre1/g/mech_mars' + dataset_cfg['sekitti']['root']
    else:
        dataset_cfg['nuscenes']['root'] = '/home/ziliang' + dataset_cfg['nuscenes']['root']
        dataset_cfg['sekitti']['root'] = '/home/ziliang' + dataset_cfg['sekitti']['root']

    # pre-training on background for motion segmentation task
    if args.mode == 'also':
        also_pretrain(cfg[args.mode], dataset_cfg, args.resume_version)
    # background test
    elif args.mode == 'uno_test':
        print("No ALSO Test Method!")


