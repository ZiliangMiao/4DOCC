# base
import argparse
import re
import os
import click
import yaml
import copy
import numpy as np
from tqdm import tqdm
import sys
import logging
from datetime import datetime
# torch
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# dataset
from nuscenes.nuscenes import NuScenes
from utils.metrics import ClassificationMetrics
from utils.deterministic import set_deterministic
from models.mos4d.models import MosNetwork
from datasets.nusc_utils import NuscDataloader
from datasets.mos4d.nusc import NuscMosDataset
from ours_script import mos_finetune


def mos4d_baseline_train(model_cfg, dataset_cfg, resume_version):
    # params
    dataset_name = model_cfg['dataset_name']
    assert dataset_name == 'nuscenes'  # TODO: only nuscenes dataset supported now
    downsample_pct = model_cfg['downsample_pct']
    train_dir = f"./logs/mos_baseline/mos4d_train/{downsample_pct}%{dataset_name}"
    os.makedirs(train_dir, exist_ok=True)
    quant_size = model_cfg['quant_size']
    batch_size = model_cfg['batch_size']
    time = round(model_cfg['n_input'] * model_cfg['time_interval'] + model_cfg['n_input'] * model_cfg['n_skip'] * model_cfg['time_interval'], 2)
    model_params = f"vs-{quant_size}_t-{time}_bs-{batch_size}"

    # dataloader
    nusc = NuScenes(dataroot=dataset_cfg["nuscenes"]["root"], version=dataset_cfg["nuscenes"]["version"])
    train_set = NuscMosDataset(nusc, model_cfg, dataset_cfg, 'train')
    val_set = NuscMosDataset(nusc, model_cfg, dataset_cfg, 'val')
    dataloader = NuscDataloader(nusc, model_cfg, train_set, val_set, True)
    dataloader.setup()
    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()

    # model
    model = MosNetwork(model_cfg, True)

    # lr_monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # tensorboard logger
    tb_logger = pl_loggers.TensorBoardLogger(train_dir, name=model_params, default_hp_metric=False)

    # checkpoint saver
    checkpoint_saver = ModelCheckpoint(
        monitor="epoch",
        verbose=True,
        save_top_k=model_cfg['num_epoch'],
        mode="max",
        filename="{epoch}",
        every_n_epochs=5,
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
        resume_model_cfg = yaml.safe_load(open(os.path.join(train_dir, model_params, f'version_{resume_version}', "hparams.yaml")))
        assert set(model_cfg) == set(resume_model_cfg)
        resume_ckpt_path = os.path.join(train_dir, model_params, f'version_{resume_version}', 'checkpoints', 'last.ckpt')
        trainer.fit(model, train_dataloader, ckpt_path=resume_ckpt_path)
    else:
        trainer.fit(model, train_dataloader)


def mos_test(cfg_test, cfg_dataset):
    # model config
    model_dir = cfg_test['model_dir']
    cfg_model = yaml.safe_load(open(os.path.join(model_dir, "hparams.yaml")))
    log_dir = os.path.join(model_dir, 'results')
    os.makedirs(log_dir, exist_ok=True)

    # dataloader
    test_dataset = cfg_test['test_dataset']
    assert test_dataset == 'nuscenes'  # TODO: only support nuscenes test now.
    nusc = NuScenes(dataroot=cfg_dataset["nuscenes"]["root"], version=cfg_dataset["nuscenes"]["version"])
    train_set = NuscMosDataset(nusc, cfg_model, cfg_dataset, 'train')
    val_set = NuscMosDataset(nusc, cfg_model, cfg_dataset, 'val')
    dataloader = NuscDataloader(nusc, cfg_model, train_set, val_set, False)
    dataloader.setup()
    test_dataloader = dataloader.test_dataloader()

    for test_epoch in cfg_test["test_epoch"]:
        # logger
        date = datetime.now().strftime('%m%d')  # %m%d-%H%M
        log_file = os.path.join(log_dir, f"epoch_{test_epoch}_{date}.txt")
        formatter = logging.Formatter(fmt='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        logger = logging.getLogger(f'test epoch {test_epoch} logger')
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.info(str(cfg_test))
        logger.info(log_file)

        # test checkpoint
        ckpt_path = os.path.join(model_dir, "checkpoints", f"epoch={test_epoch}.ckpt")

        # model
        model = MosNetwork(cfg_model, False, model_dir=model_dir, test_epoch=test_epoch, test_logger=logger, nusc=nusc)
        iou_metric = ClassificationMetrics(n_classes=3, ignore_index=0)

        # predict
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=cfg_test["num_devices"], deterministic=True)
        trainer.predict(model, dataloaders=test_dataloader, return_predictions=True, ckpt_path=ckpt_path)

        # metric-1: object-level detection rate
        det_rate = model.det_mov_obj_cnt / (model.mov_obj_num + 1e-15)
        logger.info('Metric-1 moving object detection rate: %.3f' % (det_rate * 100))

        # metric-2: object-level iou
        obj_iou_mean = torch.mean(torch.tensor(model.object_iou_list)).item()
        logger.info('Metric-2 object-level avg. moving iou: %.3f' % (obj_iou_mean * 100))

        # metric-3: sample-level iou
        sample_iou_mean = torch.mean(torch.tensor(model.sample_iou_list)).item()
        logger.info('Metric-3 sample-level avg. moving iou: %.3f' % (sample_iou_mean * 100))

        # metric-4: point-level iou
        point_iou = iou_metric.get_iou(model.accumulated_conf_mat)
        logger.info('Metric-4 point-level moving iou: %.3f' % (point_iou[2] * 100))
        logger.info('Metric-4 point-level static iou: %.3f' % (point_iou[1] * 100))

        # statistics
        logger.info(f'Number of validation samples: {len(val_set)}, Number of samples without moving points: {model.no_mov_sample_num}')


if __name__ == "__main__":
    # deterministic
    set_deterministic(666)

    # mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'finetune', 'test'], default='train')
    parser.add_argument('--resume_version', type=int, default=-1)  # -1: not resuming
    parser.add_argument('--autodl', type=bool, default=False)
    args = parser.parse_args()

    # load config
    with open("configs/mos4d.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    with open("configs/dataset.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)

    # dataset root path at different platform
    if args.autodl:
        dataset_cfg['nuscenes']['root'] = '/root/autodl-tmp' + dataset_cfg['nuscenes']['root']
    else:
        dataset_cfg['nuscenes']['root'] = '/home/user' + dataset_cfg['nuscenes']['root']

    # training from scratch
    if args.mode == 'train':
        mos4d_baseline_train(cfg[args.mode], dataset_cfg, args.resume_version)
    elif args.mode == 'finetune':
        mos_finetune(cfg[args.mode], dataset_cfg, args.resume_version)
    elif args.mode == 'test':
        mos_test(cfg[args.mode], dataset_cfg)
