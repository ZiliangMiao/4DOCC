# base
import argparse
import logging
import os
import sys
from datetime import datetime
import yaml
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# models
from models.uno.models import UnONetwork
# dataset
from nuscenes.nuscenes import NuScenes
from datasets.nusc_utils import NuscDataloader
from datasets.uno.nusc import NuscUnODataset
# lib
from utils.deterministic import set_deterministic
from utils.metrics import ClassificationMetrics


def uno_pretrain(model_cfg, dataset_cfg, resume_version):
    # model params
    dataset_name = model_cfg['dataset_name']
    assert dataset_name == 'nuscenes'
    downsample_pct = model_cfg['downsample_pct']
    pretrain_dir = f"./logs/forecast_baseline/uno/{downsample_pct}%{dataset_name}"
    os.makedirs(pretrain_dir, exist_ok=True)
    quant_size = model_cfg['quant_size']
    batch_size = model_cfg['batch_size']
    time = round(model_cfg['n_input'] * model_cfg['time_interval'] + (model_cfg['n_input'] - 1) * model_cfg['n_skip'] *
                 model_cfg['time_interval'], 2)
    model_params = f"vs-{quant_size}_t-{time}_bs-{batch_size}"

    # dataloader
    nusc = NuScenes(dataroot=dataset_cfg["nuscenes"]["root"], version=dataset_cfg["nuscenes"]["version"])
    train_set = NuscUnODataset(nusc, model_cfg, dataset_cfg, 'train')
    val_set = NuscUnODataset(nusc, model_cfg, dataset_cfg, 'val')
    dataloader = NuscDataloader(nusc, model_cfg, train_set, val_set, True)
    dataloader.setup()
    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()

    # pretrain model
    pretrain_model = UnONetwork(model_cfg, True, iters_per_epoch=len(train_dataloader))

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
        resume_model_cfg = yaml.safe_load(open(os.path.join(pretrain_dir, model_params, f'version_{resume_version}', "hparams.yaml")))
        assert set(model_cfg) == set(resume_model_cfg), "resume training: cfg dict keys are not the same."
        assert model_cfg == resume_model_cfg, f"resume training: cfg keys have different values."
        resume_ckpt_path = os.path.join(pretrain_dir, model_params, f'version_{resume_version}', 'checkpoints', 'last.ckpt')
        trainer.fit(pretrain_model, train_dataloader, ckpt_path=resume_ckpt_path)
    else:
        trainer.fit(pretrain_model, train_dataloader)


def uno_test(cfg_test, cfg_dataset):
    # test checkpoint
    model_dir = cfg_test['model_dir']
    test_epoch = cfg_test["test_epoch"]
    ckpt_path = os.path.join(model_dir, "checkpoints", f"epoch={test_epoch}.ckpt")

    # model
    cfg_model = yaml.safe_load(open(os.path.join(model_dir, "hparams.yaml")))
    model = UnONetwork(cfg_model, False, model_dir=model_dir, test_epoch=test_epoch)

    # dataloader
    test_dataset = cfg_test['test_dataset']
    assert test_dataset == 'nuscenes'
    nusc = NuScenes(dataroot=cfg_dataset["nuscenes"]["root"], version=cfg_dataset["nuscenes"]["version"])
    train_set = NuscUnODataset(nusc, cfg_model, cfg_dataset, 'train')
    val_set = NuscUnODataset(nusc, cfg_model, cfg_dataset, 'val')
    dataloader = NuscDataloader(nusc, cfg_model, train_set, val_set, False)
    dataloader.setup()
    test_dataloader = dataloader.test_dataloader()

    # logger
    log_folder = os.path.join(model_dir, 'results')
    os.makedirs(log_folder, exist_ok=True)
    date = datetime.now().strftime('%m%d-%H%M')
    log_file = os.path.join(log_folder, f"epoch_{test_epoch}_{date}.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(cfg_test))
    logging.info(log_file)

    # metrics
    metrics = ClassificationMetrics(n_classes=3, ignore_index=0)

    # predict
    trainer = Trainer(accelerator="gpu", strategy="ddp", devices=cfg_test["num_devices"], deterministic=True)
    pred_outputs = trainer.predict(model, dataloaders=test_dataloader, return_predictions=True, ckpt_path=ckpt_path)

    # pred iou and acc
    conf_mat_list = [output["confusion_matrix"] for output in pred_outputs]
    acc_conf_mat = torch.zeros(2, 2)
    for conf_mat in conf_mat_list:
        acc_conf_mat = acc_conf_mat.add(conf_mat)
    iou = metrics.get_iou(acc_conf_mat)
    free_iou, occ_iou = iou[0].item() * 100, iou[1].item() * 100
    acc = metrics.get_acc(acc_conf_mat)
    free_acc, occ_acc = acc[0].item() * 100, acc[1].item() * 100
    logging.info("Background Occ IoU/Acc: %.3f / %.3f, Free IoU/Acc: %.3f / %.3f", occ_iou, occ_acc, free_iou, free_acc)


if __name__ == "__main__":
    # deterministic
    set_deterministic(666)

    # mode
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['uno', 'uno_test'], default='uno')
    parser.add_argument('--resume_version', type=int, default=-1)  # -1: not resuming
    parser.add_argument('--autodl', type=bool, default=False)
    args = parser.parse_args()

    # load config
    with open('configs/uno.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    with open('configs/dataset.yaml', 'r') as f:
        dataset_cfg = yaml.safe_load(f)

    # dataset root path at different platform
    if args.autodl:
        dataset_cfg['nuscenes']['root'] = '/root/autodl-tmp' + dataset_cfg['nuscenes']['root']
    else:
        dataset_cfg['nuscenes']['root'] = '/home/user' + dataset_cfg['nuscenes']['root']

    # pre-training on background for motion segmentation task
    if args.mode == 'uno':
        uno_pretrain(cfg[args.mode], dataset_cfg, args.resume_version)
    # background test
    elif args.mode == 'uno_test':
        uno_test(cfg[args.mode], dataset_cfg)
