# base
import argparse
import re
import os
import click
import yaml
import copy
import numpy as np
from matplotlib import pyplot as plt
from nuscenes.utils.splits import create_splits_logs
from nuscenes.utils.geometry_utils import points_in_box
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
from datasets.nusc_utils import NuscDataloader, split_logs_to_samples, get_input_sd_toks, get_transformed_pcd
from datasets.mos4d.nusc import NuscMosDataset


def mos4d_baseline_train(model_cfg, dataset_cfg, resume_version):
    # params
    dataset_name = model_cfg['dataset_name']
    assert dataset_name == 'nuscenes'  # TODO: only nuscenes dataset supported now
    downsample_pct = model_cfg['downsample_pct']
    if model_cfg['shuffle']:
        train_dir = f"./logs/mos_baseline/mos4d_shuffle/{downsample_pct}%{dataset_name}"
    else:
        train_dir = f"./logs/mos_baseline/mos4d/{downsample_pct}%{dataset_name}"
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
        every_n_epochs=10,
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


def load_pretrained_encoder(ckpt_path, model, use_mlp_decoder:bool):
    print(f"Load pretrained encoder from checkpoint {ckpt_path}")
    checkpoint = torch.load(ckpt_path)
    pretrained_dict = checkpoint["state_dict"]
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'decoder' not in k}

    if use_mlp_decoder:
        if 'loss.weight' in pretrained_dict.keys():
            pretrained_dict.pop('loss.weight')
    else: # filter out unnecessary keys (generate new dict)
        if 'encoder.MinkUNet.final.kernel' in pretrained_dict.keys():
            pretrained_dict.pop('encoder.MinkUNet.final.kernel')
        if 'encoder.MinkUNet.final.bias' in pretrained_dict.keys():
            pretrained_dict.pop('encoder.MinkUNet.final.bias')
        if 'loss.weight' in pretrained_dict.keys():
            pretrained_dict.pop('loss.weight')
    # overwrite finetune model dict
    model_dict.update(pretrained_dict)
    # load the pretrained model dict
    model.load_state_dict(model_dict)
    return model


def mos_finetune(model_cfg, dataset_cfg, resume_version):
    # pre-training checkpoint path
    pre_method = model_cfg["pretrain_method"]
    pre_dataset = model_cfg["pretrain_dataset"]
    pre_params = model_cfg["pretrain_params"]
    pre_version = model_cfg["pretrain_version"]
    pre_epoch = model_cfg["pretrain_epoch"]
    pretrain_model_dir = f"./logs/{pre_method}/{pre_dataset}/{pre_params}/version_{pre_version}/checkpoints"
    pretrain_ckpt_name = f"epoch={pre_epoch}.ckpt"
    pretrain_ckpt_path = os.path.join(pretrain_model_dir, pretrain_ckpt_name)

    # fine-tuning params
    dataset_name = model_cfg['dataset_name']
    assert dataset_name == 'nuscenes'
    downsample_pct = model_cfg['downsample_pct']
    finetune_dir = f"./logs/{pre_method}(epoch-{pre_epoch})-mos_finetune/{pre_dataset}-{downsample_pct}%{dataset_name}"
    os.makedirs(finetune_dir, exist_ok=True)
    quant_size = model_cfg['quant_size']
    batch_size = model_cfg['batch_size']
    time = round(model_cfg['n_input'] * model_cfg['time_interval'] + (model_cfg['n_input'] - 1) * model_cfg['n_skip'] * model_cfg['time_interval'], 2)
    finetune_model_params = f"vs-{quant_size}_t-{time}_bs-{batch_size}"

    # dataloader
    nusc = NuScenes(dataroot=dataset_cfg["nuscenes"]["root"], version=dataset_cfg["nuscenes"]["version"])
    train_set = NuscMosDataset(nusc, model_cfg, dataset_cfg, 'train')
    val_set = NuscMosDataset(nusc, model_cfg, dataset_cfg, 'val')
    dataloader = NuscDataloader(nusc, model_cfg, train_set, val_set, True)
    dataloader.setup()
    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()

    # load pre-trained encoder to fine-tuning model
    finetune_model = MosNetwork(model_cfg, True)

    # lr_monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # tensorboard logger
    tb_logger = pl_loggers.TensorBoardLogger(finetune_dir, name=finetune_model_params, default_hp_metric=False)

    # checkpoint saver
    checkpoint_saver = ModelCheckpoint(
        monitor="epoch",
        verbose=True,
        save_top_k=model_cfg['num_epoch'],
        mode="max",
        filename="{epoch}",
        every_n_epochs=10,
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
        resume_model_cfg = yaml.safe_load(open(os.path.join(finetune_dir, finetune_model_params, f'version_{resume_version}', "hparams.yaml")))
        assert set(model_cfg) == set(resume_model_cfg), "resume training: cfg dict keys are not the same."
        assert model_cfg == resume_model_cfg, f"resume training: cfg keys have different values."
        resume_ckpt_path = os.path.join(finetune_dir, finetune_model_params, f'version_{resume_version}', 'checkpoints', 'last.ckpt')
        trainer.fit(finetune_model, train_dataloader, ckpt_path=resume_ckpt_path)
    else:
        finetune_model = load_pretrained_encoder(pretrain_ckpt_path, finetune_model, model_cfg['use_mlp_decoder'])
        trainer.fit(finetune_model, train_dataloader)


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

def mov_pts_statistics(cfg_dataset, cfg_model):
    nusc = NuScenes(dataroot=dataset_cfg["nuscenes"]["root"], version=dataset_cfg["nuscenes"]["version"])
    split_logs = create_splits_logs('train', nusc)
    split_logs_val = create_splits_logs('val', nusc)
    split_logs = split_logs + split_logs_val
    sample_toks = split_logs_to_samples(nusc, split_logs)
    sample_to_sd_toks_dict = get_input_sd_toks(nusc, cfg_model, sample_toks)

    mov_obj_num = 0
    pts_in_mov_obj_list = []
    for sample_tok in list(sample_to_sd_toks_dict.keys()):
        # sample and sample data
        sample = nusc.get('sample', sample_tok)
        sd_tok = sample['data']['LIDAR_TOP']
        sample_data = nusc.get('sample_data', sd_tok)

        # point cloud
        org, pcd, ts, valid_mask = get_transformed_pcd(nusc, cfg_model, sd_tok, sd_tok)

        # labels
        mos_labels_dir = os.path.join(cfg_dataset["nuscenes"]["root"], "mos_labels", cfg_dataset["nuscenes"]["version"])
        mos_label_file = os.path.join(mos_labels_dir, sd_tok + "_mos.label")
        mos_labels = torch.tensor(np.fromfile(mos_label_file, dtype=np.uint8))[valid_mask]

        # number of points in moving object
        _, bbox_list, _ = nusc.get_sample_data(sd_tok, selected_anntokens=sample['anns'], use_flat_vehicle_coordinates=False)
        for ann_tok, box in zip(sample['anns'], bbox_list):
            ann = nusc.get('sample_annotation', ann_tok)
            if ann['num_lidar_pts'] == 0: continue
            obj_pts_mask = points_in_box(box, pcd[:, :3].T)
            if np.sum(obj_pts_mask) == 0:  # not lidar points in obj bbox
                continue
            # gt and pred object labels
            gt_obj_labels = mos_labels[obj_pts_mask]
            mov_pts_mask = gt_obj_labels == 2
            num_mov_pts = torch.sum(mov_pts_mask)
            if num_mov_pts == 0:  # static object
                continue
            else:
                mov_obj_num += 1
                pts_in_mov_obj_list.append(num_mov_pts)

    # histogram
    fig_1 = plt.figure()
    ax_1 = fig_1.gca()
    hist_1 = ax_1.hist(pts_in_mov_obj_list, range=(0,400), bins=100, rwidth=0, align='left', log=True)
    ax_1.set_xlabel("Number of points in moving objects")
    ax_1.set_ylabel("Frequency (Num of moving objects)")
    fig_1.savefig("./statistics_points_in_mov_obj_3.png", dpi=1000)

    fig_2 = plt.figure()
    ax_2 = fig_2.gca()
    hist_2 = ax_2.hist(pts_in_mov_obj_list, range=(0, 200), bins=100, rwidth=0, align='left', log=True)
    ax_2.set_xlabel("Number of points in moving objects")
    ax_2.set_ylabel("Frequency (Num of moving objects)")
    fig_2.savefig("./statistics_points_in_mov_obj_4.png", dpi=1000)

    fig_3 = plt.figure()
    ax_3 = fig_3.gca()
    hist_3 = ax_3.hist(pts_in_mov_obj_list, range=(0, 200), bins=200, rwidth=0, align='left', log=True)
    ax_3.set_xlabel("Number of points in moving objects")
    ax_3.set_ylabel("Frequency (Num of moving objects)")
    fig_3.savefig("./statistics_points_in_mov_obj_5.png", dpi=1000)

    fig_4 = plt.figure()
    ax_4 = fig_4.gca()
    hist_4 = ax_4.hist(pts_in_mov_obj_list, range=(0, 400), bins=200, rwidth=0, align='left', log=True)
    ax_4.set_xlabel("Number of points in moving objects")
    ax_4.set_ylabel("Frequency (Num of moving objects)")
    fig_4.savefig("./statistics_points_in_mov_obj_6.png", dpi=1000)

    asd = 1


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
        dataset_cfg['nuscenes']['root'] = '/home/ziliang' + dataset_cfg['nuscenes']['root']

    # mov_pts_statistics(dataset_cfg, cfg[args.mode])

    # training from scratch
    if args.mode == 'train':
        mos4d_baseline_train(cfg[args.mode], dataset_cfg, args.resume_version)
    elif args.mode == 'finetune':
        mos_finetune(cfg[args.mode], dataset_cfg, args.resume_version)
    elif args.mode == 'test':
        mos_test(cfg[args.mode], dataset_cfg)
