# base
import os
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from nuscenes.utils.splits import create_splits_logs
from nuscenes.utils.geometry_utils import points_in_box

# model
import torch
from models.mos4d.models import MosNetwork
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# data
from nuscenes.nuscenes import NuScenes
from datasets.mos4d.kitti import KittiMOSDataset
from datasets.mos4d.nusc import NuscMosDataset
from datasets.dataloader import Dataloader
from datasets.nusc_utils import split_logs_to_samples, get_input_sd_toks, get_transformed_pcd

# utils
from utils.metrics import ClassificationMetrics
from utils.deterministic import set_deterministic


def build_dataloader(dataset_name, model_cfg, dataset_cfg, mode, cfg_test=None):
    if dataset_name == 'nuscenes':
        nusc = NuScenes(dataroot=dataset_cfg["nuscenes"]["root"], version=dataset_cfg["nuscenes"]["version"])
        train_set = NuscMosDataset(nusc, model_cfg, dataset_cfg, 'train')
        val_set = NuscMosDataset(nusc, model_cfg, dataset_cfg, 'val')
        dataloader = Dataloader(model_cfg, train_set, val_set, mode in ['train', 'finetune'], nusc=nusc)
    elif dataset_name == 'sekitti':
        train_set = KittiMOSDataset(model_cfg, dataset_cfg, split='train', cfg_test=cfg_test) if mode == 'test' else KittiMOSDataset(model_cfg, dataset_cfg, split='train')
        val_set = KittiMOSDataset(model_cfg, dataset_cfg, split='val', cfg_test=cfg_test) if mode == 'test' else KittiMOSDataset(model_cfg, dataset_cfg, split='val')
        dataloader = Dataloader(model_cfg, train_set, val_set, mode in ['train', 'finetune'])
    else:
        print("Not a supported dataset.")
        return None
    dataloader.setup()
    if mode == 'test':
        return dataloader.test_dataloader()
    elif mode == 'val':
        return dataloader.val_dataloader()
    elif mode in ['train', 'finetune']:
        return dataloader.train_dataloader(), dataloader.val_dataloader()
    else:
        return None


def build_trainer(model_cfg, log_dir, model_params, callbacks):
    tb_logger = pl_loggers.TensorBoardLogger(log_dir, name=model_params, default_hp_metric=False)
    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp",
        num_nodes=1,
        devices=model_cfg["num_devices"],
        logger=tb_logger,
        log_every_n_steps=1,
        max_epochs=model_cfg['num_epoch'],
        accumulate_grad_batches=model_cfg["acc_batches"],
        callbacks=callbacks,
    )
    return trainer, tb_logger


def mos4d_baseline_train(model_cfg, dataset_cfg, resume_training:bool):
    # if resume training, load original config
    if resume_training:
        resume_model_cfg = yaml.safe_load(open(os.path.join(model_cfg['resume_path'], "hparams.yaml")))
        resume_ckpt_path = os.path.join(model_cfg['resume_path'], 'checkpoints', 'last.ckpt')
        model_cfg = resume_model_cfg
        
    # params
    train_dir = f"./logs/mos_baseline/mos4d_shuffle/{model_cfg['downsample_pct']}%{model_cfg['dataset_name']}"
    os.makedirs(train_dir, exist_ok=True)
    time = round(model_cfg['n_input'] * model_cfg['time_interval'] + model_cfg['n_input'] * model_cfg['n_skip'] * model_cfg['time_interval'], 2)
    model_params = f"vs-{model_cfg['quant_size']}_t-{time}_bs-{model_cfg['batch_size']}"

    # dataloader
    train_dataloader, val_dataloader = build_dataloader(model_cfg['dataset_name'], model_cfg, dataset_cfg, 'train')

    # model
    model = MosNetwork(model_cfg, dataset_cfg, True)

    # lr_monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

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

    # trainer and tensorboard logger
    trainer, tb_logger = build_trainer(model_cfg, train_dir, model_params, [lr_monitor, checkpoint_saver])

    # training
    if resume_training:
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


def mos_finetune(model_cfg, dataset_cfg, resume_training:bool):
    if resume_training:
        resume_model_cfg = yaml.safe_load(open(os.path.join(model_cfg['resume_path'], "hparams.yaml")))
        resume_ckpt_path = os.path.join(model_cfg['resume_path'], 'checkpoints', 'last.ckpt')
        model_cfg = resume_model_cfg
     
    # pre-training checkpoint path
    pre_method = model_cfg['pretrain_method']
    pre_dataset = model_cfg['pretrain_dataset']
    pre_params = model_cfg['pretrain_params']
    pre_version = model_cfg['pretrain_version']
    pre_epoch = model_cfg['pretrain_epoch']
    pre_model_dir = f"./logs/{pre_method}/{pre_dataset}/{pre_params}/version_{pre_version}/checkpoints"
    pre_ckpt_name = f"epoch={pre_epoch}.ckpt"
    pre_ckpt_path = os.path.join(pre_model_dir, pre_ckpt_name)

    # fine-tuning params
    dataset_name = model_cfg['dataset_name']
    downsample_pct = model_cfg['downsample_pct']
    finetune_dir = f"./logs/{pre_method}(epoch-{pre_epoch})-mos_finetune/{pre_dataset}-{downsample_pct}%{dataset_name}"
    os.makedirs(finetune_dir, exist_ok=True)
    time = round(model_cfg['n_input'] * model_cfg['time_interval'] + model_cfg['n_input'] * model_cfg['n_skip'] * model_cfg['time_interval'], 2)
    finetune_model_params = f"vs-{model_cfg['quant_size']}_t-{time}_bs-{model_cfg['batch_size']}"

    # dataloader
    train_dataloader, val_dataloader = build_dataloader(dataset_name, model_cfg, dataset_cfg, 'finetune')

    # load pre-trained encoder to fine-tuning model
    finetune_model = MosNetwork(model_cfg, dataset_cfg, True)

    # lr_monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

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

    # trainer and tensorboard logger
    trainer, tb_logger = build_trainer(model_cfg, finetune_dir, finetune_model_params, [lr_monitor, checkpoint_saver])

    # training
    if resume_training:  # resume training
        trainer.fit(finetune_model, train_dataloader, ckpt_path=resume_ckpt_path)
    else:
        finetune_model = load_pretrained_encoder(pre_ckpt_path, finetune_model, model_cfg['use_mlp_decoder'])
        trainer.fit(finetune_model, train_dataloader)


def mos_eval(cfg, cfg_dataset):
    '''
    This function is for MOS evaluation, using both object-wise recall and IoU_w/o metrics. Used for both validation and testing
    Args:
        cfg:
        cfg_dataset:

    Returns:

    '''
    # model config
    model_dir = cfg['model_dir']
    cfg_model = yaml.safe_load(open(os.path.join(model_dir, "hparams.yaml")))
    log_dir = os.path.join(model_dir, 'results')
    os.makedirs(log_dir, exist_ok=True)

    # dataloader
    dataset_name = cfg['eval_dataset']
    test_dataloader, train_set, val_set, nusc = build_dataloader(dataset_name, cfg_model, cfg_dataset, 'test', cfg)
    
    # create excel file
    excel_file = os.path.join(log_dir, 'metrics.xlsx')
    if not os.path.exists(excel_file):
        # set column names
        df = pd.DataFrame(columns=['epoch', 'obj iou', 'iou'])
        df.to_excel(excel_file, index=False)
    else:
        df = pd.read_excel(excel_file)

    for test_epoch in cfg["eval_epoch"]:
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
        logger.info(str(cfg))
        logger.info(log_file)

        # test checkpoint
        ckpt_path = os.path.join(model_dir, "checkpoints", f"epoch={test_epoch}.ckpt")

        # model
        if dataset_name == 'nuscenes':
            model = MosNetwork(cfg_model, cfg_dataset, False, model_dir=model_dir, test_epoch=test_epoch, test_logger=logger, nusc=nusc, metric_obj=cfg["metric_obj"])
            iou_metric = ClassificationMetrics(n_classes=3, ignore_index=0)
        elif dataset_name == 'sekitti':
            model = MosNetwork(cfg_model, cfg_dataset, False, model_dir=model_dir, test_epoch=test_epoch, test_logger=logger, metric_obj=cfg["metric_obj"])
            iou_metric = ClassificationMetrics(n_classes=3, ignore_index=0)
        else:
            model = None
            iou_metric = None

        # predict
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=cfg["num_devices"], deterministic=True)
        trainer.predict(model, dataloaders=test_dataloader, return_predictions=True, ckpt_path=ckpt_path)

        # metric: object-level detection rate
        # det_rate = model.det_mov_obj_cnt / (model.mov_obj_num + 1e-15)
        # logger.info('Metric-1 moving object detection rate: %.3f' % (det_rate * 100))

        # metric-1: object-level iou
        obj_iou_mean = torch.mean(torch.tensor(model.object_iou_list)).item()
        logger.info('Metric-1 object-level avg. moving iou: %.3f' % (obj_iou_mean * 100))

        # metric: sample-level iou
        # sample_iou_mean = torch.mean(torch.tensor(model.sample_iou_list)).item()
        # logger.info('Metric-3 sample-level avg. moving iou: %.3f' % (sample_iou_mean * 100))

        # metric-2: point-level iou
        point_iou = iou_metric.get_iou(model.accumulated_conf_mat)
        logger.info('Metric-2 point-level moving iou: %.3f' % (point_iou[2] * 100))
        logger.info('Metric-2 point-level static iou: %.3f' % (point_iou[1] * 100))

        # statistics
        logger.info(f'Number of validation samples: {len(val_set)}, Number of samples without moving points: {model.no_mov_sample_num}')

        # append new row to dataframe
        new_row = pd.DataFrame({
            'epoch': [test_epoch],
            'obj iou': [obj_iou_mean * 100], 
            'iou': [point_iou[2].item() * 100]
        })
        df = pd.concat([df, new_row], ignore_index=True)
        # save the updated dataframe to excel
        df.to_excel(excel_file, index=False)


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


if __name__ == "__main__":
    # deterministic
    set_deterministic(666)

    # mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'finetune', 'val', 'test'], default='val')
    parser.add_argument('--resume_version', type=int, default=-1)  # -1: not resuming
    parser.add_argument('--autodl', type=bool, help="autodl server", default=False)
    parser.add_argument('--mars', type=bool, help="mars server", default=False)
    parser.add_argument('--hpc', type=bool, help="hpc server", default=False)
    args = parser.parse_args()

    # load config
    with open("configs/mos4d.yaml", "r") as f:
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

    # mov_pts_statistics(dataset_cfg, cfg[args.mode])

    # training from scratch
    if args.mode == 'train':  # train set = full org train set
        mos4d_baseline_train(cfg[args.mode], dataset_cfg, args.resume_version)
    elif args.mode == 'finetune':  # finetune train set = subset of org train set
        mos_finetune(cfg[args.mode], dataset_cfg, args.resume_version)
    elif args.mode == 'val':  # val set = a subset of org train set
        mos_eval(cfg[args.mode], dataset_cfg)
    elif args.mode == 'test':  # test set = org val set
        mos_eval(cfg[args.mode], dataset_cfg)
