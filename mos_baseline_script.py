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
from ours_script import general_pipeline
from models.mos4d.models import MosNetwork
from datasets.nusc_utils import NuscDataloader
from datasets.mos4d.nusc import NuscMosDataset
from ours_script import mos_finetune


def mos_train_from_scratch(cfg_model, cfg_dataset):
    # fine-tuning params
    dataset_name = cfg_model['dataset_name']
    assert dataset_name == 'nuscenes'  # TODO: only nuscenes dataset supported now
    downsample_pct = cfg_model['downsample_pct']
    train_dir = f"./logs/mos_baseline/mos4d_train/{downsample_pct}%{dataset_name}"

    # load pre-trained encoder to fine-tuning model
    model = MosNetwork(cfg_model, True)

    # dataloader
    nusc = NuScenes(dataroot=cfg_dataset["nuscenes"]["root"], version=cfg_dataset["nuscenes"]["version"])
    train_set = NuscMosDataset(nusc, cfg_model, cfg_dataset, 'train')
    val_set = NuscMosDataset(nusc, cfg_model, cfg_dataset, 'val')
    dataloader = NuscDataloader(nusc, cfg_model, train_set, val_set, True)
    dataloader.setup()
    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()

    # training
    general_pipeline(cfg_model, model, train_dataloader, train_dir)


def mos_test(cfg_test, cfg_dataset):
    # test checkpoint
    model_dir = cfg_test['model_dir']
    test_epoch = cfg_test["test_epoch"]
    ckpt_path = os.path.join(model_dir, "checkpoints", f"epoch={test_epoch}.ckpt")

    # model
    cfg_model = yaml.safe_load(open(os.path.join(model_dir, "hparams.yaml")))
    model = MosNetwork(cfg_model, False, model_dir=model_dir, test_epoch=test_epoch)

    # dataloader
    test_dataset = cfg_test['test_dataset']
    assert test_dataset == 'nuscenes'  # TODO: only support nuscenes test now.
    nusc = NuScenes(dataroot=cfg_dataset["nuscenes"]["root"], version=cfg_dataset["nuscenes"]["version"])
    train_set = NuscMosDataset(nusc, cfg_model, cfg_dataset, 'train')
    val_set = NuscMosDataset(nusc, cfg_model, cfg_dataset, 'val')
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

    # pred iou
    conf_mat_list = [output["confusion_matrix"] for output in pred_outputs]
    acc_conf_mat = torch.zeros(3, 3)
    for conf_mat in conf_mat_list:
        acc_conf_mat = acc_conf_mat.add(conf_mat)
    iou = metrics.get_iou(acc_conf_mat)
    sta_iou = iou[1]
    mov_iou = iou[2]
    logging.info('Moving Object IoU w/o ego vehicle (point-level): %.3f' % (sta_iou.item() * 100))
    logging.info('Moving Object IoU w/o ego vehicle (point-level): %.3f' % (mov_iou.item() * 100))

    mov_iou_list = model.get_mov_iou_list()
    mov_iou_samples = torch.tensor(mov_iou_list)
    mov_iou_mean = torch.mean(mov_iou_samples)
    mov_iou_var = torch.var(mov_iou_samples)
    logging.info('Moving Object IoU Mean (sample-level): %.3f' % (mov_iou_mean.item() * 100))
    logging.info('Moving Object IoU Var (sample-level): %.3f' % (mov_iou_var.item() * 100))


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

    # mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'finetune', 'test'], default='test')
    args = parser.parse_args()

    # load config
    with open("configs/mos4d.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    with open("configs/dataset.yaml", "r") as f:
        cfg_dataset = yaml.safe_load(f)

    # training from scratch
    if args.mode == 'train':
        mos_train_from_scratch(cfg[args.mode], cfg_dataset)
    elif args.mode == 'finetune':
        mos_finetune(cfg[args.mode], cfg_dataset)
    elif args.mode == 'test':
        mos_test(cfg[args.mode], cfg_dataset)
