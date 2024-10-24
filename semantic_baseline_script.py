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
from models.semantic.models import SemanticNetwork
from datasets.nusc_utils import NuscDataloader
from datasets.semantic.nusc import NuscSemanticDataset


def semantic_baseline_train(model_cfg, dataset_cfg, resume_version):
    # params
    dataset_name = model_cfg['dataset_name']
    assert dataset_name == 'nuscenes'  # TODO: only nuscenes dataset supported now
    downsample_pct = model_cfg['downsample_pct']
    train_dir = f"./logs/semantic_baseline/semantic/{downsample_pct}%{dataset_name}"
    os.makedirs(train_dir, exist_ok=True)
    quant_size = model_cfg['quant_size']
    batch_size = model_cfg['batch_size']
    time = round(model_cfg['n_input'] * model_cfg['time_interval'] + model_cfg['n_input'] * model_cfg['n_skip'] * model_cfg['time_interval'], 2)
    model_params = f"vs-{quant_size}_t-{time}_bs-{batch_size}"

    # dataloader
    nusc = NuScenes(dataroot=dataset_cfg["nuscenes"]["root"], version=dataset_cfg["nuscenes"]["version"])
    train_set = NuscSemanticDataset(nusc, model_cfg, dataset_cfg, 'train')
    val_set = NuscSemanticDataset(nusc, model_cfg, dataset_cfg, 'val')
    dataloader = NuscDataloader(nusc, model_cfg, train_set, val_set, True)
    dataloader.setup()
    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()

    # model
    model = SemanticNetwork(model_cfg, True)

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


def semantic_finetune(model_cfg, dataset_cfg, resume_version):
    # pre-training checkpoint path
    pre_method = model_cfg["pretrain_method"]
    pre_dataset = model_cfg["pretrain_dataset"]
    pre_params = model_cfg["pretrain_params"]
    pre_version = model_cfg["pretrain_version"]
    pre_epoch = model_cfg["pretrain_epoch"]
    pretrain_model_dir = f"./logs/{pre_method}/{pre_dataset}/{pre_params}/version_{pre_version}/checkpoints" # TODO: rename
    pretrain_ckpt_name = f"epoch={pre_epoch}.ckpt"
    pretrain_ckpt_path = os.path.join(pretrain_model_dir, pretrain_ckpt_name)

    # fine-tuning params
    dataset_name = model_cfg['dataset_name']
    assert dataset_name == 'nuscenes'
    downsample_pct = model_cfg['downsample_pct']
    finetune_dir = f"./logs/{pre_method}(epoch-{pre_epoch})-semantic_finetune/{pre_dataset}-{downsample_pct}%{dataset_name}"  # TODO: rename
    os.makedirs(finetune_dir, exist_ok=True)
    quant_size = model_cfg['quant_size']
    batch_size = model_cfg['batch_size']
    time = round(model_cfg['n_input'] * model_cfg['time_interval'] + (model_cfg['n_input'] - 1) * model_cfg['n_skip'] * model_cfg['time_interval'], 2)
    finetune_model_params = f"vs-{quant_size}_t-{time}_bs-{batch_size}"

    # dataloader
    nusc = NuScenes(dataroot=dataset_cfg["nuscenes"]["root"], version=dataset_cfg["nuscenes"]["version"])
    train_set = NuscSemanticDataset(nusc, model_cfg, dataset_cfg, 'train')
    val_set = NuscSemanticDataset(nusc, model_cfg, dataset_cfg, 'val')
    dataloader = NuscDataloader(nusc, model_cfg, train_set, val_set, True)
    dataloader.setup()
    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()

    # load pre-trained encoder to fine-tuning model
    finetune_model = SemanticNetwork(model_cfg, True)

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
        resume_model_cfg = yaml.safe_load(open(os.path.join(finetune_dir, finetune_model_params, f'version_{resume_version}', "hparams.yaml")))
        assert set(model_cfg) == set(resume_model_cfg), "resume training: cfg dict keys are not the same."
        assert model_cfg == resume_model_cfg, f"resume training: cfg keys have different values."
        resume_ckpt_path = os.path.join(finetune_dir, finetune_model_params, f'version_{resume_version}', 'checkpoints', 'last.ckpt')
        trainer.fit(finetune_model, train_dataloader, ckpt_path=resume_ckpt_path)
    else:
        finetune_model = load_pretrained_encoder(pretrain_ckpt_path, finetune_model)
        trainer.fit(finetune_model, train_dataloader)


def semantic_test(cfg_test, cfg_dataset):
    # model config
    model_dir = cfg_test['model_dir']
    cfg_model = yaml.safe_load(open(os.path.join(model_dir, "hparams.yaml")))
    log_dir = os.path.join(model_dir, 'results')
    os.makedirs(log_dir, exist_ok=True)

    # dataloader
    test_dataset = cfg_test['test_dataset']
    assert test_dataset == 'nuscenes'  # TODO: only support nuscenes test now.
    nusc = NuScenes(dataroot=cfg_dataset["nuscenes"]["root"], version=cfg_dataset["nuscenes"]["version"])
    train_set = NuscSemanticDataset(nusc, cfg_model, cfg_dataset, 'train')
    val_set = NuscSemanticDataset(nusc, cfg_model, cfg_dataset, 'val')
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
        model = SemanticNetwork(cfg_model, False, model_dir=model_dir, test_epoch=test_epoch, test_logger=logger, nusc=nusc)

        # predict
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=cfg_test["num_devices"], deterministic=True)
        trainer.predict(model, dataloaders=test_dataloader, return_predictions=True, ckpt_path=ckpt_path)

        # metric
        semantic_cls_names = list(model.cfg_semantic['labels_16'].values())[1:]  # remove noise
        iou = model.ClassificationMetrics.get_iou(model.accumulated_conf_mat)[1:]

        # metric-1: semantic class mean IoU
        for cls_name, cls_iou in zip(semantic_cls_names, iou):
            logger.info(cls_name + "IoU: %.3f", cls_iou * 100)

        logger.info("Semantic class mean IoU: %.3f", np.mean(list(iou * 100)))


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
    with open("configs/semantic.yaml", "r") as f:
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
        semantic_baseline_train(cfg[args.mode], dataset_cfg, args.resume_version)
    elif args.mode == 'finetune':
        semantic_finetune(cfg[args.mode], dataset_cfg, args.resume_version)
    elif args.mode == 'test':
        semantic_test(cfg[args.mode], dataset_cfg)
