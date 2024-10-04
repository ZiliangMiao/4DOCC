# base
import argparse
import logging
import os
from datetime import datetime
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# torch
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# models
from models.ours.models import MutualObsPretrainNetwork
# dataset
from nuscenes.nuscenes import NuScenes
from datasets.nusc_utils import NuscDataloader
from datasets.ours.nusc import NuscMopDataset
# lib
from utils.deterministic import set_deterministic
from utils.metrics import ClassificationMetrics


def statistics(cfg_model, cfg_dataset):
    nusc = NuScenes(dataroot=cfg_dataset["nuscenes"]["root"], version=cfg_dataset["nuscenes"]["version"])
    nuscenes = NuscMopDataset(nusc, cfg_model, cfg_dataset, "train")
    num_occ_percentage_per_ray = []
    num_occ_total = 0
    num_free_total = 0
    num_bg_samples_per_ray = []
    valid_rays_percentage_per_sample = []
    num_valid_rays_total = 0
    num_rays_total = 0
    for sample_idx, sample in tqdm(enumerate(nuscenes)):
        _, pcds_4d, ray_to_bg_samples_dict = sample
        valid_rays_percentage_per_sample.append(len(ray_to_bg_samples_dict) / len(pcds_4d) * 100)
        num_valid_rays_total += len(ray_to_bg_samples_dict)
        num_rays_total += len(pcds_4d[pcds_4d[:, -1] == 0])  # only ref pcd
        for ray_idx, bg_samples in ray_to_bg_samples_dict.items():
            num_occ_ray = np.sum(bg_samples[:, -1] == 2)
            num_free_ray = np.sum(bg_samples[:, -1] == 1)
            num_occ_percentage_per_ray.append((num_occ_ray / (num_occ_ray + num_free_ray)) * 100)
            num_free_total += num_free_ray
            num_occ_total += num_occ_ray
            num_bg_samples_per_ray.append(len(bg_samples))
    # statistics of occ points percentage
    plt.figure()
    plt.hist(np.array(num_occ_percentage_per_ray), bins=20, color='skyblue', alpha=1, log=True)
    plt.title('Distribution of Occ Percentage')
    plt.xlabel('Occ Percentage (/Occ + Free)')
    plt.ylabel('Frequency (Ray Samples)')
    plt.savefig('./occ percentage distribution.jpg')
    print(
        f"Free Samples: {num_free_total}; Occ Samples: {num_occ_total}; Occ Percentage: {num_occ_total / (num_occ_total + num_free_total) * 100}%")

    # statistics of valid rays
    plt.figure()
    plt.hist(np.array(valid_rays_percentage_per_sample), bins=20, color='skyblue', alpha=1, log=True)
    plt.title('Distribution of Valid Rays Percentage')
    plt.xlabel('Valid Rays Percentage')
    plt.ylabel('Frequency (Ray Samples)')
    plt.savefig('./valid rays distribution.jpg')
    print(
        f"Valid Rays: {num_valid_rays_total}; Invalid Rays: {num_rays_total - num_valid_rays_total}; Valid Percentage: {num_valid_rays_total / num_rays_total * 100}%")

    # statistics of valid rays
    plt.figure()
    plt.hist(np.array(num_bg_samples_per_ray), bins=50, color='skyblue', alpha=1, log=True)
    plt.title('Distribution of Background Samples')
    plt.xlabel('Number of Background Samples')
    plt.ylabel('Frequency (Ray Samples)')
    plt.savefig('./background samples distribution.jpg')

    # show
    plt.show()


def mutual_observation_pretrain(model_cfg, dataset_cfg, resume_version):
    # pretrain params
    dataset_name = model_cfg['dataset_name']
    assert dataset_name == 'nuscenes'
    downsample_pct = model_cfg['downsample_pct']
    unk_pct = model_cfg['unk_samples_pct']
    train_bg_mop = model_cfg['train_bg_mop_samples']
    pretrain_method = f"mop_bg_{unk_pct}%unk" if train_bg_mop else f"mop_all_{unk_pct}%unk"
    pretrain_dir = f"./logs/ours/{pretrain_method}/{downsample_pct}%{dataset_name}"
    os.makedirs(pretrain_dir, exist_ok=True)
    quant_size = model_cfg['quant_size']
    batch_size = model_cfg['batch_size']
    time = round(model_cfg['n_input'] * model_cfg['time_interval'] + (model_cfg['n_input'] - 1) * model_cfg['n_skip'] *
                 model_cfg['time_interval'], 2)
    model_params = f"vs-{quant_size}_t-{time}_bs-{batch_size}"

    # dataloader
    nusc = NuScenes(dataroot=dataset_cfg["nuscenes"]["root"], version=dataset_cfg["nuscenes"]["version"])
    train_set = NuscMopDataset(nusc, model_cfg, dataset_cfg, 'train')
    val_set = NuscMopDataset(nusc, model_cfg, dataset_cfg, 'val')
    dataloader = NuscDataloader(nusc, model_cfg, train_set, val_set, True)
    dataloader.setup()
    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()

    # pretrain model
    pretrain_model = MutualObsPretrainNetwork(model_cfg, True, iters_per_epoch=len(train_dataloader))

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


def mos_finetune(model_cfg, dataset_cfg, resume_version):
    # pre-training checkpoint path
    pre_method = model_cfg["pretrain_method"]
    pre_dataset = model_cfg["pretrain_dataset"]
    pre_params = model_cfg["pretrain_params"]
    pre_version = model_cfg["pretrain_version"]
    pre_epoch = model_cfg["pretrain_epoch"]
    pretrain_model_dir = f"./logs/ours/{pre_method}/{pre_dataset}/{pre_params}/version_{pre_version}/checkpoints"
    pretrain_ckpt_name = f"epoch={pre_epoch}.ckpt"
    pretrain_ckpt_path = os.path.join(pretrain_model_dir, pretrain_ckpt_name)

    # fine-tuning params
    dataset_name = model_cfg['dataset_name']
    assert dataset_name == 'nuscenes'  # TODO: only nuscenes dataset supported now
    downsample_pct = model_cfg['downsample_pct']
    finetune_dir = f"./logs/ours/{pre_method}(epoch-{pre_epoch})-mos_finetune/{pre_dataset}-{downsample_pct}%{dataset_name}"
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
        finetune_model = load_pretrained_encoder(pretrain_ckpt_path, finetune_model)
        trainer.fit(finetune_model, train_dataloader)


def mutual_obs_test(cfg_test, cfg_dataset):
    # cfg model
    model_dir = cfg_test['model_dir']
    cfg_model = yaml.safe_load(open(os.path.join(model_dir, "hparams.yaml")))
    log_dir = os.path.join(model_dir, 'results')
    os.makedirs(log_dir, exist_ok=True)

    # dataloader
    test_dataset = cfg_test['test_dataset']
    assert test_dataset == 'nuscenes'  # TODO: only support nuscenes test now.
    nusc = NuScenes(dataroot=cfg_dataset["nuscenes"]["root"], version=cfg_dataset["nuscenes"]["version"])
    train_set = NuscMopDataset(nusc, cfg_model, cfg_dataset, 'train')
    val_set = NuscMopDataset(nusc, cfg_model, cfg_dataset, 'val')
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

        # model
        ckpt_path = os.path.join(model_dir, "checkpoints", f"epoch={test_epoch}.ckpt")
        model = MutualObsPretrainNetwork(cfg_model, False, model_dir=model_dir, test_epoch=test_epoch, test_logger=logger, save_pred_labels=cfg_test['save_pred_labels'])

        # metrics
        metrics = ClassificationMetrics(n_classes=3, ignore_index=0)

        # predict
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=cfg_test["num_devices"], deterministic=True)
        trainer.predict(model, dataloaders=test_dataloader, return_predictions=True, ckpt_path=ckpt_path)

        # pred iou and acc
        iou = metrics.get_iou(model.accumulated_conf_mat)
        unk_iou, free_iou, occ_iou = iou[0].item() * 100, iou[1].item() * 100, iou[2].item() * 100
        acc = metrics.get_acc(model.accumulated_conf_mat)
        unk_acc, free_acc, occ_acc = acc[0].item() * 100, acc[1].item() * 100, acc[2].item() * 100
        logger.info("Mutual Observation (IoU/Acc): [Unk %.3f/%.3f], [Free %.3f/%.3f], [Occ %.3f/%.3f]",
                     unk_iou, unk_acc, free_iou, free_acc, occ_iou, occ_acc)


if __name__ == "__main__":
    # deterministic
    set_deterministic(666)

    # mode
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['mop', 'mop_test'], default='mop')
    parser.add_argument('--resume_version', type=int, default=-1)  # -1: not resuming
    parser.add_argument('--autodl', type=bool, default=False)
    parser.add_argument('--statistics', type=bool, default=False)
    args = parser.parse_args()

    # load config
    with open('configs/ours.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    with open('configs/dataset.yaml', 'r') as f:
        dataset_cfg = yaml.safe_load(f)

    # statistics of background samples
    if args.statistics:
        statistics(cfg['mop'], dataset_cfg)

    # dataset root path at different platform
    if args.autodl:
        dataset_cfg['nuscenes']['root'] = '/root/autodl-tmp' + dataset_cfg['nuscenes']['root']
    else:
        dataset_cfg['nuscenes']['root'] = '/home/user' + dataset_cfg['nuscenes']['root']

    # pre-training on background for motion segmentation task
    if args.mode == 'mop':
        mutual_observation_pretrain(cfg[args.mode], dataset_cfg, args.resume_version)
    # background test
    elif args.mode == 'mop_test':
        mutual_obs_test(cfg[args.mode], dataset_cfg)
