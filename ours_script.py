# base
import argparse
import os


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
from models.ours.models import MotionPretrainNetwork
from models.mos4d.models import MosNetwork
# dataset
from nuscenes.nuscenes import NuScenes
from datasets.nusc_utils import NuscDataloader
from datasets.ours.nusc import NuscBgDataset
from datasets.mos4d.nusc import NuscMosDataset
# lib
from utils.deterministic import set_deterministic
from utils.metrics import ClassificationMetrics


def statistics(cfg_model, cfg_dataset):
    nusc = NuScenes(dataroot=cfg_dataset["nuscenes"]["root"], version=cfg_dataset["nuscenes"]["version"])
    nuscenes = NuscBgDataset(nusc, cfg_model, cfg_dataset, "train")
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


def general_pipeline(cfg_model, model, train_dataloader, logger_dir):
    # lr_monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # checkpoint saver
    checkpoint_saver = ModelCheckpoint(
        monitor="epoch",
        verbose=True,
        save_top_k=cfg_model['num_epoch'],
        mode="max",
        filename="{epoch}",
        every_n_epochs=5,
        save_last=True,
    )

    # logger
    quant_size = cfg_model['quant_size']
    batch_size = cfg_model['batch_size']
    time = round(cfg_model['n_input'] * cfg_model['time_interval'] + (cfg_model['n_input'] - 1) * cfg_model['n_skip'] *
                 cfg_model['time_interval'], 2)
    finetune_model_params = f"vs-{quant_size}_t-{time}_bs-{batch_size}"
    os.makedirs(logger_dir, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(logger_dir, name=finetune_model_params, default_hp_metric=False)

    # trainer
    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=cfg_model["num_devices"],
        logger=tb_logger,
        log_every_n_steps=1,
        max_epochs=cfg_model['num_epoch'],
        accumulate_grad_batches=cfg_model["acc_batches"],  # accumulate batches, default=1
        callbacks=[lr_monitor, checkpoint_saver],
    )

    # training
    resume_ckpt_path = cfg_model["resume_ckpt"]
    trainer.fit(model, train_dataloader, ckpt_path=resume_ckpt_path)


def background_pretrain(cfg, cfg_dataset, mode: str):
    # model cfg
    cfg_model = cfg[mode]

    # logger
    dataset_name = cfg_model['dataset_name']
    assert dataset_name == 'nuscenes'  # TODO: only nuscenes dataset supported now
    downsample_pct = cfg_model['downsample_pct']
    pretrain_dir = f"./logs/ours/{mode}/{downsample_pct}%{dataset_name}"

    # pretrain model
    pretrain_model = MotionPretrainNetwork(cfg_model, True)

    # dataloader
    nusc = NuScenes(dataroot=cfg_dataset["nuscenes"]["root"], version=cfg_dataset["nuscenes"]["version"])
    train_set = NuscBgDataset(nusc, cfg_model, cfg_dataset, 'train')
    val_set = NuscBgDataset(nusc, cfg_model, cfg_dataset, 'val')
    dataloader = NuscDataloader(nusc, cfg_model, train_set, val_set, True)
    dataloader.setup()
    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()

    # training
    general_pipeline(cfg_model, pretrain_model, train_dataloader, pretrain_dir)


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
    pretrained_dict.pop('loss.weight')
    # overwrite finetune model dict
    model_dict.update(pretrained_dict)
    # load the pretrained model dict
    model.load_state_dict(model_dict)
    return model


def mos_finetune(cfg, cfg_dataset, mode: str):
    # model cfg
    cfg_model = cfg[mode]

    # pre-training checkpoint path
    pre_method = cfg_model["pretrain_method"]
    assert pre_method == 'bg_pretrain'
    pre_dataset = cfg_model["pretrain_dataset"]
    pre_params = cfg_model["pretrain_params"]
    pre_version = cfg_model["pretrain_version"]
    pre_epoch = cfg_model["pretrain_epoch"]
    pretrain_model_dir = f"./logs/ours/{pre_method}/{pre_dataset}/{pre_params}/version_{pre_version}/checkpoints"
    pretrain_ckpt_name = f"epoch={pre_epoch}.ckpt"
    pretrain_ckpt_path = os.path.join(pretrain_model_dir, pretrain_ckpt_name)

    # fine-tuning params
    dataset_name = cfg_model['dataset_name']
    assert dataset_name == 'nuscenes'  # TODO: only nuscenes dataset supported now
    downsample_pct = cfg_model['downsample_pct']
    finetune_dir = f"./logs/ours/{pre_method}(epoch-{pre_epoch})-{mode}/{pre_dataset}-{downsample_pct}%{dataset_name}"

    # load pre-trained encoder to fine-tuning model
    finetune_model = MosNetwork(cfg_model, True)
    finetune_model = load_pretrained_encoder(pretrain_ckpt_path, finetune_model)

    # dataloader
    nusc = NuScenes(dataroot=cfg_dataset["nuscenes"]["root"], version=cfg_dataset["nuscenes"]["version"])
    train_set = NuscMosDataset(nusc, cfg_model, cfg_dataset, 'train')
    val_set = NuscMosDataset(nusc, cfg_model, cfg_dataset, 'val')
    dataloader = NuscDataloader(nusc, cfg_model, train_set, val_set, True)
    dataloader.setup()
    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()

    # training
    general_pipeline(cfg_model, finetune_model, train_dataloader, finetune_dir)


def bg_test(cfg, test_epoch):
    a = 1


if __name__ == "__main__":
    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['bg_pretrain', 'mos_finetune', 'mos_test'], default='mos_finetune')
    args = parser.parse_args()
    set_deterministic(666)
    with open("configs/ours.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    with open("configs/dataset.yaml", "r") as f:
        cfg_dataset = yaml.safe_load(f)

    # statistics of background samples
    check_statistics = False
    if check_statistics:
        statistics(cfg['bg_pretrain'], cfg_dataset)

    # pre-training on background for motion segmentation task
    if args.mode == 'bg_pretrain':
        background_pretrain(cfg, cfg_dataset, mode=args.mode)

    # background test
    elif args.mode == 'bg_test':
        a = 1

    # fine-tuning on moving object segmentation benchmark
    elif args.mode == 'mos_finetune':
        mos_finetune(cfg, cfg_dataset, mode=args.mode)

    # test on moving object segmentation benchmark
    elif args.mode == 'mos_test':
        mos_test(cfg, test_epoch=49)
