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
from datasets.nusc_loader import NuscDataloader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import Trainer
from models.also.models import AlsoNetwork



# class Fuck(pl.LightningModule):
#
#     def __init__(self, config):
#         super().__init__()
#
#
#     def forward(self, data):
#         outputs = self.backbone(data)
#
#         if isinstance(outputs, dict):
#             for k, v in outputs.items():
#                 data[k] = v
#         else:
#             data["latents"] = outputs
#
#         return_data = self.decoder(data)
#
#         return return_data
#
#     def compute_confusion_matrix(self, output_data):
#         outputs = output_data["predictions"].squeeze(-1)
#         occupancies = output_data["occupancies"].float()
#
#         output_np = (torch.sigmoid(outputs).cpu().detach().numpy() > 0.5).astype(int)
#         target_np = occupancies.cpu().numpy().astype(int)
#         cm = confusion_matrix(
#             target_np.ravel(), output_np.ravel(), labels=list(range(2))
#         )
#         return cm
#
#     def compute_loss(self, output_data, prefix):
#
#         loss = 0
#         loss_values = {}
#         for key, value in output_data.items():
#             if "loss" in key and (self.config["loss"][key + "_lambda"] > 0):
#                 loss = loss + self.config["loss"][key + "_lambda"] * value
#                 self.log(prefix + "/" + key, value.item(), on_step=True, on_epoch=False, prog_bar=True, logger=False)
#                 loss_values[key] = value.item()
#
#         # log also the total loss
#         self.log(prefix + "/loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=False)
#
#         if self.train_cm.sum() > 0: # TODO
#             # self.log(prefix + "/iou", metrics.stats_iou_per_class(self.train_cm)[0], on_step=True, on_epoch=False,
#             #          prog_bar=True, logger=False)
#
#         return loss, loss_values
#
#     def on_train_epoch_start(self) -> None:
#         self.train_cm = np.zeros((2, 2))
#         return super().on_train_epoch_start()
#
#     def training_step(self, data, batch_idx):
#
#         if batch_idx % 10 == 0:
#             torch.cuda.empty_cache()
#
#         output_data = self.forward(data)
#         loss, individual_losses = self.compute_loss(output_data, prefix="train")
#         cm = self.compute_confusion_matrix(output_data)
#         self.train_cm += cm
#
#         individual_losses["loss"] = loss
#
#         return individual_losses
#
#     def compute_log_data(self, outputs, cm, prefix):
#
#         # compute iou
#         iou = metrics.stats_iou_per_class(cm)[0]
#         self.log(prefix + "/iou", iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#
#         log_data = {}
#         keys = outputs[0].keys()
#         for key in keys:
#             if "loss" not in key:
#                 continue
#             if key == "loss":
#                 loss = np.mean([d[key].item() for d in outputs])
#             else:
#                 loss = np.mean([d[key] for d in outputs])
#             log_data[key] = loss
#
#         log_data["iou"] = iou
#         log_data["steps"] = self.global_step
#
#         return log_data
#
#     def training_epoch_end(self, outputs):
#
#         log_data = self.compute_log_data(outputs, self.train_cm, prefix="train")
#
#         os.makedirs(self.logger.log_dir, exist_ok=True)
#         logs_file(os.path.join(self.logger.log_dir, "logs_train.csv"), self.current_epoch, log_data)
#
#         if (self.global_step > 0) and (not self.config["interactive_log"]):
#             desc = "Train " + self.get_description_string(log_data)
#             print(wblue(desc))


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
    nusc = NuScenes(dataroot=dataset_cfg["nuscenes"]["root"], version=dataset_cfg["nuscenes"]["version"])
    train_set = NuscAlsoDataset(nusc, model_cfg, dataset_cfg, 'train')
    val_set = NuscAlsoDataset(nusc, model_cfg, dataset_cfg, 'val')
    dataloader = NuscDataloader(nusc, model_cfg, train_set, val_set, True)
    dataloader.setup()
    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()

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


