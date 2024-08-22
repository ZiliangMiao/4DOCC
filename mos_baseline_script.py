# base
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
from models.mos4d import models
from datasets.mos4d import nusc as nusc_dataset
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
    train_dir = f"./logs/mos_baseline/mos4d/{downsample_pct}%{dataset_name}"

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


def mos_test(cfg, test_epoch):
    cfg["mos_test"]["test_epoch"] = test_epoch
    num_device = cfg["mos_test"]["num_devices"]
    model_dataset = cfg["mos_test"]["model_dataset"]
    model_name = cfg["mos_test"]["model_name"]
    model_version = cfg["mos_test"]["model_version"]
    model_dir = os.path.join("./logs", "mos4d", model_dataset, model_name, model_version)
    ckpt_path = os.path.join(model_dir, "checkpoints", f"{model_name}_epoch={test_epoch}.ckpt")

    # org 4dmos, load trained model's hparams
    # hparams_path = os.path.join(model_dir, "hparams.yaml")
    # hparams = yaml.safe_load(open(hparams_path))

    # dataset
    test_dataset = cfg["data"]["dataset_name"]
    if test_dataset == "nuscenes":
        nusc = NuScenes(dataroot=cfg["dataset"]["nuscenes"]["root"], version=cfg["dataset"]["nuscenes"]["version"])
        data = NuscMosDataset.NuscSequentialModule(cfg, nusc, "test")
        data.setup()
    else:
        raise ValueError("Not supported test dataset.")

    model = MosNetwork(cfg)
    # org 4dmos: Load ckeckpoint model
    # ckpt = torch.load(ckpt_path)
    # model = model.cuda()
    # model.load_state_dict(ckpt["state_dict"])
    # model.eval()
    # model.freeze()

    # testing
    test_dataloader = data.test_dataloader()

    ##############################################################################
    # test_data_list = list(test_dataloader)
    # from torch.utils.data import DataLoader, Dataset
    # class PartialDataset(Dataset):
    #     def __init__(self, data_list):
    #         self.data_list = data_list
    #     def __len__(self):
    #         return len(self.data_list)
    #     def __getitem__(self, index):
    #         return self.data_list[index]
    # def collate_fn(batch):  # define how to merge a list of samples to from a mini-batch samples
    #     sample_data_token = [item[0][0] for item in batch]
    #     point_cloud = [item[1][0] for item in batch]
    #     mos_label = [item[2][0] for item in batch]
    #     return [sample_data_token, point_cloud, mos_label]
    # # 创建自定义的Dataset对象
    # partial_dataset = PartialDataset(test_data_list)
    # # 创建DataLoader对象
    # partial_dataloader = DataLoader(partial_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    # listss = list(partial_dataloader)
    ##############################################################################

    # logger
    log_folder = os.path.join(model_dir, "results", f"epoch_{test_epoch}")
    os.makedirs(log_folder, exist_ok=True)
    date = datetime.date.today().strftime('%Y%m%d')
    log_file = os.path.join(log_folder, f"{model_name}_epoch-{test_epoch}_{date}.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(cfg))
    logging.info(log_file)

    # metrics
    metrics = ClassificationMetrics(n_classes=3, ignore_index=0)

    # predict with pytorch-lightning
    trainer = Trainer(accelerator="gpu", strategy="ddp", devices=num_device, deterministic=True)
    pred_outputs = trainer.predict(model, dataloaders=test_dataloader, return_predictions=True, ckpt_path=ckpt_path)
    # pred iou
    conf_mat_list_pred = [output["confusion_matrix"] for output in pred_outputs]
    acc_conf_mat_pred = torch.zeros(3, 3)
    for conf_mat in conf_mat_list_pred:
        acc_conf_mat_pred = acc_conf_mat_pred.add(conf_mat)
    TP_pred, FP_pred, FN_pred = metrics.get_stats(acc_conf_mat_pred)
    IOU_pred = metrics.get_iou(TP_pred, FP_pred, FN_pred)[2]
    logging.info('Final Avg. Moving Object IoU w/o ego vehicle: %f' % (IOU_pred.item() * 100))

    # # validate
    # valid_outputs = trainer.validate(model, dataloaders=test_dataloader, ckpt_path=ckpt_path)
    # # valid iou
    # conf_mat_list = model.validation_step_outputs
    # acc_conf_mat = torch.zeros(3, 3)
    # for conf_mat in conf_mat_list:
    #     acc_conf_mat = acc_conf_mat.add(conf_mat)
    # TP, FP, FN = metrics.getStats(acc_conf_mat)
    # IOU = metrics.getIoU(TP, FP, FN)[2]
    # logging.info('Final Avg. Moving Object IoU w/o ego vehicle: %f' % (IOU.item() * 100))
    # a = 1

    # # loop batch
    # num_classes = 3
    # ignore_class_idx = 0
    # moving_class_idx = 2
    # TP_mov, FP_mov, FN_mov = 0, 0, 0
    # acc_conf_mat = torch.zeros(num_classes, num_classes).cuda()
    # num_samples = 0
    # for i, batch in tqdm(enumerate(test_dataloader)):
    #     meta, point_clouds, mos_labels = batch
    #     point_clouds = [point_cloud.cuda() for point_cloud in point_clouds]
    #     curr_coords_list, curr_feats_list = model(point_clouds)
    #
    #     for batch_idx, (coords, logits) in enumerate(zip(curr_coords_list, curr_feats_list)):
    #         gt_label = mos_labels[batch_idx].cuda()
    #         if test_dataset == "nuscenes":
    #             # get ego mask
    #             curr_time_mask = point_clouds[batch_idx][:, -1] == 0.0
    #             ego_mask = NuscSequentialDataset.get_ego_mask(point_clouds[batch_idx][curr_time_mask]).cpu().numpy()
    #             # get pred mos label file name
    #             sample_data_token = meta[batch_idx]
    #             pred_label_file = os.path.join(pred_mos_labels_dir, f"{sample_data_token}_mos_pred.label")
    #         elif test_dataset == "SEKITTI":
    #             # get ego mask
    #             curr_time_mask = point_clouds[batch_idx][:, -1] == 0.0
    #             ego_mask = KittiSequentialDataset.get_ego_mask(point_clouds[batch_idx][curr_time_mask]).cpu().numpy()
    #             # get pred mos label file name
    #             seq_idx, scan_idx, _ = meta[batch_idx]
    #             pred_label_file = os.path.join(pred_mos_labels_dir, f"seq-{seq_idx}_scan-{scan_idx}_mos_pred.label")
    #         else:
    #             raise Exception("Not supported test dataset")
    #
    #         # save predictions
    #         logits[:, ignore_class_idx] = -float("inf")  # ingore: 0, i.e., unknown or noise
    #         pred_confidence = F.softmax(logits, dim=1).detach().cpu().numpy()
    #         moving_confidence = pred_confidence[:, moving_class_idx]
    #         pred_label = np.ones_like(moving_confidence, dtype=np.uint8)  # notice: dtype of mos labels is uint8
    #         pred_label[moving_confidence > 0.5] = 2
    #
    #         # calculate iou w/o ego vehicle pts
    #         cfs_mat = metrics.compute_confusion_matrix(logits[~ego_mask], gt_label[~ego_mask])
    #         acc_conf_mat = acc_conf_mat.add(cfs_mat)
    #         tp, fp, fn = metrics.getStats(cfs_mat)  # stat of current sample
    #         iou_mov = metrics.getIoU(tp, fp, fn)[moving_class_idx] * 100  # IoU of moving object (class 2)
    #         TP_mov += tp[moving_class_idx]
    #         FP_mov += fp[moving_class_idx]
    #         FN_mov += fn[moving_class_idx]
    #         # logging two iou
    #         num_samples += 1
    #         logging.info('Validation Sample Index %d, Moving Object IoU w/o ego vehicle: %f' % (num_samples, iou_mov))
    #         # save predicted labels
    #         if save_pred_wo_ego:
    #             pred_label[ego_mask] = 0  # set ego vehicle points as unknown for visualization
    #         # save pred mos label
    #         pred_label.tofile(pred_label_file)
    #     torch.cuda.empty_cache()
    # IOU_mov = metrics.getIoU(TP_mov, FP_mov, FN_mov)
    #
    # # TP, FP, FN = metrics.getStats(acc_conf_mat)
    # # IOU = metrics.getIoU(TP, FP, FN)
    #
    # logging.info('Final Avg. Moving Object IoU w/o ego vehicle: %f' % (IOU_mov * 100))


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

    # load train config
    with open("configs/mos4d.yaml", "r") as f:
        cfg_model = yaml.safe_load(f)
    with open("configs/dataset.yaml", "r") as f:
        cfg_dataset = yaml.safe_load(f)

    # training from scratch
    mos_train_from_scratch(cfg_model, cfg_dataset)

