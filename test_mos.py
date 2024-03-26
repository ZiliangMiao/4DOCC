import os
import json
import copy
import argparse
from datetime import datetime
import numpy as np
import torch
from torch import nn
import yaml

from model_mos import MosOccupancyForecastingNetwork
from utils.cal_iou import getIoU

from torch.utils.cpp_extension import load
dvr = load("dvr", sources=["lib/dvr/dvr.cpp", "lib/dvr/dvr.cu"], verbose=True)

from model_mos import get_grid_mask, get_grid_mask_voxel
from train import make_mos_dataloader

def mkdir_if_not_exists(d):
    if not os.path.exists(d):
        print(f"creating directory {d}")
        os.makedirs(d, exist_ok=True)

def test(cfg):
    # get data params
    _n_input, _n_mos_class = cfg["data"]["n_input"], cfg["data"]["n_mos_class"]
    _pc_range, _voxel_size = cfg["data"]["pc_range"], cfg["data"]["voxel_size"]
    _eval_within_grid = cfg["data"]["eval_within_grid"]
    _eval_outside_grid = cfg["data"]["eval_outside_grid"]

    # get model params
    _expt_dir = cfg["model"]["expt_dir"]
    _expt_name = cfg["model"]["expt_name"]
    _test_epoch = cfg["model"]["test_epoch"]
    _batch_size = cfg["model"]["batch_size"]
    _num_workers = cfg["model"]["num_workers"]
    _loss_type = cfg["model"]["loss_type"]

    # get device status
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_devices = torch.cuda.device_count()
    assert num_devices == cfg["model"]["num_devices"]
    assert _batch_size % num_devices == 0
    if _batch_size % num_devices != 0:
        raise RuntimeError(f"Batch size ({_batch_size}) cannot be divided by device count ({num_devices})")

    # dataset
    _dataset_name = cfg["dataset"]["name"]
    if _dataset_name.lower() == "nuscenes":
        data_loaders = make_mos_dataloader(cfg)
        val_data_loader = data_loaders[cfg["dataset"][_dataset_name]["test_split"]]

    # load the finetuned model
    model = MosOccupancyForecastingNetwork(_loss_type, _n_input, _n_mos_class, _pc_range, _voxel_size).to(device)
    finetune_model_path = os.path.join(_expt_dir, "ckpts", "finetune", f"model_epoch_{_test_epoch}.pth")
    checkpoint = torch.load(finetune_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # data parallel
    model = nn.DataParallel(model)
    model.eval()

    # log the test results
    dt = datetime.now()
    results_dir = os.path.join(_expt_dir, "results", _expt_name, f"epoch_{_test_epoch}")
    mkdir_if_not_exists(results_dir)
    logfile_path = os.path.join(results_dir, f"{dt}.txt")
    logfile = open(logfile_path, "w")

    metrics = {
        "count": 0,
        "iou": 0.0,
    }
    for i, batch in enumerate(val_data_loader):
        filenames = batch[0]
        points, tindex, mos_labels = batch[1:4]
        assert _batch_size == len(points)
        if _batch_size % num_devices != 0:
            print(f"Dropping the last batch of size {_batch_size}")
            continue

        with torch.set_grad_enabled(False):
            ret_dict = model(points, tindex, mos_labels, mode="testing", eval_within_grid=_eval_within_grid)

        # only calculate current sample data here (t = 0)
        valid_pts_list = ret_dict["points_list"]
        valid_pts_gt_labels_list = ret_dict["points_gt_labels_list"]
        valid_pts_pred_labels_list = ret_dict["points_pred_labels_list"]
        iou_list = ret_dict["iou_list"]
        # save predicted labels
        valid_pts_dir = os.path.join(_expt_dir, "results", _expt_name, "visualization", "valid_points")
        gt_labels_dir = os.path.join(_expt_dir, "results", _expt_name, "visualization", "gt_mos_labels")
        pred_labels_dir = os.path.join(_expt_dir, "results", _expt_name, "visualization", "pred_mos_labels")
        if not os.path.exists(valid_pts_dir):
            os.makedirs(valid_pts_dir, exist_ok=True)
        if not os.path.exists(gt_labels_dir):
            os.makedirs(gt_labels_dir, exist_ok=True)
        if not os.path.exists(pred_labels_dir):
            os.makedirs(pred_labels_dir, exist_ok=True)
        for j in range(_batch_size):
            valid_pts = valid_pts_list[j].cpu().numpy()
            valid_pts_gt_labels = valid_pts_gt_labels_list[j].cpu().numpy().astype(np.uint8)
            valid_pts_pred_labels = valid_pts_pred_labels_list[j].cpu().numpy().astype(np.uint8)

            # save valid points and predicted labels for visualization
            valid_pts_file = os.path.join(valid_pts_dir, filenames[j][2] + ".bin")
            valid_pts.tofile(valid_pts_file)  # np.float32
            gt_label_file = os.path.join(gt_labels_dir, filenames[j][2] + "_mos_gt.label")
            valid_pts_gt_labels.tofile(gt_label_file)  # np.uint8
            pred_label_file = os.path.join(pred_labels_dir, filenames[j][2] + "_mos_pred.label")
            valid_pts_pred_labels.tofile(pred_label_file)  # np.uint8

            # vis mos pointcloud
            # render_mos_pointcloud(points=valid_pts, gt_labels=valid_pts_gt_labels, pred_labels=valid_pts_pred_labels)

            # get the metrics
            metrics["count"] += 1
            metrics["iou"] += iou_list[j]
            print("Current Scan MOS IoU: ", iou_list[j])
        print("Batch", str(i)+"/"+str(len(val_data_loader))+":", "MOS IoU:", metrics["iou"] / metrics["count"])
    print("Final MOS IoU:", metrics["iou"] / metrics["count"])

    logfile.write("\nFinal IoU: " + str(metrics["iou"] / metrics["count"]))
    logfile.close()

if __name__ == "__main__":
    # set random seeds
    np.random.seed(0)
    torch.random.manual_seed(0)

    # load pretrain config
    with open("configs/mos_test.yaml", "r") as f:
        cfg_test = yaml.load(f)

    # test mos on nusc validation set
    test(cfg_test)

    # test IoU calculate
    # gt_labels =   np.array([0, 1, 2, 0, 1, 1, 1, 2, 2, 1])
    # pred_labels = np.array([1, 1, 2, 1, 1, 1, 2, 2, 1, 1])
    # mos_confusion_mat = confusion_matrix(y_true=gt_labels, y_pred=pred_labels, labels=[1, 2])  # 1-static, 2-moving
    # tp, fp, fn = getStat(mos_confusion_mat)  # stat of current sample
    # IoU = getIoU(tp, fp, fn)
