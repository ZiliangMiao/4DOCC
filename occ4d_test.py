import logging
import os
import sys
import datetime
import numpy as np
import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets.occ4d.common import MinkCollateFn
from models.occ4d.models import MinkOccupancyForecastingNetwork
from utils.vis.vis_occ import get_occupancy_as_pcd
from utils.occ4d_evaluation import compute_chamfer_distance, compute_chamfer_distance_inner, compute_ray_errors
from utils.deterministic import set_deterministic

from torch.utils.cpp_extension import load
dvr = load("dvr", sources=["lib/dvr/dvr.cpp", "lib/dvr/dvr.cu"], verbose=True)

def get_grid_mask(points, pc_range):
    points = points.T
    mask1 = np.logical_and(pc_range[0] <= points[0], points[0] <= pc_range[3])
    mask2 = np.logical_and(pc_range[1] <= points[1], points[1] <= pc_range[4])
    mask3 = np.logical_and(pc_range[2] <= points[2], points[2] <= pc_range[5])
    mask = mask1 & mask2 & mask3
    return mask

def get_rendered_pcds(origin, points, tindex, gt_dist, pred_dist, pc_range, eval_within_grid=False, eval_outside_grid=False):
    pcds = []
    for t in range(len(origin)):
        mask = np.logical_and(tindex == t, gt_dist > 0.0)
        if eval_within_grid:
            mask = np.logical_and(mask, get_grid_mask(points, pc_range))
        if eval_outside_grid:
            mask = np.logical_and(mask, ~get_grid_mask(points, pc_range))
        # skip the ones with no data
        if not mask.any():
            continue
        _pts = points[mask, :3]
        # use ground truth lidar points for the raycasting direction
        v = _pts - origin[t][None, :]
        d = v / np.sqrt((v ** 2).sum(axis=1, keepdims=True))
        pred_pts = origin[t][None, :] + d * pred_dist[mask][:, None]
        pcds.append(torch.from_numpy(pred_pts))
    return pcds

def get_clamped_output(origin, points, tindex, pc_range, gt_dist, eval_within_grid=False, eval_outside_grid=False, get_indices=False):
    pcds = []
    if get_indices:
        indices = []
    for t in range(len(origin)):
        mask = np.logical_and(tindex == t, gt_dist > 0.0)
        if eval_within_grid:
            mask = np.logical_and(mask, get_grid_mask(points, pc_range))
        if eval_outside_grid:
            mask = np.logical_and(mask, ~get_grid_mask(points, pc_range))
        # skip the ones with no data
        if not mask.any():
            continue
        if get_indices:
            idx = np.arange(points.shape[0])
            indices.append(idx[mask])
        _pts = points[mask, :3]
        # use ground truth lidar points for the raycasting direction
        v = _pts - origin[t][None, :]
        d = v / np.sqrt((v ** 2).sum(axis=1, keepdims=True))
        gt_pts = origin[t][None, :] + d * gt_dist[mask][:, None]
        pcds.append(torch.from_numpy(gt_pts))
    if get_indices:
        return pcds, indices
    else:
        return pcds

def make_mink_dataloaders(cfg):
    dataset_kwargs = {
        "pc_range": cfg["data"]["pc_range"],
        "input_within_pc_range": cfg["data"]["input_within_pc_range"],
        "voxel_size": cfg["data"]["voxel_size"],
        "n_input": cfg["data"]["n_input"],
        "n_output": cfg["data"]["n_output"],
        "ego_mask": cfg["data"]["ego_mask"],
        "flip": cfg["data"]["flip"],
        "fgbg_label": cfg["data"]["fgbg_label"],
    }
    data_loader_kwargs = {
        "pin_memory": False,  # NOTE
        "shuffle": cfg["data"]["shuffle"],
        "drop_last": True,
        "batch_size": cfg["model"]["batch_size"],
        "num_workers": cfg["model"]["num_workers"],
    }

    dataset_name = cfg["data"]["dataset_name"].lower()
    if dataset_name == "nuscenes":
        from datasets.occ4d.nusc import nuScenesDataset
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(cfg["dataset"][dataset_name]["version"], cfg["dataset"][dataset_name]["root"])
        data_loader = DataLoader(
                nuScenesDataset(nusc, "val", dataset_kwargs),
                collate_fn=MinkCollateFn,
                **data_loader_kwargs)
    else:
        raise NotImplementedError("Dataset " + cfg["data"]["dataset_name"] + "is not supported.")
    return data_loader

def test(cfg):
    # get params
    _batch_size = cfg["model"]["batch_size"]
    _model_dataset = cfg["model"]["model_dataset"]
    _model_name = cfg["model"]["model_name"]
    _model_version = cfg["model"]["model_version"]
    _loss_type = cfg["model"]["loss_type"]
    _test_epoch = cfg["model"]["test_epoch"]
    _dataset = cfg["data"]["dataset_name"]
    _n_input, _n_output = cfg["data"]["n_input"], cfg["data"]["n_output"]
    _pc_range, _voxel_size = cfg["data"]["pc_range"], cfg["data"]["voxel_size"]

    # dir
    _model_dir = os.path.join("./logs", "occ4d", _model_dataset, _model_name, _model_version)
    vis_dir = os.path.join(_model_dir, "results", f"epoch_{_test_epoch}", "predictions")

    # vis settings
    _write_pcd = cfg["model"]["write_pcd"]
    os.makedirs(vis_dir, exist_ok=True)
    pred_pcds_dir = os.path.join(vis_dir, "pred_pcds")
    os.makedirs(pred_pcds_dir, exist_ok=True)
    gt_pcds_dir = os.path.join(vis_dir, "gt_pcds")
    os.makedirs(gt_pcds_dir, exist_ok=True)
    occ_pcds_dir = os.path.join(vis_dir, "occ_pcds")
    os.makedirs(occ_pcds_dir, exist_ok=True)

    # get device status
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_devices = torch.cuda.device_count()
    assert num_devices == cfg["model"]["num_devices"]
    assert _batch_size % num_devices == 0
    if _batch_size % num_devices != 0:
        raise RuntimeError(f"Batch size ({_batch_size}) cannot be divided by device count ({num_devices})")

    # dataset
    data_loader = make_mink_dataloaders(cfg)

    # init model on cuda device
    model = MinkOccupancyForecastingNetwork(_loss_type, _n_input, _n_output, _pc_range, _voxel_size).to(device)

    # load trained model

    ckpt_path = f"{_model_dir}/checkpoints/{_model_name}_epoch={_test_epoch}.ckpt"
    assert os.path.exists(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)  # NOTE: ignore renderer's parameters

    # data parallel
    model = nn.DataParallel(model)
    model.eval()

    # logging
    os.makedirs(f"{_model_dir}/results/epoch_{_test_epoch}", exist_ok=True)
    date = datetime.date.today().strftime('%Y%m%d')
    log_file = os.path.join(f"{_model_dir}/results/epoch_{_test_epoch}/{date}.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(cfg))
    logging.info(log_file)

    metrics = {
        "count": 0.0,
        "chamfer_distance": 0.0,
        "chamfer_distance_inner": 0.0,
        "l1_error": 0.0,
        "absrel_error": 0.0
    }
    for i, batch in enumerate(data_loader):
        filenames = batch[0]
        input_points_4d = batch[1]
        output_origin, output_points, output_tindex = batch[2:5]  # removed output_labels as the last returned argument

        if cfg["model"]["assume_const_velo"]:
            displacement = np.array([fname[-1].numpy() for fname in filenames]).reshape((-1, 1, 3))
            output_origin = torch.zeros_like(output_origin)
            displacements = torch.arange(_n_output) + 1
            output_origin = (output_origin + displacements[None, :, None]) * displacement

        if _dataset.lower() == "nusc":
            output_labels = batch[5]
        else:
            output_labels = None

        assert _batch_size == len(input_points_4d)
        if _batch_size % num_devices != 0:
            print(f"Dropping the last batch of size {_batch_size}")
            continue

        with torch.set_grad_enabled(False):
            ret_dict = model(
                input_points_4d,
                output_origin,
                output_points,
                output_tindex,
                output_labels=output_labels,
                mode="testing",
                eval_within_grid=cfg["model"]["eval_within_grid"],
                eval_outside_grid=cfg["model"]["eval_outside_grid"])

            # iterate through the batch
            for j in range(len(output_points)):
                # save occupancy predictions
                filename = occ_pcds_dir + f"/{filenames[j][2]}"
                get_occupancy_as_pcd(ret_dict["pog"][j].detach().cpu().numpy(), 0.5,
                                     _voxel_size, _pc_range, "Oranges", filename)

                pred_pcds = get_rendered_pcds(
                    output_origin[j].cpu().numpy(),
                    output_points[j].cpu().numpy(),
                    output_tindex[j].cpu().numpy(),
                    ret_dict["gt_dist"][j].cpu().numpy(),
                    ret_dict["pred_dist"][j].cpu().numpy(),
                    _pc_range,
                    cfg["model"]["eval_within_grid"],
                    cfg["model"]["eval_outside_grid"])
                gt_pcds = get_clamped_output(
                    output_origin[j].cpu().numpy(),
                    output_points[j].cpu().numpy(),
                    output_tindex[j].cpu().numpy(),
                    _pc_range,
                    ret_dict["gt_dist"][j].cpu().numpy(),
                    cfg["model"]["eval_within_grid"],
                    cfg["model"]["eval_outside_grid"])

                # load predictions (loop in time-axis)
                for k in range(len(gt_pcds)):
                    pred_pcd = pred_pcds[k]
                    gt_pcd = gt_pcds[k]
                    origin = output_origin[j][k].cpu().numpy()

                    # get the metrics
                    metrics["count"] += 1
                    metrics["chamfer_distance"] += compute_chamfer_distance(pred_pcd, gt_pcd, device)
                    metrics["chamfer_distance_inner"] += compute_chamfer_distance_inner(pred_pcd, gt_pcd, device)
                    l1_error, absrel_error = compute_ray_errors(pred_pcd, gt_pcd, torch.from_numpy(origin), device)
                    metrics["l1_error"] += l1_error
                    metrics["absrel_error"] += absrel_error

                    # save pred_pcd as [sample_data_token]_pred.pcd
                    if _write_pcd:
                        import open3d
                        pred_pcd_file = os.path.join(pred_pcds_dir, f"{filenames[j][2]}_pred-{k}.pcd")
                        o3d_pred_pcd = open3d.geometry.PointCloud()
                        o3d_pred_pcd.points = open3d.utility.Vector3dVector(pred_pcd.numpy())
                        open3d.io.write_point_cloud(pred_pcd_file, o3d_pred_pcd)
                        gt_pcd_file = os.path.join(gt_pcds_dir, f"{filenames[j][2]}_gt-{k}.pcd")
                        o3d_gt_pcd = open3d.geometry.PointCloud()
                        o3d_gt_pcd.points = open3d.utility.Vector3dVector(gt_pcd.numpy())
                        open3d.io.write_point_cloud(gt_pcd_file, o3d_gt_pcd)
                        # print(f"Predicted pcd saved: {filenames[j][2]}_pred.pcd")
                        # print(f"Ground truth pcd saved: {filenames[j][2]}_gt.pcd")

        count = metrics["count"]
        chamfer_distance = metrics["chamfer_distance"]
        chamfer_distance_inner = metrics["chamfer_distance_inner"]
        l1_error = metrics["l1_error"]
        absrel_error = metrics["absrel_error"]
        logging.info(f"Batch {i} / {len(data_loader)}, Chamfer Distance: {chamfer_distance / count}")
        logging.info(f"Batch {i} / {len(data_loader)}, Chamfer Distance Inner: {chamfer_distance_inner / count}")
        logging.info(f"Batch {i} / {len(data_loader)}, L1 Error: {l1_error / count}")
        logging.info(f"Batch {i} / {len(data_loader)}, AbsRel Error: {absrel_error / count}")
        logging.info(f"Batch {i} / {len(data_loader)}, Count: {count}")

    logging.info("Final Chamfer Distance: " + str(metrics["chamfer_distance"] / metrics["count"]))
    logging.info("Final Chamfer Distance Inner: " + str(metrics["chamfer_distance_inner"] / metrics["count"]))
    logging.info("Final L1 Error: " + str(metrics["l1_error"] / metrics["count"]))
    logging.info("Final AbsRel Error: " + str(metrics["absrel_error"] / metrics["count"]))



if __name__ == "__main__":
    # random seed
    set_deterministic(666)

    # test point cloud forecasting
    with open("configs/occ4d_test.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    test(cfg)

    # pytorch-lightning predict
    # # get params
    # _batch_size = cfg["model"]["batch_size"]
    # _model_name = cfg["model"]["model_name"]
    # _model_version = cfg["model"]["model_version"]
    # _loss_type = cfg["model"]["loss_type"]
    # _test_epoch = cfg["model"]["test_epoch"]
    # _dataset = cfg["data"]["dataset_name"]
    # _n_input, _n_output = cfg["data"]["n_input"], cfg["data"]["n_output"]
    # _pc_range, _voxel_size = cfg["data"]["pc_range"], cfg["data"]["voxel_size"]
    #
    # # get device status
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # num_devices = torch.cuda.device_count()
    # assert num_devices == cfg["model"]["num_devices"]
    # assert _batch_size % num_devices == 0
    # if _batch_size % num_devices != 0:
    #     raise RuntimeError(f"Batch size ({_batch_size}) cannot be divided by device count ({num_devices})")
    #
    # # dataset
    # data_loader = make_mink_dataloaders(cfg)
    #
    # # init model on cuda device
    # model = MinkOccupancyForecastingNetwork(cfg).to(device)
    #
    # # load trained model
    # _model_dir = os.path.join("logs", "pretrain", _dataset, _model_name, _model_version)
    # ckpt_path = f"{_model_dir}/checkpoints/{_model_name}_epoch={_test_epoch}.ckpt"
    # # assert os.path.exists(ckpt_path)
    # # ckpt = torch.load(ckpt_path, map_location=device)
    # # model.load_state_dict(ckpt["state_dict"], strict=False)  # NOTE: ignore renderer's parameters
    #
    # trainer = Trainer(accelerator="gpu", strategy="ddp", devices=num_devices)
    # trainer.predict(model, dataloaders=data_loader, return_predictions=False, ckpt_path=ckpt_path)



