import os
import re
import json
import argparse

import torch
import numpy as np
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.common import CollateFn, MosCollateFn
from model import OccupancyForecastingNetwork
from model_mos import MosOccupancyForecastingNetwork

def make_data_loaders(cfg):
    dataset_kwargs = {
        "pc_range": cfg["data"]["pc_range"],
        "voxel_size": cfg["data"]["voxel_size"],
        "n_input": cfg["data"]["n_input"],
        "n_output": cfg["data"]["n_output"],
        "nusc_version": cfg["dataset"]["nuscenes"]["version"],
        "ego_mask": cfg["data"]["ego_mask"],
    }
    data_loader_kwargs = {
        "pin_memory": False,  # NOTE
        "shuffle": False,
        "drop_last": True,
        "batch_size": cfg["model"]["batch_size"],
        "num_workers": cfg["model"]["num_workers"],
    }

    dataset_name = cfg["dataset"]["name"]
    if dataset_name.lower() == "kitti":
        from data.kitti import KittiDataset

        data_loaders = {
            "train": DataLoader(
                KittiDataset(cfg["dataset"]["name"]["root"], cfg["dataset"]["name"]["config"], "trainval", dataset_kwargs),
                collate_fn=CollateFn,
                **data_loader_kwargs,
            ),
            "val": DataLoader(
                KittiDataset(cfg["dataset"]["name"]["root"], cfg["dataset"]["name"]["config"], "test", dataset_kwargs),
                collate_fn=CollateFn,
                **data_loader_kwargs,
            ),
        }
    elif dataset_name.lower() == "nuscenes":
        from data.nusc import nuScenesDataset
        from nuscenes.nuscenes import NuScenes

        nusc = NuScenes(cfg["dataset"][dataset_name]["version"], cfg["dataset"][dataset_name]["root"])
        Dataset = nuScenesDataset
        data_loaders = {
            "train": DataLoader(
                Dataset(nusc, "train", dataset_kwargs),
                collate_fn=CollateFn,
                **data_loader_kwargs,
            ),
            "val": DataLoader(
                Dataset(nusc, "val", dataset_kwargs),
                collate_fn=CollateFn,
                **data_loader_kwargs,
            ),
        }
    elif dataset_name.lower() == "argoverse2":
        from data.av2 import Argoverse2Dataset
        data_loaders = {
            "train": DataLoader(
                Argoverse2Dataset(cfg["dataset"][dataset_name]["root"], "train", dataset_kwargs, subsample=cfg["dataset"][dataset_name]["subsample"]),
                collate_fn=CollateFn,
                **data_loader_kwargs,
            )
        }
    else:
        raise NotImplementedError("Dataset " + cfg["dataset"]["name"] + "is not supported.")
    return data_loaders


def make_mos_dataloader(cfg):
    dataset_kwargs = {
        "pc_range": cfg["data"]["pc_range"],
        "voxel_size": cfg["data"]["voxel_size"],
        "n_input": cfg["data"]["n_input"],
        "n_output": cfg["data"]["n_output"],
        "nusc_version": cfg["dataset"]["nuscenes"]["version"],
        "ego_mask": cfg["data"]["ego_mask"],
    }
    data_loader_kwargs = {
        "pin_memory": False,  # NOTE
        "shuffle": False,
        "drop_last": True,
        "batch_size": cfg["model"]["batch_size"],
        "num_workers": cfg["model"]["num_workers"],
    }

    dataset_name = cfg["dataset"]["name"]
    if dataset_name.lower() == "nuscenes":
        from data.nusc_mos import nuScenesMosDataset
        from nuscenes.nuscenes import NuScenes

        nusc = NuScenes(cfg["dataset"][dataset_name]["version"], cfg["dataset"][dataset_name]["root"])
        Dataset = nuScenesMosDataset
        data_loaders = {
            "train": DataLoader(
                Dataset(nusc, "train", dataset_kwargs),
                collate_fn=MosCollateFn,
                **data_loader_kwargs,
            ),
            "val": DataLoader(
                Dataset(nusc, "val", dataset_kwargs),
                collate_fn=MosCollateFn,
                **data_loader_kwargs,
            ),
        }
    else:
        raise NotImplementedError("Dataset " + cfg["dataset"]["name"] + "is not supported.")
    return data_loaders

def resume_from_ckpts(ckpt_pth, model, optimizer, scheduler):
    print(f"Resume training from checkpoint {ckpt_pth}")

    checkpoint = torch.load(ckpt_pth)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = 1 + checkpoint["epoch"]
    n_iter = checkpoint["n_iter"]
    return start_epoch, n_iter

def load_pretrained_encoder(ckpt_dir, model):
    if len(os.listdir(ckpt_dir)) > 0:
        pattern = re.compile(r"model_epoch_(\d+).pth")
        epochs = []
        for f in os.listdir(ckpt_dir):
            m = pattern.findall(f)
            if len(m) > 0:
                epochs.append(int(m[0]))
        resume_epoch = max(epochs)
        ckpt_path = f"{ckpt_dir}/model_epoch_{resume_epoch}.pth"
        print(f"Load pretrained encoder from checkpoint {ckpt_path}")

        checkpoint = torch.load(ckpt_path)
        pretrained_dict = checkpoint["model_state_dict"]
        model_dict = model.state_dict()

        # filter out unnecessary keys (generate new dict)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # overwrite finetune model dict
        model_dict.update(pretrained_dict)
        # load the pretrained model dict
        model.load_state_dict(model_dict)
    return model

def pretrain(cfg):
    # get data params
    _dataset = cfg["dataset"]["name"]
    _n_input, _n_output = cfg["data"]["n_input"], cfg["data"]["n_output"]
    _pc_range, _voxel_size = cfg["data"]["pc_range"], cfg["data"]["voxel_size"]

    # get model params
    _expt_dir = cfg["model"]["expt_dir"]
    _num_epoch = cfg["model"]["num_epoch"]
    _batch_size = cfg["model"]["batch_size"]
    _num_workers = cfg["model"]["num_workers"]
    _loss_type = cfg["model"]["loss_type"]
    _lr_start, _lr_epoch, _lr_decay = cfg["model"]["lr_start"], cfg["model"]["lr_epoch"], cfg["model"]["lr_decay"]

    # get device status
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_devices = torch.cuda.device_count()
    assert num_devices == cfg["model"]["num_devices"]
    assert _batch_size % num_devices == 0
    if _batch_size % num_devices != 0:
        raise RuntimeError(f"Batch size ({_batch_size}) cannot be divided by device count ({num_devices})")

    # pretrain dataset loader
    data_loaders = make_data_loaders(cfg)

    model = OccupancyForecastingNetwork(_loss_type, _n_input, _n_output, _pc_range, _voxel_size)
    model = model.to(device)

    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=_lr_start)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=_lr_epoch, gamma=_lr_decay)

    # dump pretraining config
    model_dir = cfg["model"]["expt_dir"]
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "ckpts"), exist_ok=True)
    with open(f"{model_dir}/config.json", "w") as f:
        json.dump(cfg, f, indent=4)

    # resume
    if cfg["model"]["resume_ckpt"] is not None:
        start_epoch, n_iter = resume_from_ckpts(cfg["model"]["resume_ckpt"], model, optimizer, scheduler)
    else:
        start_epoch, n_iter = 0, 0

    # data parallel
    model = nn.DataParallel(model)

    # writer
    writer = SummaryWriter(f"{model_dir}/tf_logs")
    for epoch in range(start_epoch, _num_epoch):
        for phase in ["train"]:  # , "val"]:
            data_loader = data_loaders[phase]
            if phase == "train":
                model.train()
            else:
                model.eval()

            total_val_loss = {}
            num_batch = len(data_loader)
            num_example = len(data_loader.dataset)
            weigth_list = []
            grad_list = []
            avg_loss_50_iters = 0
            for i, batch in enumerate(data_loader):
                input_points, input_tindex = batch[1:3]
                output_origin, output_points, output_tindex = batch[3:6]
                if _dataset == "nuscenes":
                    output_labels = batch[6]
                else:
                    output_labels = None

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    loss = _loss_type
                    ret_dict = model(
                        input_points,
                        input_tindex,
                        output_origin,
                        output_points,
                        output_tindex,
                        output_labels=output_labels,
                        mode="training",
                        loss=loss
                    )

                    if phase == "train":
                        optimizer.step()

                        # params = list(model.named_parameters())
                        # # check whether weight and grad changed after optimization
                        # block1_weight = params[2][1].data
                        # block1_grad = params[2][1].grad
                        # weigth_list.append(block1_weight.cpu())
                        # grad_list.append(block1_grad.cpu())
                        # if i != 0:
                        #     equal_weight = (weigth_list[i] == weigth_list[i-1])
                        #     equal_grad = (grad_list[i] == grad_list[i-1])
                        #     equal_weight_sum = torch.sum(equal_weight)
                        #     equal_grad_sum = torch.sum(equal_grad)
                        #     a = 1

                avg_loss = ret_dict[f"{loss}_loss"].mean()
                avg_loss_50_iters += avg_loss.item() / 50
                
                if phase == "train":
                    n_iter += 1
                    # print every 50 iter:
                    if i % 50 == 49:
                        print(
                                    f"Phase: {phase}, Iter: {n_iter},",
                                    f"Epoch: {epoch}/{_num_epoch},",
                                    f"Batch: {i}/{num_batch},",
                                    f"{loss.upper()} Loss Per 50 Iters: {avg_loss_50_iters}",
                        )
                        writer.add_scalar(f"{phase}/50 iters avg l1_loss", avg_loss_50_iters, n_iter)
                        avg_loss_50_iters = 0
                    for key in ret_dict:
                        if key.endswith("loss"):
                            writer.add_scalar(f"{phase}/{key}", ret_dict[key].mean().item(), n_iter)
                # if phase == "train":
                #     n_iter += 1
                #     for key in ret_dict:
                #         if key.endswith("loss"):
                #             writer.add_scalar(
                #                     f"{phase}/{key}", ret_dict[key].mean().item(), n_iter
                #             )
                else:
                    for key in ret_dict:
                        if key.endswith("loss"):
                            if key not in total_val_loss:
                                total_val_loss[key] = 0
                            total_val_loss[key] += ret_dict[key].mean().item() * len(
                                input_points
                            )

                if phase == "train" and (i + 1) % (num_batch // 10) == 0:
                    ckpt_path = f"{_expt_dir}/ckpts/model_epoch_{epoch}_iter_{n_iter}.pth"
                    torch.save(
                            {
                                "epoch": epoch,
                                "n_iter": n_iter,
                                "model_state_dict": model.module.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                            },
                            ckpt_path,
                            _use_new_zipfile_serialization=False,
                    )

            if phase == "train":
                ckpt_path = f"{_expt_dir}/ckpts/model_epoch_{epoch}.pth"
                torch.save(
                        {
                            "epoch": epoch,
                            "n_iter": n_iter,
                            "model_state_dict": model.module.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                        },
                        ckpt_path,
                        _use_new_zipfile_serialization=False,
                )
            else:
                for key in total_val_loss:
                    mean_val_loss = total_val_loss[key] / num_example
                    writer.add_scalar(f"{phase}/{key}", mean_val_loss, n_iter)

        scheduler.step()
    #
    writer.close()

def finetune(cfg):
    # get data params
    _n_input, _n_mos_class = cfg["data"]["n_input"], cfg["data"]["n_mos_class"]
    _pc_range, _voxel_size = cfg["data"]["pc_range"], cfg["data"]["voxel_size"]

    # get model params
    _expt_dir = cfg["model"]["expt_dir"]
    _num_epoch = cfg["model"]["num_epoch"]
    _batch_size = cfg["model"]["batch_size"]
    _num_workers = cfg["model"]["num_workers"]
    _loss_type = cfg["model"]["loss_type"]
    _lr_start, _lr_epoch, _lr_decay = cfg["model"]["lr_start"], cfg["model"]["lr_epoch"], cfg["model"]["lr_decay"]

    # get device status
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_devices = torch.cuda.device_count()
    assert num_devices == cfg["model"]["num_devices"]
    assert _batch_size % num_devices == 0
    if _batch_size % num_devices != 0:
        raise RuntimeError(f"Batch size ({_batch_size}) cannot be divided by device count ({num_devices})")

    # make mos dataset loader
    data_loaders = make_mos_dataloader(cfg)

    # instantiate a finetune model
    finetune_model = MosOccupancyForecastingNetwork(
        _loss_type,
        _n_input,
        _n_mos_class,
        _pc_range,
        _voxel_size,).to(device)

    # load the pretrained encoder parameters
    pretrain_ckpt_dir = os.path.join(_expt_dir, "ckpts", "pretrain")
    assert os.path.exists(pretrain_ckpt_dir)
    finetune_model = load_pretrained_encoder(pretrain_ckpt_dir, finetune_model)

    # adam optimizer
    optimizer = torch.optim.Adam(finetune_model.parameters(), lr=_lr_start)
    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=_lr_epoch, gamma=_lr_decay)

    # data parallel
    model = nn.DataParallel(finetune_model)

    # writer
    writer = SummaryWriter(os.path.join(_expt_dir, "tf_logs"))
    n_iter = 0
    finetune_model_dir = os.path.join(_expt_dir, "ckpts", "finetune")  # save finetuned model to this directory
    os.makedirs(finetune_model_dir, exist_ok=True)
    for epoch in range(_num_epoch):
        # store loss and iou metrics while training
        metrics = {
            "count": 0,
            "nll": 0.0,
            "iou": 0.0,
        }
        for phase in ["train"]:  # , "val"] train and validation process
            data_loader = data_loaders[phase]
            if phase == "train":
                model.train()
            else:
                model.eval()

            num_batch = len(data_loader)
            for i, batch in enumerate(data_loader):
                input_points, input_tindex, mos_labels = batch[1:4]

                optimizer.zero_grad()  # clear the previous gradient
                with torch.set_grad_enabled(phase == "train"):
                    ret_dict = model(input_points, input_tindex, mos_labels, mode="training", eval_within_grid=True)
                    if phase == "train":
                        optimizer.step()

                metrics["count"] += 1
                metrics["nll"] += ret_dict["nll"]
                metrics["iou"] += ret_dict["iou"]

                avg_loss = ret_dict["nll"].mean()
                avg_iou = ret_dict["iou"].mean()
                print(f"Phase: {phase}, Iter: {n_iter},",
                      f"Epoch: {epoch}/{_num_epoch},",
                      f"Batch: {i}/{num_batch},",
                      f"NLL Loss: {avg_loss.item():.3f}",
                      f"IoU: {avg_iou.item():.3f}",)

                if phase == "train":
                    n_iter += 1
                    writer.add_scalar(f"Epoch-{epoch} steps nll-loss", metrics["nll"] / metrics["count"], n_iter)
                    writer.add_scalar(f"Epoch-{epoch} steps iou", metrics["iou"] / metrics["count"], n_iter)
                if phase == "train" and (i + 1) % (num_batch // 10) == 0:
                    finetune_ckpt_path = os.path.join(finetune_model_dir, f"model_epoch_{epoch}_iter_{n_iter}.pth")
                    torch.save(
                            {
                                "epoch": epoch,
                                "n_iter": n_iter,
                                "model_state_dict": model.module.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                            },
                            finetune_ckpt_path,
                            _use_new_zipfile_serialization=False,
                    )

            if phase == "train":  # save model_epoch_x.pth
                finetune_ckpt_path = os.path.join(finetune_model_dir, f"model_epoch_{epoch}.pth")
                torch.save(
                        {
                            "epoch": epoch,
                            "n_iter": n_iter,
                            "model_state_dict": model.module.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                        },
                        finetune_ckpt_path,
                        _use_new_zipfile_serialization=False,
                )
                # save tensorboard log to track epoch loss
                writer.add_scalar("epochs nll-loss", metrics["nll"] / metrics["count"], epoch)
                writer.add_scalar("epochs iou", metrics["iou"] / metrics["count"], epoch)
        scheduler.step()
    writer.close()

if __name__ == "__main__":
    # set random seeds
    np.random.seed(0)
    torch.random.manual_seed(0)

    # load pretrain config
    with open("configs/occ_pretrain.yaml", "r") as f:
        cfg_pretrain = yaml.safe_load(f)
    # load finetune config
    with open("configs/mos_finetune.yaml", "r") as f:
        cfg_finetune = yaml.safe_load(f)

    # point cloud forecasting pre-training
    pretrain(cfg_pretrain)
    # moving object segmentation fine-tuning
    # finetune(cfg_finetune)

    # test crossentropy loss and nll loss
    # ce_loss = nn.CrossEntropyLoss(ignore_index=0)
    # pred_feats = torch.Tensor([[0, 3, 1],
    #                            [0, 7, 1]])
    # gt_label = torch.tensor([0, 2])
    # print(ce_loss(pred_feats, gt_label))
    # # weight = torch.Tensor([0.5, 0.5])
    # nllloss = nn.NLLLoss(ignore_index=0)
    # log_prob = torch.log(torch.softmax(pred_feats, dim=-1))
    # print(nllloss(log_prob, gt_label))

