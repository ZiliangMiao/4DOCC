import os
import re
import json
import yaml
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

from datasets.occ4d.common import MinkCollateFn
from models.occ4d.models_org import MinkOccupancyForecastingNetwork

# JIT
from torch.utils.cpp_extension import load
dvr = load("dvr", sources=["lib/dvr/dvr.cpp", "lib/dvr/dvr.cu"], verbose=True, extra_cuda_cflags=['-allow-unsupported-compiler'])

def make_mink_dataloaders(cfg):
    dataset_kwargs = {
        "pc_range": cfg["data"]["pc_range"],
        "input_within_pc_range": cfg["data"]["input_within_pc_range"],
        "voxel_size": cfg["data"]["voxel_size"],
        "n_input": cfg["data"]["n_input"],
        "n_output": cfg["data"]["n_output"],
        "ego_mask": cfg["data"]["ego_mask"],
        "flip": cfg["data"]["flip"],
    }
    data_loader_kwargs = {
        "pin_memory": False,  # NOTE
        "shuffle": cfg["data"]["shuffle"],
        "drop_last": True,
        "batch_size": cfg["model"]["batch_size"],
        "num_workers": cfg["model"]["num_workers"],
    }

    dataset_name = cfg["dataset"]["name"]
    if dataset_name.lower() == "nuscenes":
        from datasets.occ4d.nusc import nuScenesDataset
        from nuscenes.nuscenes import NuScenes

        nusc = NuScenes(cfg["dataset"][dataset_name]["version"], cfg["dataset"][dataset_name]["root"])
        data_loaders = {
            "train": DataLoader(
                nuScenesDataset(nusc, "train", dataset_kwargs),
                collate_fn=MinkCollateFn,
                **data_loader_kwargs,
            ),
            "val": DataLoader(
                nuScenesDataset(nusc, "val", dataset_kwargs),
                collate_fn=MinkCollateFn,
                **data_loader_kwargs,
            ),
        }
    elif dataset_name.lower() == "kitti":
        raise NotImplementedError("KITTI is not supported now, wait for data.kitti_mink.py.")
        from datasets.kitti import KittiDataset
        data_loaders = {
            "train": DataLoader(
                KittiDataset(cfg["dataset"]["name"]["root"], cfg["dataset"]["name"]["config"], "trainval", dataset_kwargs),
                collate_fn=MinkCollateFn,
                **data_loader_kwargs,
            ),
            "val": DataLoader(
                KittiDataset(cfg["dataset"]["name"]["root"], cfg["dataset"]["name"]["config"], "test", dataset_kwargs),
                collate_fn=MinkCollateFn,
                **data_loader_kwargs,
            ),
        }
    elif dataset_name.lower() == "argoverse2":
        raise NotImplementedError("Argoverse is not supported now, wait for data.av2_mink.py.")
        from datasets.av2 import Argoverse2Dataset
        data_loaders = {
            "train": DataLoader(
                Argoverse2Dataset(cfg["dataset"][dataset_name]["root"], "train", dataset_kwargs, subsample=cfg["dataset"][dataset_name]["subsample"]),
                collate_fn=MinkCollateFn,
                **data_loader_kwargs,
            )
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
    _dataset_name = cfg["dataset"]["name"]
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # num_devices = torch.cuda.device_count()
    num_devices = cfg["model"]["num_devices"]
    assert num_devices == cfg["model"]["num_devices"]
    assert _batch_size % num_devices == 0
    if _batch_size % num_devices != 0:
        raise RuntimeError(f"Batch size ({_batch_size}) cannot be divided by device count ({num_devices})")

    # pretrain dataset loader
    data_loaders = make_mink_dataloaders(cfg)
    model = MinkOccupancyForecastingNetwork(_loss_type, _n_input, _n_output, _pc_range, _voxel_size)
    model = model.to(device)

    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=_lr_start)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=_lr_epoch, gamma=_lr_decay)

    # dump pretraining config
    model_dir = cfg["model"]["expt_dir"]
    os.makedirs(model_dir, exist_ok=True)
    with open(f"{model_dir}/config.json", "w") as f:
        json.dump(cfg, f, indent=4)

    # resume training
    if cfg["model"]["resume_ckpt"] is not None:
        start_epoch, n_iter = resume_from_ckpts(cfg["model"]["resume_ckpt"], model, optimizer, scheduler)
    else:
        start_epoch, n_iter = 0, 0

    # data parallel
    model = nn.DataParallel(model, device_ids=[0])

    # if use distributed data parallel (ddp)
    # dist.init_process_group(backend='nccl')
    # local_rank = int(os.environ['LOCAL_RANK'])
    # torch.cuda.set_device(local_rank)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # writer
    writer = SummaryWriter(f"{model_dir}/tf_logs")
    for epoch in range(start_epoch, _num_epoch):
        # training phase:
        data_loader = data_loaders["train"]
        model.train()

        num_batch = len(data_loader)
        avg_loss_50_iters = 0
        total_val_loss = {}
        for batch_index, batch_data in enumerate(data_loader):
            input_points_4d = batch_data[1]
            output_origin, output_points, output_tindex = batch_data[2:5]
            output_labels = batch_data[5] if _dataset_name == "nuscenes" and cfg["data"]["fgbg_label"] else None

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                ret_dict = model(
                    input_points_4d,
                    output_origin,
                    output_points,
                    output_tindex,
                    output_labels=output_labels,
                    mode="training",
                    loss=_loss_type
                )  # do backward during model forward
                optimizer.step()
                torch.cuda.empty_cache()
                n_iter += 1

            # logging: printer and writer
            avg_loss_50_iters += ret_dict[f"{_loss_type}_loss"].item() / 50
            if batch_index % 50 == 49:
                print(f"Train Iter: {n_iter},",
                      f"Epoch: {epoch}/{_num_epoch},",
                      f"Batch: {batch_index}/{num_batch},",
                      f"{_loss_type} Loss Per 50 Iters: {avg_loss_50_iters}",)
                writer.add_scalar("train/50 iters avg l1_loss", avg_loss_50_iters, n_iter)
                avg_loss_50_iters = 0
            for ret_item in ret_dict:
                if ret_item.endswith("loss"):
                    writer.add_scalar(f"train/{ret_item}", ret_dict[ret_item].item(), n_iter)

            if (batch_index + 1) % (num_batch // 10) == 0:  # // meams divide + floor
                os.makedirs(os.path.join(_expt_dir, "ckpts"), exist_ok=True)
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

        os.makedirs(os.path.join(_expt_dir, "ckpts"), exist_ok=True)
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

        # validation phase:
        data_loader = data_loaders["val"]
        model.eval()

        total_val_loss = {}
        num_batch = len(data_loader)
        num_example = len(data_loader.dataset)

        avg_loss_50_iters = 0
        for batch_index, batch_data in enumerate(data_loader):
            input_points_4d = batch_data[1]
            output_origin, output_points, output_tindex = batch_data[2:5]
            if _dataset_name == "nuscenes":
                output_labels = batch_data[5]
            else:
                output_labels = None

            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                loss = _loss_type
                ret_dict = model(
                    input_points_4d,
                    output_origin,
                    output_points,
                    output_tindex,
                    output_labels=output_labels,
                    mode="training",
                    loss=loss
                )

            avg_loss = ret_dict[f"{loss}_loss"].mean()
            avg_loss_50_iters += avg_loss.item() / 50
            for ret_item in ret_dict:
                if ret_item.endswith("loss"):
                    if ret_item not in total_val_loss:
                        total_val_loss[ret_item] = 0
                    total_val_loss[ret_item] += ret_dict[ret_item].mean().item() * len(input_points_4d)
        for ret_item in total_val_loss:
            mean_val_loss = total_val_loss[ret_item] / num_example
            writer.add_scalar(f"validation/{ret_item}", mean_val_loss, n_iter)
        scheduler.step()
    #
    writer.close()
    # if use ddp
    dist.destroy_process_group()

if __name__ == "__main__":
    # set random seeds
    np.random.seed(666)
    torch.random.manual_seed(666)

    # load pretrain config
    with open("configs/occ4d.yaml", "r") as f:
        cfg_pretrain = yaml.safe_load(f)
    pretrain(cfg_pretrain)

    # moving object segmentation fine-tuning (finetune at mos project, not 4docc project)
    # with open("./configs/mos_finetune.yaml", "r") as f:
    #     cfg_finetune = yaml.safe_load(f)
    # finetune(cfg_finetune)