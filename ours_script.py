# base
import argparse
import logging
import os
import sys
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
# dataset
from nuscenes.nuscenes import NuScenes
from datasets.nusc_utils import NuscSequentialModule
from datasets.ours.nusc import NuscBgDataset as nusc_bg_dataset
from datasets.mos4d.nusc import NuscMosDataset as nusc_mos_dataset
# lib
from utils.deterministic import set_deterministic
from utils.metrics import ClassificationMetrics


def statistics(cfg_model, cfg_dataset):
    nusc = NuScenes(dataroot=cfg_dataset["nuscenes"]["root"], version=cfg_dataset["nuscenes"]["version"])
    nuscenes = nusc_bg_dataset(nusc, cfg_model, cfg_dataset, "train")
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


def background_pretrain(cfg, cfg_dataset, mode: str):
    cfg_model = cfg[mode]

    # pre-training params
    dataset_name = cfg_model['dataset_name']
    assert dataset_name == 'nuscenes', "Only nuscenes dataset supported now!"
    downsample_pct = cfg_model['downsample_pct']
    quant_size = cfg_model['quant_size']
    time_interval = cfg_model['time_interval']
    n_input = cfg_model['n_input']
    n_skip = cfg_model['n_skip']
    time = round(n_input * time_interval + (n_input - 1) * n_skip * time_interval, 2)
    num_epoch = cfg_model['num_epoch']
    batch_size = cfg_model['batch_size']

    # pretrain model
    from models.ours import models
    pretrain_model = models.MotionPretrainNetwork(cfg_model, True)

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_saver = ModelCheckpoint(
        monitor="epoch",
        verbose=True,
        save_top_k=num_epoch,
        mode="max",
        filename="{epoch}",
        every_n_epochs=5,
        save_last=False,
    )

    # Logger
    pretrain_dir = f"./logs/ours/{mode}/{downsample_pct}%{dataset_name}"
    pretrain_model_params = f"vs-{quant_size}_t-{time}_bs-{batch_size}"
    os.makedirs(pretrain_dir, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(pretrain_dir, name=pretrain_model_params, default_hp_metric=False)

    # Load data and construct data loader
    nusc = NuScenes(dataroot=cfg_dataset["nuscenes"]["root"], version=cfg_dataset["nuscenes"]["version"])
    train_set = nusc_bg_dataset(nusc, cfg_model, cfg_dataset, 'train')
    val_set = nusc_bg_dataset(nusc, cfg_model, cfg_dataset, 'val')
    dataloader = NuscSequentialModule(nusc, cfg_model, train_set, val_set, True)
    dataloader.setup()
    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()

    # Setup trainer
    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=cfg_model["num_devices"],
        logger=tb_logger,
        log_every_n_steps=1,
        max_epochs=num_epoch,
        accumulate_grad_batches=cfg_model["acc_batches"],  # accumulate batches, default=1
        callbacks=[lr_monitor, checkpoint_saver],
    )

    # Training
    resume_ckpt_path = cfg_model["resume_ckpt"]
    trainer.fit(pretrain_model, train_dataloader, ckpt_path=resume_ckpt_path)


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
    cfg_model = cfg[mode]

    # pre-training params
    pre_method = cfg_model["pretrain_method"]
    pre_dataset = cfg_model["pretrain_dataset"]
    pre_params = cfg_model["pretrain_params"]
    pre_version = cfg_model["pretrain_version"]
    pre_epoch = cfg_model["pretrain_epoch"]
    pretrain_model_dir = f"./logs/ours/{pre_method}/{pre_dataset}/{pre_params}/version_{pre_version}/checkpoints"
    pretrain_ckpt_name = f"epoch={pre_epoch}.ckpt"
    pretrain_ckpt_path = os.path.join(pretrain_model_dir, pretrain_ckpt_name)

    # fine-tuning params
    dataset_name = cfg_model['dataset_name']
    assert dataset_name == 'nuscenes', "Only nuscenes dataset supported now!"
    downsample_pct = cfg_model['downsample_pct']
    quant_size = cfg_model['quant_size']
    time_interval = cfg_model['time_interval']
    n_input = cfg_model['n_input']
    n_skip = cfg_model['n_skip']
    time = round(n_input * time_interval + (n_input - 1) * n_skip * time_interval, 2)
    num_epoch = cfg_model['num_epoch']
    batch_size = cfg_model['batch_size']

    # load pre-trained encoder to fine-tuning model
    assert pre_method in ['bg_pretrain', 'occ4d']
    from models.mos4d import models
    finetune_model = models.MOSNet(cfg_model, True)
    finetune_model = load_pretrained_encoder(pretrain_ckpt_path, finetune_model)

    # add callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_saver = ModelCheckpoint(
        monitor="epoch",
        verbose=True,
        save_top_k=num_epoch,
        mode="max",
        filename="{epoch}",
        every_n_epochs=5,
        save_last=True,
    )

    # logger
    finetune_dir = f"./logs/ours/{pre_method}(epoch-{pre_epoch})-{mode}/{pre_dataset}-{downsample_pct}%{dataset_name}"
    finetune_model_params = f"vs-{quant_size}_t-{time}_bs-{batch_size}"
    os.makedirs(finetune_dir, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(finetune_dir, name=finetune_model_params, default_hp_metric=False)

    # Load data and construct data loader
    nusc = NuScenes(dataroot=cfg_dataset["nuscenes"]["root"], version=cfg_dataset["nuscenes"]["version"])
    train_set = nusc_mos_dataset(nusc, cfg_model, cfg_dataset, 'train')
    val_set = nusc_mos_dataset(nusc, cfg_model, cfg_dataset, 'val')
    dataloader = NuscSequentialModule(nusc, cfg_model, train_set, val_set, True)
    dataloader.setup()
    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()

    # setup trainer
    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=cfg_model["num_devices"],
        logger=tb_logger,
        log_every_n_steps=1,
        max_epochs=num_epoch,
        accumulate_grad_batches=cfg_model["acc_batches"],  # accumulate batches, default=1
        callbacks=[lr_monitor, checkpoint_saver],
    )

    # fine-tuning
    resume_ckpt_path = cfg_model["resume_ckpt"]
    trainer.fit(finetune_model, train_dataloader, ckpt_path=resume_ckpt_path)


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
        data = nusc_mos_dataset.NuscSequentialModule(cfg, nusc, "test")
        data.setup()
    else:
        raise ValueError("Not supported test dataset.")

    model = models.MOSNet(cfg)
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
