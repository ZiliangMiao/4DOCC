# base
import os
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from nuscenes.utils.splits import create_splits_logs
from nuscenes.utils.geometry_utils import points_in_box

# model
import torch
from models.mos4d.models import MosNetwork
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# data
from nuscenes.nuscenes import NuScenes
from datasets.nusc_utils import split_logs_to_samples, get_input_sd_toks, get_transformed_pcd

# utils
from utils.metrics import ClassificationMetrics
from utils.deterministic import set_deterministic


def build_trainer(model_cfg, log_dir, model_params, callbacks):
    tb_logger = pl_loggers.TensorBoardLogger(log_dir, name=model_params, default_hp_metric=False)
    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp",
        num_nodes=1,
        devices=model_cfg["num_devices"],
        logger=tb_logger,
        log_every_n_steps=1,
        max_epochs=model_cfg['num_epoch'],
        accumulate_grad_batches=model_cfg["acc_batches"],
        callbacks=callbacks,
    )
    return trainer


def mos4d_baseline_train(model_cfg, dataset_cfg, resume_training:bool):
    # if resume training, load original config
    if resume_training:
        resume_model_cfg = yaml.safe_load(open(os.path.join(model_cfg['resume_path'], "hparams.yaml")))
        resume_ckpt_path = os.path.join(model_cfg['resume_path'], 'checkpoints', 'last.ckpt')
        model_cfg = resume_model_cfg
        
    # params
    train_dir = f"./logs/mos_baseline/mos4d_shuffle/{model_cfg['downsample_pct']}%{model_cfg['dataset_name']}"
    os.makedirs(train_dir, exist_ok=True)
    time = round(model_cfg['n_input'] * model_cfg['time_interval'] + model_cfg['n_input'] * model_cfg['n_skip'] * model_cfg['time_interval'], 2)
    model_params = f"vs-{model_cfg['quant_size']}_t-{time}_bs-{model_cfg['batch_size']}"

    # dataloader
    from datasets.dataloader import build_dataloader
    nusc = NuScenes(dataroot=dataset_cfg["nuscenes"]["root"], version=dataset_cfg["nuscenes"]["version"], verbose=False)
    train_dataloader = build_dataloader(model_cfg, dataset_cfg, 'train', nusc)

    # model
    model = MosNetwork(model_cfg, dataset_cfg, True)

    # lr_monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

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

    # trainer and tensorboard logger
    trainer = build_trainer(model_cfg, train_dir, model_params, [lr_monitor, checkpoint_saver])

    # training
    if resume_training:
        trainer.fit(model, train_dataloader, ckpt_path=resume_ckpt_path)
    else:
        trainer.fit(model, train_dataloader)


def load_pretrained_encoder(ckpt_path, model, use_mlp_decoder:bool):
    print(f"Load pretrained encoder from checkpoint {ckpt_path}")
    checkpoint = torch.load(ckpt_path)
    pretrained_dict = checkpoint["state_dict"]
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'decoder' not in k}

    if use_mlp_decoder:
        if 'loss.weight' in pretrained_dict.keys():
            pretrained_dict.pop('loss.weight')
    else: # filter out unnecessary keys (generate new dict)
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


def mos_finetune(model_cfg, dataset_cfg, resume_training:bool):
    if resume_training:
        resume_model_cfg = yaml.safe_load(open(os.path.join(model_cfg['resume_path'], "hparams.yaml")))
        resume_ckpt_path = os.path.join(model_cfg['resume_path'], 'checkpoints', 'last.ckpt')
        model_cfg = resume_model_cfg
     
    # pre-training checkpoint path
    pre_method = model_cfg['pretrain_method']
    pre_dataset = model_cfg['pretrain_dataset']
    pre_params = model_cfg['pretrain_params']
    pre_version = model_cfg['pretrain_version']
    pre_epoch = model_cfg['pretrain_epoch']
    pre_model_dir = f"./logs/{pre_method}/{pre_dataset}/{pre_params}/version_{pre_version}/checkpoints"
    pre_ckpt_name = f"epoch={pre_epoch}.ckpt"
    pre_ckpt_path = os.path.join(pre_model_dir, pre_ckpt_name)

    # fine-tuning params
    dataset_name = model_cfg['dataset_name']
    downsample_pct = model_cfg['downsample_pct']
    finetune_dir = f"./logs/{pre_method}(epoch-{pre_epoch})-mos_finetune/{pre_dataset}-{downsample_pct}%{dataset_name}"
    os.makedirs(finetune_dir, exist_ok=True)
    time = round(model_cfg['n_input'] * model_cfg['time_interval'] + model_cfg['n_input'] * model_cfg['n_skip'] * model_cfg['time_interval'], 2)
    finetune_model_params = f"vs-{model_cfg['quant_size']}_t-{time}_bs-{model_cfg['batch_size']}"

    # dataloader
    from datasets.dataloader import build_dataloader
    nusc = NuScenes(dataroot=dataset_cfg["nuscenes"]["root"], version=dataset_cfg["nuscenes"]["version"], verbose=False)
    train_dataloader = build_dataloader(model_cfg, dataset_cfg, 'finetune', nusc=nusc)

    # load pre-trained encoder to fine-tuning model
    finetune_model = MosNetwork(model_cfg, dataset_cfg, True)

    # lr_monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

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

    # trainer and tensorboard logger
    trainer = build_trainer(model_cfg, finetune_dir, finetune_model_params, [lr_monitor, checkpoint_saver])

    # training
    if resume_training:  # resume training
        trainer.fit(finetune_model, train_dataloader, ckpt_path=resume_ckpt_path)
    else:
        finetune_model = load_pretrained_encoder(pre_ckpt_path, finetune_model, model_cfg['use_mlp_decoder'])
        trainer.fit(finetune_model, train_dataloader)


def mos_eval(mode, cfg_eval, cfg_dataset):
    '''
    This function is for MOS evaluation, using both object-wise recall and IoU_w/o metrics. Used for both validation and testing
    Args:
        cfg_eval:
        cfg_dataset:

    Returns:

    '''
    # path
    model_dir = cfg_eval['model_dir']
    if mode == 'val':
        results_dir = os.path.join(model_dir, 'val_results')
    elif mode == 'test':
        results_dir = os.path.join(model_dir, 'test_results')
    else:
        print("Not a valid evaluation mode.")
        return None
    os.makedirs(results_dir, exist_ok=True)

    # config
    cfg_model = yaml.safe_load(open(os.path.join(model_dir, "hparams.yaml")))

    # dataloader
    from datasets.dataloader import build_dataloader
    nusc = NuScenes(dataroot=dataset_cfg["nuscenes"]["root"], version=dataset_cfg["nuscenes"]["version"], verbose=False)
    eval_dataloader = build_dataloader(cfg_model, cfg_dataset, mode, nusc)
    
    # create xlsx file
    excel_file = os.path.join(results_dir, 'results.xlsx')
    if not os.path.exists(excel_file):
        df = pd.DataFrame(columns=['epoch', 'recall', 'iou'])
        df.to_excel(excel_file, index=False)
    else:
        df = pd.read_excel(excel_file)

    # evaluation
    for epoch in cfg_eval["eval_epoch"]:
        date = datetime.now().strftime('%m%d')  # %m%d-%H%M
        log_file = os.path.join(results_dir, f"epoch_{epoch}_{date}.txt")
        formatter = logging.Formatter(fmt='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        logger = logging.getLogger(f'test epoch {epoch} logger')
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.info(str(cfg_eval))
        logger.info(log_file)

        # checkpoint
        ckpt_path = os.path.join(model_dir, "checkpoints", f"epoch={epoch}.ckpt")

        # model
        model = MosNetwork(cfg_model, cfg_dataset, False, model_dir=model_dir, eval_epoch=epoch, logger=logger, save_pred=cfg_eval['save_pred'], nusc=nusc)
        iou_metric = ClassificationMetrics(n_classes=3, ignore_index=0)

        # predict
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=cfg_eval["num_devices"], deterministic=True)
        trainer.predict(model, dataloaders=eval_dataloader, return_predictions=True, ckpt_path=ckpt_path)

        # metric-1: object-level iou
        obj_recall = torch.mean(torch.tensor(model.object_iou_list)).item()
        logger.info('Metric-1 object-level recall: %.3f' % (obj_recall * 100))

        # metric-2: point-level iou
        point_iou = iou_metric.get_iou(model.accumulated_conf_mat)
        logger.info('Metric-2 point-level moving iou: %.3f' % (point_iou[2] * 100))
        logger.info('Metric-2 point-level static iou: %.3f' % (point_iou[1] * 100))

        # append new row to dataframe
        new_row = pd.DataFrame({
            'epoch': [epoch],
            'recall': [obj_recall * 100],
            'iou': [point_iou[2].item() * 100]
        })
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(excel_file, index=False)  # save the updated dataframe to excel


def mov_pts_statistics(cfg_dataset, cfg_model):
    nusc = NuScenes(dataroot=dataset_cfg["nuscenes"]["root"], version=dataset_cfg["nuscenes"]["version"], verbose=False)
    split_logs = create_splits_logs('train', nusc)
    split_logs_val = create_splits_logs('val', nusc)
    split_logs = split_logs + split_logs_val
    sample_toks = split_logs_to_samples(nusc, split_logs)
    sample_to_sd_toks_dict = get_input_sd_toks(nusc, cfg_model, sample_toks)

    mov_obj_num = 0
    pts_in_mov_obj_list = []
    for sample_tok in list(sample_to_sd_toks_dict.keys()):
        # sample and sample data
        sample = nusc.get('sample', sample_tok)
        sd_tok = sample['data']['LIDAR_TOP']
        sample_data = nusc.get('sample_data', sd_tok)

        # point cloud
        org, pcd, ts, valid_mask = get_transformed_pcd(nusc, cfg_model, sd_tok, sd_tok)

        # labels
        mos_labels_dir = os.path.join(cfg_dataset["nuscenes"]["root"], "mos_labels", cfg_dataset["nuscenes"]["version"])
        mos_label_file = os.path.join(mos_labels_dir, sd_tok + "_mos.label")
        mos_labels = torch.tensor(np.fromfile(mos_label_file, dtype=np.uint8))[valid_mask]

        # number of points in moving object
        _, bbox_list, _ = nusc.get_sample_data(sd_tok, selected_anntokens=sample['anns'], use_flat_vehicle_coordinates=False)
        for ann_tok, box in zip(sample['anns'], bbox_list):
            ann = nusc.get('sample_annotation', ann_tok)
            if ann['num_lidar_pts'] == 0: continue
            obj_pts_mask = points_in_box(box, pcd[:, :3].T)
            if np.sum(obj_pts_mask) == 0:  # not lidar points in obj bbox
                continue
            # gt and pred object labels
            gt_obj_labels = mos_labels[obj_pts_mask]
            mov_pts_mask = gt_obj_labels == 2
            num_mov_pts = torch.sum(mov_pts_mask)
            if num_mov_pts == 0:  # static object
                continue
            else:
                mov_obj_num += 1
                pts_in_mov_obj_list.append(num_mov_pts)

    # histogram
    fig_1 = plt.figure()
    ax_1 = fig_1.gca()
    hist_1 = ax_1.hist(pts_in_mov_obj_list, range=(0,400), bins=100, rwidth=0, align='left', log=True)
    ax_1.set_xlabel("Number of points in moving objects")
    ax_1.set_ylabel("Frequency (Num of moving objects)")
    fig_1.savefig("./statistics_points_in_mov_obj_3.png", dpi=1000)

    fig_2 = plt.figure()
    ax_2 = fig_2.gca()
    hist_2 = ax_2.hist(pts_in_mov_obj_list, range=(0, 200), bins=100, rwidth=0, align='left', log=True)
    ax_2.set_xlabel("Number of points in moving objects")
    ax_2.set_ylabel("Frequency (Num of moving objects)")
    fig_2.savefig("./statistics_points_in_mov_obj_4.png", dpi=1000)

    fig_3 = plt.figure()
    ax_3 = fig_3.gca()
    hist_3 = ax_3.hist(pts_in_mov_obj_list, range=(0, 200), bins=200, rwidth=0, align='left', log=True)
    ax_3.set_xlabel("Number of points in moving objects")
    ax_3.set_ylabel("Frequency (Num of moving objects)")
    fig_3.savefig("./statistics_points_in_mov_obj_5.png", dpi=1000)

    fig_4 = plt.figure()
    ax_4 = fig_4.gca()
    hist_4 = ax_4.hist(pts_in_mov_obj_list, range=(0, 400), bins=200, rwidth=0, align='left', log=True)
    ax_4.set_xlabel("Number of points in moving objects")
    ax_4.set_ylabel("Frequency (Num of moving objects)")
    fig_4.savefig("./statistics_points_in_mov_obj_6.png", dpi=1000)


if __name__ == "__main__":
    # deterministic
    set_deterministic(666)

    # mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'finetune', 'val', 'test'], default='test')
    parser.add_argument('--resume_version', type=int, default=-1)  # -1: not resuming
    parser.add_argument('--autodl', type=bool, help="autodl server", default=False)
    parser.add_argument('--mars', type=bool, help="mars server", default=False)
    parser.add_argument('--hpc', type=bool, help="hpc server", default=False)
    args = parser.parse_args()

    # load config
    with open("configs/mos4d.yaml", "r") as f:
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

    # mov_pts_statistics(dataset_cfg, cfg[args.mode])

    # training
    if args.mode == 'train':  # train set = full org train set
        mos4d_baseline_train(cfg[args.mode], dataset_cfg, args.resume_version)
    elif args.mode == 'finetune':  # finetune train set = subset of org train set
        mos_finetune(cfg[args.mode], dataset_cfg, args.resume_version)
    elif args.mode == 'val':  # val set = a subset of org train set
        mos_eval(args.mode, cfg[args.mode], dataset_cfg)
    elif args.mode == 'test':  # test set = org val set
        mos_eval(args.mode, cfg[args.mode], dataset_cfg)


    # def plot_metrics_vs_epoch():
    #     import pandas as pd
    #     import matplotlib.pyplot as plt
    #
    #     # 读取test和validation的Excel文件
    #     test_file_path = "/home/ziliang/Projects/4DOCC/logs/iccv/nusc10/511-49-nusc10-iccv/vs-0.1_t-3.0_bs-4/version_0/results/metrics.xlsx"
    #     val_file_path = "/home/ziliang/Projects/4DOCC/logs/iccv/nusc10/511-49-nusc10-iccv/vs-0.1_t-3.0_bs-4/version_0/val_results/results.xlsx"
    #     df_test = pd.read_excel(test_file_path)
    #     df_val = pd.read_excel(val_file_path)
    #
    #     # 创建图形
    #     plt.figure(figsize=(12, 8))
    #
    #     # 绘制test set的结果
    #     plt.plot(df_test.iloc[:, 0], df_test.iloc[:, 1], 'b-', label='Test Recall', marker='o')
    #     plt.plot(df_test.iloc[:, 0], df_test.iloc[:, 2], 'r-', label='Test IoU', marker='s')
    #
    #     # 绘制validation set的结果
    #     plt.plot(df_val.iloc[:, 0], df_val.iloc[:, 1], 'b--', label='Val Recall', marker='^')
    #     plt.plot(df_val.iloc[:, 0], df_val.iloc[:, 2], 'r--', label='Val IoU', marker='v')
    #
    #     # 设置图形属性
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Metric')
    #     plt.title('Metrics vs Epoch (Test & Validation Sets)')
    #     plt.grid(True)
    #     plt.legend()
    #
    #     # 保存图形
    #     # plt.savefig('metrics_vs_epoch.png', dpi=300, bbox_inches='tight')
    #     plt.show()
    #     plt.close()
    # # 调用函数
    # plot_metrics_vs_epoch()
    
