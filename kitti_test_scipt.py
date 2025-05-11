# base
import argparse
import os
import yaml
# torch
import torch
from pytorch_lightning import Trainer
# dataset
from datasets.mos4d.kitti_test import KittiMOSDataset, KittiDataloader
from utils.deterministic import set_deterministic
from models.mos4d.models_test import MosNetwork
from utils.vis.vis_occ_or_esdf import seq_idx


def kitti_test(cfg_test, cfg_dataset):
    # model config
    model_dir = cfg_test['model_dir']
    cfg_model = yaml.safe_load(open(os.path.join(model_dir, "hparams.yaml")))
    log_dir = os.path.join(model_dir, 'results')
    os.makedirs(log_dir, exist_ok=True)

    # construct model
    test_epoch = cfg_test["eval_epoch"][0]
    ckpt_path = os.path.join(model_dir, "checkpoints", f"epoch={test_epoch}.ckpt")
    trainer = Trainer(accelerator="gpu", strategy="ddp", devices=cfg_test["num_devices"], deterministic=True)

    # loop each test sequences
    for test_seq in cfg_dataset['sekitti']['test']:
        # dataloader
        test_set = KittiMOSDataset(cfg_model, dataset_cfg, cfg_test, test_seq)  # 'train', 'val', 'test'
        dataloader = KittiDataloader(cfg_model, test_set)
        dataloader.setup()
        test_dataloader = dataloader.test_dataloader()

        # model
        model = MosNetwork(cfg_model, cfg_dataset, model_dir=model_dir, test_epoch=test_epoch, test_seq=test_seq)

        # predict
        trainer.predict(model, dataloaders=test_dataloader, return_predictions=True, ckpt_path=ckpt_path)


if __name__ == "__main__":
    # deterministic
    set_deterministic(666)

    # mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'finetune', 'test'], default='test')
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
        dataset_cfg['sekitti']['root'] = '/root/autodl-tmp' + dataset_cfg['sekitti']['root']
    elif args.mars:
        dataset_cfg['sekitti']['root'] = '/home/miaozl' + dataset_cfg['sekitti']['root']
    elif args.hpc:
        dataset_cfg['sekitti']['root'] = '/lustre1/g/mech_mars' + dataset_cfg['sekitti']['root']
    else:
        dataset_cfg['sekitti']['root'] = '/home/ziliang' + dataset_cfg['sekitti']['root']

    # training from scratch
    if args.mode == 'test':
        kitti_test(cfg[args.mode], dataset_cfg)
