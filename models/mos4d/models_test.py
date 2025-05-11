import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from nuscenes.utils.geometry_utils import points_in_box
from pytorch_lightning import LightningModule
import MinkowskiEngine as ME

from datasets.kitti_utils import load_mos_labels
from models.backbone import MinkUNetBackbone
from utils.metrics import ClassificationMetrics


#######################################
# Lightning Module
#######################################


class MosNetwork(LightningModule):
    def __init__(self, cfg_model: dict, cfg_dataset: dict, **kwargs):
        super().__init__()
        self.cfg_model = cfg_model
        self.cfg_dataset = cfg_dataset
        self.n_mos_cls = 3  # 0 -> unknown, 1 -> static, 2 -> moving
        self.ignore_class_idx = 0  # ignore unknown class when calculating scores

        # loss
        weight = [0.0, 1.0, 1.0]
        weight = torch.Tensor([w / sum(weight) for w in weight])  # ignore unknown class when calculate loss
        self.loss = nn.NLLLoss(weight=weight)

        # encoder - decoder
        if self.cfg_model['use_mlp_decoder']:
            self.encoder = MOSModel(cfg_model, self.cfg_model['pretrain_featdim'])
            self.decoder = MOSHead(in_dim=self.cfg_model['pretrain_featdim'], planes=[128, 64, 32, 16, self.n_mos_cls])
        else:
            self.encoder = MOSModel(cfg_model, self.n_mos_cls)

        # save predictions
        self.model_dir = kwargs['model_dir']
        self.test_epoch = kwargs['eval_epoch']
        self.pred_dir = os.path.join(self.model_dir, f"sequences_epoch_{self.test_epoch}", str(kwargs['test_seq']).zfill(2), "predictions")
        os.makedirs(self.pred_dir, exist_ok=True)

    def forward(self, batch: dict):
        # unfold batch data
        meta_batch, pcds_batch = batch

        # encoder
        sparse_featmap = self.encoder(pcds_batch)

        # decoder
        softmax = nn.Softmax(dim=1)
        mos_probs_batch = []
        for batch_idx in range(len(pcds_batch)):
            coords = sparse_featmap.coordinates_at(batch_index=batch_idx)
            feats = sparse_featmap.features_at(batch_index=batch_idx)
            ref_time_mask = coords[:, -1] == 0
            mos_feats = feats[ref_time_mask]
            # TODO: task MLP head
            if self.cfg_model['use_mlp_decoder']:
                mos_feats = self.decoder(mos_feats)
            mos_feats[:, self.ignore_class_idx] = -float("inf")
            mos_probs = softmax(mos_feats)
            mos_probs_batch.append(mos_probs)
        return mos_probs_batch


    def predict_step(self, batch: tuple, batch_idx):
        # unfold batch data and model forward
        meta_batch, pcds_batch = batch
        mos_probs_batch = self.forward(batch)

        # iterate each sample
        for scan_idx, pcds_4d, mos_probs in zip(meta_batch, pcds_batch, mos_probs_batch):
            # save predicted labels
            mos_pred_file = os.path.join(self.pred_dir, f"{scan_idx}.label")
            pred_labels = torch.argmax(mos_probs, dim=1).type(torch.uint32).detach().cpu().numpy()
            pred_labels.tofile(mos_pred_file)

            # logger
            print(f"Test scan: %s", scan_idx)
        torch.cuda.empty_cache()


#######################################
# Modules
#######################################


class MOSModel(nn.Module):
    def __init__(self, cfg_model: dict, n_mos_cls: int):
        super().__init__()

        # backbone network
        self.MinkUNet = MinkUNetBackbone(in_channels=1, out_channels=n_mos_cls, D=cfg_model['pos_dim'])

        dx = dy = dz = cfg_model["quant_size"]
        dt = 1  # TODO: should be cfg_model["time_interval"], handle different lidar frequency of kitti and nuscenes
        self.quant = torch.Tensor([dx, dy, dz, dt])
        self.scene_bbox = cfg_model["scene_bbox"]

    def forward(self, pcds_4d_batch):
        # quantized 4d pcd and initialized features
        self.quant = self.quant.type_as(pcds_4d_batch[0])
        quant_4d_pcds = [torch.div(pcd, self.quant) for pcd in pcds_4d_batch]
        feats = [0.5 * torch.ones(len(pcd), 1).type_as(pcd) for pcd in pcds_4d_batch]

        # sparse collate, tensor field, net calculation
        coords, feats = ME.utils.sparse_collate(quant_4d_pcds, feats)
        tensor_field = ME.TensorField(features=feats, coordinates=coords,
                                      quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        sparse_input = tensor_field.sparse()
        sparse_output = self.MinkUNet(sparse_input)

        # TODO: point-wise sparse feature output
        sparse_featmap = sparse_output.slice(tensor_field)
        sparse_featmap.coordinates[:, 1:] = torch.mul(sparse_featmap.coordinates[:, 1:], self.quant)
        return sparse_featmap


class MOSHead(nn.Module):
    def __init__(self, in_dim, planes, **kwargs):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('linear0', nn.Linear(in_dim, planes[0])),
            ('relu0', nn.ReLU(inplace=False)),
            ('linear1', nn.Linear(planes[0], planes[1])),
            ('relu1', nn.ReLU(inplace=False)),
            ('linear2', nn.Linear(planes[1], planes[2])),
            ('relu2', nn.ReLU(inplace=False)),
            ('linear3', nn.Linear(planes[2], planes[3])),
            ('relu3', nn.ReLU(inplace=False)),
            ('final', nn.Linear(planes[3], planes[4])),
        ]))

    def forward(self, x):
        if type(x) is list:  # input batch data, not concatenated data
            probs_list = []
            for x_i in x:
                probs_list.append(self.block(x_i))
            return probs_list
        else:
            return self.block(x)






