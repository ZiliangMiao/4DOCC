import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import yaml
from nuscenes.utils.geometry_utils import points_in_box
from pytorch_lightning import LightningModule
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock
from models.backbone import MinkUNetBackbone
from utils.metrics import ClassificationMetrics


#######################################
# Lightning Module
#######################################


class SemanticNetwork(LightningModule):
    def __init__(self, cfg_model: dict, train_flag: bool, **kwargs):
        super().__init__()
        if train_flag:
            self.save_hyperparameters(cfg_model)
        self.cfg_model = cfg_model
        self.n_cls = 17
        self.ignore_class_idx = [0]  # ignore unknown class when calculating scores

        # semantic learning map
        with open("configs/semantic_learning_map.yaml", "r") as f:
            self.cfg_semantic = yaml.safe_load(f)

        # encoder - decoder
        if self.cfg_model['use_mlp_decoder']:
            self.encoder = SemanticModel(cfg_model, self.cfg_model['pretrain_featdim'])
            self.decoder = SemanticHead(in_dim=self.cfg_model['pretrain_featdim'], planes=[128, 64, 32, 16, self.n_cls])
        else:
            self.encoder = SemanticModel(cfg_model, self.n_cls)

        # metrics
        self.ClassificationMetrics = ClassificationMetrics(self.n_cls, self.ignore_class_idx)

        # loss
        weight = [0.0 if i in self.ignore_class_idx else 1.0 for i in range(self.n_cls)]
        weight = torch.Tensor([w / sum(weight) for w in weight])  # ignore unknown class when calculate loss
        self.loss = nn.NLLLoss(weight=weight)

        # epoch accumulated
        self.epoch_conf_mat = torch.zeros(self.n_cls, self.n_cls)

        # save predictions
        if not train_flag:
            self.model_dir = kwargs['model_dir']
            self.test_epoch = kwargs['test_epoch']
            self.test_logger = kwargs['test_logger']
            self.nusc = kwargs['nusc']
            self.pred_dir = os.path.join(self.model_dir, "predictions", f"epoch_{self.test_epoch}")
            os.makedirs(self.pred_dir, exist_ok=True)

            # metrics
            self.accumulated_conf_mat = torch.zeros(self.n_cls, self.n_cls)  # to calculate point-level avg. iou

    def forward(self, batch: dict):
        # unfold batch data
        meta_batch, pcds_batch, _ = batch

        # encoder
        softmax = nn.Softmax(dim=1)
        sparse_featmap = self.encoder(pcds_batch)
        semantic_probs_batch = []
        for batch_idx in range(len(pcds_batch)):
            coords = sparse_featmap.coordinates_at(batch_index=batch_idx)
            feats = sparse_featmap.features_at(batch_index=batch_idx)
            ref_time_mask = coords[:, -1] == 0
            semantic_feats = feats[ref_time_mask]
            # TODO: add a MLP decoder
            if self.cfg_model['use_mlp_decoder']:
                semantic_feats = self.decoder(semantic_feats)
            semantic_feats[:, self.ignore_class_idx] = -float("inf")
            semantic_probs = softmax(semantic_feats)
            semantic_probs_batch.append(semantic_probs)
        return semantic_probs_batch

    def configure_optimizers(self):
        lr_start = self.cfg_model["lr_start"]
        lr_epoch = self.cfg_model["lr_epoch"]
        lr_decay = self.cfg_model["lr_decay"]
        weight_decay = self.cfg_model["weight_decay"]
        optimizer = torch.optim.Adam(self.parameters(), lr=lr_start, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_epoch, gamma=lr_decay)
        return [optimizer], [scheduler]

    def get_loss(self, semantic_probs, semantic_labels):
        semantic_labels = semantic_labels.long()
        assert len(semantic_labels) == len(semantic_probs)
        semantic_prob_log = torch.log(semantic_probs.clamp(min=1e-8))
        loss = self.loss(semantic_prob_log, semantic_labels)  # dtype of torch.nllloss must be torch.long
        return loss

    def training_step(self, batch: tuple, batch_idx, dataloader_index=0):
        # model_dict = self.state_dict()  # check state dict
        sd_toks_batch, pcds_batch, semantic_labels_batch = batch
        semantic_probs_batch = self.forward(batch)  # encoder & decoder
        semantic_probs = torch.cat(semantic_probs_batch, dim=0)
        pred_labels = torch.argmax(semantic_probs, axis=1)
        semantic_labels = torch.cat(semantic_labels_batch, dim=0)
        loss = self.get_loss(semantic_probs, semantic_labels)

        # metrics
        conf_mat = self.ClassificationMetrics.compute_conf_mat(pred_labels, semantic_labels)
        iou = self.ClassificationMetrics.get_iou(conf_mat)
        semantic_cls_names = list(self.cfg_semantic['labels_16'].values())

        # logging
        self.log("loss", loss.item(), on_step=True, prog_bar=True, logger=True)
        miou_list = []
        for iou_cls, cls_name in zip(iou, semantic_cls_names):
            miou_list.append(iou_cls.item() * 100)
            self.log(cls_name, iou_cls.item() * 100, on_step=True, prog_bar=True, logger=True)
        self.log("mIoU", np.mean(miou_list), on_step=True, prog_bar=True, logger=True)

        # logging
        self.epoch_conf_mat += conf_mat
        torch.cuda.empty_cache()
        return loss

    def on_train_epoch_end(self):
        # metrics in one epoch
        iou = self.ClassificationMetrics.get_iou(self.epoch_conf_mat)[1:]  # remove noise
        semantic_cls_names = list(self.cfg_semantic['labels_16'].values())[1:]  # remove noise

        # logging
        miou_list = []
        for iou_cls, cls_name in zip(iou, semantic_cls_names):
            miou_list.append(iou_cls.item() * 100)
            self.log('epoch_'+cls_name, iou_cls.item() * 100, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_mIoU", np.mean(miou_list), on_epoch=True, prog_bar=True, logger=True)

        # clean
        self.epoch_conf_mat.zero_()
        torch.cuda.empty_cache()

    def validation_step(self, batch: tuple, batch_idx):
        a = 1

    def on_validation_epoch_end(self):
        a = 1

    def predict_step(self, batch: tuple, batch_idx):
        sd_toks_batch, pcds_batch, semantic_labels_batch = batch
        semantic_probs_batch = self.forward(batch)  # encoder & decoder

        for sd_tok, pcds_4d, semantic_probs, semantic_labels in zip(sd_toks_batch, pcds_batch, semantic_probs_batch, semantic_labels_batch):
            # prediction
            pred_labels = torch.argmax(semantic_probs, axis=1)

            # metric
            conf_mat = self.ClassificationMetrics.compute_conf_mat(pred_labels, semantic_labels)
            self.accumulated_conf_mat += conf_mat
            iou = self.ClassificationMetrics.get_iou(conf_mat)[1:]

            # save predicted labels
            semantic_pred_file = os.path.join(self.pred_dir, f"{sd_tok}_semantic_pred.label")
            pred_labels = torch.argmax(semantic_probs, dim=1).type(torch.uint8).detach().cpu().numpy()
            pred_labels.tofile(semantic_pred_file)

            # logging
            self.test_logger.info("Val sd tok: %s, mIoU: %.3f", sd_tok, np.mean(list(iou * 100)))
        torch.cuda.empty_cache()


#######################################
# Modules
#######################################


class SemanticModel(nn.Module):
    def __init__(self, cfg_model: dict, n_cls: int):
        super().__init__()

        # backbone network
        self.n_cls = n_cls
        self.MinkUNet = MinkUNetBackbone(in_channels=1, out_channels=self.n_cls, D=cfg_model['pos_dim'])

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

class SemanticHead(nn.Module):
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






