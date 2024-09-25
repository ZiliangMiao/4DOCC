import logging
import os
from collections import OrderedDict
from random import sample as random_sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from pytorch_lightning import LightningModule
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock
from lib.minkowski.minkunet import MinkUNetBase
from utils.metrics import ClassificationMetrics
from datasets.ours.nusc import NuscBgDataset
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


#######################################
# Lightning Module
#######################################


class UnONetwork(LightningModule):
    def __init__(self, cfg_model: dict, train_flag: bool, **kwargs):
        super().__init__()
        if train_flag:
            self.save_hyperparameters(cfg_model)
        self.cfg_model = cfg_model
        self.n_uno_cls = self.cfg_model['num_cls']
        self.pos_dim = self.cfg_model['pos_dim']
        self.feat_dim = self.cfg_model['feat_dim']
        self.hidden_size = self.cfg_model['hidden_size']
        self.iters_per_epoch = kwargs['iters_per_epoch']

        # normal encoder
        self.encoder = MotionEncoder(self.cfg_model, self.n_uno_cls)

        # uno decoder
        self.offset_predictor = OffsetPredictor(self.pos_dim, self.feat_dim, self.hidden_size, 3)
        self.decoder = UnODecoder(self.pos_dim, self.feat_dim * 2, self.hidden_size, self.n_uno_cls)

        # metrics
        self.ClassificationMetrics = ClassificationMetrics(self.n_uno_cls, ignore_index=[])

        # pytorch lightning training output
        self.training_step_outputs = []
        self.validation_step_outputs = []

        # save predictions
        if not train_flag:
            self.model_dir = kwargs['model_dir']
            self.test_epoch = kwargs['test_epoch']
            self.pred_dir = os.path.join(self.model_dir, "predictions", f"epoch_{self.test_epoch}")
            os.makedirs(self.pred_dir, exist_ok=True)

    def forward(self, batch):
        # unfold batch: [(ref_sd_tok, uno_sd_toks), pcds_4d, (uno_query_points, uno_labels)]
        meta_batch, pcds_batch, uno_samples_batch = batch

        # encoder
        dense_featmap = self.encoder(pcds_batch)

        # get bg samples
        uno_labels_batch = []
        uno_feats_batch = []
        for batch_idx, uno_samples in enumerate(uno_samples_batch):
            # ref_sd_tok = meta_batch[batch_idx][0]
            # uno_sd_toks = meta_batch[batch_idx][1]
            uno_query_points_4d = uno_samples[0]  # x, y, z, ts
            uno_labels = uno_samples[1]

            # features interpolation
            feats = dense_featmap  # (uno_query_points_4d)

            # offset prediction
            pos_offset = self.offset_predictor(uno_query_points_4d, feats)

            # offset features interpolation
            offset_feats = dense_featmap  # (uno_query_points_4d + pos_offset)

            # aggregated feats
            agg_feats = torch.cat([feats, offset_feats], dim=0)

            # collate to batch
            uno_feats_batch.append(agg_feats)
            uno_labels_batch.append(uno_labels)
        # stack batch data
        uno_feats = torch.stack(uno_feats_batch)
        uno_labels = torch.stack(uno_labels_batch)

        # decoder
        uno_probs = self.decoder(uno_feats)
        return uno_probs, uno_labels

    def configure_optimizers(self):
        lr_start = self.cfg_model["lr_start"]
        weight_decay = self.cfg_model["weight_decay"]
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr_start, weight_decay=weight_decay)
        lr_max = self.cfg_model["lr_max"]
        lr_min = self.cfg_model["lr_min"]
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                  warmup_steps=0.02 * self.cfg_model['num_epoch'] * self.iters_per_epoch,
                                                  first_cycle_steps=self.cfg_model['num_epoch'] * self.iters_per_epoch,
                                                  cycle_mult=1.0,  # period
                                                  max_lr=lr_max,
                                                  min_lr=lr_min,
                                                  gamma=1.0)  # lr_max decrease rate
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_epoch, gamma=lr_decay)
        # return [optimizer], [scheduler]  # TODO: default scheduler interval is 'epoch'

    def get_loss(self, mutual_probs, mutual_labels, mutual_confidence):
        mutual_labels = mutual_labels.long()
        assert len(mutual_labels) == len(mutual_probs)
        log_prob = torch.log(mutual_probs.clamp(min=1e-8))
        loss_func = nn.NLLLoss(reduction='none')
        loss = loss_func(log_prob, mutual_labels)  # dtype of torch.nllloss must be torch.long
        loss = torch.mean(loss * mutual_confidence)  # TODO: add confidence weight
        return loss

    def training_step(self, batch: tuple, batch_idx, dataloader_index=0):
        # model_dict = self.state_dict()  # check state dict
        mutual_probs_batch, mutual_labels_batch, mutual_confidence_batch = self.forward(batch)  # encoder & decoder
        mutual_probs = torch.cat(mutual_probs_batch, dim=0)
        pred_labels = torch.argmax(mutual_probs, axis=1)
        mutual_labels = torch.cat(mutual_labels_batch, dim=0)
        mutual_confidence = torch.cat(mutual_confidence_batch, dim=0)
        loss = self.get_loss(mutual_probs, mutual_labels, mutual_confidence)

        # metrics
        conf_mat = self.ClassificationMetrics.compute_conf_mat(pred_labels, mutual_labels)
        iou = self.ClassificationMetrics.get_iou(conf_mat)
        unk_iou, free_iou, occ_iou = iou[0], iou[1], iou[2]
        acc = self.ClassificationMetrics.get_acc(conf_mat)
        unk_acc, free_acc, occ_acc = acc[0], acc[1], acc[2]

        # logging
        self.log("loss", loss.item(), on_step=True, prog_bar=True, logger=True)
        self.log("unk_iou", unk_iou.item() * 100, on_step=True, prog_bar=True, logger=True)
        self.log("free_iou", free_iou.item() * 100, on_step=True, prog_bar=True, logger=True)
        self.log("occ_iou", occ_iou.item() * 100, on_step=True, prog_bar=True, logger=True)
        self.log("unk_acc", unk_acc.item() * 100, on_step=True, prog_bar=True, logger=True)
        self.log("free_acc", free_acc.item() * 100, on_step=True, prog_bar=True, logger=True)
        self.log("occ_acc", occ_acc.item() * 100, on_step=True, prog_bar=True, logger=True)
        self.training_step_outputs.append({"loss": loss.item(), "confusion_matrix": conf_mat})
        torch.cuda.empty_cache()
        return loss

    def on_train_epoch_end(self):
        conf_mat_list = [output["confusion_matrix"] for output in self.training_step_outputs]
        acc_conf_mat = torch.zeros(self.n_uno_cls, self.n_uno_cls)
        for conf_mat in conf_mat_list:
            acc_conf_mat = acc_conf_mat.add(conf_mat)

        # metrics in one epoch
        iou = self.ClassificationMetrics.get_iou(acc_conf_mat)
        unk_iou, free_iou, occ_iou = iou[0], iou[1], iou[2]
        acc = self.ClassificationMetrics.get_acc(acc_conf_mat)
        unk_acc, free_acc, occ_acc = acc[0], acc[1], acc[2]
        self.log("epoch_unk_iou", unk_iou.item() * 100, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_free_iou", free_iou.item() * 100, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_occ_iou", occ_iou.item() * 100, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_unk_acc", unk_acc.item() * 100, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_free_acc", free_acc.item() * 100, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_occ_acc", occ_acc.item() * 100, on_epoch=True, prog_bar=True, logger=True)

        # clean
        self.training_step_outputs = []
        torch.cuda.empty_cache()

    def predict_step(self, batch: tuple, batch_idx):
        # model_dict = self.state_dict()  # check state dict
        meta_batch, _, _ = batch
        mutual_probs_batch, mutual_labels_batch, mutual_confidence_batch = self.forward(batch)  # encoder & decoder

        # iterate each batch data for predicted label saving
        acc_conf_mat = torch.zeros(self.n_uno_cls, self.n_uno_cls)
        for meta, mutual_probs, mutual_labels in zip(meta_batch, mutual_probs_batch, mutual_labels_batch):
            sd_tok = meta[0]
            pred_labels = torch.argmax(mutual_probs, axis=1)

            # metrics
            conf_mat = self.ClassificationMetrics.compute_conf_mat(pred_labels, mutual_labels)
            iou = self.ClassificationMetrics.get_iou(conf_mat)
            unk_iou, free_iou, occ_iou = iou[0].item()*100, iou[1].item()*100, iou[2].item()*100
            acc = self.ClassificationMetrics.get_acc(conf_mat)
            unk_acc, free_acc, occ_acc = acc[0].item()*100, acc[1].item()*100, acc[2].item()*100

            # update confusion matrix
            acc_conf_mat = acc_conf_mat.add(conf_mat)

            # save predicted labels for visualization
            pred_file = os.path.join(self.pred_dir, f"{sd_tok}_bg_pred.label")
            pred_labels = pred_labels.type(torch.uint8).detach().cpu().numpy()
            pred_labels.tofile(pred_file)

            # logger
            logging.info("Val sample data (IoU/Acc): %s, [Unk %.3f/%.3f], [Occ %.3f/%.3f], [Free %.3f/%.3f]",
                         sd_tok, unk_iou, unk_acc, free_iou, free_acc, occ_iou, occ_acc)
        torch.cuda.empty_cache()
        return {"confusion_matrix": acc_conf_mat.detach().cpu()}


#######################################
# Modules
#######################################


class MinkUNet14(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)
    # PLANES = (8, 16, 32, 64, 64, 32, 16, 8)
    PLANES = (8, 32, 128, 256, 256, 128, 32, 8)
    INIT_DIM = 8


class MotionEncoder(nn.Module):
    def __init__(self, cfg_model: dict, n_classes: int):
        super().__init__()
        # backbone network
        self.feat_dim = cfg_model["feat_dim"]
        self.MinkUNet = MinkUNet14(in_channels=1, out_channels=self.feat_dim, D=cfg_model['pos_dim'])  # D: UNet spatial dim

        dx = dy = dz = cfg_model["quant_size"]
        dt = 1  # TODO: cfg_model["time_interval"]
        self.quant = torch.Tensor([dx, dy, dz, dt])

        self.scene_bbox = cfg_model["scene_bbox"]
        featmap_size = cfg_model["featmap_size"]
        z_height = int((self.scene_bbox[5] - self.scene_bbox[2]) / dz)
        y_length = int((self.scene_bbox[4] - self.scene_bbox[1]) / dy)
        x_width = int((self.scene_bbox[3] - self.scene_bbox[0]) / dx)
        b_size = cfg_model["batch_size"]
        self.featmap_shape = [b_size, x_width, y_length, z_height, cfg_model["n_input"]]

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

        # TODO: dense feature map output (with interpolation)
        featmap_shape = torch.Size([self.featmap_shape[0], 1, self.featmap_shape[1], self.featmap_shape[2], self.featmap_shape[3], self.featmap_shape[4]])
        dense_featmap, _, _ = sparse_output.dense(shape=featmap_shape, min_coordinate=torch.IntTensor([self.scene_bbox[0], self.scene_bbox[1], self.scene_bbox[2], -self.cfg_model["n_input"]+1]))
        return dense_featmap


class OffsetPredictor(nn.Module):
    def __init__(self, pos_dim, feat_dim, hidden_size, out_pos_dim):
        super().__init__()
        self.pos_linear_proj = nn.Linear(pos_dim, hidden_size)
        self.feat_linear_proj = nn.Linear(feat_dim, hidden_size)
        self.res_block = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(hidden_size, hidden_size)),
            ('relu', nn.ReLU(inplace=False)),
            ('linear', nn.Linear(hidden_size, hidden_size)),
        ]))
        self.relu = nn.ReLU(inplace=False)
        self.final = nn.Linear(hidden_size, out_pos_dim)

    def forward(self, pos, feat):
        pos_proj = self.pos_linear_proj(pos)
        feat_proj = self.feat_linear_proj(feat)
        # residual blocks
        input = pos_proj + feat_proj
        out = self.relu(input + self.res_block(input))
        pos_offset = self.final(out)
        return pos_offset


class UnODecoder(nn.Module):
    def __init__(self, pos_dim, feat_dim, hidden_size, num_cls, **kwargs):
        super().__init__()
        self.pos_linear_proj = nn.Linear(pos_dim, hidden_size)
        self.feat_linear_proj = nn.Linear(feat_dim, hidden_size)
        self.res_block_1 = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(hidden_size, hidden_size)),
            ('relu', nn.ReLU(inplace=False)),
            ('linear', nn.Linear(hidden_size, hidden_size)),
        ]))
        self.res_block_2 = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(hidden_size, hidden_size)),
            ('relu', nn.ReLU(inplace=False)),
            ('linear', nn.Linear(hidden_size, hidden_size)),
        ]))
        self.res_block_3 = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(hidden_size, hidden_size)),
            ('relu', nn.ReLU(inplace=False)),
            ('linear', nn.Linear(hidden_size, hidden_size)),
        ]))
        self.relu = nn.ReLU(inplace=False)
        self.final = nn.Sequential(OrderedDict([
            ('final', nn.Linear(hidden_size, num_cls)),
            ('softmax', nn.Softmax(dim=1)),
        ]))

    def forward(self, pos, feat):
        pos_proj = self.pos_linear_proj(pos)
        feat_proj = self.feat_linear_proj(feat)
        # residual blocks
        out_1 = self.relu(pos_proj + self.res_block_1(pos_proj + feat_proj))
        out_2 = self.relu(out_1 + self.res_block_2(out_1 + feat_proj))
        out_3 = self.relu(out_2 + self.res_block_3(out_2 + feat_proj))
        occ_probs = self.final(out_3)
        return occ_probs

