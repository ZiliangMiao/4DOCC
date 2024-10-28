import logging
import os
from collections import OrderedDict
from random import sample as random_sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from lib.minkowski.resnet import ResNetBase
from pytorch_lightning import LightningModule
import MinkowskiEngine as ME
from models.backbone import MinkUNetBackbone
from utils.metrics import ClassificationMetrics
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


#######################################
# Lightning Module
#######################################


class MutualObsPretrainNetwork(LightningModule):
    def __init__(self, cfg_model: dict, train_flag: bool, **kwargs):
        super().__init__()
        if train_flag:
            self.save_hyperparameters(cfg_model)
        self.cfg_model = cfg_model
        self.n_mutual_cls = self.cfg_model['num_cls']
        if train_flag:
            self.iters_per_epoch = kwargs['iters_per_epoch']

        # encoder and decoder
        self.encoder = MotionEncoder(self.cfg_model)
        self.pe = PositionalEncoding(feat_dim=self.cfg_model["feat_dim"], pos_dim=self.cfg_model['pos_dim'])
        self.decoder = BackgroundFieldMLP(in_dim=self.cfg_model["feat_dim"] * 2, planes=[256, 256, 128, 64, 32, 16, self.n_mutual_cls])

        # metrics
        self.ClassificationMetrics = ClassificationMetrics(self.n_mutual_cls, ignore_index=[])

        # pytorch lightning training output
        self.epoch_mo_conf_mat = torch.zeros(self.n_mutual_cls, self.n_mutual_cls)
        self.epoch_co_conf_mat = torch.zeros(self.n_mutual_cls, self.n_mutual_cls)

        # save predictions
        if not train_flag:
            self.model_dir = kwargs['model_dir']
            self.test_epoch = kwargs['test_epoch']

            # logger
            self.test_logger = kwargs['test_logger']

            # save pred labels
            self.pred_dir = os.path.join(self.model_dir, "predictions", f"epoch_{self.test_epoch}")
            os.makedirs(self.pred_dir, exist_ok=True)
            self.save_pred_labels = kwargs['save_pred_labels']

            # metrics
            self.accumulated_conf_mat = torch.zeros(self.n_mutual_cls, self.n_mutual_cls)  # to calculate point-level avg. iou

    def forward(self, batch):
        # mo_rays_idx, mo_pts_4d, mo_labels, mo_confidence, co_rays_idx, co_pts_4d, co_labels, co_confidence
        meta_batch, pcds_batch, samples_batch = batch

        # encoder
        sparse_featmap = self.encoder(pcds_batch)

        # get mutual observation and current observation samples
        mo_feats_batch, mo_labels_batch, mo_confidence_batch = [], [], []
        co_feats_batch, co_labels_batch, co_confidence_batch = [], [], []
        for batch_idx, samples in enumerate(samples_batch):
            # get point-wise features
            coords = sparse_featmap.coordinates_at(batch_index=batch_idx)
            feats = sparse_featmap.features_at(batch_index=batch_idx)
            feats = feats[coords[:, -1] == 0]

            # unfold samples
            if self.cfg_model['train_co_samples']:
                mo_rays_idx, mo_pts_4d, mo_labels, mo_confidence, co_rays_idx, co_pts_4d, co_labels, co_confidence = samples
            else:
                mo_rays_idx, mo_pts_4d, mo_labels, mo_confidence = samples

            # mutual observation feats and labels
            mo_feats = feats[mo_rays_idx]
            mo_pe_feats = self.pe(mo_pts_4d)
            mo_feats = torch.cat((mo_feats, mo_pe_feats), dim=1)
            mo_feats_batch.append(mo_feats)
            mo_labels_batch.append(mo_labels)
            mo_confidence_batch.append(mo_confidence)

            # current observation feats and labels
            if self.cfg_model['train_co_samples']:
                co_feats = feats[co_rays_idx]
                co_pe_feats = self.pe(co_pts_4d)
                co_feats = torch.cat((co_feats, co_pe_feats), dim=1)
                co_feats_batch.append(co_feats)
                co_labels_batch.append(co_labels)
                co_confidence_batch.append(co_confidence)

        # decoder TODO: bug
        mo_probs_batch = self.decoder(mo_feats_batch)  # logits -> softmax -> probs
        if self.cfg_model['train_co_samples']:
            # decoder
            co_probs_batch = self.decoder(co_feats_batch)
            return mo_probs_batch, mo_labels_batch, mo_confidence_batch, co_probs_batch, co_labels_batch, co_confidence_batch
        return mo_probs_batch, mo_labels_batch, mo_confidence_batch

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
        # return [optimizer], [scheduler]  # default scheduler interval is 'epoch'

    def get_loss(self, probs, labels, confidence):
        labels = labels.long()
        assert len(labels) == len(probs)
        log_prob = torch.log(probs.clamp(min=1e-8))
        loss_func = nn.NLLLoss(reduction='none')
        loss = loss_func(log_prob, labels)  # dtype of torch.nllloss must be torch.long
        loss = torch.mean(loss * confidence)
        return loss

    def training_step(self, batch: tuple, batch_idx, dataloader_index=0):
        if self.cfg_model['train_co_samples']:
            (mo_probs_batch, mo_labels_batch, mo_confidence_batch,
             co_probs_batch, co_labels_batch, co_confidence_batch) = self.forward(batch)
        else:
            mo_probs_batch, mo_labels_batch, mo_confidence_batch = self.forward(batch)

        # mutual observation prediction loss
        mo_probs = torch.cat(mo_probs_batch, dim=0)
        mo_labels = torch.cat(mo_labels_batch, dim=0)
        mo_pred_labels = torch.argmax(mo_probs, axis=1)
        mo_confidence = torch.cat(mo_confidence_batch, dim=0)
        loss = self.get_loss(mo_probs, mo_labels, mo_confidence)

        # metrics
        mo_conf_mat = self.ClassificationMetrics.compute_conf_mat(mo_pred_labels, mo_labels)
        self.epoch_mo_conf_mat += mo_conf_mat
        mo_iou = self.ClassificationMetrics.get_iou(mo_conf_mat)
        mo_unk_iou, mo_free_iou, mo_occ_iou = mo_iou[0], mo_iou[1], mo_iou[2]
        # mo_acc = self.ClassificationMetrics.get_acc(conf_mat)
        # mo_unk_acc, mo_free_acc, mo_occ_acc = mo_acc[0], mo_acc[1], mo_acc[2]

        # current observation occupancy prediction loss
        if self.cfg_model['train_co_samples']:
            co_probs = torch.cat(co_probs_batch, dim=0)
            co_labels = torch.cat(co_labels_batch, dim=0)
            co_pred_labels = torch.argmax(co_probs, axis=1)
            co_confidence = torch.cat(co_confidence_batch, dim=0)
            co_loss = self.get_loss(co_probs, co_labels, co_confidence)

            # loss
            loss = loss + co_loss

            # metrics
            co_conf_mat = self.ClassificationMetrics.compute_conf_mat(co_pred_labels, co_labels)
            self.epoch_co_conf_mat += co_conf_mat
            co_iou = self.ClassificationMetrics.get_iou(co_conf_mat)
            co_free_iou, co_occ_iou = co_iou[1], co_iou[2]

        # logging
        self.log("loss", loss.item(), on_step=True, prog_bar=True, logger=True)
        self.log("mo_unk_iou", mo_unk_iou.item() * 100, on_step=True, prog_bar=True, logger=True)
        self.log("mo_free_iou", mo_free_iou.item() * 100, on_step=True, prog_bar=True, logger=True)
        self.log("mo_occ_iou", mo_occ_iou.item() * 100, on_step=True, prog_bar=True, logger=True)
        if self.cfg_model['train_co_samples']:
            self.log("co_free_iou", co_free_iou.item() * 100, on_step=True, prog_bar=True, logger=True)
            self.log("co_occ_iou", co_occ_iou.item() * 100, on_step=True, prog_bar=True, logger=True)

        # clean cuda memory
        torch.cuda.empty_cache()
        return loss

    def on_train_epoch_end(self):
        # metrics in one epoch
        mo_iou = self.ClassificationMetrics.get_iou(self.epoch_mo_conf_mat)
        mo_unk_iou, mo_free_iou, mo_occ_iou = mo_iou[0], mo_iou[1], mo_iou[2]
        co_iou = self.ClassificationMetrics.get_iou(self.epoch_co_conf_mat)
        co_free_iou, co_occ_iou = co_iou[1], co_iou[2]

        # log
        self.log("epoch_mo_unk_iou", mo_unk_iou.item() * 100, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_mo_free_iou", mo_free_iou.item() * 100, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_mo_occ_iou", mo_occ_iou.item() * 100, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_co_free_iou", co_free_iou.item() * 100, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_co_occ_iou", co_occ_iou.item() * 100, on_epoch=True, prog_bar=True, logger=True)

        # clean cuda memory
        self.epoch_mo_conf_mat = self.epoch_mo_conf_mat.zero_()
        self.epoch_co_conf_mat = self.epoch_co_conf_mat.zero_()
        torch.cuda.empty_cache()


    def predict_step(self, batch: tuple, batch_idx, dataloader_idx=0):
        if self.pred_occ:
            # TODO: predict occupancy states of the current ray
            # model_dict = self.state_dict()  # check state dict
            meta_batch, _, _ = batch
            occ_probs_batch, occ_labels_batch, _ = self.forward(batch)  # encoder & decoder

            # iterate each batch data for predicted label saving
            for meta, occ_probs, occ_labels in zip(meta_batch, occ_probs_batch, occ_labels_batch):
                sd_tok = meta[0]
                pred_labels = torch.argmax(occ_probs, axis=1)

                # metrics
                conf_mat = self.ClassificationMetrics.compute_conf_mat(pred_labels, occ_labels)
                iou = self.ClassificationMetrics.get_iou(conf_mat)
                unk_iou, free_iou, occ_iou = iou[0].item() * 100, iou[1].item() * 100, iou[2].item() * 100
                acc = self.ClassificationMetrics.get_acc(conf_mat)
                unk_acc, free_acc, occ_acc = acc[0].item() * 100, acc[1].item() * 100, acc[2].item() * 100

                # update confusion matrix
                self.accumulated_conf_mat += conf_mat

                if self.save_pred_labels:
                    # save predicted labels for visualization
                    pred_file = os.path.join(self.pred_dir, f"{sd_tok}_occ_pred.label")
                    pred_labels = pred_labels.type(torch.uint8).detach().cpu().numpy()
                    pred_labels.tofile(pred_file)

                # logger
                self.test_logger.info(
                    "Val sample data (IoU/Acc): %s, [Unk %.3f/%.3f], [Free %.3f/%.3f], [Occ %.3f/%.3f]",
                    sd_tok, unk_iou, unk_acc, free_iou, free_acc, occ_iou, occ_acc)
        torch.cuda.empty_cache()

        if self.pred_mop:
            # TODO: predict unknown occupancy states with mutual observations
            # model_dict = self.state_dict()  # check state dict
            meta_batch, _, _ = batch
            mutual_probs_batch, mutual_labels_batch, _ = self.forward(batch)  # encoder & decoder

            # iterate each batch data for predicted label saving
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
                self.accumulated_conf_mat += conf_mat

                if self.save_pred_labels:
                    # save predicted labels for visualization
                    pred_file = os.path.join(self.pred_dir, f"{sd_tok}_moco_pred.label")
                    pred_labels = pred_labels.type(torch.uint8).detach().cpu().numpy()
                    pred_labels.tofile(pred_file)

                # logger
                self.test_logger.info("Val sample data (IoU/Acc): %s, [Unk %.3f/%.3f], [Free %.3f/%.3f], [Occ %.3f/%.3f]",
                             sd_tok, unk_iou, unk_acc, free_iou, free_acc, occ_iou, occ_acc)
        torch.cuda.empty_cache()


#######################################
# Modules
#######################################


class MotionEncoder(nn.Module):
    def __init__(self, cfg_model: dict):
        super().__init__()
        # backbone network
        self.feat_dim = cfg_model["feat_dim"]
        self.MinkUNet = MinkUNetBackbone(in_channels=1, out_channels=self.feat_dim, D=cfg_model['pos_dim'])  # D: UNet spatial dim

        # input point cloud quantization
        self.quant_size_s = cfg_model["quant_size"]
        self.quant_size_t = 1  # TODO: cfg_model["time_interval"]
        self.quant = torch.Tensor([self.quant_size_s, self.quant_size_s, self.quant_size_s, self.quant_size_t])


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

        # slice to copy voxel features to original points
        sparse_featmap = sparse_output.slice(tensor_field)
        sparse_featmap.coordinates[:, 1:] = torch.mul(sparse_featmap.coordinates[:, 1:], self.quant)
        return sparse_featmap


class PositionalEncoding(nn.Module):
    def __init__(self, feat_dim, pos_dim):
        # d_model是每个词embedding后的维度
        super(PositionalEncoding, self).__init__()
        self.feat_dim = feat_dim
        self.pos_dim = pos_dim
        # self.dropout = nn.Dropout(p=dropout)
        # self.register_buffer('pe', pe)  # will not train this parameter -> use self.pe to call

    def forward(self, points_4d):
        pe_xyzt = []
        for i in range(self.pos_dim):
            assert self.feat_dim % self.pos_dim == 0
            i_dim = int(self.feat_dim / self.pos_dim)
            assert i_dim % 2 == 0, "dimension should be even"
            pe_i = torch.zeros(len(points_4d), i_dim).cuda()
            position = points_4d[:, i].unsqueeze(-1)
            div_term = torch.exp(torch.arange(0, i_dim, 2).float() * (-math.log(10000.0) / i_dim)).cuda()
            # 高级切片方式，即从0开始，两个步长取一个。即奇数和偶数位置赋值不一样
            pe_i[:, 0::2] = torch.sin(position * div_term)
            pe_i[:, 1::2] = torch.cos(position * div_term)
            pe_xyzt.append(pe_i)
            # self.dropout(pos_4d)
        return torch.hstack(pe_xyzt)


class BackgroundFieldMLP(nn.Module):
    def __init__(self, in_dim, planes, **kwargs):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('linear0', nn.Linear(in_dim, planes[0])),
            # ('bn0', nn.BatchNorm1d(planes[0])),
            ('relu0', nn.ReLU(inplace=False)),
            ('linear1', nn.Linear(planes[0], planes[1])),
            # ('bn1', nn.BatchNorm1d(planes[1])),
            ('relu1', nn.ReLU(inplace=False)),
            ('linear2', nn.Linear(planes[1], planes[2])),
            # ('bn2', nn.BatchNorm1d(planes[2])),
            ('relu2', nn.ReLU(inplace=False)),
            ('linear3', nn.Linear(planes[2], planes[3])),
            # ('bn3', nn.BatchNorm1d(planes[3])),
            ('relu3', nn.ReLU(inplace=False)),
            ('linear4', nn.Linear(planes[3], planes[4])),
            # ('bn4', nn.BatchNorm1d(planes[4])),
            ('relu4', nn.ReLU(inplace=False)),
            ('linear5', nn.Linear(planes[4], planes[5])),
            # ('bn5', nn.BatchNorm1d(planes[5])),
            ('relu5', nn.ReLU(inplace=False)),
            ('final', nn.Linear(planes[5], planes[6])),
            ('softmax', nn.Softmax(dim=1)),
        ]))

    def forward(self, x):
        if type(x) is list:  # input batch data, not concatenated data
            probs_list = []
            for x_i in x:
                probs_list.append(self.block(x_i))
            return probs_list
        else:
            return self.block(x)

