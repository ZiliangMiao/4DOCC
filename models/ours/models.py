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


class MotionPretrainNetwork(LightningModule):
    def __init__(self, cfg_model: dict, train_flag: bool, **kwargs):
        super().__init__()
        if train_flag:
            self.save_hyperparameters(cfg_model)
        self.cfg_model = cfg_model
        self.n_mutual_cls = self.cfg_model['num_cls']
        if train_flag:
            self.iters_per_epoch = kwargs['iters_per_epoch']

        # encoder and decoder
        self.encoder = MotionEncoder(self.cfg_model, self.n_mutual_cls)
        self.pe = PositionalEncoding(feat_dim=self.cfg_model["feat_dim"], pos_dim=self.cfg_model['pos_dim'])
        self.decoder = BackgroundFieldMLP(in_dim=self.cfg_model["feat_dim"], planes=[256, 128, 64, 32, 16, self.n_mutual_cls])

        # metrics
        self.ClassificationMetrics = ClassificationMetrics(self.n_mutual_cls, ignore_index=[])

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
        # unfold batch: [(ref_sd_tok, mutual_sd_toks), pcds_4d, (mutual_obs_rays_idx, mutual_obs_pts, mutual_obs_depth, mutual_obs_ts, mutual_obs_labels, mutual_obs_confidence)]
        meta_batch, pcds_batch, mutual_samples_batch = batch

        # encoder
        sparse_featmap = self.encoder(pcds_batch)

        # get bg samples
        mutual_labels_batch = []
        mutual_feats_batch = []
        mutual_confidence_batch = []
        for batch_idx, mutual_samples in enumerate(mutual_samples_batch):
            # ref_sd_tok = meta_batch[batch_idx][0]
            # mutual_sd_toks = meta_batch[batch_idx][1]
            mutual_obs_rays_idx = mutual_samples[0]
            mutual_obs_pts = mutual_samples[1]
            mutual_obs_ts = mutual_samples[2]
            mutual_obs_labels = mutual_samples[3]
            mutual_obs_confidence = mutual_samples[4]

            # point-wise features
            coords = sparse_featmap.coordinates_at(batch_index=batch_idx)
            feats = sparse_featmap.features_at(batch_index=batch_idx)
            ref_time_mask = coords[:, -1] == 0
            # coords = coords[ref_time_mask]
            query_feats = feats[ref_time_mask][mutual_obs_rays_idx]

            # mutual obs points positional encoding
            mutual_pts_4d = torch.hstack((mutual_obs_pts, mutual_obs_ts.reshape(-1, 1)))
            pe_feats = self.pe(mutual_pts_4d)
            mutual_obs_feats = query_feats + pe_feats

            # collate to batch
            mutual_feats_batch.append(mutual_obs_feats)
            mutual_labels_batch.append(mutual_obs_labels)
            mutual_confidence_batch.append(mutual_obs_confidence)
        # decoder
        mutual_probs_batch = self.decoder(mutual_feats_batch)  # logits -> softmax -> probs
        return mutual_probs_batch, mutual_labels_batch, mutual_confidence_batch

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
        acc_conf_mat = torch.zeros(self.n_mutual_cls, self.n_mutual_cls)
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
        acc_conf_mat = torch.zeros(self.n_mutual_cls, self.n_mutual_cls)
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
        z_height = int((self.scene_bbox[5] - self.scene_bbox[2]) / featmap_size)
        y_length = int((self.scene_bbox[4] - self.scene_bbox[1]) / featmap_size)
        x_width = int((self.scene_bbox[3] - self.scene_bbox[0]) / featmap_size)
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

        # TODO: point-wise sparse feature output
        sparse_featmap = sparse_output.slice(tensor_field)
        sparse_featmap.coordinates[:, 1:] = torch.mul(sparse_featmap.coordinates[:, 1:], self.quant)

        # TODO: dense feature map output (with interpolation)
        # featmap_shape = torch.Size([self.featmap_shape[0], 1, self.featmap_shape[1], self.featmap_shape[2], self.featmap_shape[3], self.featmap_shape[4]])
        # dense_featmap, _, _ = sparse_output.dense(shape=featmap_shape, min_coordinate=torch.IntTensor([0, 0, 0, 0]))
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
            ('relu0', nn.ReLU(inplace=False)),
            ('linear1', nn.Linear(planes[0], planes[1])),
            ('relu1', nn.ReLU(inplace=False)),
            ('linear2', nn.Linear(planes[1], planes[2])),
            ('relu2', nn.ReLU(inplace=False)),
            ('linear3', nn.Linear(planes[2], planes[3])),
            ('relu3', nn.ReLU(inplace=False)),
            ('linear4', nn.Linear(planes[3], planes[4])),
            ('relu4', nn.ReLU(inplace=False)),
            ('final', nn.Linear(planes[4], planes[5])),
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

