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
from utils.metrics import ClassificationMetrics
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from models.backbone import MinkUNetBackbone


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
        self.xy_offset = torch.nn.parameter.Parameter(torch.Tensor([self.cfg_model["scene_bbox"][0],
                                                                    self.cfg_model["scene_bbox"][1]])[None:],
                                                      requires_grad=False)
        self.offset_predictor = OffsetPredictor(self.pos_dim, self.feat_dim, self.hidden_size, 2)  # TODO: only output x and y pos
        self.decoder = UnODecoder(self.pos_dim, self.feat_dim * 2, self.hidden_size, self.n_uno_cls)

        # metrics
        self.ClassificationMetrics = ClassificationMetrics(self.n_uno_cls, ignore_index=[])

        # pytorch lightning training output
        self.training_step_outputs = []

        # save predictions
        if not train_flag:
            self.model_dir = kwargs['model_dir']
            self.test_epoch = kwargs['test_epoch']
            self.pred_dir = os.path.join(self.model_dir, "predictions", f"epoch_{self.test_epoch}")
            os.makedirs(self.pred_dir, exist_ok=True)

    def forward(self, batch):
        # unfold batch: [(ref_sd_tok, uno_sd_toks), pcds_4d, (uno_pts_4d, uno_labels)]
        meta_batch, pcds_batch, uno_samples_batch = batch

        # encoder
        dense_featmap = self.encoder(pcds_batch)

        # uno query points and occupancy labels
        uno_points_batch = []
        uno_labels_batch = []
        for uno_samples in uno_samples_batch:
            uno_pts_4d = uno_samples[0]  # x, y, z, ts
            uno_labels = uno_samples[1]
            uno_points_batch.append(uno_pts_4d)
            uno_labels_batch.append(uno_labels)
        uno_pts_4d = torch.stack(uno_points_batch)
        uno_labels = torch.stack(uno_labels_batch)

        # bilinear feature interpolation
        uno_pts_yx = torch.stack([uno_pts_4d[:, :, 1], uno_pts_4d[:, :, 0]], dim=2)
        pts_min = self.cfg_model["scene_bbox"][0]
        pts_max = self.cfg_model["scene_bbox"][3]
        query_pts = ((uno_pts_yx - pts_min) / (pts_max - pts_min) * 2 - 1).unsqueeze(-2)  # normalize to [-1, 1]
        query_pts = torch.clip(query_pts, min=-1, max=1)
        feats = F.grid_sample(input=dense_featmap, grid=query_pts, mode='bilinear', padding_mode='zeros',
                              align_corners=False).squeeze(-1)
        feats = torch.permute(feats, (0, 2, 1)).contiguous()

        # offset prediction
        pos_offset_xy = self.offset_predictor(uno_pts_4d, feats)
        pos_offset_yx = torch.stack([pos_offset_xy[:, :, 1], pos_offset_xy[:, :, 0]], dim=2)

        # offset features interpolation
        query_offset_pts = ((uno_pts_yx + pos_offset_yx - pts_min) / (pts_max - pts_min) * 2 - 1).unsqueeze(-2)  # normalize to [-1, 1]
        query_offset_pts = torch.clip(query_offset_pts, min=-1, max=1)
        offset_feats = F.grid_sample(input=dense_featmap, grid=query_offset_pts, mode='bilinear', padding_mode='zeros',
                                     align_corners=False).squeeze(-1)
        offset_feats = torch.permute(offset_feats, (0, 2, 1)).contiguous()

        # aggregated feats
        uno_feats = torch.cat([feats, offset_feats], dim=2)

        # decoder
        uno_probs = self.decoder(uno_pts_4d, uno_feats)
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

    def get_loss(self, uno_probs, uno_labels):
        uno_labels = uno_labels.long()
        assert len(uno_labels) == len(uno_probs)
        log_prob = torch.log(uno_probs.clamp(min=1e-8))
        loss_func = nn.NLLLoss()
        loss = loss_func(log_prob, uno_labels)  # dtype of torch.nllloss must be torch.long
        return loss

    def training_step(self, batch: tuple, batch_idx, dataloader_index=0):
        # model_dict = self.state_dict()  # check state dict
        uno_probs_batch, uno_labels_batch = self.forward(batch)  # encoder & decoder
        uno_probs = uno_probs_batch.view(-1, self.n_uno_cls)
        uno_labels = uno_labels_batch.view(-1)
        pred_labels = torch.argmax(uno_probs, axis=1)
        loss = self.get_loss(uno_probs, uno_labels)

        # metrics
        conf_mat = self.ClassificationMetrics.compute_conf_mat(pred_labels, uno_labels)
        iou = self.ClassificationMetrics.get_iou(conf_mat)
        free_iou, occ_iou = iou[0].item() * 100, iou[1].item() * 100
        acc = self.ClassificationMetrics.get_acc(conf_mat)
        free_acc, occ_acc = acc[0].item() * 100, acc[1].item() * 100

        # logging
        self.log("loss", loss.item(), on_step=True, prog_bar=True, logger=True)
        self.log("free_iou", free_iou, on_step=True, prog_bar=True, logger=True)
        self.log("occ_iou", occ_iou, on_step=True, prog_bar=True, logger=True)
        self.log("free_acc", free_acc, on_step=True, prog_bar=True, logger=True)
        self.log("occ_acc", occ_acc, on_step=True, prog_bar=True, logger=True)
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
        free_iou, occ_iou = iou[0].item() * 100, iou[1].item() * 100
        acc = self.ClassificationMetrics.get_acc(acc_conf_mat)
        free_acc, occ_acc = acc[0].item() * 100, acc[1].item() * 100
        self.log("epoch_free_iou", free_iou, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_occ_iou", occ_iou, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_free_acc", free_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_occ_acc", occ_acc, on_epoch=True, prog_bar=True, logger=True)

        # clean
        self.training_step_outputs = []
        torch.cuda.empty_cache()

    def predict_step(self, batch: tuple, batch_idx):
        # model_dict = self.state_dict()  # check state dict
        a = -1


#######################################
# Modules
#######################################


class DenseFeatHead(nn.Module):
    def __init__(self, in_channels, out_channels, featmap_shape, D):
        super().__init__()
        # extra conv for feature map size correction
        self.conv0k2s2 = ME.MinkowskiConvolution(in_channels, out_channels,
                                                 kernel_size=2, stride=[2, 2, 2, 1], dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(out_channels)
        self.conv1k2s2 = ME.MinkowskiConvolution(out_channels, out_channels,
                                                 kernel_size=2, stride=[2, 2, 2, 1], dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(out_channels)

        # pooling block (for z and t dimension)
        self.avg_pool_t = ME.MinkowskiAvgPooling(kernel_size=[1, 1, 1, featmap_shape[5]],
                                                 stride=[1, 1, 1, featmap_shape[5]], dimension=D)
        self.avg_pool_z = ME.MinkowskiAvgPooling(kernel_size=[1, 1, featmap_shape[4], 1],
                                                 stride=[1, 1, featmap_shape[4], 1], dimension=D)

        # to dense, dense featmap shape
        self.dense_featmap_shape = [featmap_shape[0], featmap_shape[1], featmap_shape[2], featmap_shape[3], 1, 1]

        # dense block, for denser feature map TODO: output feature map size o = (i - k) + 2p + 1
        self.dense_conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3, stride=1)),
            ('relu', nn.ReLU(inplace=False)),
            ('conv1', nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3, stride=1)),
            ('relu', nn.ReLU(inplace=False)),
            ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3, stride=1)),
            ('relu', nn.ReLU(inplace=False)),
        ]))

    def forward(self, sparse_input):
        # extra conv layer (feature map size correction)
        sparse_out = self.conv0k2s2(sparse_input)
        sparse_out = self.bn0(sparse_out)
        sparse_out = self.conv1k2s2(sparse_out)
        sparse_out = self.bn1(sparse_out)

        # avg pooling (for z and t dimension)
        sparse_out_xyz = self.avg_pool_t(sparse_out)
        sparse_out_xy = self.avg_pool_z(sparse_out_xyz)

        # sparse to dense
        dense_featmap_xy, _, _ = sparse_out_xy.dense(shape=torch.Size(self.dense_featmap_shape),
                                                     min_coordinate=torch.IntTensor([0, 0, 0, 0]))  # [B, F, 350, 350, 1, 1]
        dense_featmap_xy = torch.squeeze(torch.squeeze(dense_featmap_xy, -1), -1)  # B, F, X, Y

        # dense conv layers (increase receptive field)
        dense_featmap_xy = self.dense_conv_block(dense_featmap_xy)
        # dense_featmap_np = dense_featmap_xy.detach().cpu().numpy()[0]
        return dense_featmap_xy


class MotionEncoder(nn.Module):
    def __init__(self, cfg_model: dict, n_classes: int):
        super().__init__()
        # quantization offset
        self.scene_bbox = cfg_model["scene_bbox"]
        self.offset = torch.nn.parameter.Parameter(
            torch.Tensor([self.scene_bbox[0], self.scene_bbox[1], self.scene_bbox[2], -cfg_model['n_input'] + 1])[None:],
            requires_grad=False)

        # features params
        self.feat_dim = cfg_model["feat_dim"]
        z_height = int((self.scene_bbox[5] - self.scene_bbox[2]) / cfg_model["featmap_size"])
        y_length = int((self.scene_bbox[4] - self.scene_bbox[1]) / cfg_model["featmap_size"])
        x_width = int((self.scene_bbox[3] - self.scene_bbox[0]) / cfg_model["featmap_size"])
        b_size = cfg_model["batch_size"]
        t_size = cfg_model['n_input']
        self.featmap_shape = [b_size, self.feat_dim, x_width, y_length, z_height, t_size]
        self.dense_featmap_shape = [b_size, self.feat_dim, x_width, y_length, 1, 1]  # TODO: squeeze z and t dimension

        # backbone network
        self.MinkUNet = MinkUNetBackbone(in_channels=1, out_channels=self.feat_dim, D=cfg_model['pos_dim'])
        self.DenseFeatHead = DenseFeatHead(in_channels=self.feat_dim, out_channels=self.feat_dim,
                                           featmap_shape=self.featmap_shape, D=cfg_model['pos_dim'])

        # input point cloud quantization
        self.quant_size_s = cfg_model["quant_size"]
        self.quant_size_t = 1  # TODO: cfg_model["time_interval"]
        self.quant = torch.Tensor([self.quant_size_s, self.quant_size_s, self.quant_size_s, self.quant_size_t])


    def forward(self, pcds_4d_batch):
        # quantized 4d pcd and initialized features
        self.quant = self.quant.type_as(pcds_4d_batch[0])
        quant_4d_pcds = [torch.div(pcd - self.offset, self.quant, rounding_mode=None) for pcd in pcds_4d_batch]
        feats = [0.5 * torch.ones(len(pcd), 1).type_as(pcd) for pcd in pcds_4d_batch]

        # sparse collate, tensor field, net calculation
        coords, feats = ME.utils.sparse_collate(quant_4d_pcds, feats)
        sparse_input = ME.SparseTensor(features=feats, coordinates=coords,
                                      quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

        # TODO: filter quantized input which outside quantized scene bbox ##############################################
        quant_coords = sparse_input.coordinates
        quant_feats = sparse_input.features

        # filter index out of bound
        x_max = torch.max(quant_coords[:, 1])
        y_max = torch.max(quant_coords[:, 2])
        z_max = torch.max(quant_coords[:, 3])
        quant_x_max = int((self.scene_bbox[3] - self.scene_bbox[0]) / self.quant_size_s)
        quant_y_max = int((self.scene_bbox[4] - self.scene_bbox[1]) / self.quant_size_s)
        quant_z_max = int((self.scene_bbox[5] - self.scene_bbox[2]) / self.quant_size_s)
        if x_max >= quant_x_max or y_max >= quant_y_max or z_max >= quant_z_max:
            print(f"\nin max coords out of bound, filter and generate new sparse tensor: {x_max}, {y_max}, {z_max}")
            quant_valid_mask = torch.logical_and(quant_coords[:, 1] >= 0, quant_coords[:, 1] < quant_x_max)
            quant_valid_mask = torch.logical_and(quant_valid_mask,
                               torch.logical_and(quant_coords[:, 2] >= 0, quant_coords[:, 2] < quant_y_max))
            quant_valid_mask = torch.logical_and(quant_valid_mask,
                               torch.logical_and(quant_coords[:, 3] >= 0, quant_coords[:, 3] < quant_z_max))
            # update valid sparse input
            sparse_input = ME.SparseTensor(features=quant_feats[quant_valid_mask].reshape(-1, 1),
                                           coordinates=quant_coords[quant_valid_mask],
                                           quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        ################################################################################################################

        # forward
        sparse_output = self.MinkUNet(sparse_input)

        # to dense featmap
        dense_featmap = self.DenseFeatHead(sparse_output)
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
        input = pos_proj + feat_proj
        out_1 = self.relu(input + self.res_block_1(input))
        out_2 = self.relu(out_1 + self.res_block_2(out_1 + feat_proj))
        out_3 = self.relu(out_2 + self.res_block_3(out_2 + feat_proj))
        uno_probs = self.final(out_3)
        return uno_probs

