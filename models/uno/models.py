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
from lib.minkowski.resnet import ResNetBase
from utils.metrics import ClassificationMetrics
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
        self.xy_offset = torch.nn.parameter.Parameter(torch.Tensor([self.cfg_model["scene_bbox"][0],
                                                            self.cfg_model["scene_bbox"][1]])[None:],
                                                      requires_grad=False)
        self.featmap_size = self.cfg_model['featmap_size']
        self.offset_predictor = OffsetPredictor(self.pos_dim, self.feat_dim, self.hidden_size, self.pos_dim)
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
        query_pts = torch.clip(uno_pts_yx / self.cfg_model["scene_bbox"][3], min=-1, max=1).unsqueeze(-2)
        # x_max = torch.max(query_pts[:, :, 0])
        # y_max = torch.max(query_pts[:, :, 1])
        # query_pts = query_pts.unsqueeze(-2)
        feats = F.grid_sample(input=dense_featmap, grid=query_pts, mode='bilinear', padding_mode='zeros', align_corners=False)
        feats = torch.permute(feats.squeeze(-1), (0, 2, 1))

        # offset prediction
        pos_offset_4d = self.offset_predictor(uno_pts_4d, feats)

        # offset features interpolation
        pos_offset_yx = torch.stack([pos_offset_4d[:, :, 1], pos_offset_4d[:, :, 0]], dim=2)
        query_offset_pts = torch.clip((uno_pts_yx + pos_offset_yx) / self.cfg_model["scene_bbox"][3], min=-1, max=1).unsqueeze(-2)
        # xx_max = torch.max(query_offset_pts[:, :, 0])
        # yy_max = torch.max(query_offset_pts[:, :, 1])
        # query_offset_pts = query_offset_pts.unsqueeze(-2)
        offset_feats = F.grid_sample(input=dense_featmap, grid=query_offset_pts, mode='bilinear', padding_mode='zeros', align_corners=False)
        offset_feats = torch.permute(offset_feats.squeeze(-1), (0, 2, 1))

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


class MinkUNetUno(ResNetBase):
    BLOCK = BasicBlock
    # PLANES = (8, 16, 32, 64, 64, 32, 16, 8)
    PLANES = (8, 32, 128, 256, 256, 128, 32, 8)
    EXTRA_PLANES = (8, 8)
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)
    INIT_DIM = 8
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        # TODO: extra conv for correct feature map size
        self.extra_conv0 = ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion, self.EXTRA_PLANES[0], kernel_size=2, stride=[2, 2, 2, 1], dimension=D)
        self.extra_bn0 = ME.MinkowskiBatchNorm(self.inplanes)
        self.extra_conv1 = ME.MinkowskiConvolution(self.EXTRA_PLANES[0], self.EXTRA_PLANES[1], kernel_size=2, stride=[2, 2, 2, 1], dimension=D)
        self.extra_bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.pooling = ME.MinkowskiAvgPooling(kernel_size=[5, 5, 5, 5], stride=[2, 2, 2, 2], dimension=D)

        # final layer
        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def get_max_coords(self, sparse_output):
        feat_coords = torch.cat(sparse_output.decomposed_coordinates, dim=0)
        x_quant_max = torch.max(feat_coords[:, 0])
        y_quant_max = torch.max(feat_coords[:, 1])
        z_quant_max = torch.max(feat_coords[:, 2])
        t_quant_max = torch.max(feat_coords[:, 3])
        return x_quant_max, y_quant_max, z_quant_max, t_quant_max

    def forward(self, x):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        x_quant_max, y_quant_max, z_quant_max, t_quant_max = self.get_max_coords(out_p1)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        x_quant_max, y_quant_max, z_quant_max, t_quant_max = self.get_max_coords(out_b1p2)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        x_quant_max, y_quant_max, z_quant_max, t_quant_max = self.get_max_coords(out_b2p4)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        x_quant_max, y_quant_max, z_quant_max, t_quant_max = self.get_max_coords(out_b3p8)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        x_quant_max, y_quant_max, z_quant_max, t_quant_max = self.get_max_coords(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        out = self.block8(out)

        # # extra conv
        # out = self.extra_conv0(out)
        # out = self.extra_bn0(out)
        # out = self.relu(out)
        # out = self.extra_conv1(out)
        # out = self.extra_bn1(out)
        # out = self.relu(out)

        # pooling
        out = self.pooling(out)
        out = self.pooling(out)

        return self.final(out)


class MotionEncoder(nn.Module):
    def __init__(self, cfg_model: dict, n_classes: int):
        super().__init__()
        # backbone network
        self.feat_dim = cfg_model["feat_dim"]
        self.MinkUNet = MinkUNetUno(in_channels=1, out_channels=self.feat_dim, D=cfg_model['pos_dim'])  # D: UNet spatial dim

        # input point cloud quantization
        dx = dy = dz = cfg_model["quant_size"]
        dt = 1  # TODO: cfg_model["time_interval"]
        self.quant = torch.Tensor([dx, dy, dz, dt])

        # quantization offset
        self.scene_bbox = cfg_model["scene_bbox"]
        self.offset = torch.nn.parameter.Parameter(
            torch.Tensor([self.scene_bbox[0], self.scene_bbox[1], self.scene_bbox[2], -cfg_model['n_input'] + 1])[None:],
            requires_grad=False)

        # dense feature map
        self.featmap_size = cfg_model["featmap_size"]
        z_height = int((self.scene_bbox[5] - self.scene_bbox[2]) / self.featmap_size)
        y_length = int((self.scene_bbox[4] - self.scene_bbox[1]) / self.featmap_size)
        x_width = int((self.scene_bbox[3] - self.scene_bbox[0]) / self.featmap_size)
        b_size = cfg_model["batch_size"]
        t_size = cfg_model['n_input']
        self.featmap_shape = [b_size, self.feat_dim, x_width, y_length, z_height, t_size]

        # dense conv after sparse to dense feature map
        self.dense_conv_block = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=3, padding=1, stride=1)),
            ('relu', nn.ReLU(inplace=False)),
            ('conv_2', nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=3, padding=1, stride=1)),
            ('relu', nn.ReLU(inplace=False)),
            ('conv_3', nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=3, padding=1, stride=1)),
            ('relu', nn.ReLU(inplace=False)),
        ]))

    def forward(self, pcds_4d_batch):
        # quantized 4d pcd and initialized features
        self.quant = self.quant.type_as(pcds_4d_batch[0])
        quant_4d_pcds = [torch.div(pcd - self.offset, self.quant) for pcd in pcds_4d_batch]

        # for quant_pcd in quant_4d_pcds:
        #     x_max = torch.max(quant_pcd[:, 0])
        #     y_max = torch.max(quant_pcd[:, 1])
        #     z_max = torch.max(quant_pcd[:, 2])
        #     t_max = torch.max(quant_pcd[:, 3])

        feats = [0.5 * torch.ones(len(pcd), 1).type_as(pcd) for pcd in pcds_4d_batch]

        # sparse collate, tensor field, net calculation
        coords, feats = ME.utils.sparse_collate(quant_4d_pcds, feats)
        tensor_field = ME.TensorField(features=feats, coordinates=coords,
                                      quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        sparse_input = tensor_field.sparse()
        sparse_output = self.MinkUNet(sparse_input)

        # sparse_featmap = sparse_output.slice(tensor_field)
        # sparse_featmap.coordinates[:, 1:] = torch.mul(sparse_featmap.coordinates[:, 1:], self.quant)

        # dense feature map output
        dense_featmap, _, _ = sparse_output.dense(shape=torch.Size(self.featmap_shape), min_coordinate=torch.IntTensor([0, 0, 0, 0]))
        dense_featmap = torch.mean(dense_featmap, dim=-1, keepdim=False)
        dense_featmap = torch.mean(dense_featmap, dim=-1, keepdim=False)

        # dense conv to increase receptive field
        dense_featmap = self.dense_conv_block(dense_featmap)
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

