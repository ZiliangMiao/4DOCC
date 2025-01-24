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
# from torch_geometric.nn import radius as search_radius
# import torch_cluster.radius as search_radius
import open3d as o3d


#######################################
# Lightning Module
#######################################


class AlsoNetwork(LightningModule):
    def __init__(self, cfg_model: dict, train_flag: bool, **kwargs):
        super().__init__()
        if train_flag:
            self.save_hyperparameters(cfg_model)
        self.cfg_model = cfg_model
        self.n_cls = self.cfg_model['num_cls']
        self.pos_dim = self.cfg_model['pos_dim']
        self.feat_dim = self.cfg_model['feat_dim']
        self.iters_per_epoch = kwargs['iters_per_epoch']

        # also encoder
        self.encoder = AlsoEncoder(self.cfg_model)

        # also decoder
        self.decoder = AlsoDecoder(self.cfg_model)

        # metrics
        self.ClassificationMetrics = ClassificationMetrics(self.n_cls, ignore_index=[])

        # pytorch lightning training output
        self.epoch_acc_conf_mat = torch.zeros(self.n_cls, self.n_cls)

        # save predictions
        if not train_flag:
            self.model_dir = kwargs['model_dir']
            self.test_epoch = kwargs['test_epoch']
            self.pred_dir = os.path.join(self.model_dir, "predictions", f"epoch_{self.test_epoch}")
            os.makedirs(self.pred_dir, exist_ok=True)

    def forward(self, batch):
        # unbatch data
        meta_batch, pcds_batch, also_samples_batch = batch

        # encoder
        sparse_featmap_batch = self.encoder(pcds_batch)

        # decoder
        also_probs_batch = []
        also_labels_batch = []
        for batch_idx, also_samples in enumerate(also_samples_batch):
            coords = sparse_featmap_batch.coordinates_at(batch_index=batch_idx)
            feats = sparse_featmap_batch.features_at(batch_index=batch_idx)
            feats = feats[coords[:, -1] == 0]

            # decoder
            also_probs, also_labels = self.decoder(also_samples[0], feats, also_samples[1], also_samples[2])
            also_probs_batch.append(also_probs)
            also_labels_batch.append(also_labels)
        return also_probs_batch, also_labels_batch

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

    def get_loss(self, probs, labels):
        labels = labels.long()  # dtype of torch.nllloss must be torch.long
        assert len(labels) == len(probs)
        log_prob = torch.log(probs.clamp(min=1e-15))
        loss_func = nn.NLLLoss()
        loss = loss_func(log_prob, labels)
        return loss

    def training_step(self, batch: tuple, batch_idx, dataloader_index=0):
        # model_dict = self.state_dict()  # check state dict
        # recons_loss = F.binary_cross_entropy_with_logits(x[:, 0], also_labels.float())
        also_probs_batch, also_labels_batch = self.forward(batch)  # encoder & decoder
        also_probs = torch.concat(also_probs_batch).view(-1, self.n_cls)
        also_labels = torch.concat(also_labels_batch).view(-1)
        pred_labels = torch.argmax(also_probs, axis=-1)
        loss = self.get_loss(also_probs, also_labels)

        # metrics
        conf_mat = self.ClassificationMetrics.compute_conf_mat(pred_labels, also_labels)
        iou = self.ClassificationMetrics.get_iou(conf_mat)
        free_iou, occ_iou = iou[0].item() * 100, iou[1].item() * 100
        acc = self.ClassificationMetrics.get_acc(conf_mat)
        free_acc, occ_acc = acc[0].item() * 100, acc[1].item() * 100
        self.epoch_acc_conf_mat += conf_mat  # add conf mat to epoch accumulated conf mat

        # logging
        self.log("loss", loss.item(), on_step=True, prog_bar=True, logger=True)
        self.log("free_iou", free_iou, on_step=True, prog_bar=True, logger=True)
        self.log("occ_iou", occ_iou, on_step=True, prog_bar=True, logger=True)
        self.log("free_acc", free_acc, on_step=True, prog_bar=True, logger=True)
        self.log("occ_acc", occ_acc, on_step=True, prog_bar=True, logger=True)

        # cuda memory clean
        del pred_labels, also_labels
        torch.cuda.empty_cache()
        return loss

    def on_train_epoch_end(self):
        # metrics in one epoch
        iou = self.ClassificationMetrics.get_iou(self.epoch_acc_conf_mat)
        free_iou, occ_iou = iou[0].item() * 100, iou[1].item() * 100
        acc = self.ClassificationMetrics.get_acc(self.epoch_acc_conf_mat)
        free_acc, occ_acc = acc[0].item() * 100, acc[1].item() * 100
        self.log("epoch_free_iou", free_iou, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_occ_iou", occ_iou, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_free_acc", free_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_occ_acc", occ_acc, on_epoch=True, prog_bar=True, logger=True)

        # clean
        self.epoch_acc_conf_mat = self.epoch_acc_conf_mat.zero_()
        torch.cuda.empty_cache()

    def predict_step(self, batch: tuple, batch_idx):
        # model_dict = self.state_dict()  # check state dict
        a = -1


#######################################
# Modules
#######################################


class AlsoEncoder(nn.Module):
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


def search_radius(points, query_points, radius):
    """
    Args:
        points: (N, 3) 源点云
        query_points: (M, 3) 查询点
        radius: 搜索半径
    Returns:
        row: 查询点的索引
        col: 源点云中邻居点的索引
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 构建KD树
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    rows = []
    cols = []

    # 对每个查询点进行半径搜索
    for i, query in enumerate(query_points):
        # 返回[k, idx, dist]
        _, idx, _ = kdtree.search_radius_vector_3d(query, radius)
        rows.extend([i] * len(idx))
        cols.extend(idx)

    return torch.tensor(rows), torch.tensor(cols)


class AlsoDecoder(nn.Module):
    def __init__(self, cfg_model: dict):
        super().__init__()
        # feature dimension
        self.latent_size = cfg_model["feat_dim"]
        self.out_channels = 2 # [free_prob, occ_prob]

        # layers of the decoder
        self.fc_in = torch.nn.Linear(self.latent_size + 3, self.latent_size)
        mlp_layers = [torch.nn.Linear(self.latent_size, self.latent_size) for _ in range(2)]
        self.mlp_layers = nn.ModuleList(mlp_layers)
        self.fc_out = torch.nn.Linear(self.latent_size, self.out_channels)
        self.activation = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

        # radius search
        self.radius = cfg_model["radius"]

    def forward(self, pcd, feats, also_pts, also_labels):
        # get the data
        pos_source = pcd
        pos_target = also_pts

        # neighborhood search
        # radii = torch.Tensor([self.radius, self.radius, self.radius])
        # idx = ml3d.ops.radius_search(pos_source, pos_target, radii,
        #                        points_row_splits=torch.LongTensor([0, len(pos_source)]),
        #                        queries_row_splits=torch.LongTensor([0, len(pos_target)]))
        row, col = search_radius(points=pos_source.to('cpu'), query_points=pos_target.to('cpu'), radius=self.radius)

        # compute reltive position between query and input point cloud and the corresponding latent vectors
        pos_relative = pos_target[row] - pos_source[col]
        latents_relative = feats[col]

        x = torch.cat([latents_relative, pos_relative], dim=1)

        # Decoder layers
        x = self.fc_in(x.contiguous())
        for i, l in enumerate(self.mlp_layers):
            x = l(self.activation(x))
        x = self.fc_out(x)
        probs = self.softmax(x)
        return probs, also_labels[row]
