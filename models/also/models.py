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
from torch_geometric.nn import radius as search_radius
# import torch_cluster.radius as search_radius
import open3d.ml.tf as ml3d

#######################################
# Lightning Module
#######################################


class AlsoNetwork(LightningModule):
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

        # also encoder
        self.encoder = AlsoEncoder(self.cfg_model)

        # also decoder
        self.decoder = AlsoDecoder(self.cfg_model)

        # metrics
        self.ClassificationMetrics = ClassificationMetrics(self.n_uno_cls, ignore_index=[])

        # pytorch lightning training output
        self.epoch_acc_conf_mat = torch.zeros(self.n_uno_cls, self.n_uno_cls)

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

            # batch data
            curr_pcd = also_samples[0]
            also_pts = also_samples[1]  # x, y, z
            also_labels = also_samples[2]
            also_labels_batch.append(also_labels)

            # decoder
            also_probs = self.decoder(curr_pcd, feats, also_pts)
            also_probs_batch.append(also_probs)
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

    def get_loss(self, uno_probs, uno_labels):
        uno_labels = uno_labels.long()  # dtype of torch.nllloss must be torch.long
        assert len(uno_labels) == len(uno_probs)
        log_prob = torch.log(uno_probs.clamp(min=1e-15))
        loss_func = nn.NLLLoss()
        loss = loss_func(log_prob, uno_labels)
        return loss

    def training_step(self, batch: tuple, batch_idx, dataloader_index=0):
        # model_dict = self.state_dict()  # check state dict
        # recons_loss = F.binary_cross_entropy_with_logits(x[:, 0], also_labels.float())
        uno_probs_batch, uno_labels_batch = self.forward(batch)  # encoder & decoder
        uno_probs = uno_probs_batch.view(-1, self.n_uno_cls)
        uno_labels = uno_labels_batch.view(-1)
        pred_labels = torch.argmax(uno_probs, axis=-1)
        loss = self.get_loss(uno_probs, uno_labels)

        # metrics
        conf_mat = self.ClassificationMetrics.compute_conf_mat(pred_labels, uno_labels)
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
        del pred_labels, uno_labels
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


class AlsoDecoder(nn.Module):
    def __init__(self, cfg_model: dict):
        super().__init__()
        # feature dimension
        self.latent_size = cfg_model["feat_dim"]
        self.out_channels = cfg_model["feat_dim"]

        # layers of the decoder
        self.fc_in = torch.nn.Linear(self.latent_size + 3, self.latent_size)
        mlp_layers = [torch.nn.Linear(self.latent_size, self.latent_size) for _ in range(2)]
        self.mlp_layers = nn.ModuleList(mlp_layers)
        self.fc_out = torch.nn.Linear(self.latent_size, self.out_channels)
        self.activation = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

        # radius search
        self.radius = cfg_model["radius"]

    def forward(self, pcd, feats, also_pts):
        # get the data
        pos_source = pcd
        pos_target = also_pts

        # neighborhood search
        # radii = torch.Tensor([self.radius, self.radius, self.radius])
        # idx = ml3d.ops.radius_search(pos_source, pos_target, radii,
        #                        points_row_splits=torch.LongTensor([0, len(pos_source)]),
        #                        queries_row_splits=torch.LongTensor([0, len(pos_target)]))
        row, col = search_radius(x=pos_source, y=pos_target, r=self.radius)

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

        # reconstruction loss
        logits = x[:, 0]
        return logits
