import os
from collections import OrderedDict

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
from datasets.ours.nusc import NuscSequentialDataset


#######################################
# Lightning Module
#######################################


class MotionPretrainNetwork(LightningModule):
    def __init__(self, cfg_model: dict, train_flag: bool):
        super().__init__()
        # parameters
        self.save_hyperparameters(cfg_model)
        self.cfg_model = cfg_model
        self.n_bg_cls = 2

        # encoder and decoder
        self.encoder = MotionEncoder(self.cfg_model, self.n_bg_cls)
        self.pe = PositionalEncoding(feat_dim=self.cfg_model["feat_dim"], pos_dim=self.cfg_model['pos_dim'])
        self.decoder = BackgroundFieldMLP(in_dim=self.cfg_model["feat_dim"], planes=[256, 128, 64, 32, 16, self.n_bg_cls])

        # metrics
        self.ClassificationMetrics = ClassificationMetrics(self.n_bg_cls, ignore_index=[])

        # pytorch lightning training output
        self.training_step_outputs = []
        self.validation_step_outputs = []

        # loss
        self.loss = nn.NLLLoss(weight=torch.Tensor(self.cfg_model['bg_cls_weight']))

        # save predictions
        if not train_flag:  # TODO: modify the trained model name, and the test directory
            model_dataset = self.cfg_model["model_dataset"]
            test_epoch = self.cfg_model["test_epoch"]
            model_dir = os.path.join("./logs", "ours", model_dataset)
            self.bg_pred_dir = os.path.join(model_dir, "results", f"epoch_{test_epoch}", "predictions", "ours_pred")
            os.makedirs(self.bg_pred_dir, exist_ok=True)

    def forward(self, batch):
        # unfold batch: [(ref_sd_tok, num_rays_all, num_bg_samples_all, num_bg_samples_per_ray_list), pcds_4d, ray_to_bg_samples_dict]
        meta_batch, pcds_batch, bg_samples_batch = batch

        # encoder
        sparse_featmap = self.encoder(pcds_batch)

        # get bg samples
        bg_labels_batch = []
        bg_feats_batch = []
        for batch_idx, bg_samples_dict in enumerate(bg_samples_batch):
            # point-wise features
            coords = sparse_featmap.coordinates_at(batch_index=batch_idx)
            feats = sparse_featmap.features_at(batch_index=batch_idx)
            ref_time_mask = coords[:, -1] == 0
            query_points_idx = np.array(list(bg_samples_dict.keys()))
            query_feats = feats[ref_time_mask][query_points_idx]
            # query_points = pcds_batch[batch_idx][query_points_idx]
            # query_points_with_quant_error = coords[ref_time_mask][query_points_idx]

            # bg_samples
            bg_samples = torch.from_numpy(np.concatenate(list(bg_samples_dict.values()))).cuda()
            bg_points_4d = bg_samples[:, 0:self.cfg_model['pos_dim']]
            bg_labels = bg_samples[:, -1] - 1  # TODO: 1:free, 2:occ -> 0: free, 1: occ

            # bg points positional encoding
            num_bg_samples_per_ray_list = meta_batch[batch_idx][3]
            pe_feats = self.pe(bg_points_4d)
            query_feats = torch.repeat_interleave(query_feats, torch.tensor(num_bg_samples_per_ray_list).cuda(), dim=0)
            bg_feats = query_feats + pe_feats

            # collate to batch
            bg_feats_batch.append(bg_feats)
            bg_labels_batch.append(bg_labels)
        bg_feats = torch.cat(bg_feats_batch, dim=0)
        bg_labels = torch.cat(bg_labels_batch, dim=0)
        # decoder
        bg_probs = self.decoder(bg_feats)  # logits -> softmax -> probs
        return bg_probs, bg_labels

    def configure_optimizers(self):
        # TODO: will be call only at training stage, do not need 'train_flag'
        lr_start = self.cfg_model["lr_start"]
        lr_epoch = self.cfg_model["lr_epoch"]
        lr_decay = self.cfg_model["lr_decay"]
        weight_decay = self.cfg_model["weight_decay"]
        optimizer = torch.optim.Adam(self.parameters(), lr=lr_start, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_epoch, gamma=lr_decay)
        return [optimizer], [scheduler]

    def get_loss(self, bg_probs, bg_labels):
        bg_labels = bg_labels.long()
        assert len(bg_labels) == len(bg_probs)
        bg_prob_log = torch.log(bg_probs.clamp(min=1e-8))
        loss = self.loss(bg_prob_log, bg_labels)  # dtype of torch.nllloss must be torch.long
        return loss

    def training_step(self, batch: tuple, batch_idx, dataloader_index=0):
        # model_dict = self.state_dict()  # check state dict

        bg_probs, bg_labels = self.forward(batch)  # encoder & decoder
        loss = self.get_loss(bg_probs, bg_labels)  # TODO: bg loss only

        # metrics
        conf_mat = self.ClassificationMetrics.compute_conf_mat(bg_probs.detach(), bg_labels)
        free_iou, occ_iou = self.ClassificationMetrics.get_iou(conf_mat)
        free_acc, occ_acc = self.ClassificationMetrics.get_acc(conf_mat)

        # logging
        self.log("loss", loss.item(), on_step=True, prog_bar=True, logger=True)
        self.log("free_iou", free_iou.item() * 100, on_step=True, prog_bar=True, logger=True)
        self.log("occ_iou", occ_iou.item() * 100, on_step=True, prog_bar=True, logger=True)
        self.log("free_acc", free_acc.item() * 100, on_step=True, prog_bar=True, logger=True)
        self.log("occ_acc", occ_acc.item() * 100, on_step=True, prog_bar=True, logger=True)
        self.training_step_outputs.append({"loss": loss.item(), "confusion_matrix": conf_mat})
        torch.cuda.empty_cache()
        return loss

    def on_train_epoch_end(self):
        conf_mat_list = [output["confusion_matrix"] for output in self.training_step_outputs]
        acc_conf_mat = torch.zeros(self.n_bg_cls, self.n_bg_cls)
        for conf_mat in conf_mat_list:
            acc_conf_mat = acc_conf_mat.add(conf_mat)

        # metrics in one epoch
        tp, fp, fn = self.ClassificationMetrics.get_stats(acc_conf_mat)  # stat of current sample
        iou = self.ClassificationMetrics.get_iou(tp, fp, fn)
        free_iou = iou[0]
        occ_iou = iou[1]
        free_acc, occ_acc, acc = self.ClassificationMetrics.get_acc(acc_conf_mat)
        self.log("epoch_free_iou", free_iou.item() * 100, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_occ_iou", occ_iou.item() * 100, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_free_acc", free_acc.item() * 100, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_occ_acc", occ_acc.item() * 100, on_epoch=True, prog_bar=True, logger=True)

        # clean
        self.training_step_outputs = []
        torch.cuda.empty_cache()

    def validation_step(self, batch: tuple, batch_idx):
        # unfold batch data
        _, point_clouds, mos_labels = batch
        _, curr_feats_list = self.forward(point_clouds)
        conf_mat = self.get_confusion_matrix(curr_feats_list, mos_labels)
        self.validation_step_outputs.append(conf_mat.detach().cpu())
        torch.cuda.empty_cache()
        return {"confusion_matrix": conf_mat.detach().cpu()}

    def on_validation_epoch_end(self):
        conf_mat_list = self.validation_step_outputs
        acc_conf_mat = torch.zeros(self.n_bg_cls, self.n_bg_cls)
        for conf_mat in conf_mat_list:
            acc_conf_mat = acc_conf_mat.add(conf_mat)
        tp, fp, fn = self.ClassificationMetrics.get_stats(acc_conf_mat)  # stat of current sample
        iou = self.ClassificationMetrics.get_iou(tp, fp, fn)[self.mov_class_idx]
        self.log("val_iou", iou.item() * 100, on_epoch=True, logger=True)

        # clean
        self.validation_step_outputs = []
        torch.cuda.empty_cache()

    def save_mos_pred(self, pred_logits, mos_pred_file):
        assert pred_logits is not list
        # Set ignored classes to -inf to not influence softmax
        ignore_idx = self.ignore_class_idx[0]
        pred_logits[:, ignore_idx] = -float("inf")
        pred_softmax = F.softmax(pred_logits, dim=1)
        pred_labels = torch.argmax(pred_softmax, dim=1).type(torch.uint8).detach().cpu().numpy()
        # save mos pred labels
        pred_labels.tofile(mos_pred_file)

    # func "predict_step" is called by "trainer.predict"
    def predict_step(self, batch: tuple, batch_idx):
        # unfold batch data
        sample_data_tokens, point_clouds, mos_labels = batch
        batch_size = len(point_clouds)

        # network prediction
        curr_coords_list, curr_feats_list = self.forward(point_clouds)
        # loop batch data list
        acc_conf_mat = torch.zeros(self.n_bg_cls, self.n_bg_cls)
        for i, (curr_feats, mos_label) in enumerate(zip(curr_feats_list, mos_labels)):
            # get ego mask
            curr_time_mask = point_clouds[i][:, -1] == (self.cfg_model["n_input"] - 1)
            ego_mask = NuscSequentialDataset.get_ego_mask(point_clouds[i][curr_time_mask])
            # save mos pred (with ego vehicle points)
            mos_pred_file = os.path.join(self.bg_pred_dir, f"{sample_data_tokens[i]}_mos_pred.label")
            self.save_mos_pred(curr_feats, mos_pred_file)
            # compute confusion matrix (without ego vehicle points)
            conf_mat = self.get_confusion_matrix([curr_feats[~ego_mask]], [mos_label[~ego_mask]])
            acc_conf_mat = acc_conf_mat.add(conf_mat)
            # compute iou metric
            tp, fp, fn = self.ClassificationMetrics.get_stats(conf_mat)  # stat of current sample
            iou = self.ClassificationMetrics.get_iou(tp, fp, fn)[self.mov_class_idx]
            print(
                f"Validation Sample Index {i + batch_idx * batch_size}, Moving Object IoU w/o ego vehicle: {iou.item() * 100}")
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

        # TODO: quantization resolution
        dx = dy = dz = cfg_model["quant_size"]
        dt = 1  # TODO: should be cfg_model["time_interval"], handle different lidar frequency of kitti and nuscenes
        self.quant = torch.Tensor([dx, dy, dz, dt])

        # TODO: feature map shape
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

        # TODO: dense feature map output
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

    def forward(self, bg_points_4d):
        pe_xyzt = []
        for i in range(self.pos_dim):
            assert self.feat_dim % self.pos_dim == 0
            i_dim = int(self.feat_dim / self.pos_dim)
            pe_i = torch.zeros(len(bg_points_4d), i_dim).cuda()
            position = bg_points_4d[:, i].unsqueeze(-1)
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
        return self.block(x)

