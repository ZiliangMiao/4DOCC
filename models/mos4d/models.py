import os
import torch
import torch.nn as nn
import sys
import logging
from datetime import datetime

from pytorch_lightning import LightningModule
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock
from lib.minkowski.minkunet import MinkUNetBase
from utils.metrics import ClassificationMetrics


#######################################
# Lightning Module
#######################################


class MosNetwork(LightningModule):
    def __init__(self, cfg_model: dict, train_flag: bool, **kwargs):
        super().__init__()
        if train_flag:
            self.save_hyperparameters(cfg_model)
        self.cfg_model = cfg_model
        self.n_mos_cls = 3  # 0 -> unknown, 1 -> static, 2 -> moving
        self.ignore_class_idx = [0]  # ignore unknown class when calculating scores
        self.mov_class_idx = 2
        # self.dt_prediction = self.cfg_model["time_interval"]

        # only encoder, no decoder
        self.encoder = MOSModel(cfg_model, self.n_mos_cls)

        # metrics
        self.ClassificationMetrics = ClassificationMetrics(self.n_mos_cls, self.ignore_class_idx)

        # pytorch lightning training output
        self.training_step_outputs = []
        self.validation_step_outputs = []

        # loss
        weight = [0.0 if i in self.ignore_class_idx else 1.0 for i in range(self.n_mos_cls)]
        weight = torch.Tensor([w / sum(weight) for w in weight])  # ignore unknown class when calculate loss
        self.loss = nn.NLLLoss(weight=weight)

        # save predictions
        if not train_flag:
            self.model_dir = kwargs['model_dir']
            self.test_epoch = kwargs['test_epoch']
            self.pred_dir = os.path.join(self.model_dir, "predictions", f"epoch_{self.test_epoch}")
            os.makedirs(self.pred_dir, exist_ok=True)
            self.mov_iou_list = []
            self.num_sample_wo_mov = 0
            self.test_logger = kwargs['test_logger']

    def forward(self, batch: dict):
        # unfold batch data
        meta_batch, pcds_batch, mos_labels_batch = batch

        # encoder
        softmax = nn.Softmax(dim=1)
        sparse_featmap = self.encoder(pcds_batch)
        mos_probs_batch = []
        for batch_idx in range(len(pcds_batch)):
            coords = sparse_featmap.coordinates_at(batch_index=batch_idx)
            feats = sparse_featmap.features_at(batch_index=batch_idx)
            ref_time_mask = coords[:, -1] == 0
            mos_feats = feats[ref_time_mask]
            mos_feats[:, self.ignore_class_idx] = -float("inf")
            mos_probs = softmax(mos_feats)
            mos_probs_batch.append(mos_probs)
        return mos_probs_batch

    def configure_optimizers(self):
        lr_start = self.cfg_model["lr_start"]
        lr_epoch = self.cfg_model["lr_epoch"]
        lr_decay = self.cfg_model["lr_decay"]
        weight_decay = self.cfg_model["weight_decay"]
        optimizer = torch.optim.Adam(self.parameters(), lr=lr_start, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_epoch, gamma=lr_decay)
        return [optimizer], [scheduler]

    def get_loss(self, mos_probs, mos_labels):
        mos_labels = mos_labels.long()
        assert len(mos_labels) == len(mos_probs)
        bg_prob_log = torch.log(mos_probs.clamp(min=1e-8))
        loss = self.loss(bg_prob_log, mos_labels)  # dtype of torch.nllloss must be torch.long
        return loss

    def get_mov_iou_list(self):
        return self.mov_iou_list

    def get_num_sample_wo_mov(self):
        return self.num_sample_wo_mov

    def training_step(self, batch: tuple, batch_idx, dataloader_index=0):
        # model_dict = self.state_dict()  # check state dict

        sd_toks_batch, pcds_batch, mos_labels_batch = batch
        mos_probs_batch = self.forward(batch)  # encoder & decoder
        mos_probs = torch.cat(mos_probs_batch, dim=0)
        mos_labels = torch.cat(mos_labels_batch, dim=0)
        loss = self.get_loss(mos_probs, mos_labels)

        # metrics
        conf_mat = self.ClassificationMetrics.compute_conf_mat(mos_probs.detach(), mos_labels)
        iou = self.ClassificationMetrics.get_iou(conf_mat)
        sta_iou, mov_iou = iou[1], iou[2]

        # logging
        self.log("loss", loss.item(), on_step=True, prog_bar=True, logger=True)
        self.log("sta_iou", sta_iou.item() * 100, on_step=True, prog_bar=True, logger=True)
        self.log("mov_iou", mov_iou.item() * 100, on_step=True, prog_bar=True, logger=True)
        self.training_step_outputs.append({"loss": loss.item(), "confusion_matrix": conf_mat})
        torch.cuda.empty_cache()
        return loss

    def on_train_epoch_end(self):
        conf_mat_list = [output["confusion_matrix"] for output in self.training_step_outputs]
        acc_conf_mat = torch.zeros(self.n_mos_cls, self.n_mos_cls)
        for conf_mat in conf_mat_list:
            acc_conf_mat = acc_conf_mat.add(conf_mat)

        # metrics in one epoch
        iou = self.ClassificationMetrics.get_iou(acc_conf_mat)
        sta_iou, mov_iou = iou[1], iou[2]
        self.log("epoch_sta_iou", sta_iou.item() * 100, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_mov_iou", mov_iou.item() * 100, on_epoch=True, prog_bar=True, logger=True)

        # clean
        self.training_step_outputs = []
        torch.cuda.empty_cache()

    def validation_step(self, batch: tuple, batch_idx):
        # unfold batch data
        _, point_clouds, mos_labels = batch
        probs = self.forward(point_clouds)
        conf_mat = self.get_confusion_matrix(curr_feats_list, mos_labels)
        self.validation_step_outputs.append(conf_mat.detach().cpu())
        torch.cuda.empty_cache()
        return {"confusion_matrix": conf_mat.detach().cpu()}

    def on_validation_epoch_end(self):
        conf_mat_list = self.validation_step_outputs
        acc_conf_mat = torch.zeros(self.n_mos_cls, self.n_mos_cls)
        for conf_mat in conf_mat_list:
            acc_conf_mat = acc_conf_mat.add(conf_mat)
        tp, fp, fn = self.ClassificationMetrics.getStats(acc_conf_mat)  # stat of current sample
        iou = self.ClassificationMetrics.getIoU(tp, fp, fn)[self.mov_class_idx]
        self.log("val_iou", iou.item() * 100,  on_epoch=True, logger=True)

        # clean
        self.validation_step_outputs = []
        torch.cuda.empty_cache()

    def predict_step(self, batch: tuple, batch_idx):
        # unfold batch data
        sd_tok_batch, pcds_batch, mos_labels_batch = batch

        # network prediction
        mos_probs_batch = self.forward(batch)

        # iterate each batch data for predicted label saving
        acc_conf_mat = torch.zeros(self.n_mos_cls, self.n_mos_cls)

        for sd_tok, mos_probs, mos_labels in zip(sd_tok_batch, mos_probs_batch, mos_labels_batch):
            # metrics
            conf_mat = self.ClassificationMetrics.compute_conf_mat(mos_probs.detach(), mos_labels)
            iou = self.ClassificationMetrics.get_iou(conf_mat)
            sta_iou, mov_iou = iou[1], iou[2]

            # update confusion matrix
            acc_conf_mat = acc_conf_mat.add(conf_mat)

            # save predicted labels
            mos_pred_file = os.path.join(self.pred_dir, f"{sd_tok}_mos_pred.label")
            pred_labels = torch.argmax(mos_probs, dim=1).type(torch.uint8).detach().cpu().numpy()
            pred_labels.tofile(mos_pred_file)

            # logger
            self.test_logger.info("Val sd tok: %s, Moving IoU: %.3f", sd_tok, mov_iou.item() * 100)

            # TODO: method robustness at different scene (calculate sample level IoU avg.), ignore samples that have no moving points
            num_mov_pts = conf_mat[2][0] + conf_mat[2][1] + conf_mat[2][2]
            num_sta_pts = conf_mat[1][0] + conf_mat[1][1] + conf_mat[1][2]
            num_unk_pts = conf_mat[0][0] + conf_mat[0][1] + conf_mat[0][2]
            assert len(mos_probs) == num_mov_pts + num_sta_pts + num_unk_pts
            if num_mov_pts != 0:
                self.mov_iou_list.append(mov_iou.item())
            else:
                self.num_sample_wo_mov += 1

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


class MOSModel(nn.Module):
    def __init__(self, cfg_model: dict, n_classes: int):
        super().__init__()

        # backbone network
        self.n_mos_cls = 3  # 0: static, 1: moving
        self.MinkUNet = MinkUNet14(in_channels=1, out_channels=self.n_mos_cls, D=cfg_model['pos_dim'])

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

        # point-wise sparse feature output
        sparse_featmap = sparse_output.slice(tensor_field)
        sparse_featmap.coordinates[:, 1:] = torch.mul(sparse_featmap.coordinates[:, 1:], self.quant)
        return sparse_featmap






