import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pytorch_lightning import LightningModule
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock
from lib.minkowski.minkunet import MinkUNetBase
from utils.metrics import ClassificationMetrics
from datasets.mos4d.nusc import NuscMosDataset


#######################################
# Lightning Module
#######################################


class MosNetwork(LightningModule):
    def __init__(self, cfg_model: dict, train_flag: bool):
        super().__init__()
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
            model_dataset = self.cfg_model["model"]["model_dataset"]
            model_name = self.cfg_model["model"]["model_name"]
            model_version = self.cfg_model["model"]["model_version"]
            test_epoch = self.cfg_model["model"]["test_epoch"]
            model_dir = os.path.join("./logs", "mos4d", model_dataset, model_name, model_version)
            self.mos_pred_dir = os.path.join(model_dir, "results", f"epoch_{test_epoch}", "predictions", "mos_pred")
            os.makedirs(self.mos_pred_dir, exist_ok=True)

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
        mos_probs = torch.cat(mos_probs_batch, dim=0)
        mos_labels = torch.cat(mos_labels_batch, dim=0)
        return mos_probs, mos_labels

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

    def training_step(self, batch: tuple, batch_idx, dataloader_index=0):
        # model_dict = self.state_dict()  # check state dict

        mos_probs, mos_labels = self.forward(batch)  # encoder & decoder
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


    # 暂时修改， 为了测试predict和valid输出为什么不同
    # def validation_step(self, batch: tuple, batch_idx):
    #     # unfold batch data
    #     sample_data_tokens, point_clouds, mos_labels = batch
    #     batch_size = len(point_clouds)
    #     # network prediction
    #     curr_coords_list, curr_feats_list = self.forward(point_clouds)
    #     # loop batch data list
    #     acc_conf_mat = torch.zeros(self.n_classes, self.n_classes)
    #     for i, (curr_feats, mos_label) in enumerate(zip(curr_feats_list, mos_labels)):
    #         # get ego mask
    #         curr_time_mask = point_clouds[i][:, -1] == 0.0
    #         ego_mask = NuscSequentialDataset.get_ego_mask(point_clouds[i][curr_time_mask])
    #         # compute confusion matrix
    #         conf_mat = self.get_confusion_matrix([curr_feats[~ego_mask]], [mos_label[~ego_mask]])  # input is lists
    #         acc_conf_mat = acc_conf_mat.add(conf_mat)
    #         # compute iou metric
    #         tp, fp, fn = self.ClassificationMetrics.getStats(conf_mat)  # stat of current sample
    #         iou = self.ClassificationMetrics.getIoU(tp, fp, fn)[self.mov_class_idx]
    #         print(f"Validation Sample Index {i + batch_idx * batch_size}, Moving Object IoU w/o ego vehicle: {iou.item() * 100}")
    #     self.validation_step_outputs.append(acc_conf_mat.detach().cpu())
    #     torch.cuda.empty_cache()
    #     return {"confusion_matrix": acc_conf_mat.detach().cpu()}
    #
    # def on_validation_epoch_end(self):
    #     conf_mat_list = self.validation_step_outputs
    #     acc_conf_mat = torch.zeros(self.n_classes, self.n_classes)
    #     for conf_mat in conf_mat_list:
    #         acc_conf_mat = acc_conf_mat.add(conf_mat)
    #     tp, fp, fn = self.ClassificationMetrics.getStats(acc_conf_mat)  # stat of current sample
    #     iou = self.ClassificationMetrics.getIoU(tp, fp, fn)[self.mov_class_idx]
    #     self.log("val_iou", iou.item() * 100, on_epoch=True, logger=True)
    #     return self.validation_step_outputs

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
        acc_conf_mat = torch.zeros(self.n_mos_cls, self.n_mos_cls)
        for conf_mat in conf_mat_list:
            acc_conf_mat = acc_conf_mat.add(conf_mat)
        tp, fp, fn = self.ClassificationMetrics.getStats(acc_conf_mat)  # stat of current sample
        iou = self.ClassificationMetrics.getIoU(tp, fp, fn)[self.mov_class_idx]
        self.log("val_iou", iou.item() * 100,  on_epoch=True, logger=True)

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
        acc_conf_mat = torch.zeros(self.n_mos_cls, self.n_mos_cls)
        for i, (curr_feats, mos_label) in enumerate(zip(curr_feats_list, mos_labels)):
            # get ego mask
            curr_time_mask = point_clouds[i][:, -1] == (self.n_input - 1)
            ego_mask = NuscMosDataset.get_ego_mask(point_clouds[i][curr_time_mask])
            # save mos pred (with ego vehicle points)
            mos_pred_file = os.path.join(self.mos_pred_dir, f"{sample_data_tokens[i]}_mos_pred.label")
            self.save_mos_pred(curr_feats, mos_pred_file)
            # compute confusion matrix (without ego vehicle points)
            conf_mat = self.get_confusion_matrix([curr_feats[~ego_mask]], [mos_label[~ego_mask]])
            acc_conf_mat = acc_conf_mat.add(conf_mat)
            # compute iou metric
            tp, fp, fn = self.ClassificationMetrics.getStats(conf_mat)  # stat of current sample
            iou = self.ClassificationMetrics.getIoU(tp, fp, fn)[self.mov_class_idx]
            print(f"Validation Sample Index {i + batch_idx * batch_size}, Moving Object IoU w/o ego vehicle: {iou.item() * 100}")
        torch.cuda.empty_cache()
        return {"confusion_matrix": acc_conf_mat.detach().cpu()}



        # for batch_idx in range(len(batch[0])):
        #     sample_data_token = sample_data_tokens[batch_idx]
        #     mos_label = mos_labels[batch_idx].cpu().detach().numpy()
        #     step = 0  # only evaluate the performance of current timestamp
        #     coords = out.coordinates_at(batch_idx)
        #     logits = out.features_at(batch_idx)
        #
        #     t = round(-step * self.dt_prediction, 3)
        #     mask = coords[:, -1].isclose(torch.tensor(t))
        #     masked_logits = logits[mask]
        #     masked_logits[:, self.ignore_class_idx] = -float("inf")  # ingore: 0, i.e., unknown or noise
        #
        #     pred_softmax = F.softmax(masked_logits, dim=1)
        #     pred_softmax = pred_softmax.detach().cpu().numpy()
        #     assert pred_softmax.shape[1] == 3
        #     assert pred_softmax.shape[0] >= 0
        #     sum = np.sum(pred_softmax[:, 1:3], axis=1)
        #     assert np.isclose(sum, np.ones_like(sum)).all()
        #     moving_confidence = pred_softmax[:, 2]
        #
        #     # directly output the mos label, without any bayesian strategy (do not need confidences_to_labels.py file)
        #     pred_label = np.ones_like(moving_confidence, dtype=np.uint8)  # notice: dtype of nusc labels are always uint8
        #     pred_label[moving_confidence > 0.5] = 2
        #     pred_label_dir = os.path.join(self.test_datapath, "4dmos_sekitti_pred", self.version)
        #     os.makedirs(pred_label_dir, exist_ok=True)
        #     pred_label_file = os.path.join(pred_label_dir, sample_data_token + "_mos_pred.label")
        #     pred_label.tofile(pred_label_file)
        # torch.cuda.empty_cache()

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






