import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from nuscenes.utils.geometry_utils import points_in_box
from pytorch_lightning import LightningModule
import MinkowskiEngine as ME

from datasets.kitti_utils import load_mos_labels
from models.backbone import MinkUNetBackbone
from utils.metrics import ClassificationMetrics


#######################################
# Lightning Module
#######################################


class MosNetwork(LightningModule):
    def __init__(self, cfg_model: dict, cfg_dataset: dict, train_flag: bool, **kwargs):
        super().__init__()
        if train_flag:
            self.save_hyperparameters(cfg_model)
        self.cfg_model = cfg_model
        self.cfg_dataset = cfg_dataset
        self.n_mos_cls = 3  # 0 -> unknown, 1 -> static, 2 -> moving
        self.ignore_class_idx = [0]  # ignore unknown class when calculating scores
        # self.dt_prediction = self.cfg_model["time_interval"]

        # encoder - decoder
        if self.cfg_model['use_mlp_decoder']:
            self.encoder = MOSModel(cfg_model, self.cfg_model['pretrain_featdim'])
            self.decoder = MOSHead(in_dim=self.cfg_model['pretrain_featdim'], planes=[128, 64, 32, 16, self.n_mos_cls])
        else:
            self.encoder = MOSModel(cfg_model, self.n_mos_cls)

        # metrics
        self.ClassificationMetrics = ClassificationMetrics(self.n_mos_cls, self.ignore_class_idx)

        # epoch accumulation
        self.epoch_conf_mat = torch.zeros(self.n_mos_cls, self.n_mos_cls)

        # loss
        weight = [0.0 if i in self.ignore_class_idx else 1.0 for i in range(self.n_mos_cls)]
        weight = torch.Tensor([w / sum(weight) for w in weight])  # ignore unknown class when calculate loss
        self.loss = nn.NLLLoss(weight=weight)

        # save predictions
        if not train_flag:
            if self.cfg_model['dataset_name'] == 'nuscenes':
                self.nusc = kwargs['nusc']
            self.model_dir = kwargs['model_dir']
            self.test_epoch = kwargs['test_epoch']
            self.test_logger = kwargs['test_logger']
            self.pred_dir = os.path.join(self.model_dir, "predictions", f"epoch_{self.test_epoch}")
            os.makedirs(self.pred_dir, exist_ok=True)

            # metrics
            self.metric_obj = kwargs['metric_obj']
            self.accumulated_conf_mat = torch.zeros(self.n_mos_cls, self.n_mos_cls)  # to calculate point-level avg. iou
            self.sample_iou_list = []  # to calculate sample-level avg. iou
            self.object_iou_list = []  # to calculate object-level avg. iou
            self.mov_obj_num = 0  # number of all moving objects
            self.det_mov_obj_cnt = 0  # count of detected moving objects
            self.no_mov_sample_num = 0  # number of samples that have no moving objects

    def forward(self, batch: dict):
        # unfold batch data
        meta_batch, pcds_batch, _ = batch

        # encoder
        sparse_featmap = self.encoder(pcds_batch)

        # decoder
        softmax = nn.Softmax(dim=1)
        mos_probs_batch = []
        for batch_idx in range(len(pcds_batch)):
            coords = sparse_featmap.coordinates_at(batch_index=batch_idx)
            feats = sparse_featmap.features_at(batch_index=batch_idx)
            ref_time_mask = coords[:, -1] == 0
            mos_feats = feats[ref_time_mask]
            # TODO: task MLP head
            if self.cfg_model['use_mlp_decoder']:
                mos_feats = self.decoder(mos_feats)
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

    def training_step(self, batch: tuple, batch_idx, dataloader_index=0):
        # model_dict = self.state_dict()  # check state dict
        _, pcds_batch, mos_labels_batch = batch
        mos_probs_batch = self.forward(batch)  # encoder & decoder
        mos_probs = torch.cat(mos_probs_batch, dim=0)
        pred_labels = torch.argmax(mos_probs, axis=1)
        mos_labels = torch.cat(mos_labels_batch, dim=0)
        loss = self.get_loss(mos_probs, mos_labels)

        # metrics
        conf_mat = self.ClassificationMetrics.compute_conf_mat(pred_labels, mos_labels)
        iou = self.ClassificationMetrics.get_iou(conf_mat)
        sta_iou, mov_iou = iou[1], iou[2]

        # logging
        self.log("loss", loss.item(), on_step=True, prog_bar=True, logger=True)
        self.log("sta_iou", sta_iou.item() * 100, on_step=True, prog_bar=True, logger=True)
        self.log("mov_iou", mov_iou.item() * 100, on_step=True, prog_bar=True, logger=True)
        self.epoch_conf_mat += conf_mat
        torch.cuda.empty_cache()
        return loss

    def on_train_epoch_end(self):
        # metrics in one epoch
        iou = self.ClassificationMetrics.get_iou(self.epoch_conf_mat)
        sta_iou, mov_iou = iou[1], iou[2]
        self.log("epoch_sta_iou", sta_iou.item() * 100, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_mov_iou", mov_iou.item() * 100, on_epoch=True, prog_bar=True, logger=True)

        # clean
        self.epoch_conf_mat.zero_()
        torch.cuda.empty_cache()

    def validation_step(self, batch: tuple, batch_idx):
        a = 1

    def on_validation_epoch_end(self):
        a = 1

    def predict_step(self, batch: tuple, batch_idx):
        # unfold batch data and model forward
        meta_batch, pcds_batch, mos_labels_batch = batch
        mos_probs_batch = self.forward(batch)

        # iterate each sample
        for meta_info, pcds_4d, mos_probs, mos_labels in zip(meta_batch, pcds_batch, mos_probs_batch, mos_labels_batch):
            # meta
            sd_tok = None
            scan_idx = None
            valid_mask = None
            if self.cfg_model['dataset_name'] == 'nuscenes':
                sd_tok = meta_info
            elif self.cfg_model['dataset_name'] == 'sekitti':
                scan_idx = meta_info[0]
                valid_mask = meta_info[1].to('cpu')

            # labels
            mos_labels = mos_labels.cpu()
            pred_labels = torch.argmax(mos_probs, axis=1).cpu()

            # TODO: mean moving object IoU
            if self.metric_obj and self.cfg_model['dataset_name'] == 'nuscenes':
                mov_obj_num = 0
                det_mov_obj_cnt = 0
                ref_time_mask = pcds_4d[:, -1] == 0
                pcd = pcds_4d[ref_time_mask].cpu().numpy()
                sample_data = self.nusc.get('sample_data', sd_tok)
                sample = self.nusc.get("sample", sample_data['sample_token'])
                _, bbox_list, _ = self.nusc.get_sample_data(sd_tok, selected_anntokens=sample['anns'], use_flat_vehicle_coordinates=False)
                for ann_tok, box in zip(sample['anns'], bbox_list):
                    ann = self.nusc.get('sample_annotation', ann_tok)
                    if ann['num_lidar_pts'] == 0: continue
                    obj_pts_mask = points_in_box(box, pcd[:, :3].T)
                    if np.sum(obj_pts_mask) == 0:  # not lidar points in obj bbox
                        continue
                    # gt and pred object labels
                    gt_obj_labels = mos_labels[obj_pts_mask]
                    mov_pts_mask = gt_obj_labels == 2
                    if torch.sum(mov_pts_mask) == 0:  # static object
                        continue
                    else:
                        mov_obj_num += 1

                    # metric-1: object-level detection rate
                    pred_obj_mos = pred_labels[obj_pts_mask]
                    tp_mask = torch.logical_and(mov_pts_mask, gt_obj_labels == pred_obj_mos)
                    if torch.sum(tp_mask) >= 1:  # TODO: need a percentage threshold
                        det_mov_obj_cnt += 1

                    # metric-2: object-level iou
                    obj_conf_mat = self.ClassificationMetrics.compute_conf_mat(pred_obj_mos, gt_obj_labels)
                    obj_iou = self.ClassificationMetrics.get_iou(obj_conf_mat)
                    obj_mov_iou = obj_iou[2].item()
                    self.object_iou_list.append(obj_mov_iou)

                # metric-1: object-level detection rate
                self.det_mov_obj_cnt += det_mov_obj_cnt
                self.mov_obj_num += mov_obj_num
                det_rate = det_mov_obj_cnt / (mov_obj_num + 1e-15)

                # metric-3: sample-level iou
                sample_conf_mat = self.ClassificationMetrics.compute_conf_mat(pred_labels, mos_labels)
                sample_iou = self.ClassificationMetrics.get_iou(sample_conf_mat)
                sample_mov_iou = sample_iou[2].item()

                num_mov_pts = sample_conf_mat[2][0] + sample_conf_mat[2][1] + sample_conf_mat[2][2]
                if num_mov_pts != 0:
                    self.sample_iou_list.append(sample_mov_iou)
                else:
                    self.no_mov_sample_num += 1

                # metric-4: point-level iou
                self.accumulated_conf_mat += sample_conf_mat

                # save predicted labels
                mos_pred_file = os.path.join(self.pred_dir, f"{sd_tok}_mos_pred.label")
                pred_labels = torch.argmax(mos_probs, dim=1).type(torch.uint8).detach().cpu().numpy()
                pred_labels.tofile(mos_pred_file)

                # logger
                self.test_logger.info(f"Val sd tok: %s, Sample IoU: %.3f, Moving obj num: %d, Det rate: %.3f", sd_tok, sample_mov_iou * 100, mov_obj_num, det_rate * 100)

            elif self.metric_obj and self.cfg_model['dataset_name'] == 'sekitti':
                # get file path
                root = self.cfg_dataset['sekitti']['root']
                path_to_seq = os.path.join(root, str(self.cfg_dataset['sekitti']['val'][0]).zfill(2))
                scan_labels_file = os.path.join(path_to_seq, 'labels', scan_idx.zfill(6) + ".label")
                scan_labels = np.fromfile(scan_labels_file, dtype=np.int32)

                # split semantic labels and instance labels
                # sem_labels = scan_labels & 0xFFFF  # semantic label in lower half
                obj_labels = scan_labels >> 16
                obj_labels = obj_labels.astype(np.int32)[valid_mask]
                unq_obj = np.unique(obj_labels)

                # loop each instance
                mov_obj_num = 0
                for obj in unq_obj:
                    obj_mask = obj_labels == obj
                    gt_obj_mos = mos_labels[obj_mask]
                    mov_pts_mask = gt_obj_mos == 2
                    if torch.sum(mov_pts_mask) == 0:  # static object
                        continue
                    else:
                        mov_obj_num += 1

                    # metric-1: object-level iou
                    pred_obj_mos = pred_labels[obj_mask]
                    obj_conf_mat = self.ClassificationMetrics.compute_conf_mat(pred_obj_mos, gt_obj_mos)
                    obj_iou = self.ClassificationMetrics.get_iou(obj_conf_mat)
                    obj_mov_iou = obj_iou[2].item()
                    self.object_iou_list.append(obj_mov_iou)

                # metric-1: object-level iou
                self.mov_obj_num += mov_obj_num

                # metric-2: point-level iou
                sample_conf_mat = self.ClassificationMetrics.compute_conf_mat(pred_labels, mos_labels)
                sample_iou = self.ClassificationMetrics.get_iou(sample_conf_mat)
                sample_mov_iou = sample_iou[2].item()
                self.accumulated_conf_mat += sample_conf_mat

                # save predicted labels
                mos_pred_file = os.path.join(self.pred_dir, f"{scan_idx}.label")
                pred_labels = torch.argmax(mos_probs, dim=1).type(torch.uint8).detach().cpu().numpy()
                pred_labels.tofile(mos_pred_file)

                # logger
                self.test_logger.info(f"Val scan: %s, Sample IoU: %.3f, Moving obj num: %d",
                                      scan_idx, sample_mov_iou * 100, mov_obj_num)

            else:
                # metric-3: sample-level iou
                sample_conf_mat = self.ClassificationMetrics.compute_conf_mat(pred_labels, mos_labels)
                sample_iou = self.ClassificationMetrics.get_iou(sample_conf_mat)
                sample_mov_iou = sample_iou[2].item()

                num_mov_pts = sample_conf_mat[2][0] + sample_conf_mat[2][1] + sample_conf_mat[2][2]
                if num_mov_pts != 0:
                    self.sample_iou_list.append(sample_mov_iou)
                else:
                    self.no_mov_sample_num += 1

                # metric-4: point-level iou
                self.accumulated_conf_mat += sample_conf_mat

                # save predicted labels
                mos_pred_file = os.path.join(self.pred_dir, f"{scan_idx}_mos_pred.label")
                pred_labels = torch.argmax(mos_probs, dim=1).type(torch.uint8).detach().cpu().numpy()
                pred_labels.tofile(mos_pred_file)

                # logger
                self.test_logger.info(f"Val sd tok: %s, Sample IoU: %.3f", scan_idx, sample_mov_iou * 100)
        torch.cuda.empty_cache()


#######################################
# Modules
#######################################


class MOSModel(nn.Module):
    def __init__(self, cfg_model: dict, n_mos_cls: int):
        super().__init__()

        # backbone network
        self.MinkUNet = MinkUNetBackbone(in_channels=1, out_channels=n_mos_cls, D=cfg_model['pos_dim'])

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

        # TODO: point-wise sparse feature output
        sparse_featmap = sparse_output.slice(tensor_field)
        sparse_featmap.coordinates[:, 1:] = torch.mul(sparse_featmap.coordinates[:, 1:], self.quant)
        return sparse_featmap


class MOSHead(nn.Module):
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
            ('final', nn.Linear(planes[3], planes[4])),
        ]))

    def forward(self, x):
        if type(x) is list:  # input batch data, not concatenated data
            probs_list = []
            for x_i in x:
                probs_list.append(self.block(x_i))
            return probs_list
        else:
            return self.block(x)






