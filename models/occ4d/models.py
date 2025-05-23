import logging
import os
import sys
from collections import OrderedDict
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import MinkowskiEngine as ME
from models.backbone import MinkUNetBackbone
from models.occ4d.occ4d_evaluation import compute_chamfer_distance, compute_chamfer_distance_inner, compute_ray_errors
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

# JIT
from torch.utils.cpp_extension import load
dvr = load("dvr", sources=["./lib/dvr/dvr.cpp", "./lib/dvr/dvr.cu"], verbose=True,
           extra_cuda_cflags=['-allow-unsupported-compiler'])


#######################################
# Lightning Modules
#######################################


class Occ4dNetwork(LightningModule):
    def __init__(self, cfg_model: dict, train_flag: bool, **kwargs):
        super().__init__()
        if train_flag:
            self.save_hyperparameters(cfg_model)

        # params
        self.cfg_model = cfg_model
        self.loss_type = cfg_model["loss_type"]
        assert self.loss_type in ["l1", "l2", "absrel"]
        self.scene_bbox = cfg_model['scene_bbox']
        self.t_size = cfg_model['n_input']
        self.b_size = cfg_model["batch_size"]

        # featmap
        self.offset = torch.nn.parameter.Parameter(
            torch.Tensor([self.scene_bbox[0], self.scene_bbox[1], self.scene_bbox[2]])[None:],
            requires_grad=False)
        self.featmap_size = cfg_model['featmap_size']
        self.z_height = int((self.scene_bbox[5] - self.scene_bbox[2]) / self.featmap_size)  # 45
        self.y_length = int((self.scene_bbox[4] - self.scene_bbox[1]) / self.featmap_size)  # 700
        self.x_width = int((self.scene_bbox[3] - self.scene_bbox[0]) / self.featmap_size)  # 700
        self.featmap_shape = [self.b_size, self.x_width, self.y_length, self.z_height, self.t_size]
        self.feat_dim = self.z_height * self.t_size  # 45 * 6 = 270

        # encoder, decoder, dvr
        self.iters_per_epoch = kwargs['iters_per_epoch']
        self.encoder = OccEncoder(cfg_model=self.cfg_model, feat_dim=self.feat_dim)
        self.decoder = OccDenseDecoder(cfg_model=self.cfg_model, featmap_shape=self.featmap_shape)
        self.dvr = DifferentiableVolumeRendering(loss_type=self.loss_type, voxel_size=self.featmap_size,
                                                 output_grid=[self.t_size, self.z_height, self.y_length, self.x_width])

        # manual optimization
        self.automatic_optimization = False

        # pytorch lightning training output
        self.epoch_acc_l1_loss = 0
        self.epoch_acc_l2_loss = 0
        # self.epoch_acc_absrel_loss = 0
        self.iters_acc_loss = 0

        # save predictions
        if not train_flag:
            self.model_dir = kwargs['model_dir']
            self.test_epoch = kwargs['eval_epoch']
            self.pred_dir = os.path.join(self.model_dir, "predictions", f"epoch_{self.test_epoch}")
            os.makedirs(self.pred_dir, exist_ok=True)

    def forward(self, batch):
        # unpack batch data
        meta_batch, pcds_batch, occ4d_future_batch = batch
        future_org_batch = []
        future_pcd_batch = []
        future_tindex_batch = []

        # quantize future orgs and pcds
        # max_num_pcd = max([len(occ4d_sample[1]) for occ4d_sample in occ4d_future_batch])
        for occ4d_sample in occ4d_future_batch:
            # quantization
            future_orgs, future_pcds, future_tindex = occ4d_sample
            future_orgs_grid = torch.div(future_orgs - self.offset, self.featmap_size, rounding_mode=None)
            future_pcds_grid = torch.div(future_pcds - self.offset, self.featmap_size, rounding_mode=None)

            # # padding nan and -1, for batch stack
            # future_pcds_grid_pad = F.pad(future_pcds_grid, (0, 0, 0, max_num_pcd - len(future_pcds_grid)),
            #                              mode="constant", value=float("nan"))
            # future_tindex_pad = F.pad(future_tindex, (0, max_num_pcd - len(future_tindex)),
            #                              mode="constant", value=-1)

            # append
            future_org_batch.append(future_orgs_grid)
            future_pcd_batch.append(future_pcds_grid)
            future_tindex_batch.append(future_tindex)
        future_org = torch.stack(future_org_batch)  # B, T, 3
        future_pcd = torch.stack(future_pcd_batch)  # B, N, 3
        future_tindex = torch.stack(future_tindex_batch)  # B, N, 3

        # encoder
        sparse_featmap = self.encoder(pcds_batch)

        # decoder
        dense_occ_sigma = self.decoder(sparse_featmap)  # B, T, Z, Y, X

        # dvr depth rendering
        dense_occ_sigma, pred_dist, gt_dist, grad_sigma = self.dvr.dvr_render(dense_occ_sigma, future_org, future_pcd, future_tindex)
        return dense_occ_sigma, pred_dist, gt_dist, grad_sigma

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

    def get_loss(self, gt_dist, pred_dist):
        # valid distance
        valid = gt_dist >= 0
        gt_dist = gt_dist[valid]
        pred_dist = pred_dist[valid]
        # get loss
        l1_loss = torch.mean(torch.abs(gt_dist - pred_dist))
        l2_loss = torch.mean(torch.pow(gt_dist - pred_dist, 2) / 2)
        # absrel_loss = torch.mean(torch.abs(gt_dist - pred_dist) / (gt_dist + 1e-15))
        return l1_loss, l2_loss

    def training_step(self, batch: tuple, batch_idx, dataloader_index=0):
        # encoder, decoder, volume rendering
        dense_occ_sigma, pred_dist, gt_dist, grad_sigma = self.forward(batch)

        # get loss
        l1_loss, l2_loss = self.get_loss(gt_dist, pred_dist)

        # logging
        self.log("l1_loss", l1_loss.item(), on_step=True, prog_bar=True, logger=True)
        self.log("l2_loss", l2_loss.item(), on_step=True, prog_bar=True, logger=True)
        # self.log("absrel_loss", absrel_loss.item(), on_step=True, prog_bar=True, logger=True)
        self.epoch_acc_l1_loss += l1_loss.item()
        self.epoch_acc_l2_loss += l2_loss.item()
        # self.epoch_acc_absrel_loss += absrel_loss.item()

        # iters accumulated loss
        self.iters_acc_loss += l1_loss.item() / 50
        if (self.global_step + 1) % 50 == 0:  # self.current_batch
            self.log("train_50_iters_acc_loss", self.iters_acc_loss, on_step=True, prog_bar=True, logger=True)
            self.iters_acc_loss = 0

        # step learning rate scheduler
        sch = self.lr_schedulers()
        sch.step()

        # manual optimization
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(dense_occ_sigma, gradient=grad_sigma)
        opt.step()

        # clear cuda memory
        del dense_occ_sigma, pred_dist, gt_dist, grad_sigma
        torch.cuda.empty_cache()

    def on_train_epoch_end(self):
        # epoch logging
        epoch_l1_loss = self.epoch_acc_l1_loss / len(self.train_dataloader())
        epoch_l2_loss = self.epoch_acc_l2_loss / len(self.train_dataloader())
        # epoch_absrel_loss = self.epoch_acc_absrel_loss / len(self.train_dataloader())
        self.log("epoch_l1_loss", epoch_l1_loss, on_epoch=True, prog_bar=True)
        self.log("epoch_l2_loss", epoch_l2_loss, on_epoch=True, prog_bar=True)
        # self.log("epoch_absrel_loss", epoch_absrel_loss, on_epoch=True, prog_bar=True)

        # clear cuda memory
        self.epoch_acc_l1_loss = 0
        self.epoch_acc_l2_loss = 0
        # self.epoch_acc_absrel_loss = 0
        torch.cuda.empty_cache()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:  # TODO: need to rewrite
        # unpack batch data
        if self.cfg_model["data"]["fgbg_label"]:
            net_input, output_origin, output_points, output_tindex, output_labels, meta = self.unpack_batch(batch)
        else:
            net_input, output_origin, output_points, output_tindex, meta = self.unpack_batch(batch)

        # if assume constant velocity
        if self.cfg_model["model"]["assume_const_velo"]:
            # meta: (ref_scene_token, ref_sample_token, ref_sd_token, displacement)
            # displacement = input_origin[current_index] - input_origin[current_index - 1]
            displacement = torch.concat([fname[-1] for fname in meta]).reshape((-1, 1, 3))
            output_origin = torch.zeros_like(output_origin)
            displacements = (torch.arange(self.n_output) + 1).to(self.device)
            output_origin = (output_origin + displacements[None, :, None]) * displacement

        # forward (dvr.render_forward)
        net_output = self.forward(net_input)
        sigma, pred_dist, gt_dist = self.dvr_render_forward(net_output, output_origin, output_points, output_tindex)
        pog = 1 - torch.exp(-sigma)

        # visualize occupancy and predicted point cloud
        os.makedirs(self.vis_dir, exist_ok=True)
        pred_pcds_dir = os.path.join(self.vis_dir, "pred_pcds")
        os.makedirs(pred_pcds_dir, exist_ok=True)
        gt_pcds_dir = os.path.join(self.vis_dir, "gt_pcds")
        os.makedirs(gt_pcds_dir, exist_ok=True)
        occ_pred_dir = os.path.join(self.vis_dir, "occ_pred")
        os.makedirs(occ_pred_dir, exist_ok=True)
        # occ_pcd = get_occupancy_as_pcd(pog, 0.01, self.voxel_size, self.pc_range, "Oranges")

        # logging
        logging.basicConfig(filename=self.log_file, level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(self.cfg_model))
        logging.info(self.log_file)

        # evaluation
        metrics = {
            "count": 0.0,
            "chamfer_distance": 0.0,
            "chamfer_distance_inner": 0.0,
            "l1_error": 0.0,
            "absrel_error": 0.0
        }
        # iterate through the batch
        for i in range(len(output_points)):
            pred_pcds = get_rendered_pcds(
                output_origin[i].cpu().numpy(),
                output_points[i].cpu().numpy(),
                output_tindex[i].cpu().numpy(),
                gt_dist[i].cpu().numpy(),
                pred_dist[i].cpu().numpy(),
                self.pc_range,
                self.cfg_model["model"]["eval_within_grid"],
                self.cfg_model["model"]["eval_outside_grid"])
            gt_pcds = get_clamped_output(
                output_origin[i].cpu().numpy(),
                output_points[i].cpu().numpy(),
                output_tindex[i].cpu().numpy(),
                self.pc_range,
                gt_dist[i].cpu().numpy(),
                self.cfg_model["model"]["eval_within_grid"],
                self.cfg_model["model"]["eval_outside_grid"])

            # load predictions (loop in time-axis)
            for j in range(len(gt_pcds)):
                pred_pcd = pred_pcds[j]
                gt_pcd = gt_pcds[j]
                origin = output_origin[i][j].cpu().numpy()

                # get the metrics
                metrics["count"] += 1
                metrics["chamfer_distance"] += compute_chamfer_distance(pred_pcd, gt_pcd, self.device)
                metrics["chamfer_distance_inner"] += compute_chamfer_distance_inner(pred_pcd, gt_pcd, self.device)
                l1_error, absrel_error = compute_ray_errors(pred_pcd, gt_pcd, torch.from_numpy(origin), self.device)
                metrics["l1_error"] += l1_error
                metrics["absrel_error"] += absrel_error

                # save pred_pcd as [sample_data_token]_pred.pcd
                if self.cfg_model["model"]["write_pcd"]:
                    import open3d
                    pred_pcd_file = os.path.join(pred_pcds_dir, f"{meta[i][2]}_pred.pcd")
                    o3d_pred_pcd = open3d.geometry.PointCloud()
                    o3d_pred_pcd.points = open3d.utility.Vector3dVector(pred_pcd.numpy())
                    open3d.io.write_point_cloud(pred_pcd_file, o3d_pred_pcd)
                    # print(f"Predicted pcd saved: {meta[i][2]}_pred.pcd")
                    gt_pcd_file = os.path.join(gt_pcds_dir, f"{meta[i][2]}_gt.pcd")
                    o3d_gt_pcd = open3d.geometry.PointCloud()
                    o3d_gt_pcd.points = open3d.utility.Vector3dVector(gt_pcd.numpy())
                    open3d.io.write_point_cloud(gt_pcd_file, o3d_gt_pcd)
                    # print(f"Ground truth pcd saved: {meta[i][2]}_gt.pcd")

        count = metrics["count"]
        chamfer_distance = metrics["chamfer_distance"]
        chamfer_distance_inner = metrics["chamfer_distance_inner"]
        l1_error = metrics["l1_error"]
        absrel_error = metrics["absrel_error"]
        logging.info(f"\nBatch {i}, Chamfer Distance: {chamfer_distance / count}")
        logging.info(f"\nBatch {i}, Chamfer Distance Inner: {chamfer_distance_inner / count}")
        logging.info(f"\nBatch {i}, L1 Error: {l1_error / count}")
        logging.info(f"\nBatch {i}, AbsRel Error: {absrel_error / count}")
        logging.info(f"\nBatch {i}, Count: {count}")


#######################################
# Torch Modules
#######################################


class OccEncoder(nn.Module):
    def __init__(self, cfg_model: dict, feat_dim):
        super().__init__()
        # quantization offset
        self.scene_bbox = cfg_model["scene_bbox"]
        self.offset = torch.nn.parameter.Parameter(
            torch.Tensor([self.scene_bbox[0], self.scene_bbox[1], self.scene_bbox[2], -cfg_model['n_input'] + 1])[None:],
            requires_grad=False)

        # input point cloud quantization
        self.quant_size_s = cfg_model["quant_size"]
        self.quant_size_t = 1  # TODO: cfg_model["time_interval"]
        self.quant = torch.Tensor([self.quant_size_s, self.quant_size_s, self.quant_size_s, self.quant_size_t])

        # backbone network
        self.feat_dim = feat_dim
        self.MinkUNet = MinkUNetBackbone(in_channels=1, out_channels=self.feat_dim, D=cfg_model['pos_dim'])


    def forward(self, pcds_4d_batch):
        # quantized 4d pcd and initialized features
        self.quant = self.quant.type_as(pcds_4d_batch[0])
        quant_4d_pcds = [torch.div(pcd - self.offset, self.quant, rounding_mode=None) for pcd in pcds_4d_batch]
        feats = [torch.full((len(pcd), 1), 0.5).type_as(pcd) for pcd in pcds_4d_batch]

        # sparse collate, tensor field, net calculation
        coords, feats = ME.utils.sparse_collate(quant_4d_pcds, feats)
        sparse_input = ME.SparseTensor(features=feats, coordinates=coords,
                                      quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

        # # TODO: filter quantized input which outside quantized scene bbox ############################################
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
        # ##############################################################################################################

        # forward
        sparse_featmap = self.MinkUNet(sparse_input)
        return sparse_featmap


class OccDenseDecoder(nn.Module):
    def __init__(self, cfg_model: dict, featmap_shape):
        super().__init__()
        # features params
        self.b_size, self.x_width, self.y_length, self.z_height, self.t_size = tuple(featmap_shape)
        self.feat_dim = self.t_size * self.z_height
        self.pos_dim = cfg_model['pos_dim']

        # extra conv for feature map size correction
        self.conv0k3s2 = ME.MinkowskiConvolution(self.feat_dim, self.feat_dim,
                                                 kernel_size=3, stride=[2, 2, 2, 1], dimension=self.pos_dim)
        self.bn0 = ME.MinkowskiBatchNorm(self.feat_dim)
        self.conv1k3s1 = ME.MinkowskiConvolution(self.feat_dim, self.feat_dim,
                                                 kernel_size=3, stride=[1, 1, 1, 1], dimension=self.pos_dim)
        self.bn1 = ME.MinkowskiBatchNorm(self.feat_dim)

        # pooling block (for z and t dimension)
        self.avg_pool_t = ME.MinkowskiSumPooling(kernel_size=[1, 1, 1, self.t_size],
                                                 stride=[1, 1, 1, self.t_size], dimension=self.pos_dim)
        self.avg_pool_z = ME.MinkowskiSumPooling(kernel_size=[1, 1, self.z_height, 1],
                                                 stride=[1, 1, self.z_height, 1], dimension=self.pos_dim)

        # dense block, for denser feature map, output feature map size o = (i - k) + 2p + 1
        self.dense_conv_planes = [self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim]
        self.dense_conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(self.feat_dim, self.dense_conv_planes[0], kernel_size=3, padding=1, stride=1)),
            ('bn0', nn.BatchNorm2d(self.dense_conv_planes[0])),
            ('relu0', nn.ReLU(inplace=False)),
            ('conv1', nn.Conv2d(self.dense_conv_planes[0], self.dense_conv_planes[1], kernel_size=3, padding=1, stride=1)),
            ('bn1', nn.BatchNorm2d(self.dense_conv_planes[1])),
            ('relu1', nn.ReLU(inplace=False)),
            ('conv2', nn.Conv2d(self.dense_conv_planes[1], self.dense_conv_planes[2], kernel_size=3, padding=1, stride=1)),
            ('bn2', nn.BatchNorm2d(self.dense_conv_planes[2])),
            ('relu2', nn.ReLU(inplace=False)),
            ('conv3', nn.Conv2d(self.dense_conv_planes[2], self.dense_conv_planes[3], kernel_size=3, padding=1, stride=1)),
            ('bn3', nn.BatchNorm2d(self.dense_conv_planes[3])),
            ('relu3', nn.ReLU(inplace=False)),
            ('conv4', nn.Conv2d(self.dense_conv_planes[3], self.dense_conv_planes[4], kernel_size=3, padding=1, stride=1)),
        ]))

    def forward(self, sparse_featmap):
        # extra conv layer (feature map size correction)
        sparse_featmap_s2 = self.conv0k3s2(sparse_featmap)  # tensor stride = 2
        sparse_featmap_s2 = self.bn0(sparse_featmap_s2)
        sparse_featmap_s2 = self.conv1k3s1(sparse_featmap_s2)
        sparse_featmap_s2 = self.bn1(sparse_featmap_s2)

        # avg pooling (for z and t dimension)
        sparse_featmap_xyz = self.avg_pool_t(sparse_featmap_s2)
        sparse_featmap_xy = self.avg_pool_z(sparse_featmap_xyz)

        # sparse to dense
        dense_featmap_xy, _, _ = sparse_featmap_xy.dense(shape=torch.Size([self.b_size, self.feat_dim, self.x_width, self.y_length, 1, 1]),
                                                         min_coordinate=torch.IntTensor([0, 0, 0, 0]))
        dense_featmap_xy = torch.squeeze(torch.squeeze(dense_featmap_xy, -1), -1)  # B, F, X, Y

        # dense conv layers (increase receptive field)
        dense_featmap_xy = self.dense_conv_block(dense_featmap_xy)  # B, F, X, Y

        # dense occupancy logits (logits: before activation; probs: after activation)
        dense_occ_logits = (torch.permute(dense_featmap_xy, (0, 2, 3, 1)).contiguous()
                            .view(self.b_size, self.x_width, self.y_length, self.z_height, self.t_size))
        dense_occ_sigma = F.relu(dense_occ_logits, inplace=True)  # B, X, Y, Z, T
        dense_occ_sigma = torch.permute(dense_occ_sigma, (0, 4, 3, 2, 1)).contiguous()  # B, T, Z, Y, X
        return dense_occ_sigma


class DifferentiableVolumeRendering(nn.Module):
    def __init__(self, loss_type, voxel_size, output_grid):
        super().__init__()
        self.loss_type = loss_type
        self.voxel_size = voxel_size
        self.output_grid = output_grid

    def dvr_render(self, dense_occ_sigma, future_org, future_pcd, future_tindex):
        assert dense_occ_sigma.requires_grad  # for training
        pred_dist, gt_dist, grad_sigma = dvr.render(
            dense_occ_sigma,
            future_org,
            future_pcd,
            future_tindex,
            self.loss_type
        )

        # take care of nans and infs if any
        invalid = torch.isnan(grad_sigma)
        grad_sigma[invalid] = 0.0
        invalid = torch.isnan(pred_dist)
        pred_dist[invalid] = -1.0
        gt_dist[invalid] = -1.0
        invalid = torch.isinf(pred_dist)
        pred_dist[invalid] = -1.0
        gt_dist[invalid] = -1.0

        # recover depth scale
        pred_dist *= self.voxel_size
        gt_dist *= self.voxel_size
        return dense_occ_sigma, pred_dist, gt_dist, grad_sigma

    def dvr_render_forward(self, dense_occ_sigma, future_org, future_pcd, future_tindex):
        assert dense_occ_sigma.requires_grad is False  # for validation or test
        pred_dist, gt_dist = dvr.render_forward(
            dense_occ_sigma,
            future_org,
            future_pcd,
            future_tindex,
            self.output_grid,
            "test"  # original setting: "train" for train and validation, "test" for test
        )

        # take care of nans if any
        invalid = torch.isnan(pred_dist)
        pred_dist[invalid] = 0.0
        gt_dist[invalid] = 0.0

        # recover depth scale
        pred_dist *= self.voxel_size
        gt_dist *= self.voxel_size
        return dense_occ_sigma, pred_dist, gt_dist


#######################################
# Static Methods
#######################################

def get_grid_mask(points_all, pc_range):
    masks = []
    for batch in range(points_all.shape[0]):
        points = points_all[batch].T
        mask1 = torch.logical_and(pc_range[0] <= points[0], points[0] < pc_range[3])
        mask2 = torch.logical_and(pc_range[1] <= points[1], points[1] < pc_range[4])
        mask3 = torch.logical_and(pc_range[2] <= points[2], points[2] < pc_range[5])
        mask = mask1 & mask2 & mask3
        masks.append(mask)
    # print("shape of mask being returned", mask.shape)
    return torch.stack(masks)


def get_rendered_pcds(origin, points, tindex, gt_dist, pred_dist, pc_range, eval_within_grid=False, eval_outside_grid=False):
    pcds = []
    for t in range(len(origin)):
        mask = np.logical_and(tindex == t, gt_dist > 0.0)
        if eval_within_grid:
            mask = np.logical_and(mask, get_grid_mask(points, pc_range))
        if eval_outside_grid:
            mask = np.logical_and(mask, ~get_grid_mask(points, pc_range))
        # skip the ones with no data
        if not mask.any():
            continue
        _pts = points[mask, :3]
        # use ground truth lidar points for the raycasting direction
        v = _pts - origin[t][None, :]
        d = v / np.sqrt((v ** 2).sum(axis=1, keepdims=True))
        pred_pts = origin[t][None, :] + d * pred_dist[mask][:, None]
        pcds.append(torch.from_numpy(pred_pts))
    return pcds


def get_clamped_output(origin, points, tindex, pc_range, gt_dist, eval_within_grid=False, eval_outside_grid=False, get_indices=False):
    pcds = []
    if get_indices:
        indices = []
    for t in range(len(origin)):
        mask = np.logical_and(tindex == t, gt_dist > 0.0)
        if eval_within_grid:
            mask = np.logical_and(mask, get_grid_mask(points, pc_range))
        if eval_outside_grid:
            mask = np.logical_and(mask, ~get_grid_mask(points, pc_range))
        # skip the ones with no data
        if not mask.any():
            continue
        if get_indices:
            idx = np.arange(points.shape[0])
            indices.append(idx[mask])
        _pts = points[mask, :3]
        # use ground truth lidar points for the raycasting direction
        v = _pts - origin[t][None, :]
        d = v / np.sqrt((v ** 2).sum(axis=1, keepdims=True))
        gt_pts = origin[t][None, :] + d * gt_dist[mask][:, None]
        pcds.append(torch.from_numpy(gt_pts))
    if get_indices:
        return pcds, indices
    else:
        return pcds



