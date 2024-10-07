import datetime
import logging
import os
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock
from lib.minkowski.minkunet import MinkUNetBase

# occlusion rendering decoder
from models.ours_depth_rendering.occlusion_decoder import OcclusionDecoder

# evaluation
from models.occ4d.occ4d_evaluation import compute_chamfer_distance, compute_chamfer_distance_inner, compute_ray_errors


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


def point_quantization(batch, cfg):
    # unpack batch
    (meta_info, in_points_4d, out_origin, out_points_4d) = batch
    device = in_points_4d.device

    # scene scaling
    scene_bbox = cfg["data"]["scene_bbox"]
    voxel_size = cfg["data"]["voxel_size"]
    scene_offset = torch.nn.parameter.Parameter(
        torch.Tensor([scene_bbox[0], scene_bbox[1], scene_bbox[2]])[None, None, :], requires_grad=False).to(device)
    scene_offset_t = torch.nn.parameter.Parameter(
        torch.Tensor([scene_bbox[0], scene_bbox[1], scene_bbox[2], 0.0])[None, None, :], requires_grad=False).to(device)
    scene_scaler = torch.nn.parameter.Parameter(
        torch.Tensor([voxel_size, voxel_size, voxel_size])[None, None, :], requires_grad=False).to(device)
    scene_scaler_t = torch.nn.parameter.Parameter(
        torch.Tensor([voxel_size, voxel_size, voxel_size, 1.0])[None, None, :], requires_grad=False).to(device)

    # quantization
    in_points_quant = torch.div((in_points_4d - scene_offset_t), scene_scaler_t)
    out_origin_quant = torch.div((out_origin - scene_offset), scene_scaler).float()
    out_points_quant = torch.div((out_points_4d - scene_offset_t), scene_scaler_t).float()
    return meta_info, in_points_quant, out_origin_quant, out_points_quant


#######################################
# Torch Modules
#######################################
class MinkUNet14(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)
    # PLANES = (8, 16, 32, 64, 64, 32, 16, 8)
    PLANES = (8, 32, 128, 256, 256, 128, 32, 8)
    INIT_DIM = 8


class SparseEncoder4D(nn.Module):
    def __init__(self, cfg, feat_volume_shape, in_channels, out_channels):
        super(SparseEncoder4D, self).__init__()
        self.cfg = cfg
        self.voxel_size = self.cfg["data"]["voxel_size"]
        self.feat_volume_shape = feat_volume_shape
        self.MinkUNet = MinkUNet14(in_channels=in_channels, out_channels=out_channels, D=4)
        self.quantization = torch.Tensor([self.voxel_size, self.voxel_size, self.voxel_size, 1.0]).to(device='cuda')

    def forward(self, batch):
        [B, F, T, Z, Y, X] = self.feat_volume_shape

        # quantization
        meta_info, in_points_quant, out_origin_quant, out_points_quant = point_quantization(batch, self.cfg)

        # sparse collate, TODO: need a feature initialization method here
        features = torch.ones(in_points_quant.shape[0], in_points_quant.shape[1], F).type_as(in_points_quant)
        coords, feats = ME.utils.sparse_collate([torch.squeeze(in_points_quant)], [torch.squeeze(features)])
        tensor_field = ME.TensorField(features=feats, coordinates=coords, quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        s_input = tensor_field.sparse()
        # s_input = ME.SparseTensor(coordinates=coords, features=feats, quantization_mode=QuantMode.UNWEIGHTED_AVERAGE)

        # model prediction
        # TODO: check feat volume dimension, minkowski sparse tensor [coord=[B, X, Y, Z, T]; feat=[F]]
        s_pred = self.MinkUNet(s_input)  # B, X, Y, Z, T; F
        s_out_coords = s_pred.slice(tensor_field).coordinates
        s_out_feats = s_pred.slice(tensor_field).features

        # to dense feature volume
        # TODO: check feat volume dimension, minkowski dense tensor [B, F, X, Y, Z, T]
        feat_volume, min_coord, tensor_stride = s_pred.dense(shape=torch.Size([B, F, X, Y, Z, T]),
                                              min_coordinate=torch.IntTensor([0, 0, 0, 0]), contract_stride=True)

        # TODO: check feat volume dimension, torch conv dense tensor [B, F, T, Z, Y, X]
        feat_volume = torch.squeeze(feat_volume.permute(0, 1, 5, 4, 3, 2).contiguous()).reshape(B, F, T, Z, Y, X)
        return feat_volume


#######################################
# Lightning Modules
#######################################
class MinkOccForecastNet(LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        # dataset params
        self.dataset = self.cfg["data"]["dataset_name"].lower()
        assert self.cfg["data"]["fgbg_label"] is False

        # 4d scene params
        self.voxel_size = cfg["data"]["voxel_size"]
        self.scene_bbox = cfg["data"]["scene_bbox"]  # list
        self.t_scans = cfg["data"]["t_scans"]
        self.z_height = int((self.scene_bbox[5] - self.scene_bbox[2]) / self.voxel_size)
        self.y_length = int((self.scene_bbox[4] - self.scene_bbox[1]) / self.voxel_size)
        self.x_width = int((self.scene_bbox[3] - self.scene_bbox[0]) / self.voxel_size)
        # TODO: add timestamp to scene_bbox, [-5, -4, -3, -2, -1, 0 | 1, 2, 3, 4, 5, 6]
        self.quantization = torch.nn.parameter.Parameter(
            torch.Tensor([self.voxel_size, self.voxel_size, self.voxel_size, 1.0]), requires_grad=False
        )

        # loss params
        self.loss_type = self.cfg["model"]["loss_type"].lower()
        assert self.loss_type in ["l1", "l2", "absrel"]
        self.loss_weight = self.cfg["model"]["loss_weight"]  # list

        # model params
        self.b_batch = self.cfg["model"]["batch_size"]
        assert self.b_batch == 1  # only batch size 1 available now
        self.f_feat = self.cfg["data"]["feat_dim"]
        self.feat_volume_shape = [self.b_batch, self.f_feat, self.t_scans, self.z_height, self.y_length, self.x_width]
        print("shape of feature volume:", self.feat_volume_shape)

        # 4d sparse encoder
        self.encoder = SparseEncoder4D(self.cfg, self.feat_volume_shape, in_channels=self.f_feat, out_channels=self.f_feat)

        # rendering based decoder
        self.decoder = OcclusionDecoder(self.cfg, self.feat_volume_shape)

        # NOTE: initialize the linear predictor (no bias) over history
        self.linear = torch.nn.Conv3d(in_channels=self.f_feat * self.t_scans, out_channels=self.f_feat * self.t_scans,
                                      kernel_size=3, stride=1, padding=1, bias=True)

        # pytorch-lightning training params
        if self.cfg["mode"] == "pretrain" or "finetune":
            self.save_hyperparameters(cfg)
            self.lr_start = self.cfg["model"]["lr_start"]
            self.lr_epoch = self.cfg["model"]["lr_epoch"]
            self.lr_decay = self.cfg["model"]["lr_decay"]
            self.automatic_optimization = True  # TODO: activate manual optimization or not
            self.training_step_outputs = []
            self.validation_step_outputs = []
            self.iters_acc_loss = 0
        elif self.cfg["mode"] == "test":
            # visualization directory
            model_dataset = self.cfg["model"]["model_dataset"]
            model_name = self.cfg["model"]["model_name"]  # for test.cfg only
            model_version = self.cfg["model"]["model_version"]
            test_epoch = self.cfg["model"]["test_epoch"]
            model_dir = os.path.join("../../logs", "pretrain", model_dataset, model_name, model_version)
            self.vis_dir = os.path.join(model_dir, "results", f"epoch_{test_epoch}", "visualization")
            # log file directory
            date = datetime.date.today().strftime('%Y%m%d')
            self.log_file = os.path.join(model_dir, "results", f"epoch_{test_epoch}/{date}.txt")


    def forward(self, batch):
        ###########################################################################################
        # # generate dense occ input, w/ linear skip connection layer
        # assert self.batch_size == len(_input)
        # past_point_clouds = [torch.div(point_cloud, self.quantization) for point_cloud in _input]
        # features = [
        #     torch.ones(len(point_cloud), self.feat_dim).type_as(point_cloud)
        #     for point_cloud in past_point_clouds
        # ]  # TODO: refer to SHINE-mapping, use random feature initialization, feat_dim=8 or 32
        # coords, feats = ME.utils.sparse_collate(past_point_clouds, features)
        # s_input = ME.SparseTensor(coordinates=coords, features=feats,
        #                           quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        # d_input, _, _ = s_input.dense(shape=torch.Size([B, F, X, Y, Z, T]),
        #                               min_coordinate=torch.IntTensor([0, 0, 0, 0]))
        # # reshape B-0, F-1, X-2, Y-3, Z-4, T-5 to the output of original 4docc, B, F * T, Z, Y, X
        # d_input = d_input.permute(0, 1, 5, 4, 3, 2).contiguous()
        # d_input = torch.reshape(d_input, (B, F * T, Z, Y, X))
        # _li_output = self.linear(d_input)
        # _li_output = torch.reshape(_li_output, (B, F, T, Z, Y, X))
        ###########################################################################################

        # 4d sparse encoder
        curr_feat_volume = self.encoder(batch)
        # depth + occlusion rendering based decoder
        rendered_dict = self.decoder(batch=batch, curr_feat_volume=curr_feat_volume)
        return rendered_dict


    def get_losses(self, rendered_dict):
        loss_dict = {}
        if self.loss_weight.get("depth", 0.0) > 0.0:
            depth_loss = 0
            T = len(rendered_dict["gt_depth"])
            for t_idx in range(T):
                gt_depth = rendered_dict["gt_depth"][t_idx]
                rendered_depth = rendered_dict["depth"][t_idx]
                valid_depth_mask = gt_depth >= 0
                valid_depth_cnt = torch.sum(valid_depth_mask) if torch.sum(valid_depth_mask) != 0 else 1
                depth_loss += torch.sum((gt_depth - rendered_depth)[valid_depth_mask] ** 2) / valid_depth_cnt
            loss_dict["depth_loss"] = depth_loss

        if self.loss_weight.get("occlusion", 0.0) > 0.0:
            gt_occlusion = rendered_dict["gt_occlusion"][t_idx]
            rendered_occlusion = [0.0]
            valid_occlusion_mask = gt_occlusion >= 0.5
            valid_occlusion_cnt = torch.sum(valid_occlusion_mask) if torch.sum(valid_occlusion_mask) != 0 else 1
            occlusion_loss = torch.sum((gt_occlusion - rendered_occlusion)[valid_occlusion_mask] ** 2) / valid_occlusion_cnt
            loss_dict["occlusion_loss"] = occlusion_loss
        return loss_dict


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_start)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_epoch, gamma=self.lr_decay)
        return [optimizer], [scheduler]


    def training_step(self, batch: tuple, batch_idx):
        # network forward: sparse encoder + rendering decoder
        rendered_dict = self.forward(batch)

        # compute training losses
        # TODO: normalized loss? or loss in meters
        loss_dict = self.get_losses(rendered_dict)
        depth_loss = loss_dict["depth_loss"]

        # log training losses
        self.log("depth_loss", depth_loss.item(), on_step=True)  # logger=True default
        self.training_step_outputs.append({"depth_loss": depth_loss.item()})

        # iters accumulated loss
        self.iters_acc_loss += depth_loss / 50
        if (self.global_step + 1) % 50 == 0:  # self.current_batch
            self.log("train_50_iters_acc_loss", self.iters_acc_loss, prog_bar=True)
            self.iters_acc_loss = 0

        # # manual backward and optimization
        # opt = self.optimizers()
        # opt.zero_grad()
        # self.manual_backward(sigma, gradient=grad_sigma)
        # opt.step()
        torch.cuda.empty_cache()
        return depth_loss


    def on_train_epoch_end(self):
        # step lr per each epoch
        sch = self.lr_schedulers()
        sch.step()

        # epoch logging
        epoch_loss_list = []
        for train_loss_dict in self.training_step_outputs:
            epoch_loss_list.append(train_loss_dict[f"train_{self.loss_type}_loss"])
        self.log("train_epoch_loss", np.array(epoch_loss_list).mean(), on_epoch=True, prog_bar=True)

        # clear
        self.training_step_outputs = []

    def validation_step(self, batch: tuple, batch_idx):
        # unpack batch data
        if self.cfg["data"]["fgbg_label"]:
            net_input, output_origin, output_points, output_tindex, output_labels, _ = self.unpack_batch(batch, self.cfg)
        else:
            net_input, output_origin, output_points, output_tindex, _ = self.unpack_batch(batch, self.cfg)

        # forward
        net_output = self.forward(net_input)

        # compute validation losses
        l1_loss, l2_loss, absrel_loss = self.get_losses(gt_dist, pred_dist)

        # store validation loss
        val_loss_dict = {"val_l1_loss": l1_loss.item(),
                         "val_l2_loss": l1_loss.item(),
                         "val_absrel_loss": absrel_loss.item()}  # .item(): tensor -> float
        self.log("val_l1_loss_step", l1_loss.item(), on_step=True, prog_bar=True, logger=False)
        self.validation_step_outputs.append(val_loss_dict)
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        val_l1_loss_list = []
        val_l2_loss_list = []
        val_absrel_loss_list = []
        for val_loss_dict in self.validation_step_outputs:
            val_l1_loss_list.append(val_loss_dict["val_l1_loss"])
            val_l2_loss_list.append(val_loss_dict["val_l2_loss"])
            val_absrel_loss_list.append(val_loss_dict["val_absrel_loss"])
        val_l1_loss = np.array(val_l1_loss_list).mean()
        val_l2_loss = np.array(val_l2_loss_list).mean()
        val_absrel_loss = np.array(val_absrel_loss_list).mean()
        self.log("val_l1_loss", val_l1_loss, on_epoch=True, prog_bar=True)
        self.log("val_l2_loss", val_l2_loss, on_epoch=True)
        self.log("val_absrel_loss", val_absrel_loss, on_epoch=True)

        # clear
        self.validation_step_outputs = []

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # unpack batch data
        if self.cfg["data"]["fgbg_label"]:
            net_input, output_origin, output_points, output_tindex, output_labels, meta_info = self.unpack_batch(batch, self.cfg)
        else:
            net_input, output_origin, output_points, output_tindex, meta_info = self.unpack_batch(batch, self.cfg)

        # if assume constant velocity
        if self.cfg["model"]["assume_const_velo"]:
            # meta_info: ref_sd_token, displacement
            # displacement = input_origin[current_index] - input_origin[current_index - 1]
            displacement = torch.concat([fname[-1] for fname in meta_info]).reshape((-1, 1, 3))
            output_origin = torch.zeros_like(output_origin)
            displacements = (torch.arange(self.t_scans) + 1).to(self.device)
            output_origin = (output_origin + displacements[None, :, None]) * displacement

        # forward
        net_output = self.forward(net_input)
        pog = 1 - torch.exp(-sigma)

        # visualize occupancy and predicted point cloud
        os.makedirs(self.vis_dir, exist_ok=True)
        pred_pcds_dir = os.path.join(self.vis_dir, "pred_pcds")
        os.makedirs(pred_pcds_dir, exist_ok=True)
        gt_pcds_dir = os.path.join(self.vis_dir, "gt_pcds")
        os.makedirs(gt_pcds_dir, exist_ok=True)
        occ_pred_dir = os.path.join(self.vis_dir, "occ_pred")
        os.makedirs(occ_pred_dir, exist_ok=True)
        # occ_pcd = get_occupancy_as_pcd(pog, 0.01, self.voxel_size, self.scene_bbox, "Oranges")

        # logging
        logging.basicConfig(filename=self.log_file, level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(self.cfg))
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
                self.scene_bbox,
                self.cfg["model"]["eval_within_grid"],
                self.cfg["model"]["eval_outside_grid"])
            gt_pcds = get_clamped_output(
                output_origin[i].cpu().numpy(),
                output_points[i].cpu().numpy(),
                output_tindex[i].cpu().numpy(),
                self.scene_bbox,
                gt_dist[i].cpu().numpy(),
                self.cfg["model"]["eval_within_grid"],
                self.cfg["model"]["eval_outside_grid"])

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
                if self.cfg["model"]["write_pcd"]:
                    import open3d
                    pred_pcd_file = os.path.join(pred_pcds_dir, f"{meta_info[i][2]}_pred.pcd")
                    o3d_pred_pcd = open3d.geometry.PointCloud()
                    o3d_pred_pcd.points = open3d.utility.Vector3dVector(pred_pcd.numpy())
                    open3d.io.write_point_cloud(pred_pcd_file, o3d_pred_pcd)
                    # print(f"Predicted pcd saved: {meta_info[i][2]}_pred.pcd")
                    gt_pcd_file = os.path.join(gt_pcds_dir, f"{meta_info[i][2]}_gt.pcd")
                    o3d_gt_pcd = open3d.geometry.PointCloud()
                    o3d_gt_pcd.points = open3d.utility.Vector3dVector(gt_pcd.numpy())
                    open3d.io.write_point_cloud(gt_pcd_file, o3d_gt_pcd)
                    # print(f"Ground truth pcd saved: {meta_info[i][2]}_gt.pcd")

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
