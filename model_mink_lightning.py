from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock
from lib.minkowski.minkunet import MinkUNetBase

# JIT
from torch.utils.cpp_extension import load
dvr = load("dvr", sources=["lib/dvr/dvr.cpp", "lib/dvr/dvr.cu"], verbose=True,
           extra_cuda_cflags=['-allow-unsupported-compiler'])

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
    def __init__(self, in_channels, out_channels, voxel_size, output_grid):
        super(SparseEncoder4D, self).__init__()
        self.voxel_size = voxel_size
        self.output_grid = output_grid
        self.MinkUNet = MinkUNet14(in_channels=in_channels, out_channels=out_channels, D=4)
        self.quantization = torch.Tensor([self.voxel_size, self.voxel_size, self.voxel_size, 1.0]).to(device='cuda')

    def forward(self, input_points_4d):
        batch_size = len(input_points_4d)
        past_point_clouds = [torch.div(point_cloud, self.quantization) for point_cloud in input_points_4d]
        features = [
            torch.ones(len(point_cloud), 1).type_as(point_cloud)
            for point_cloud in past_point_clouds
        ]
        coords, feats = ME.utils.sparse_collate(past_point_clouds, features)

        tensor_field = ME.TensorField(features=feats, coordinates=coords.type_as(feats),
                                      quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
        s_input = tensor_field.sparse()
        # s_input = ME.SparseTensor(coordinates=coords, features=feats,
        #                           quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)

        s_prediction = self.MinkUNet(s_input)  # B, X, Y, Z, T; F
        s_out_coords = s_prediction.slice(tensor_field).coordinates
        s_out_feats = s_prediction.slice(tensor_field).features
        [T, Z, Y, X] = self.output_grid

        dense_F = torch.zeros(torch.Size([batch_size, 1, X, Y, Z, T]), dtype=torch.float32, device='cuda:0')
        coords = s_out_coords[:, 1:]
        tcoords = coords.t().long()
        batch_indices = s_out_coords[:, 0].long()
        exec(
            "dense_F[batch_indices, :, "
            + ", ".join([f"tcoords[{i}]" for i in range(len(tcoords))])
            + "] = s_out_feats"
        )

        # output check
        # out_coord = s_out.coordinates.cpu().numpy()
        # pred_coord = s_prediction.coordinates.cpu().numpy()
        # pred_feats = s_prediction.features.cpu().detach().numpy()
        # x_min = torch.max(pred_coord[:, 1])
        # y_min = torch.max(pred_coord[:, 2])
        # z_min = torch.max(pred_coord[:, 3])
        # t_min = torch.max(pred_coord[:, 4])

        # occ_feats, min_coord, tensor_stride = s_prediction.dense(shape=torch.Size([batch_size, 1, X, Y, Z, T]),
        #                                       min_coordinate=torch.IntTensor([0, 0, 0, 0]), contract_stride=True)

        # reshape B-0, F-1, X-2, Y-3, Z-4, T-5 to the output of original 4docc
        output = torch.squeeze(dense_F.permute(0, 5, 4, 3, 2, 1).contiguous())
        return output


class DenseDecoder3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(DenseDecoder3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv3d(
            #     in_channels,
            #     out_channels,
            #     kernel_size=kernel_size,
            #     padding=padding,
            #     stride=stride,
            # ),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

#######################################
# Lightning Modules
#######################################

class MinkOccupancyForecastingNetwork(LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = cfg["dataset"]["name"].lower()
        self.loss_type = cfg["model"]["loss_type"].lower()
        assert self.loss_type in ["l1", "l2", "absrel"]

        self.n_input = cfg["data"]["n_input"]
        self.n_output = cfg["data"]["n_output"]
        self.pc_range = cfg["data"]["pc_range"]
        self.voxel_size = cfg["data"]["voxel_size"]

        self.n_height = int((self.pc_range[5] - self.pc_range[2]) / self.voxel_size)
        self.n_length = int((self.pc_range[4] - self.pc_range[1]) / self.voxel_size)
        self.n_width = int((self.pc_range[3] - self.pc_range[0]) / self.voxel_size)
        self.input_grid = [self.n_input, self.n_height, self.n_length, self.n_width]
        print("input grid:", self.input_grid)
        self.output_grid = [self.n_output, self.n_height, self.n_length, self.n_width]
        print("output grid:", self.output_grid)

        self.offset = torch.nn.parameter.Parameter(
            torch.Tensor(self.pc_range[:3])[None, None, :], requires_grad=False
        )
        self.offset_t = torch.nn.parameter.Parameter(
            torch.Tensor([self.pc_range[0], self.pc_range[1], self.pc_range[2], 0.0])[None, :], requires_grad=False
        )
        self.scaler = torch.nn.parameter.Parameter(
            torch.Tensor([self.voxel_size] * 3)[None, None, :], requires_grad=False
        )

        # with 4d sparse encoder, feature dimension = 1
        self.encoder = SparseEncoder4D(in_channels=1, out_channels=1, voxel_size=self.voxel_size,
                                       output_grid=self.output_grid)
        self.decoder = DenseDecoder3D(in_channels=self.n_input, out_channels=self.n_output, kernel_size=3)

        # NOTE: initialize the linear predictor (no bias) over history
        # self.linear = torch.nn.Conv2d(
        #     _in_channels, _out_channels, (3, 3), stride=1, padding=1, bias=True
        # )

        # pytorch-lightning settings
        self.save_hyperparameters(cfg)
        self.lr_start = self.cfg["model"]["lr_start"]
        self.lr_epoch = self.cfg["model"]["lr_epoch"]
        self.lr_decay = self.cfg["model"]["lr_decay"]
        self.automatic_optimization = False  # activate manual optimization
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.iters_acc_loss = 0
        self.epoch_acc_loss = 0


    def forward(self, input_points_4d, output_origin, output_points, output_tindex):
        # preprocess input/output points
        input_points = [(points_4d - self.offset_t) for points_4d in input_points_4d]
        output_origin = ((output_origin - self.offset) / self.scaler).float()
        output_points = ((output_points - self.offset) / self.scaler).float()

        _input = input_points
        batch_size = len(_input)
        _en_output = self.encoder(_input).reshape(batch_size, self.n_input, self.n_height, self.n_length, self.n_width)
        _de_output = self.decoder(_en_output)  # minkowski unet as encoder
        _output = _de_output.reshape(batch_size, self.n_input, self.n_height, self.n_length, self.n_width)
        # w/ skip connection
        # _output = self.linear(_input) + self.decoder(self.encoder(_input))
        sigma = F.relu(_output, inplace=True)

        # dvr rendering
        if sigma.requires_grad:
            pred_dist, gt_dist, grad_sigma = dvr.render(
                sigma,
                output_origin,
                output_points,
                output_tindex,
                self.loss_type
            )
            # take care of nans and infs if any
            grad_sigma[torch.isnan(grad_sigma)] = 0.0
            pred_dist[torch.isinf(pred_dist)] = 0.0
            gt_dist[torch.isinf(pred_dist)] = 0.0
            pred_dist[torch.isnan(pred_dist)] = 0.0
            gt_dist[torch.isnan(pred_dist)] = 0.0
            return sigma, pred_dist, gt_dist, grad_sigma
        else:
            pred_dist, gt_dist = dvr.render_forward(
                sigma,
                output_origin,
                output_points,
                output_tindex,
                self.output_grid,
                "test"  # what does "train" and "test" means here?
            )
            # take care of nans if any
            pred_dist[torch.isnan(pred_dist)] = 0.0
            gt_dist[torch.isnan(pred_dist)] = 0.0
            return sigma, pred_dist, gt_dist

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_start)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_epoch, gamma=self.lr_decay)
        return [optimizer], [scheduler]

    def training_step(self, batch: tuple, batch_idx, dataloader_idx=0):
        # unfold batch data
        meta = batch[0]
        input_points_4d = batch[1]
        output_origin, output_points, output_tindex = batch[2:5]
        output_labels = batch[5] if self.dataset_name == "nuscenes" and self.cfg["data"]["fgbg_label"] else None

        # forward
        sigma, pred_dist, gt_dist, grad_sigma = self.forward(input_points_4d, output_origin, output_points, output_tindex)
        pred_dist *= self.voxel_size
        gt_dist *= self.voxel_size

        # compute training losses
        l1_loss = torch.abs(gt_dist - pred_dist)
        l2_loss = ((gt_dist - pred_dist) ** 2) / 2
        absrel_loss = torch.abs(gt_dist - pred_dist) / gt_dist
        valid = gt_dist >= 0
        valid_pts_count = torch.sum(valid) if torch.sum(valid) != 0 else 1
        l1_loss = torch.sum(l1_loss[valid]) / valid_pts_count
        l2_loss = torch.sum(l2_loss[valid]) / valid_pts_count
        absrel_loss = torch.sum(absrel_loss[valid]) / valid_pts_count

        # log training losses
        self.log("train_l1_loss", l1_loss.detach().cpu())
        self.log("train_l2_loss", l2_loss.detach().cpu())
        self.log("train_absrel_loss", absrel_loss.detach().cpu())
        loss_dict = {"train_l1_loss": l1_loss.detach().cpu(),
                     "train_l2_loss": l1_loss.detach().cpu(),
                     "train_absrel_loss": absrel_loss.detach().cpu()}
        self.training_step_outputs.append(loss_dict)
        # iters accumulated loss
        self.iters_acc_loss += loss_dict[f"train_{self.loss_type}_loss"] / 50
        if (self.global_step + 1) % 50 == 0:  # self.current_batch
            self.log("train_50_iters_acc_loss", self.iters_acc_loss, prog_bar=True)
            self.iters_acc_loss = 0
        # epoch accumulated loss
        num_train_batches = self.trainer.num_training_batches
        self.epoch_acc_loss += loss_dict[f"train_{self.loss_type}_loss"] / num_train_batches

        # manual backward and optimization
        sch = self.lr_schedulers()
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(sigma, gradient=grad_sigma)
        opt.step()
        sch.step()
        torch.cuda.empty_cache()

    def on_train_epoch_end(self):
        self.log("train_epoch_acc_loss", self.epoch_acc_loss, on_epoch=True, logger=True, prog_bar=True)
        self.epoch_acc_loss = 0


# if mode in ["testing", "plotting"]:
#
#     if loss in ["l1", "l2", "absrel"]:
#         sigma = F.relu(output, inplace=True)
#         pred_dist, gt_dist = dvr.render_forward(
#             sigma, output_origin, output_points, output_tindex, self.output_grid, "test")
#         pog = 1 - torch.exp(-sigma)
#
#         pred_dist = pred_dist.detach()
#         gt_dist = gt_dist.detach()
#
#     #
#     pred_dist *= self.voxel_size
#     gt_dist *= self.voxel_size
#
#     if mode == "testing":
#         if eval_within_grid:
#             inner_grid_mask = get_grid_mask(output_points_orig, self.pc_range)
#         if eval_outside_grid:
#             outer_grid_mask = ~ get_grid_mask(output_points_orig, self.pc_range)
#
#         # L1 distance and friends
#         mask = gt_dist > 0
#         if eval_within_grid:
#             mask = torch.logical_and(mask, inner_grid_mask)
#         if eval_outside_grid:
#             mask = torch.logical_and(mask, outer_grid_mask)
#         count = mask.sum()
#         l1_loss = torch.abs(gt_dist - pred_dist)
#         l2_loss = ((gt_dist - pred_dist) ** 2) / 2
#         absrel_loss = torch.abs(gt_dist - pred_dist) / gt_dist
#
#         train_ret_dict["l1_loss"] = l1_loss[mask].sum() / count
#         train_ret_dict["l2_loss"] = l2_loss[mask].sum() / count
#         train_ret_dict["absrel_loss"] = absrel_loss[mask].sum() / count
#
#         train_ret_dict["gt_dist"] = gt_dist
#         train_ret_dict["pred_dist"] = pred_dist
#         train_ret_dict['pog'] = pog.detach()
#         train_ret_dict["sigma"] = sigma.detach()
#
#     if mode == "plotting":
#         train_ret_dict["gt_dist"] = gt_dist
#         train_ret_dict["pred_dist"] = pred_dist
#         train_ret_dict["pog"] = pog
#
# elif mode == "dumping":
#     if loss in ["l1", "l2", "absrel"]:
#         sigma = F.relu(output, inplace=True)
#         pog = 1 - torch.exp(-sigma)
#
#     pog_max, _ = pog.max(dim=1)
#     train_ret_dict["pog_max"] = pog_max
