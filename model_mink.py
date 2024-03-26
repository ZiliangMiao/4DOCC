import math
import time

import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

import MinkowskiEngine as ME
from lib.minkowski.minkunet import MinkUNet14

# JIT
from torch.utils.cpp_extension import load

dvr = load("dvr", sources=["lib/dvr/dvr.cpp", "lib/dvr/dvr.cu"], verbose=True, extra_cuda_cflags=['-allow-unsupported-compiler'])


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


def conv3x3(in_channels, out_channels, bias=False):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
    )


def deconv3x3(in_channels, out_channels, stride):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        output_padding=1,
        bias=False,
    )


def maxpool2x2(stride):
    return nn.MaxPool2d(kernel_size=2, stride=stride, padding=0)


def relu(inplace=True):
    return nn.ReLU(inplace=inplace)


def bn(num_features):
    return nn.BatchNorm2d(num_features=num_features)


class ConvBlock(nn.Module):
    def __init__(self, num_layer, in_channels, out_channels, max_pool=False):
        super(ConvBlock, self).__init__()

        layers = []
        for i in range(num_layer):
            _in_channels = in_channels if i == 0 else out_channels
            layers.append(conv3x3(_in_channels, out_channels))
            layers.append(bn(out_channels))
            layers.append(relu())

        if max_pool:
            layers.append(maxpool2x2(stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class CustomMinkUNet(MinkUNet14):
        PLANES = (8, 16, 32, 64, 64, 32, 16, 8)
        INIT_DIM = 8

class Encoder(nn.Module):
    def __init__(self, output_grid):
        super(Encoder, self).__init__()
        self.output_grid = output_grid

        self.MinkUNet = CustomMinkUNet(in_channels=1, out_channels=1, D=4)

        self.quantization = torch.Tensor([0.2, 0.2, 0.2, 1.0]).to(device='cuda')  # x y z t

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

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            deconv3x3(in_channels, 128, stride=2),
            bn(128),
            relu(),
            conv3x3(128, 128),
            bn(128),
            relu(),
            deconv3x3(128, 64, stride=2),
            bn(64),
            relu(),
            conv3x3(64, 64),
            bn(64),
            relu(),
            conv3x3(64, out_channels, bias=True),
        )

    def forward(self, x):
        return self.block(x)


class MinkOccupancyForecastingNetwork(nn.Module):
    def __init__(self, loss_type, n_input, n_output, pc_range, voxel_size):

        super(MinkOccupancyForecastingNetwork, self).__init__()

        self.loss_type = loss_type.lower()
        assert self.loss_type in ["l1", "l2", "absrel"]

        self.n_input = n_input
        self.n_output = n_output

        self.n_height = int((pc_range[5] - pc_range[2]) / voxel_size)
        self.n_length = int((pc_range[4] - pc_range[1]) / voxel_size)
        self.n_width = int((pc_range[3] - pc_range[0]) / voxel_size)

        self.input_grid = [self.n_input, self.n_height, self.n_length, self.n_width]
        print("input grid:", self.input_grid)

        self.output_grid = [self.n_output, self.n_height, self.n_length, self.n_width]
        print("output grid:", self.output_grid)

        self.pc_range = pc_range
        self.voxel_size = voxel_size

        self.offset = torch.nn.parameter.Parameter(
            torch.Tensor(self.pc_range[:3])[None, None, :], requires_grad=False
        )
        self.offset_t = torch.nn.parameter.Parameter(
            torch.Tensor([self.pc_range[0], self.pc_range[1], self.pc_range[2], 0.0])[None, :], requires_grad=False
        )

        self.scaler = torch.nn.parameter.Parameter(
            torch.Tensor([self.voxel_size] * 3)[None, None, :], requires_grad=False
        )

        # _in_channels = self.n_input * self.n_height
        self.encoder = Encoder(self.output_grid)

        # NOTE: initialize the linear predictor (no bias) over history
        # _out_channels = self.n_output * self.n_height
        # self.linear = torch.nn.Conv2d(
        #     _in_channels, _out_channels, (3, 3), stride=1, padding=1, bias=True
        # )
        # self.decoder = Decoder(self.encoder.out_channels, _out_channels)

    def forward(
        self,
        input_points_4d,
        output_origin_orig,
        output_points_orig,
        output_tindex,
        output_labels=None,
        loss=None,
        mode="training",
        eval_within_grid=False,
        eval_outside_grid=False
    ):
        if loss == None:
            loss = self.loss_type

        if eval_within_grid:
            inner_grid_mask = get_grid_mask(output_points_orig, self.pc_range)
        if eval_outside_grid:
            outer_grid_mask = ~ get_grid_mask(output_points_orig, self.pc_range)

        # preprocess input/output points
        input_points = [(points_4d - self.offset_t) for points_4d in input_points_4d]
        output_origin = ((output_origin_orig - self.offset) / self.scaler).float()
        output_points = ((output_points_orig - self.offset) / self.scaler).float()

        # w/ skip connection
        # _output = self.linear(_input) + self.decoder(self.encoder(_input))
        _input = input_points
        _output = self.encoder(_input)  # minkowski unet as encoder + decoder
        output = _output.reshape(len(_input), self.n_input, self.n_height, self.n_length, self.n_width)

        ret_dict = {}

        if mode == "training":
            if loss in ["l1", "l2", "absrel"]:
                sigma = F.relu(output, inplace=True)
                # sigma_max, sigma_min = sigma.max(), sigma.min()

                if sigma.requires_grad:
                    pred_dist, gt_dist, grad_sigma = dvr.render(
                        sigma,
                        output_origin,
                        output_points,
                        output_tindex,
                        loss
                    )
                    # take care of nans and infs if any
                    invalid = torch.isnan(grad_sigma)
                    grad_sigma[invalid] = 0.0
                    invalid = torch.isnan(pred_dist)
                    pred_dist[invalid] = 0.0
                    gt_dist[invalid] = 0.0
                    invalid = torch.isinf(pred_dist)
                    pred_dist[invalid] = 0.0
                    gt_dist[invalid] = 0.0
                    sigma.backward(grad_sigma)
                else:
                    pred_dist, gt_dist = dvr.render_forward(
                        sigma,
                        output_origin,
                        output_points,
                        output_tindex,
                        self.output_grid,
                        "train"
                    )
                    # take care of nans if any
                    invalid = torch.isnan(pred_dist)
                    pred_dist[invalid] = 0.0
                    gt_dist[invalid] = 0.0

                pred_dist *= self.voxel_size
                gt_dist *= self.voxel_size

                # compute training losses
                valid = gt_dist >= 0
                count = valid.sum()
                l1_loss = torch.abs(gt_dist - pred_dist)
                l2_loss = ((gt_dist - pred_dist) ** 2) / 2
                absrel_loss = torch.abs(gt_dist - pred_dist) / gt_dist

                # record training losses
                if count == 0:
                    count = 1
                ret_dict["l1_loss"] = l1_loss[valid].sum() / count
                ret_dict["l2_loss"] = l2_loss[valid].sum() / count
                ret_dict["absrel_loss"] = absrel_loss[valid].sum() / count

            else:
                raise RuntimeError(f"Unknown loss type: {loss}")

        elif mode in ["testing", "plotting"]:

            if loss in ["l1", "l2", "absrel"]:
                sigma = F.relu(output, inplace=True)
                pred_dist, gt_dist = dvr.render_forward(
                    sigma, output_origin, output_points, output_tindex, self.output_grid, "test")
                pog = 1 - torch.exp(-sigma)

                pred_dist = pred_dist.detach()
                gt_dist = gt_dist.detach()

            #
            pred_dist *= self.voxel_size
            gt_dist *= self.voxel_size

            if mode == "testing":
                # L1 distance and friends
                mask = gt_dist > 0
                if eval_within_grid:
                    mask = torch.logical_and(mask, inner_grid_mask)
                if eval_outside_grid:
                    mask = torch.logical_and(mask, outer_grid_mask)
                count = mask.sum()
                l1_loss = torch.abs(gt_dist - pred_dist)
                l2_loss = ((gt_dist - pred_dist) ** 2) / 2
                absrel_loss = torch.abs(gt_dist - pred_dist) / gt_dist

                ret_dict["l1_loss"] = l1_loss[mask].sum() / count
                ret_dict["l2_loss"] = l2_loss[mask].sum() / count
                ret_dict["absrel_loss"] = absrel_loss[mask].sum() / count

                ret_dict["gt_dist"] = gt_dist
                ret_dict["pred_dist"] = pred_dist
                ret_dict['pog'] = pog.detach()
                ret_dict["sigma"] = sigma.detach()

            if mode == "plotting":
                ret_dict["gt_dist"] = gt_dist
                ret_dict["pred_dist"] = pred_dist
                ret_dict["pog"] = pog

        elif mode == "dumping":
            if loss in ["l1", "l2", "absrel"]:
                sigma = F.relu(output, inplace=True)
                pog = 1 - torch.exp(-sigma)

            pog_max, _ = pog.max(dim=1)
            ret_dict["pog_max"] = pog_max

        else:
            raise RuntimeError(f"Unknown mode: {mode}")

        return ret_dict