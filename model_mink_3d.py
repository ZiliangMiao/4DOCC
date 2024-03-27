import math
import time

import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from lib.minkowski.resnet import ResNetBase

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

class MinkUNet14(ResNetBase):
    BLOCK = BasicBlock
    PLANES = (8, 16, 32, 64, 64, 32, 16, 8)
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)
    INIT_DIM = 8
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        out = self.block8(out)

        return self.final(out)

class Backbone3D(nn.Module):
    def __init__(self, output_grid):
        super(Backbone3D, self).__init__()
        self.output_grid = output_grid
        self.MinkUNet = MinkUNet14(in_channels=2, out_channels=2, D=3)
        self.quantization = torch.Tensor([0.2, 0.2, 0.2]).to(device='cuda')  # x y z t

    def forward(self, input_points_4d):
        batch_size = len(input_points_4d)

        points_coords = []
        points_time_feats = []
        for points_4d in input_points_4d:
            points_coords.append(torch.div(points_4d[:, 0:3], self.quantization))  # 3d coords
            t_range = self.output_grid[0]
            t_feats = torch.zeros(len(points_4d), t_range).to(device='cuda')
            for i in range(t_range):
                t_mask = points_4d[:, -1] == i
                t_feats[t_mask, i] = 1
            points_time_feats.append(t_feats)

        # initialize minkowski engine sparse tensor input
        coords, feats = ME.utils.sparse_collate(points_coords, points_time_feats)
        tf_input = ME.TensorField(features=feats, coordinates=coords.type_as(feats),
                                      quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
        s_input = tf_input.sparse()
        # s_input = ME.SparseTensor(coordinates=coords, features=feats,
        #                           quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)

        #
        s_prediction = self.MinkUNet(s_input)  # B, X, Y, Z, T; F
        [T, Z, Y, X] = self.output_grid

        # d_occ_feats, min_coord, tensor_stride = s_prediction.dense(shape=torch.Size([batch_size, 2, X, Y, Z]),
        #                                       min_coordinate=torch.IntTensor([0, 0, 0]), contract_stride=True)

        s_out_coords = s_prediction.slice(tf_input).coordinates
        s_out_feats = s_prediction.slice(tf_input).features
        d_occ_feats = torch.zeros(torch.Size([batch_size, 2, X, Y, Z]), dtype=torch.float32, device='cuda:0')
        coords = s_out_coords[:, 1:]
        tcoords = coords.t().long()
        batch_indices = s_out_coords[:, 0].long()
        exec(
            "d_occ_feats[batch_indices, :, "
            + ", ".join([f"tcoords[{i}]" for i in range(len(tcoords))])
            + "] = s_out_feats"
        )

        # permute (B-0, F-1, X-2, Y-3, Z-4) to original 4docc output (B, T, Z, Y, X)
        output = torch.squeeze(d_occ_feats.permute(0, 1, 4, 3, 2).contiguous())
        # dense field check
        # feats_max = torch.max(s_prediction.features)
        # feats_min = torch.min(s_prediction.features)
        # d_occ_max = torch.max(output)
        # d_occ_min = torch.min(output)
        # num_zero = torch.sum(output == 0)
        # output_check = output.detach().cpu().numpy()
        return output

class MinkOccupancyForecastingNetwork3D(nn.Module):
    def __init__(self, loss_type, n_input, n_output, pc_range, voxel_size):

        super(MinkOccupancyForecastingNetwork3D, self).__init__()

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
        self.encoder = Backbone3D(self.output_grid)

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