import math
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# JIT
from torch.utils.cpp_extension import load

from utils.cal_iou import getIoU

dvr = load("dvr", sources=["lib/dvr/dvr.cpp", "lib/dvr/dvr.cu"], verbose=True, extra_cuda_cflags=['-allow-unsupported-compiler'])


def get_grid_mask(points_all, pc_range):
    masks = []
    for batch in range(points_all.shape[0]):
        points = points_all[batch].T
        mask1 = torch.logical_and(pc_range[0] < points[0], points[0] < pc_range[3])  # meters, "=" is deleted
        mask2 = torch.logical_and(pc_range[1] < points[1], points[1] < pc_range[4])
        mask3 = torch.logical_and(pc_range[2] < points[2], points[2] < pc_range[5])
        mask = mask1 & mask2 & mask3
        masks.append(mask)

    # print("shape of mask being returned", mask.shape)
    return torch.stack(masks)

def get_grid_mask_voxel(points_discrete):
    voxel_range = [0, 0, 0, 699, 699, 44]
    masks = []
    for batch in range(points_discrete.shape[0]):
        points = points_discrete[batch].T
        mask1 = torch.logical_and(voxel_range[0] <= points[0], points[0] <= voxel_range[3])  # voxel index
        mask2 = torch.logical_and(voxel_range[1] <= points[1], points[1] <= voxel_range[4])
        mask3 = torch.logical_and(voxel_range[2] <= points[2], points[2] <= voxel_range[5])
        mask = mask1 & mask2 & mask3
        masks.append(mask)
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


class Encoder(nn.Module):
    def __init__(self, in_channels, num_layers, num_filters):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = num_filters[4]

        # Block 1-4
        _in_channels = self.in_channels
        self.block1 = ConvBlock(
            num_layers[0], _in_channels, num_filters[0], max_pool=True
        )
        self.block2 = ConvBlock(
            num_layers[1], num_filters[0], num_filters[1], max_pool=True
        )
        self.block3 = ConvBlock(
            num_layers[2], num_filters[1], num_filters[2], max_pool=True
        )
        self.block4 = ConvBlock(num_layers[3], num_filters[2], num_filters[3])

        # Block 5 (aggregation here)
        _in_channels = sum(num_filters[0:4])
        self.block5 = ConvBlock(num_layers[4], _in_channels, num_filters[4])

    def forward(self, x):
        N, C, H, W = x.shape

        # the first 4 blocks
        c1 = self.block1(x)
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4 = self.block4(c3)

        # upsample and concat
        _H, _W = H // 4, W // 4
        c1_interp = F.interpolate(
            input=c1, size=(_H, _W), mode="bilinear", align_corners=True
        )
        c2_interp = F.interpolate(
            input=c2, size=(_H, _W), mode="bilinear", align_corners=True
        )
        c3_interp = F.interpolate(
            input=c3, size=(_H, _W), mode="bilinear", align_corners=True
        )
        c4_interp = F.interpolate(
            input=c4, size=(_H, _W), mode="bilinear", align_corners=True
        )

        #
        c4_aggr = torch.cat((c1_interp, c2_interp, c3_interp, c4_interp), dim=1)
        c5 = self.block5(c4_aggr)

        return c5


class MosDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MosDecoder, self).__init__()

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
            relu(),  # (700 * 700) * 64

            # output block 1
            conv3x3(64, out_channels, bias=True),

            # output block 2: add another upsampling block, feature map to (1400 * 1400)
            # deconv3x3(64, 32, stride=2),
            # bn(32),
            # relu(),
            # conv3x3(32, 32),
            # bn(32),
            # relu(),  # feature map: 1400 x 1400
            #
            # conv3x3(32, 64, bias=True),
            # bn(64),
            # relu(),
            # conv3x3(64, out_channels, bias=True),
        )

    def forward(self, x):
        return self.block(x)


class MosOccupancyForecastingNetwork(nn.Module):
    def __init__(self, loss_type, n_input, n_mos_class, pc_range, voxel_size):

        super(MosOccupancyForecastingNetwork, self).__init__()

        self.loss_type = loss_type.lower()
        assert self.loss_type in ["nll"]

        self.n_input = n_input
        self.n_mos_class = n_mos_class

        self.n_height = int((pc_range[5] - pc_range[2]) / voxel_size)
        self.n_length = int((pc_range[4] - pc_range[1]) / voxel_size)
        self.n_width = int((pc_range[3] - pc_range[0]) / voxel_size)

        self.input_grid = [self.n_input, self.n_height, self.n_length, self.n_width]
        print("input grid:", self.input_grid)

        self.pc_range = pc_range
        self.voxel_size = voxel_size

        self.offset = torch.nn.parameter.Parameter(
            torch.Tensor(self.pc_range[:3])[None, None, :], requires_grad=False
        )
        self.scaler = torch.nn.parameter.Parameter(
            torch.Tensor([self.voxel_size] * 3)[None, None, :], requires_grad=False
        )

        _in_channels = self.n_input * self.n_height
        self.encoder = Encoder(_in_channels, [2, 2, 3, 6, 5], [32, 64, 128, 256, 256])

        _out_channels = self.n_input * self.n_height * self.n_mos_class
        self.mos_linear = torch.nn.Conv2d(_in_channels, _out_channels, (3, 3), stride=1, padding=1, bias=True)
        self.mos_decoder = MosDecoder(self.encoder.out_channels, _out_channels)

        # important data
        self.occ_mos_feats = None
        self.points_discrete = None
        self.points_tindex = None
        self.inner_points_mask = None
        self.curr_points_mask = None

    def set_threshs(self, threshs):
        self.threshs = torch.nn.parameter.Parameter(torch.Tensor(threshs), requires_grad=False)

    def query_current_points_in_occ(self):
        mask = torch.logical_and(self.curr_points_mask, self.inner_points_mask)
        points_mos_feats_list = []
        # query the points mos state
        batch_size = len(self.points_discrete)
        for i in range(batch_size):
            points_query = self.points_discrete[i][mask[i]]
            points_query_t = self.points_tindex[i][mask[i]]
            points_query_z = points_query[:, 2]
            points_query_y = points_query[:, 1]
            points_query_x = points_query[:, 0]
            points_mos_feats = torch.transpose(self.occ_mos_feats[i][:, points_query_t, points_query_z, points_query_y, points_query_x], 0, 1)
            points_mos_feats_list.append(points_mos_feats)
        return points_mos_feats_list

    def forward(
        self,
        points,
        tindex,
        mos_labels,
        mode="training",
        eval_within_grid=False,
        eval_outside_grid=False
    ):
        # preprocess input points: offset (-70, -70, -4.5), scalar (0.2, 0.2, 0.2)
        points_float_discrete = ((points - self.offset) / self.scaler).float()  # to float voxel index [x, y, z]

        # init occupancy: 0-unknown, 1-occupied (init_cuda_kernel method in dvr.cu)
        input_occupancy = dvr.init(points_float_discrete, tindex, self.input_grid)
        N, T, Z, Y, X = input_occupancy.shape # n, t, z, y, x
        assert T == self.n_input and Z == self.n_height

        # batch size N; feature map: (X, Y); feature channel: (Z * T)
        _input = input_occupancy.reshape(N, -1, Y, X)  # standard occ grid: 2 * 45 * 700 * 700
        # w/ skip connection
        _output = self.mos_linear(_input) + self.mos_decoder(self.encoder(_input))

        # calculate negative log likelihood loss (NLLLoss)
        ret_dict = {}  # return dict
        self.occ_mos_feats = _output.reshape(N, self.n_mos_class, T, Z, Y, X)
        # get mos class prob of each point
        self.points_discrete = points_float_discrete.long()
        self.points_tindex = tindex.long()
        if eval_within_grid:
            # get grid mask (meters)
            inner_grid_mask = get_grid_mask(points, self.pc_range)
            # get voxel mask, make sure all points inside the voxel grid
            inner_voxel_mask = get_grid_mask_voxel(self.points_discrete)
            self.inner_points_mask = torch.logical_and(inner_grid_mask, inner_voxel_mask)
        elif eval_outside_grid:
            outer_grid_mask = ~ get_grid_mask(points, self.pc_range)

        # 此处应该只query当前帧, 舍弃历史点云, 网络不需要有预测历史mos的能力
        # 因为每个batch有效数据量不一致, 取inner_mask之后无法维持batch_size维度, 自动cat起来降低维度了
        self.curr_points_mask = self.points_tindex == 0  # current sample timestamp index = 0?
        mask = torch.logical_and(self.curr_points_mask, self.inner_points_mask)
        points_mos_feats_list = self.query_current_points_in_occ()

        if mode == "training":
            if self.loss_type.lower() in ["nll"]:
                points_mos_feats = torch.cat(points_mos_feats_list)
                points_mos_labels = mos_labels[mask].long()
                softmax = nn.Softmax(dim=1)
                points_mos_prob = softmax(points_mos_feats)  # only take static and moving feats, ignore unknown feats
                points_mos_log_prob = torch.log(points_mos_prob.clamp(min=1e-8))
                weight = torch.Tensor([0.0, 0.5, 0.5]).cuda()  # ignore unknown class when calculate loss
                nlloss = nn.NLLLoss(weight=weight)
                loss = nlloss(points_mos_log_prob, points_mos_labels)
                ret_dict["nll"] = loss.cpu().detach().numpy()

                # 1(negative)-static, 2(positive)-moving
                pred_labels = torch.argmax(points_mos_prob[:, 1:3], dim=1).to(torch.uint8) + 1  # from [0, 1] to [1, 2]
                gt_labels = points_mos_labels.to(torch.uint8)
                IoU = getIoU(gt_labels, pred_labels, [1, 2]) * 100  # 1-static, 2-moving
                ret_dict["iou"] = IoU
                return ret_dict
            else:
                raise RuntimeError(f"Unknown loss type")

        elif mode == "testing":
            points_list = []
            points_gt_labels_list = []
            points_pred_labels_list = []
            iou_list = []
            # query the points mos state
            batch_size = len(self.points_discrete)
            for i in range(batch_size):
                # org points in ref frame, mask ego vehicle, within grid, current timestamp
                points_i = points[i][mask[i]]
                # points mos features (d=3)
                points_mos_feats_i = points_mos_feats_list[i]
                # points mos probability (n_mos_class = 3)
                softmax = nn.Softmax(dim=1)
                points_mos_prob_i = softmax(points_mos_feats_i)
                # points mos gt labels
                points_gt_labels_i = mos_labels[i][mask[i]]
                # points mos pred labels 1(negative)-static, 2(positive)-moving
                points_pred_labels_i = torch.argmax(points_mos_prob_i[:, 1:3], dim=1).to(torch.uint8) + 1  # from [0, 1] to [1, 2]
                IoU = getIoU(points_gt_labels_i, points_pred_labels_i, [1, 2]) * 100  # 1-static, 2-moving
                # append to list
                points_list.append(points_i)
                points_gt_labels_list.append(points_gt_labels_i)
                points_pred_labels_list.append(points_pred_labels_i)
                iou_list.append(IoU)
            ret_dict["points_list"] = points_list
            ret_dict["points_gt_labels_list"] = points_gt_labels_list
            ret_dict["points_pred_labels_list"] = points_pred_labels_list
            ret_dict["iou_list"] = iou_list
            return ret_dict

        elif mode == "plotting":
            ret_dict["gt_dist"] = 1
            ret_dict["pred_dist"] = 1
            ret_dict["pog"] = 1
            return ret_dict
        else:
            raise RuntimeError(f"Unknown mode: {mode}")
