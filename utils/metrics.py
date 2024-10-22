#!/usr/bin/env python3
# @file      metrics.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved

import torch
import torch.nn as nn


class ClassificationMetrics(nn.Module):
    def __init__(self, n_classes, ignore_index):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def compute_conf_mat(self, pred_labels: torch.Tensor, gt_labels: torch.Tensor):
        from torcheval.metrics import MulticlassConfusionMatrix
        metric = MulticlassConfusionMatrix(self.n_classes)
        metric.update(pred_labels.long(), gt_labels.long())
        conf_mat = metric.compute()
        return conf_mat

    def get_stats(self, conf_mat):
        # get TP, FP, FN of both two classes
        # ignore_mask = torch.Tensor(self.ignore_index).long()
        # conf_mat[:, ignore_mask] = 0
        tp = conf_mat.diag()
        fp = conf_mat.sum(dim=1) - tp
        fn = conf_mat.sum(dim=0) - tp
        return tp, fp, fn

    def get_iou(self, conf_mat):
        tp, fp, fn = self.get_stats(conf_mat)
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        return iou

    def get_acc(self, conf_mat):
        tp, fp, fn = self.get_stats(conf_mat)
        acc = tp / (tp + fp)
        return acc
