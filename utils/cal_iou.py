import numpy as np
import torch
from sklearn.metrics import confusion_matrix

def getStat(confusion_matrix):
    tp = np.diagonal(confusion_matrix)
    fp = np.sum(confusion_matrix, axis=1) - tp
    fn = np.sum(confusion_matrix, axis=0) - tp
    return tp[1], fp[1], fn[1]

def getIoU(gt_labels, pred_labels, valid_labels=None):
    if torch.is_tensor(gt_labels):
        gt_labels = gt_labels.cpu().detach().numpy()
        pred_labels = pred_labels.cpu().detach().numpy()
    mos_confusion_mat = confusion_matrix(y_true=gt_labels, y_pred=pred_labels, labels=valid_labels)  # 1-static, 2-moving
    tp, fp, fn = getStat(mos_confusion_mat)  # stat of current sample
    intersection = tp
    union = tp + fp + fn + 1e-15
    IoU = intersection / union
    return IoU