import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .crf import DenseCRF
from .gwrp import global_weighted_rank_pooling
from joblib import Parallel, delayed


class SeedingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, seg_out, cam):
        _, H, W = cam.shape
        seg_out = F.interpolate(seg_out, (H, W), mode='bilinear')
        loss = self.criterion(seg_out, cam)
        return loss


class ExpansionLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, seg_out, label):
        prob = torch.softmax(seg_out, dim=1)
        y_gwrp = global_weighted_rank_pooling(prob, label, self.device)

        pos = label.nonzero()
        neg = (label == 0.).nonzero()
        p, _ = pos.shape
        n, _ = neg.shape

        pos_loss = - torch.sum(torch.log(y_gwrp[pos[:, 0], pos[:, 1]])) / p
        neg_loss = - torch.sum(torch.log(1 - y_gwrp[neg[:, 0], neg[:, 1]])) / n

        return pos_loss + neg_loss


class ConstrainToBoundaryLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.crf = DenseCRF()
        self.device = device

    def forward(self, img, seg_out, label):
        # downscale image
        _, _, h, w = seg_out.shape
        img = F.interpolate(img, (h, w), mode='bilinear')
        # image: (N, 3, h, w) -> (N, h, w, 3)
        img = (img * 255).to('cpu').numpy().astype(np.uint8).transpose(0, 2, 3, 1)
        prob = torch.softmax(seg_out, dim=1)    # shape => (N, C, h, w)
        probmap = prob.to('cpu').data.numpy()

        # CRF
        Q = Parallel(n_jobs=-2)([
            delayed(self.crf)(*pair) for pair in zip(img, probmap)
        ])
        Q = torch.tensor(Q).to(self.device)    # shape => (N, C, h, w)

        # ignore all the classes except classes in images
        neg = (label == 0.).nonzero()
        Q[neg[:, 0], neg[:, 1]] = 0.

        # the number of pixels of classes which images have
        num = label.sum() * h * w
        log = torch.log((Q + 1e-7) / prob)
        loss = torch.sum(Q * log) / num
        return loss


class ObjectLoss(nn.Module):
    def __init__(self, device):
        self.device = device

    def forward(self, seg_out, aff_label, obj_seg):
        n, c, h, w = seg_out.shape
        # shape => (n, 1, h, w)
        obj_seg = F.interpolate(obj_seg, (h, w), mode='bilinear')
        label = torch.zeros((n, c, h, w)).float
        # reverse binary label
        label[:, 0] = torch.where(
            obj_seg == 1, torch.tensor([0.]), torch.tensor([1.]))
