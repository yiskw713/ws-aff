import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .crf import DenseCRF
from .gwrp import global_weighted_rank_pooling
from joblib import Parallel, delayed


class SeedingLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super().__init__()
        self.threshold = threshold
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, seg_out, cam):
        # the values of cam after normalized. shape => (N, C, H', W')
        # ignore all classes images do not have (-100)
        cam = torch.where(cam > 0.2, torch.tensor(1), torch.tensor(-100))
        loss = self.loss(seg_out, cam)

        return loss


class ExpansionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, seg_out, label):
        prob = torch.softmax(seg_out, dim=1)
        y_gwrp = global_weighted_rank_pooling(prob, label)

        pos = label.nonzero()
        neg = (label == 0.).nonzero()
        p, _ = pos.shape
        n, _ = neg.shape

        pos_loss = - torch.log(y_gwrp[pos[:, 0], pos[:, 1]]) / p
        neg_loss = - torch.log(1 - y_gwrp[neg[:, 0], neg[:, 1]]) / n

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
        prob = torch.softmax(seg_out)    # shape => (N, C, h, w)
        probmap = prob.numpy()

        # CRF
        Q = Parallel(n_jobs=-1)([
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
