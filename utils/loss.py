import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .crf import DenseCRF
from .gwrp import global_weighted_rank_pooling
from joblib import Parallel, delayed


class SeedingLoss(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        if config.class_weight_flag:
            if config.target == 'affordance':
                print("use class weight of affordance label")
                class_weight = torch.tensor([
                    5.9797e-04, 2.3909e-01, 9.9345e-01, 2.2071e+00, 1.0000e+00, 3.8656e+00, 4.0732e+00, 6.4024e-01]).to(device)
            elif config.target == 'objcect':
                print("use class weight for object label")
                class_weight = torch.tensor([0.0020, 1.0000]).to(device)
            else:
                print('class weight will not be used')
                class_weight = None
        else:
            print('class weight will not be used')
            class_weight = None

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weight, ignore_index=255)

    def forward(self, seg_out, cam):
        _, H, W = cam.shape
        seg_out = F.interpolate(
            seg_out, (H, W), mode='bilinear', align_corners=True)
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

        pos_loss = - \
            torch.sum(torch.log(y_gwrp[pos[:, 0], pos[:, 1]] + 1e-7)) / p
        neg_loss = - \
            torch.sum(torch.log(1 - y_gwrp[neg[:, 0], neg[:, 1]] + 1e-7)) / n

        return pos_loss + neg_loss


class ConstrainToBoundaryLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.crf = DenseCRF()
        self.device = device

    def forward(self, img, seg_out, label):
        # downscale image
        _, _, h, w = seg_out.shape
        img = F.interpolate(img, (h, w), mode='bilinear', align_corners=True)
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


class SelfLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, seg_out):
        result = torch.softmax(seg_out, dim=1)
        _, result = result.max(dim=1)    # shape => (N, H, W)
        loss = F.cross_entropy(seg_out, result)
        return loss


# class ObjectLoss(nn.Module):
#     def __init__(self, device):
        # super().__init__()
#         self.device = device

#     def forward(self, seg_out, aff_label, obj_seg):
#         n, c, h, w = seg_out.shape
#         # shape => (n, 1, h, w)
#         obj_seg = F.interpolate(obj_seg, (h, w), mode='bilinear')
#         label = torch.zeros((n, c, h, w)).float
#         # reverse binary label
#         label[:, 0] = torch.where(
#             obj_seg == 1, torch.tensor([0.]), torch.tensor([1.]))


""" how to calculate class weight 
import glob
import numpy as np
import torch
from PIL import Image

path = glob.glob("./part-affordance-dataset/tools/*/*aff_cam_label.png")

cnt_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 255:0}

for p in path:
    img = Image.open(p)
    img = np.asarray(img)
    num, cnt = np.unique(img, return_counts=True)

    for n, c in zip(num, cnt):
        cnt_dict[n] += c
class_num = torch.tensor([7321360010, 18311032, 4406809, 1983549,
                        4377947, 1132548, 1074818, 6837994])
total = class_num.sum().item()
frequency = class_num.float() / total
median = torch.median(frequency)
class_weight = median / frequency

class_weight = torch.tensor([
    5.9797e-04, 2.3909e-01, 9.9345e-01, 2.2071e+00, 1.0000e+00, 3.8656e+00, 4.0732e+00, 6.4024e-01])


# object class weight
path = glob.glob("./part-affordance-dataset/tools/*/*obj_cam_label.png")

cnt_dict = {}
for i in range(18):
    cnt_dict[i]=0
cnt_dict[255] = 0

for p in path:
    img = Image.open(p)
    img = np.asarray(img)
    num, cnt = np.unique(img, return_counts=True)
    
    for n, c in zip(num, cnt):
        cnt_dict[n] += c

class_num = []
class_num.append(cnt_dict[0])
c = 0
for i in range(1,18):
    c += cnt_dict[i]
class_num.append(c)
print(class_num)

class_num = torch.tensor(class_num)
total = class_num.sum().item()
frequency = class_num.float() / total
median = torch.median(frequency)
class_weight = median / frequency
print(class_weight)

class_weight = torch.tensor([0.0020, 1.0000])

"""
