import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGG16(nn.Module):
    def __init__(self, obj_classes, aff_classes):
        super().__init__()

        vgg = models.vgg16_bn(pretrained=True)
        self.vgg = vgg.features[:-1]

        self.obj_conv = nn.Conv2d(
            512, 1024, kernel_size=3, stride=1, padding=1
        )

        self.aff_conv = nn.Conv2d(
            512, 1024, kernel_size=3, stride=1, padding=1
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.obj_fc = nn.Linear(1024, obj_classes)

        self.aff_fc = nn.Linear(1024, aff_classes)

        for m in [self.obj_conv, self.aff_conv, self.obj_fc, self.aff_fc]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.vgg(x)

        # object prediction
        x_obj = self.obj_conv(x)
        x_obj = self.gap(x_obj)
        x_obj = x_obj.view(x_obj.shape[0], -1)
        x_obj = self.obj_fc(x_obj)

        # affordance prediction
        x_aff = self.aff_conv(x)
        x_aff = self.gap(x_aff)
        x_aff = x_aff.view(x_aff.shape[0], -1)
        x_aff = self.aff_fc(x_aff)


        return [x_obj, x_aff]