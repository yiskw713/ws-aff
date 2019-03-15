import torch.nn as nn
import torch.nn.functional as F


class MSC(nn.Module):
    """
    Multi-scale inputs
    """

    def __init__(self, model, scales=[0.5, 0.75]):
        super().__init__()
        self.model = model
        self.scales = scales

    def forward(self, x):
        ys = []

        # original images
        y = self.model(x)
        ys.append(y)

        # Scaled images
        for s in self.scales:
            xx = F.interpolate(
                x, scale_factor=s, mode="bilinear", align_corners=False)
            yy, _, _ = self.model(xx)
            ys.append(yy)

        return ys
