import torch


def global_weighted_rank_pooling(feats, d=0.996):
    n, c, h, w = feats.shape

    desc_inds = torch.argsort(
        feats.reshape(n, c, -1), dim=2, descending=True)   # (n, c, h*w)
    ds = torch.zeros_like(desc_inds, dtype=torch.float) + d    # (n, c, h*w)
    nn, cc, inds = torch.meshgrid(
        torch.arange(n), torch.arange(c), torch.arange(h * w)
    )
    weights = torch.pow(ds, inds.float())    # (n, c, h*w)
    z_dc = torch.sum(weights[0, 0])    # (h*w,)
    desc_feats = feats.reshape(n, c, -1)[nn, cc, desc_inds]
    y_gwrp = torch.sum(weights * desc_feats / z_dc, dim=2)
    return y_gwrp
