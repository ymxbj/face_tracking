"""Auxiliary loss helpers used by the tracker.

The original Pixel3DMM repository ships a ``UVLoss`` class that operates
on the predicted UV map; this open-source release does not run UV-map
prediction, so ``UVLoss`` and the related helpers have been removed.
"""

import torch


def get_albedo_loss(gt, pred, mask):
    gt_albedo = gt[:, :3, :, :].permute(0, 2, 3, 1)
    albedo_loss = (
        (gt_albedo - pred.permute(0, 2, 3, 1)) * mask[:, 0, ...].unsqueeze(-1)
    ).abs().mean()
    return albedo_loss


def get_pos_map_loss(gt, pred, mask):
    gt_pos_map = gt.permute(0, 2, 3, 1)
    tmp = pred
    tmp *= 4
    tmp = torch.stack([-tmp[:, 0, ...], tmp[:, 2, ...], tmp[:, 1, ...]], dim=1)
    tmp /= 1.25

    tmp[:, 1] += 0.2
    l_map = gt_pos_map - tmp.permute(0, 2, 3, 1)
    valid = l_map < 0.015
    pos_map_loss = (l_map * valid.float() * mask).abs().mean()
    return pos_map_loss


def get_pos_map_loss_corresp(gt, pred, omit_mean=False):
    tmp = pred
    tmp *= 4
    tmp = torch.stack([-tmp[:, 0], tmp[:, 2], tmp[:, 1]], dim=1)
    tmp /= 1.25

    tmp[:, 1] += 0.2
    outliers = (gt - tmp).abs().sum(dim=-1) > 0.066
    if omit_mean:
        pos_map_loss = (gt - tmp) * (~outliers).float().unsqueeze(-1)
    else:
        pos_map_loss = ((gt - tmp)[~outliers, :]).abs().mean()
    return pos_map_loss
