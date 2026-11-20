import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weights(y, n_classes=7):
    """Inverse-frequency class weights, normalised to mean 1."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = len(y) / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class FocalLoss(nn.Module):
    """Focal loss written from scratch: FL = -w_t * (1 - p_t)^gamma * log(p_t)."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1.0 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()
