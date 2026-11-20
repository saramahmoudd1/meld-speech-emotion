import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    """Pooled-feature model: one branch per modality, concat, then a classifier head."""

    def __init__(self, modality="fusion", d_in=768, hidden=256, dropout=0.3, n_classes=7):
        super().__init__()
        self.use_a = modality in ("audio", "fusion")
        self.use_t = modality in ("text", "fusion")

        def branch():
            return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout))

        if self.use_a:
            self.a = branch()
        if self.use_t:
            self.t = branch()
        k = hidden * (int(self.use_a) + int(self.use_t))
        self.head = nn.Sequential(nn.Linear(k, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, xa, xt):
        parts = []
        if self.use_a:
            parts.append(self.a(xa))
        if self.use_t:
            parts.append(self.t(xt))
        return self.head(torch.cat(parts, dim=-1))
