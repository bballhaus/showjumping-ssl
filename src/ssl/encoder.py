"""R(2+1)D-18 encoder used for all SSL + downstream heads."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class R2Plus1DEncoder(nn.Module):
    """R(2+1)D-18 trunk + a 2-layer MLP projector for InfoNCE."""

    def __init__(self, embed_dim: int = 128, kinetics_init: bool = False):
        super().__init__()
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if kinetics_init else None
        backbone = r2plus1d_18(weights=weights)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, embed_dim),
        )
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C=3, T, H, W) -> (B, feat_dim)."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        z = self.projector(f)
        return nn.functional.normalize(z, dim=-1)

