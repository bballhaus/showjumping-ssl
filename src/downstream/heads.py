"""Downstream heads on top of the R(2+1)D encoder.

Two small heads sit on the frozen (or fine-tuned) encoder features, each
conditioned on fence type (concatenated one-hot), per the proposal:
  - OutcomeHead: 3-way classifier {clean, knockdown, refusal}
  - DistanceHead: scalar regressor for takeoff distance d

Type conditioning is the project's central control: the ideal d and the
difficulty differ between verticals and oxers, so the head sees type explicitly
rather than having to infer it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .dataset import OUTCOMES, TYPES


class OutcomeHead(nn.Module):
    def __init__(self, feat_dim: int, n_types: int = len(TYPES), n_classes: int = len(OUTCOMES),
                 use_type: bool = True):
        super().__init__()
        self.use_type = use_type
        in_dim = feat_dim + (n_types if use_type else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, feat: torch.Tensor, type_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feat, type_onehot], dim=1) if self.use_type else feat
        return self.net(x)


class DistanceHead(nn.Module):
    def __init__(self, feat_dim: int, n_types: int = len(TYPES), use_type: bool = True):
        super().__init__()
        self.use_type = use_type
        in_dim = feat_dim + (n_types if use_type else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, feat: torch.Tensor, type_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feat, type_onehot], dim=1) if self.use_type else feat
        return self.net(x).squeeze(-1)
