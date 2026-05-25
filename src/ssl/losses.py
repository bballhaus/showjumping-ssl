"""InfoNCE contrastive loss + cross-entropy on the 6-way permutation classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Symmetric SimCLR-style InfoNCE.

    z1, z2: (B, D), already L2-normalized.
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)            # (2B, D)
    sim = z @ z.t() / tau                     # (2B, 2B)
    sim.fill_diagonal_(-1e4)
    targets = torch.arange(2 * B, device=z.device)
    targets = (targets + B) % (2 * B)         # positive partner is offset by B
    return F.cross_entropy(sim, targets)


class TemporalOrderHead(nn.Module):
    def __init__(self, feat_dim: int, n_perms: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, n_perms),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f)
