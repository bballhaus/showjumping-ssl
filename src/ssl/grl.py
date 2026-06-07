"""Domain-adversarial pieces for suppressing the venue confound.

The milestone's key finding was that the dominant axis of variation in the
embedding space is *which broadcast a clip came from* (arena, lighting, footing,
crowd), not approach quality. A domain-adversarial head (DANN, Ganin & Lempitsky
2015) fights this: a classifier tries to predict the source video from the
features, while a gradient-reversal layer makes the encoder learn features that
*defeat* that classifier — i.e. features that are venue-invariant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Identity forward; multiplies gradient by -lambd on the backward pass."""
    return _GradReverse.apply(x, lambd)


def dann_lambda(step: int, total_steps: int, gamma: float = 10.0) -> float:
    """DANN schedule: ramp the reversal strength from 0 -> 1 over training so the
    adversary doesn't destabilize the encoder before it has useful features."""
    p = step / max(1, total_steps)
    return 2.0 / (1.0 + float(torch.exp(torch.tensor(-gamma * p)))) - 1.0


class DomainHead(nn.Module):
    """Predicts source-video id from encoder features (through the GRL)."""

    def __init__(self, feat_dim: int, n_domains: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, n_domains),
        )

    def forward(self, f: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        return self.net(grad_reverse(f, lambd))

