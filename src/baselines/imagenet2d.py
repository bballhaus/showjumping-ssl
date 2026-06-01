"""ImageNet 2D-CNN baseline.

A frame-based ResNet-18 (ImageNet-pretrained) is applied to each sampled frame
and the per-frame features are averaged into a clip embedding. This is the
"appearance, no motion model" baseline from the proposal — it tests how much the
3D motion encoder buys over a strong 2D image backbone.

The wrapper exposes the same `.features(x)` / `.feat_dim` interface as the
R(2+1)D encoder so it drops straight into `downstream.train.run(encoder=...)`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from ..downstream.train import run


class ImageNet2DEncoder(nn.Module):
    """ResNet-18 trunk applied per frame, mean-pooled over time."""

    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.net = net

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, T, H, W) -> (B, feat_dim), averaged over T."""
        B, C, T, H, W = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        f = self.net(frames)
        return f.view(B, T, -1).mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("data/annotations/labels.csv"))
    ap.add_argument("--clips", type=Path, default=Path("data/clips"))
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--no-type", action="store_true")
    ap.add_argument("--task", choices=["both", "outcome", "distance"], default="both")
    ap.add_argument("--n-labels", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--group-by-venue", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    enc = ImageNet2DEncoder()
    metrics = run(args.labels, args.clips, encoder=enc, finetune=args.finetune,
                  use_type=not args.no_type, task=args.task, n_labels=args.n_labels,
                  epochs=args.epochs, group_by_venue=args.group_by_venue,
                  device=args.device, seed=args.seed)
    print(json.dumps({"baseline": "imagenet2d", **metrics}, indent=2))


if __name__ == "__main__":
    main()
