"""Small CNN that classifies a cropped fence as vertical vs. oxer.

Replaces the width/height heuristic in `fence_type.py` once typed labels exist.
A vertical is a single plane of poles; an oxer has front+back poles, so the
appearance (depth, shadow, two pole lines) carries signal the box aspect ratio
misses. We train an ImageNet-pretrained ResNet-18 on the cropped fence region
because the labeled set is tiny (tens of boxes).

Labels come from `fences.csv` pole_count: 1 -> vertical, 2+ -> oxer (the same
convention the annotator buttons write).

Usage:
    python -m src.preprocess.fence_type_cnn train --clips data/clips \
        --fences data/annotations/fences.csv --out checkpoints/fence_type.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

CLASSES = ("vertical", "oxer")


def _build_model() -> nn.Module:
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def _crop(clip: Path, box, frame_idx, size: int = 64, pad: float = 0.15) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fi = n // 2 if frame_idx is None or (isinstance(frame_idx, float) and np.isnan(frame_idx)) \
        or int(frame_idx) < 0 else min(int(frame_idx), n - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - pad * bw)); x2 = int(min(w, x2 + pad * bw))
    y1 = int(max(0, y1 - pad * bh)); y2 = int(min(h, y2 + pad * bh))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_tensor(crop: np.ndarray) -> torch.Tensor:
    x = crop.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


def _load_dataset(clips_dir: Path, fences_csv: Path):
    df = pd.read_csv(fences_csv)
    xs, ys = [], []
    for _, r in df.iterrows():
        pc = r.get("pole_count")
        if pd.isna(pc):
            continue
        label = 1 if int(pc) >= 2 else 0  # oxer=1, vertical=0
        clip = clips_dir / f"{r['clip_id']}.mp4"
        if not clip.exists():
            continue
        crop = _crop(clip, (r["x1"], r["y1"], r["x2"], r["y2"]), r.get("frame"))
        if crop is None:
            continue
        xs.append(_to_tensor(crop))
        ys.append(label)
    if not xs:
        raise SystemExit(f"[fence_type_cnn] no usable typed crops from {fences_csv}")
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def train(clips_dir: Path, fences_csv: Path, out_path: Path, epochs: int = 40,
          lr: float = 1e-4, val_frac: float = 0.2, device: str = "cpu", seed: int = 0) -> None:
    torch.manual_seed(seed)
    X, y = _load_dataset(clips_dir, fences_csv)
    n = len(X)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = max(1, int(n * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = _build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Class weights for the vertical/oxer imbalance.
    counts = torch.bincount(y[tr_idx], minlength=2).float().clamp(min=1)
    w = (counts.sum() / counts).to(device)

    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(X[tr_idx].to(device))
        loss = nn.functional.cross_entropy(logits, y[tr_idx].to(device), weight=w)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(X[val_idx].to(device)).argmax(1).cpu()
        acc = (pred == y[val_idx]).float().mean().item()
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0:
            print(f"epoch {ep} loss={loss.item():.3f} val_acc={acc:.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "classes": CLASSES}, out_path)
    print(f"[fence_type_cnn] best val_acc={best_acc:.3f}; wrote {out_path}")


class FenceTypeCNN:
    """Inference wrapper. `classify(frame_bgr, box)` -> 'vertical' | 'oxer'."""

    def __init__(self, ckpt_path: Path, device: str = "cpu"):
        self.device = device
        self.model = _build_model().to(device).eval()
        sd = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(sd["model"])

    @torch.no_grad()
    def classify_crop(self, crop_rgb: np.ndarray) -> str:
        x = _to_tensor(crop_rgb).unsqueeze(0).to(self.device)
        return CLASSES[int(self.model(x).argmax(1))]


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pt = sub.add_parser("train")
    pt.add_argument("--clips", type=Path, default=Path("data/clips"))
    pt.add_argument("--fences", type=Path, default=Path("data/annotations/fences.csv"))
    pt.add_argument("--out", type=Path, default=Path("checkpoints/fence_type.pt"))
    pt.add_argument("--epochs", type=int, default=40)
    pt.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    if args.cmd == "train":
        train(args.clips, args.fences, args.out, epochs=args.epochs, device=args.device)


if __name__ == "__main__":
    main()
