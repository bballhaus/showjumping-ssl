"""Train the downstream outcome classifier + d regressor on encoder features.

Supports the experimental knobs the project needs:
  - encoder source: SSL checkpoint, Kinetics init, ImageNet-2D baseline, or random
  - frozen vs. fine-tuned encoder
  - n_labels: train on a subset (drives the sample-efficiency curves)
  - use_type: type conditioning on/off (the type-conditioning ablation)

`run(...)` returns a metrics dict so the sweep/ablation drivers can call it
directly; the CLI just prints/saves that dict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..eval.metrics import classification_report, d_mae
from ..ssl.embed import load_encoder
from .dataset import OUTCOMES, TYPES, LabeledClipDataset
from .heads import DistanceHead, OutcomeHead
from .split import make_split, subsample_train


def _type_str(onehot: torch.Tensor) -> str:
    if onehot.sum() == 0:
        return "unknown"
    return TYPES[int(onehot.argmax())]


def run(labels_csv: Path, clips_dir: Path, ckpt: Path | None = None,
        kinetics_init: bool = False, finetune: bool = False, use_type: bool = True,
        n_labels: int | None = None, task: str = "both", epochs: int = 30,
        lr: float = 1e-3, batch_size: int = 8, val_frac: float = 0.25,
        group_by_venue: bool = False, device: str = "cuda", seed: int = 0,
        train_ids: list[str] | None = None, val_ids: list[str] | None = None,
        encoder: nn.Module | None = None) -> dict:
    torch.manual_seed(seed)
    if train_ids is None or val_ids is None:
        train_ids, val_ids = make_split(labels_csv, val_frac=val_frac,
                                        group_by_venue=group_by_venue, seed=seed)
    if n_labels is not None:
        train_ids = subsample_train(labels_csv, train_ids, n_labels, seed=seed)

    tr = LabeledClipDataset(labels_csv, clips_dir, clip_ids=train_ids)
    va = LabeledClipDataset(labels_csv, clips_dir, clip_ids=val_ids)
    if len(tr) == 0 or len(va) == 0:
        raise SystemExit(f"[downstream] empty split (train={len(tr)}, val={len(va)})")
    tr_dl = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
    va_dl = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=2)

    # Injected encoder (baselines) or one of the R(2+1)D init paths.
    enc = encoder if encoder is not None else load_encoder(
        ckpt if ckpt else Path("///none"), device, kinetics_init=kinetics_init)
    enc = enc.to(device)
    enc.train(finetune)
    for p in enc.parameters():
        p.requires_grad = finetune

    do_cls = task in ("both", "outcome")
    do_reg = task in ("both", "distance")
    cls_head = OutcomeHead(enc.feat_dim, use_type=use_type).to(device) if do_cls else None
    reg_head = DistanceHead(enc.feat_dim, use_type=use_type).to(device) if do_reg else None

    # Class-weighted CE for the outcome imbalance.
    counts = torch.tensor(tr.class_counts(), dtype=torch.float32).clamp(min=1)
    cls_w = (counts.sum() / counts).to(device)

    params: list = []
    if cls_head is not None:
        params += list(cls_head.parameters())
    if reg_head is not None:
        params += list(reg_head.parameters())
    if finetune:
        params += list(enc.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    for _ in range(epochs):
        enc.train(finetune)
        for b in tr_dl:
            x = b["x"].to(device)
            ty = b["type_onehot"].to(device)
            with torch.set_grad_enabled(finetune):
                feat = enc.features(x)
            if not finetune:
                feat = feat.detach()
            loss = torch.zeros((), device=device)
            if cls_head is not None:
                logits = cls_head(feat, ty)
                loss = loss + nn.functional.cross_entropy(
                    logits, b["y_outcome"].to(device), weight=cls_w)
            if reg_head is not None:
                pred = reg_head(feat, ty)
                d = b["d"].to(device)
                valid = b["d_valid"].to(device)
                if valid.sum() > 0:
                    loss = loss + (((pred - d) ** 2) * valid).sum() / valid.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    # Evaluate.
    enc.eval()
    if cls_head is not None:
        cls_head.eval()
    if reg_head is not None:
        reg_head.eval()
    y_true, y_pred, types, d_true, d_pred, d_valid = [], [], [], [], [], []
    with torch.no_grad():
        for b in va_dl:
            x = b["x"].to(device)
            ty = b["type_onehot"].to(device)
            feat = enc.features(x)
            types += [_type_str(o) for o in b["type_onehot"]]
            if cls_head is not None:
                y_pred += cls_head(feat, ty).argmax(1).cpu().tolist()
                y_true += b["y_outcome"].tolist()
            if reg_head is not None:
                d_pred += reg_head(feat, ty).cpu().tolist()
                d_true += b["d"].tolist()
                d_valid += b["d_valid"].tolist()

    metrics: dict = {"n_train": len(tr), "n_val": len(va)}
    if cls_head is not None:
        metrics.update(classification_report(y_true, y_pred, types=types))
    if reg_head is not None:
        metrics.update(d_mae(d_true, d_pred, valid=d_valid, types=types))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("data/annotations/labels.csv"))
    ap.add_argument("--clips", type=Path, default=Path("data/clips"))
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--kinetics-init", action="store_true")
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--no-type", action="store_true", help="Disable type conditioning.")
    ap.add_argument("--task", choices=["both", "outcome", "distance"], default="both")
    ap.add_argument("--n-labels", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--group-by-venue", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    metrics = run(args.labels, args.clips, ckpt=args.ckpt, kinetics_init=args.kinetics_init,
                  finetune=args.finetune, use_type=not args.no_type, n_labels=args.n_labels,
                  task=args.task, epochs=args.epochs, group_by_venue=args.group_by_venue,
                  device=args.device, seed=args.seed)
    print(json.dumps(metrics, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
