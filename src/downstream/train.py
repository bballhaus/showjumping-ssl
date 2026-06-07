"""Train the downstream outcome classifier + d regressor on encoder features.

Supports the experimental knobs the project needs:
  - encoder source: SSL checkpoint, Kinetics init, ImageNet-2D baseline, or random
  - frozen vs. fine-tuned encoder
  - n_labels: train on a subset (drives the sample-efficiency curves)
  - use_type: type conditioning on/off (the type-conditioning ablation)

Frozen encoders cache their per-clip features (see `_FEAT_CACHE`): the feature
vector is deterministic, so it is computed once per (encoder, clip) and reused
across every epoch and every sweep run, turning the sweeps' dominant cost
(re-decoding + re-encoding clips 30x per run) into a single pass. Fine-tuned runs
change the encoder each step and so decode per epoch as before.

`run(...)` returns a metrics dict so the sweep/ablation drivers can call it
directly; the CLI just prints/saves that dict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..eval.metrics import classification_report, d_mae
from ..ssl.embed import load_encoder
from .dataset import OUTCOME_TO_IDX, OUTCOMES, TYPES, LabeledClipDataset, type_onehot
from .heads import DistanceHead, OutcomeHead
from .split import make_split, subsample_train


_FEAT_CACHE: dict[str, dict[str, torch.Tensor]] = {}


def _type_str(onehot: torch.Tensor) -> str:
    if onehot.sum() == 0:
        return "unknown"
    return TYPES[int(onehot.argmax())]


def _cache_key(ckpt, kinetics_init, encoder, n_frames, size) -> str | None:
    """Stable key for a frozen encoder, or None when its weights are not
    reproducible (random init) and therefore must not be cached across runs."""
    if ckpt is not None:
        base = f"ckpt:{ckpt}"
    elif kinetics_init:
        base = "kinetics"
    elif encoder is not None:
        base = f"enc:{type(encoder).__name__}"
    else:
        return None
    return f"{base}|nf{n_frames}|sz{size}"


def _ensure_features(enc, labels_csv, clips_dir, clip_ids, device, key,
                     n_frames, size) -> dict[str, torch.Tensor]:
    """{clip_id: feature vector} for clip_ids, decoding+encoding only the clips
    not already cached under `key`. key=None disables the cache (always recompute)."""
    store = _FEAT_CACHE.setdefault(key, {}) if key is not None else {}
    missing = [c for c in clip_ids if c not in store]
    if missing:
        ds = LabeledClipDataset(labels_csv, clips_dir, clip_ids=missing,
                                n_frames=n_frames, size=size)
        dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
        enc.eval()
        with torch.no_grad():
            for b in dl:
                feats = enc.features(b["x"].to(device)).cpu()
                for cid, fv in zip(b["clip_id"], feats):
                    store[str(cid)] = fv
    return store


def _assemble(df: pd.DataFrame, feats: dict[str, torch.Tensor]):
    """Stack cached features + labels in df order into training tensors."""
    ids = df["clip_id"].astype(str).tolist()
    X = torch.stack([feats[c] for c in ids])
    y = torch.tensor([OUTCOME_TO_IDX[o] for o in df["outcome"]], dtype=torch.long)
    ty = torch.stack([type_onehot(t) for t in df["type"]])
    dv = pd.to_numeric(df["d_meters"], errors="coerce")
    d = torch.tensor([float(v) if pd.notna(v) else 0.0 for v in dv], dtype=torch.float32)
    d_valid = torch.tensor([1.0 if pd.notna(v) else 0.0 for v in dv], dtype=torch.float32)
    return X, y, ty, d, d_valid


def run(labels_csv: Path, clips_dir: Path, ckpt: Path | None = None,
        kinetics_init: bool = False, finetune: bool = False, use_type: bool = True,
        n_labels: int | None = None, task: str = "both", epochs: int = 30,
        lr: float = 1e-3, batch_size: int = 8, val_frac: float = 0.25,
        group_by_venue: bool = False, device: str = "cuda", seed: int = 0,
        train_ids: list[str] | None = None, val_ids: list[str] | None = None,
        encoder: nn.Module | None = None, return_preds: bool = False) -> dict:
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

    def _step(feat, ty, y_outcome, d, d_valid):
        loss = torch.zeros((), device=device)
        if cls_head is not None:
            loss = loss + nn.functional.cross_entropy(cls_head(feat, ty), y_outcome, weight=cls_w)
        if reg_head is not None and d_valid.sum() > 0:
            pred = reg_head(feat, ty)
            loss = loss + (((pred - d) ** 2) * d_valid).sum() / d_valid.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    feats: dict | None = None
    if finetune:
        tr_dl = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
        for _ in range(epochs):
            enc.train(True)
            for b in tr_dl:
                feat = enc.features(b["x"].to(device))
                _step(feat, b["type_onehot"].to(device), b["y_outcome"].to(device),
                      b["d"].to(device), b["d_valid"].to(device))
    else:
        key = _cache_key(ckpt, kinetics_init, encoder, tr.n_frames, tr.size)
        feats = _ensure_features(enc, labels_csv, clips_dir,
                                 list(dict.fromkeys(train_ids + val_ids)), device, key,
                                 tr.n_frames, tr.size)
        X, y, ty, d, dv = _assemble(tr.df, feats)
        tr_dl = DataLoader(TensorDataset(X, y, ty, d, dv), batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for fx, fy, fty, fd, fdv in tr_dl:
                _step(fx.to(device), fty.to(device), fy.to(device),
                      fd.to(device), fdv.to(device))

    enc.eval()
    if cls_head is not None:
        cls_head.eval()
    if reg_head is not None:
        reg_head.eval()
    y_true, y_pred, types, d_true, d_pred, d_valid = [], [], [], [], [], []
    with torch.no_grad():
        if finetune:
            for b in DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=2):
                feat = enc.features(b["x"].to(device))
                ty = b["type_onehot"].to(device)
                types += [_type_str(o) for o in b["type_onehot"]]
                if cls_head is not None:
                    y_pred += cls_head(feat, ty).argmax(1).cpu().tolist()
                    y_true += b["y_outcome"].tolist()
                if reg_head is not None:
                    d_pred += reg_head(feat, ty).cpu().tolist()
                    d_true += b["d"].tolist()
                    d_valid += b["d_valid"].tolist()
        else:
            X, y, ty, d, dv = _assemble(va.df, feats)
            for i in range(0, len(X), batch_size):
                fx = X[i:i + batch_size].to(device)
                fty = ty[i:i + batch_size].to(device)
                types += [_type_str(o) for o in ty[i:i + batch_size]]
                if cls_head is not None:
                    y_pred += cls_head(fx, fty).argmax(1).cpu().tolist()
                if reg_head is not None:
                    d_pred += reg_head(fx, fty).cpu().tolist()
            if cls_head is not None:
                y_true += y.tolist()
            if reg_head is not None:
                d_true += d.tolist()
                d_valid += dv.tolist()

    metrics: dict = {"n_train": len(tr), "n_val": len(va)}
    if cls_head is not None:
        metrics.update(classification_report(y_true, y_pred, types=types))
    if reg_head is not None:
        metrics.update(d_mae(d_true, d_pred, valid=d_valid, types=types))
    if return_preds:
        metrics["y_true"] = list(y_true)
        metrics["y_pred"] = list(y_pred)
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

