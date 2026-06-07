"""Sample-efficiency sweep: the project's headline experiment.

For each method and each label budget n in {25, 50, 100, 200}, train the
downstream heads and record val accuracy / macro-F1 / d-MAE, averaged over
seeds. The val split is held fixed across methods and n (same seed) so the only
thing that changes is the encoder and the number of training labels.

Methods:
  ssl          : the SSL-pretrained R(2+1)D encoder (frozen)   [--ckpt]
  kinetics     : Kinetics-400 init R(2+1)D (frozen)
  from_scratch : random-init R(2+1)D trained end-to-end
  imagenet2d   : ImageNet ResNet-18, frame-averaged (frozen)

Writes a tidy CSV (method, n, seed, metric columns) for the curve plotter.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ..baselines.imagenet2d import ImageNet2DEncoder
from ..downstream.split import leave_one_out_splits, make_split
from ..downstream.train import run

DEFAULT_NS = (25, 50, 100, 200)
METHODS = ("ssl", "kinetics", "from_scratch", "imagenet2d")


def _run_method(method: str, labels_csv: Path, clips_dir: Path, ckpt: Path | None,
                n: int, train_ids, val_ids, use_type: bool, epochs: int,
                device: str, seed: int) -> dict:
    common = dict(labels_csv=labels_csv, clips_dir=clips_dir, n_labels=n,
                  use_type=use_type, task="both", epochs=epochs, device=device,
                  seed=seed, train_ids=train_ids, val_ids=val_ids)
    if method == "ssl":
        return run(ckpt=ckpt, finetune=False, **common)
    if method == "kinetics":
        return run(ckpt=None, kinetics_init=True, finetune=False, **common)
    if method == "from_scratch":
        return run(ckpt=None, kinetics_init=False, finetune=True, **common)
    if method == "imagenet2d":
        return run(encoder=ImageNet2DEncoder(), finetune=False, **common)
    raise ValueError(method)


def sweep(labels_csv: Path, clips_dir: Path, out_csv: Path, ckpt: Path | None = None,
          methods=METHODS, ns=DEFAULT_NS, seeds=(0, 1, 2), use_type: bool = True,
          group_by_venue: bool = False, val_venue: str | None = None,
          epochs: int = 30, device: str = "cuda") -> None:
    """Sample-efficiency sweep.

    val_venue pins the val set to one held-out physical venue (e.g. "tryon" or
    "madrid"), the honest cross-venue curve: the held-out backdrop never appears
    in train, and n varies on the remaining venues. Leave it None for the
    in-domain / group_by_venue behavior. val_venue overrides group_by_venue.
    """
    fixed_split = None
    if val_venue is not None:
        folds = {h: (tr, va) for h, tr, va in
                 leave_one_out_splits(labels_csv, level="venue")}
        if val_venue not in folds:
            raise SystemExit(f"[sweep] val_venue '{val_venue}' not in {sorted(folds)}")
        fixed_split = folds[val_venue]
        print(f"[sweep] held-out venue={val_venue}: "
              f"train={len(fixed_split[0])} val={len(fixed_split[1])}")

    fields = ["method", "n", "seed", "n_train", "n_val", "accuracy", "macro_f1",
              "d_mae", "d_mae_vertical", "d_mae_oxer"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for seed in seeds:
            if fixed_split is not None:
                train_ids, val_ids = fixed_split
            else:
                train_ids, val_ids = make_split(labels_csv, group_by_venue=group_by_venue, seed=seed)
            for n in ns:
                for method in methods:
                    m = _run_method(method, labels_csv, clips_dir, ckpt, n, train_ids,
                                    val_ids, use_type, epochs, device, seed)
                    row = {"method": method, "n": n, "seed": seed, **m}
                    w.writerow(row)
                    f.flush()
                    print(f"[sweep] {method:12s} n={n:3d} seed={seed} "
                          f"acc={m.get('accuracy', float('nan')):.3f} "
                          f"f1={m.get('macro_f1', float('nan')):.3f} "
                          f"d_mae={m.get('d_mae', float('nan')):.3f}")
    print(f"[sweep] wrote {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("data/annotations/labels.csv"))
    ap.add_argument("--clips", type=Path, default=Path("data/clips"))
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/encoder.pt"))
    ap.add_argument("--out", type=Path, default=Path("data/results/sample_efficiency.csv"))
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--ns", nargs="+", type=int, default=list(DEFAULT_NS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--no-type", action="store_true")
    ap.add_argument("--group-by-venue", action="store_true")
    ap.add_argument("--val-venue", type=str, default=None,
                    help="Pin val to one held-out physical venue (honest cross-venue curve).")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    sweep(args.labels, args.clips, args.out, ckpt=args.ckpt, methods=tuple(args.methods),
          ns=tuple(args.ns), seeds=tuple(args.seeds), use_type=not args.no_type,
          group_by_venue=args.group_by_venue, val_venue=args.val_venue,
          epochs=args.epochs, device=args.device)


if __name__ == "__main__":
    main()

