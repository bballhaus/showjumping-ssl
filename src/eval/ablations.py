"""Ablation runner.

Two ablation families from the proposal/milestone:

  1. Type conditioning. Run the downstream heads with type conditioning on vs.
     off (same encoder) to measure what explicit fence type buys.

  2. Pretext tasks. Compare SSL encoders pretrained with different objective
     combinations. You first train the variants, e.g.:
       python -m src.ssl.train --lambda-order 0 ... -> checkpoints/info_only.pt
       python -m src.ssl.train ...                  -> checkpoints/both.pt
     then pass them here as name=path pairs:
       python -m src.eval.ablations --ckpts both=checkpoints/both.pt \
           info_only=checkpoints/info_only.pt

Writes one tidy CSV row per (ablation, setting, seed).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ..downstream.train import run

FIELDS = ["ablation", "setting", "seed", "n_train", "n_val", "accuracy",
          "macro_f1", "d_mae", "d_mae_vertical", "d_mae_oxer"]


def _emit(w, csvfile, ablation, setting, seed, m):
    w.writerow({"ablation": ablation, "setting": setting, "seed": seed, **m})
    csvfile.flush()
    print(f"[ablation] {ablation:12s} {setting:18s} seed={seed} "
          f"acc={m.get('accuracy', float('nan')):.3f} "
          f"f1={m.get('macro_f1', float('nan')):.3f} "
          f"d_mae={m.get('d_mae', float('nan')):.3f}")


def run_ablations(labels_csv: Path, clips_dir: Path, out_csv: Path,
                  ckpts: dict[str, Path], type_ckpt: Path | None,
                  seeds=(0, 1, 2), group_by_venue: bool = False,
                  epochs: int = 30, device: str = "cuda") -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    common = dict(labels_csv=labels_csv, clips_dir=clips_dir, task="both",
                  epochs=epochs, group_by_venue=group_by_venue, device=device)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()

        if type_ckpt is not None:
            for seed in seeds:
                for use_type in (True, False):
                    m = run(ckpt=type_ckpt, finetune=False, use_type=use_type,
                            seed=seed, **common)
                    _emit(w, f, "type_cond", "with_type" if use_type else "no_type", seed, m)

        for name, ckpt in ckpts.items():
            for seed in seeds:
                m = run(ckpt=ckpt, finetune=False, use_type=True, seed=seed, **common)
                _emit(w, f, "pretext", name, seed, m)
    print(f"[ablation] wrote {out_csv}")


def _parse_ckpts(pairs: list[str]) -> dict[str, Path]:
    out = {}
    for p in pairs or []:
        name, _, path = p.partition("=")
        out[name] = Path(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("data/annotations/labels.csv"))
    ap.add_argument("--clips", type=Path, default=Path("data/clips"))
    ap.add_argument("--out", type=Path, default=Path("data/results/ablations.csv"))
    ap.add_argument("--ckpts", nargs="*", default=[],
                    help="Pretext variants as name=path (e.g. both=checkpoints/encoder.pt).")
    ap.add_argument("--type-ckpt", type=Path, default=Path("checkpoints/encoder.pt"),
                    help="Encoder used for the type-conditioning ablation.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--group-by-venue", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    run_ablations(args.labels, args.clips, args.out, _parse_ckpts(args.ckpts),
                  args.type_ckpt, seeds=tuple(args.seeds),
                  group_by_venue=args.group_by_venue, epochs=args.epochs, device=args.device)


if __name__ == "__main__":
    main()

