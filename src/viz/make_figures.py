"""Driver that produces every figure referenced by milestone/milestone.tex.

Safe to run with partial data: each step is independently guarded and prints a
warning if its inputs are missing rather than crashing the whole pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .d_distribution import plot as plot_d
from .training_curve import plot_curves
from .tsne import plot_tsne


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/encoder.pt"))
    ap.add_argument("--log", type=Path, default=Path("checkpoints/train_log.csv"))
    ap.add_argument("--emb", type=Path, default=Path("data/embeddings.npz"))
    ap.add_argument("--csv", type=Path, default=Path("data/annotations/auto.csv"))
    ap.add_argument("--out", type=Path, default=Path("milestone/figures"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.log.exists():
        plot_curves(args.log, args.out / "training_curve.png")
    else:
        print(f"[make_figures] missing {args.log}")

    if args.csv.exists():
        plot_d(args.csv, args.out / "d_distribution.png")
    else:
        print(f"[make_figures] missing {args.csv}")

    if args.emb.exists():
        plot_tsne(args.emb, args.csv if args.csv.exists() else None,
                  color_by="type", out_path=args.out / "tsne_type.png",
                  title="t-SNE of SSL clip embeddings (by fence type)")
    else:
        print(f"[make_figures] missing {args.emb}; run src.ssl.embed first")


if __name__ == "__main__":
    main()
