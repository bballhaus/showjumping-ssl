"""Plot SSL training curves from train_log.csv (contrastive + order loss)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_curves(log_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(log_csv)
    if df.empty:
        print(f"[curve] {log_csv} is empty")
        return
    # Per-epoch averages for a smoother plot.
    by_epoch = df.groupby("epoch")[["loss_contrastive", "loss_order", "loss_total"]].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    by_epoch["loss_contrastive"].plot(ax=ax, label="InfoNCE", marker="o")
    by_epoch["loss_order"].plot(ax=ax, label="Temporal order (CE)", marker="s")
    by_epoch["loss_total"].plot(ax=ax, label="Total", marker="^", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("SSL pretraining loss")
    ax.legend(frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[curve] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=Path("checkpoints/train_log.csv"))
    ap.add_argument("--out", type=Path, default=Path("milestone/figures/training_curve.png"))
    args = ap.parse_args()
    plot_curves(args.log, args.out)


if __name__ == "__main__":
    main()
