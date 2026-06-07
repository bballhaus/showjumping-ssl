"""Histogram of geometric takeoff-distance estimates, split by fence type."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["d_meters"])
    df = df[df["d_meters"].between(0.5, 4.0)]
    if df.empty:
        print(f"[d_dist] no usable rows in {csv_path}")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for ftype, color in [("vertical", "#1f77b4"), ("oxer", "#ff7f0e")]:
        sub = df[df["type"] == ftype]
        if not sub.empty:
            ax.hist(sub["d_meters"], bins=15, alpha=0.6, label=f"{ftype} (n={len(sub)})",
                    color=color, edgecolor="white")
    ax.axvline(2.1, ls="--", c="grey", lw=1, label="ideal vert ~2.1 m")
    ax.axvline(2.6, ls=":", c="grey", lw=1, label="ideal oxer ~2.6 m")
    ax.set_xlabel("Estimated takeoff distance d (m)")
    ax.set_ylabel("Count")
    ax.set_title("Geometric d, by fence type")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[d_dist] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("data/annotations/auto.csv"))
    ap.add_argument("--out", type=Path, default=Path("milestone/figures/d_distribution.png"))
    args = ap.parse_args()
    plot(args.csv, args.out)


if __name__ == "__main__":
    main()

