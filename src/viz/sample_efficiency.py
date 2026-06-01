"""Plot sample-efficiency curves from the sweep CSV.

X-axis: number of training labels n. Y-axis: a metric (default accuracy). One
line per method, mean over seeds with a shaded min/max band. This is the figure
that tests the core hypothesis: SSL matching from-scratch with fewer labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot(results_csv: Path, out_path: Path, metric: str = "accuracy") -> None:
    df = pd.read_csv(results_csv)
    if df.empty or metric not in df.columns:
        print(f"[sample_eff] {results_csv} empty or missing '{metric}'")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, g in df.groupby("method"):
        agg = g.groupby("n")[metric].agg(["mean", "min", "max"]).sort_index()
        ax.plot(agg.index, agg["mean"], marker="o", label=method)
        ax.fill_between(agg.index, agg["min"], agg["max"], alpha=0.15)
    ax.set_xlabel("Number of training labels (n)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Sample efficiency: {metric.replace('_', ' ')} vs. labels")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[sample_eff] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("data/results/sample_efficiency.csv"))
    ap.add_argument("--metric", type=str, default="accuracy")
    ap.add_argument("--out", type=Path, default=Path("milestone/figures/sample_efficiency.png"))
    args = ap.parse_args()
    plot(args.csv, args.out, metric=args.metric)


if __name__ == "__main__":
    main()
