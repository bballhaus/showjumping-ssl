"""Bar chart of ablation results: mean macro-F1 per setting, one panel per
ablation axis (type-conditioning, pretext task, ...). Averages over seeds with a
min/max error bar so the spread from the tiny val set is visible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot(results_csv: Path, out_path: Path, metric: str = "macro_f1") -> None:
    df = pd.read_csv(results_csv)
    if df.empty or metric not in df.columns:
        print(f"[ablation_bar] {results_csv} empty or missing '{metric}'")
        return

    groups = list(df.groupby("ablation"))
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 4),
                             squeeze=False)
    for ax, (ablation, g) in zip(axes[0], groups):
        agg = g.groupby("setting")[metric].agg(["mean", "min", "max"])
        order = agg["mean"].sort_values(ascending=False).index
        agg = agg.loc[order]
        x = range(len(agg))
        yerr = [agg["mean"] - agg["min"], agg["max"] - agg["mean"]]
        ax.bar(x, agg["mean"], color="#4c72b0", width=0.6)
        ax.errorbar(x, agg["mean"], yerr=yerr, fmt="none", ecolor="black", capsize=4)
        for i, v in enumerate(agg["mean"]):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_xticks(list(x))
        ax.set_xticklabels(agg.index, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(str(ablation))
        ax.set_ylabel(metric.replace("_", " "))
    fig.suptitle("Ablations: mean macro-F1 (error bars = min/max over seeds)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[ablation_bar] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("data/results/ablations.csv"))
    ap.add_argument("--metric", type=str, default="macro_f1")
    ap.add_argument("--out", type=Path, default=Path("milestone/figures/ablation_bar.png"))
    args = ap.parse_args()
    plot(args.csv, args.out, metric=args.metric)


if __name__ == "__main__":
    main()
