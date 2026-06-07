"""Dataset-composition panel for the paper/slides: clips per venue, clips per
source video, outcome balance, and fence-type balance — all from labels.csv.

Venue is resolved from the clip id via src/data/venues.py rather than the
labels "video" column, because that column splits on the first underscore and so
mislabels video ids that themselves contain underscores (e.g. h5_V5GYf7iM).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..data.venues import venue_of


def _video_id(clip_id: str) -> str:
    return str(clip_id).rsplit("_", 1)[0]


def _bar(ax, counts: pd.Series, title: str, color: str) -> None:
    counts = counts.sort_values(ascending=False)
    ax.bar(range(len(counts)), counts.values, color=color, width=0.7)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=25, ha="right", fontsize=8)
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, str(int(v)), ha="center", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("clips")


def plot(labels_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(labels_csv)
    if df.empty:
        print(f"[dataset_stats] {labels_csv} is empty")
        return
    df["venue"] = df["clip_id"].map(lambda c: venue_of(_video_id(c)))
    df["video"] = df["clip_id"].map(_video_id)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    _bar(axes[0, 0], df["venue"].value_counts(), "Clips per venue", "#4c72b0")
    _bar(axes[0, 1], df["video"].value_counts(), "Clips per source video", "#55a868")
    _bar(axes[1, 0], df["outcome"].fillna("none").value_counts(),
         "Outcome balance", "#c44e52")
    _bar(axes[1, 1], df["type"].fillna("unknown").value_counts(),
         "Fence-type balance", "#8172b3")
    n_d = int(pd.to_numeric(df["d_meters"], errors="coerce").notna().sum())
    fig.suptitle(f"Dataset composition: {len(df)} labeled clips, "
                 f"{df['venue'].nunique()} venues, {n_d} with takeoff distance d")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[dataset_stats] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("data/annotations/labels.csv"))
    ap.add_argument("--out", type=Path, default=Path("milestone/figures/dataset_stats.png"))
    args = ap.parse_args()
    plot(args.labels, args.out)


if __name__ == "__main__":
    main()

