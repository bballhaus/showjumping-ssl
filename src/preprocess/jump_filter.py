"""Rank clips by how much they look like a real jump, from cached tracks only.

A cantering/walking horse barely changes its box-bottom height; a jump drives
the box bottom up sharply at takeoff. Scoring the strongest normalized upward
burst of each clip's hoof-line (lift), gated on the horse actually being visible
(coverage) and continuously tracked across the burst (small takeoff gap),
separates real jumps from arena filler with no video and no YOLO. Writes a
ranked jump_candidates.csv so the outcome annotator can be fed real jumps
best-first and the canter clips dropped.

Run locally:  python -m src.preprocess.jump_filter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .detect import Box
from .geometry import _interior_smooth, detect_takeoff_frame, takeoff_gap


def _traj(entry: dict) -> list[tuple[int, Box]]:
    return [(int(fi), Box(x1, y1, x2, y2, score=s, label="horse"))
            for fi, x1, y1, x2, y2, s in entry["boxes"]]


def score_clip(traj: list[tuple[int, Box]], frame_count: int, stride: int) -> dict | None:
    if len(traj) < 5:
        return None
    ys = _interior_smooth(np.array([b.y2 for _, b in traj], dtype=float))
    median_h = float(np.median([b.h for _, b in traj])) or 1.0
    dy = np.diff(ys)
    rise = -float(dy.min())
    max_samples = max(1, frame_count // max(stride, 1) + 1)
    tf = detect_takeoff_frame(traj)
    return {
        "takeoff_frame": tf,
        "lift": rise / median_h,
        "coverage": len(traj) / max_samples,
        "gap": takeoff_gap(traj, tf),
        "n_det": len(traj),
    }


def rank(tracks_path: Path, out_csv: Path,
         min_coverage: float = 0.4, max_gap: int = 4,
         min_lift: float = 0.0) -> pd.DataFrame:
    data = json.loads(tracks_path.read_text())
    stride = int(data.get("stride", 2))
    rows = []
    for clip_id, entry in data["clips"].items():
        s = score_clip(_traj(entry), int(entry.get("frame_count", 0)), stride)
        if s is None:
            continue
        s["clip_id"] = clip_id
        s["is_jump"] = bool(s["coverage"] >= min_coverage
                            and s["gap"] <= max_gap
                            and s["lift"] >= min_lift)
        rows.append(s)
    df = (pd.DataFrame(rows)
          .sort_values(["is_jump", "lift"], ascending=[False, False])
          .reset_index(drop=True))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def report(df: pd.DataFrame) -> None:
    n_jump = int(df["is_jump"].sum())
    print(f"[jump_filter] {len(df)} scored clips, {n_jump} pass as jumps")
    q = df["lift"].quantile([0.5, 0.75, 0.9, 0.95]).round(3).to_dict()
    print(f"[jump_filter] lift percentiles  p50={q[0.5]}  p75={q[0.75]}  "
          f"p90={q[0.9]}  p95={q[0.95]}")
    print("[jump_filter] top 5 by lift:")
    for _, r in df.head(5).iterrows():
        print(f"    {r['clip_id']:28s} lift {r['lift']:.2f}  "
              f"cov {r['coverage']:.2f}  gap {int(r['gap'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=Path, default=Path("data/annotations/tracks.json"))
    ap.add_argument("--out", type=Path, default=Path("data/annotations/jump_candidates.csv"))
    ap.add_argument("--min-coverage", type=float, default=0.4)
    ap.add_argument("--max-gap", type=int, default=4)
    ap.add_argument("--min-lift", type=float, default=0.0)
    args = ap.parse_args()

    if not args.tracks.exists():
        print(f"[jump_filter] missing {args.tracks} - run cache_tracks on Colab first")
        return
    df = rank(args.tracks, args.out, min_coverage=args.min_coverage,
              max_gap=args.max_gap, min_lift=args.min_lift)
    report(df)
    print(f"[jump_filter] wrote {args.out}")


if __name__ == "__main__":
    main()
