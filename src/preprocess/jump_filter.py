"""Rank clips by how much they look like a real jump, from cached tracks only.

A cantering/walking horse barely changes its box-bottom height; a jump lifts the
box bottom well above its baseline and holds it there for the airborne arc. The
score is that sustained elevation (lift) measured on a median-despiked, detrended
hoof-line, requiring it to persist over several samples (air) so a single bad
YOLO frame can't fake a takeoff, and gated on the horse being visible (coverage),
continuously tracked (small takeoff gap), and stably sized (low size_cv, i.e. the
box isn't flickering between the horse, a rider, or a neighbouring horse). All
from cached tracks.json, no video and no YOLO. Writes a ranked
jump_candidates.csv so the outcome annotator gets real jumps best-first.

Run locally:  python -m src.preprocess.jump_filter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .detect import Box
from .geometry import _interior_smooth, _moving_average, detect_takeoff_frame, takeoff_gap


def _traj(entry: dict) -> list[tuple[int, Box]]:
    return [(int(fi), Box(x1, y1, x2, y2, score=s, label="horse"))
            for fi, x1, y1, x2, y2, s in entry["boxes"]]


def _median3(x: np.ndarray) -> np.ndarray:
    """3-tap median that despikes single-frame detection outliers, endpoints kept."""
    out = x.astype(float).copy()
    if len(x) >= 3:
        out[1:-1] = np.median(np.stack([x[:-2], x[1:-1], x[2:]]), axis=0)
    return out


def score_clip(traj: list[tuple[int, Box]], frame_count: int, stride: int) -> dict | None:
    if len(traj) < 6:
        return None
    heights = np.array([b.h for _, b in traj], dtype=float)
    median_h = float(np.median(heights)) or 1.0
    ys = _interior_smooth(_median3(np.array([b.y2 for _, b in traj], dtype=float)))
    base = _moving_average(ys, max(5, len(ys) // 2))
    resid = ys - base
    apex = int(np.argmin(resid))
    lift = -float(resid[apex]) / median_h
    air = int(np.sum(resid <= 0.5 * resid[apex])) if resid[apex] < 0 else 0
    size_cv = float(np.std(heights) / (np.mean(heights) + 1e-6))
    tf = detect_takeoff_frame(traj)
    max_samples = max(1, frame_count // max(stride, 1) + 1)
    return {
        "takeoff_frame": tf,
        "lift": lift,
        "air": air,
        "size_cv": size_cv,
        "coverage": len(traj) / max_samples,
        "gap": takeoff_gap(traj, tf),
        "n_det": len(traj),
    }


def rank(tracks_path: Path, out_csv: Path,
         min_coverage: float = 0.5, max_gap: int = 4, min_lift: float = 0.03,
         min_air: int = 2, max_size_cv: float = 0.6) -> pd.DataFrame:
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
                            and s["lift"] >= min_lift
                            and s["air"] >= min_air
                            and s["size_cv"] <= max_size_cv)
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
    print("[jump_filter] top 10 by lift (jumps only):")
    for _, r in df[df["is_jump"]].head(10).iterrows():
        print(f"    {r['clip_id']:28s} lift {r['lift']:.2f}  air {int(r['air'])}  "
              f"cov {r['coverage']:.2f}  size_cv {r['size_cv']:.2f}  gap {int(r['gap'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=Path, default=Path("data/annotations/tracks.json"))
    ap.add_argument("--out", type=Path, default=Path("data/annotations/jump_candidates.csv"))
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--max-gap", type=int, default=4)
    ap.add_argument("--min-lift", type=float, default=0.03)
    ap.add_argument("--min-air", type=int, default=2)
    ap.add_argument("--max-size-cv", type=float, default=0.6)
    args = ap.parse_args()

    if not args.tracks.exists():
        print(f"[jump_filter] missing {args.tracks} - run cache_tracks on Colab first")
        return
    df = rank(args.tracks, args.out, min_coverage=args.min_coverage,
              max_gap=args.max_gap, min_lift=args.min_lift,
              min_air=args.min_air, max_size_cv=args.max_size_cv)
    report(df)
    print(f"[jump_filter] wrote {args.out}")


if __name__ == "__main__":
    main()

