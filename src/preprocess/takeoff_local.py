"""Replay takeoff detection + geometry from cached tracks — no YOLO, no video.

Reads `data/annotations/tracks.json` (written on Colab by `cache_tracks`) and the
hand-drawn fence boxes in `data/annotations/fences.csv`, recomputes `takeoff_frame`
and `d` with the current `geometry.py`, rewrites `auto.csv`, prints the regression
guard, and saves a `y2(t)` diagnostic grid marking where each takeoff was picked.

Because every input is cached, this runs in seconds on a laptop with only
numpy/pandas/matplotlib — so the takeoff detector can be tuned without Colab.

Run locally:  python -m src.preprocess.takeoff_local
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .detect import Box
from .fence_type import classify_fence
from .geometry import (
    _interior_smooth,
    _moving_average,
    detect_takeoff_frame,
    meters_per_pixel,
    takeoff_distance,
    takeoff_gap,
    takeoff_observable,
)
from .run_pipeline import load_fence_annotations


def _traj(clip_entry: dict) -> list[tuple[int, Box]]:
    out: list[tuple[int, Box]] = []
    for fi, x1, y1, x2, y2, score in clip_entry["boxes"]:
        out.append((int(fi), Box(x1, y1, x2, y2, score=score, label="horse")))
    return out


def _last_sampled_idx(clip_entry: dict, stride: int) -> int:
    n = int(clip_entry.get("frame_count", 0))
    return max(1, ((n - 1) // stride) * stride)


def recompute(tracks_path: Path, fences_path: Path, out_csv: Path) -> pd.DataFrame:
    data = json.loads(tracks_path.read_text())
    stride = int(data.get("stride", 2))
    fences = load_fence_annotations(fences_path)

    rows = []
    for clip_id, entry in data["clips"].items():
        traj = _traj(entry)
        row = {
            "clip_id": clip_id,
            "n_horse_detections": len(traj),
            "type": "",
            "d_meters": float("nan"),
            "mpp": float("nan"),
            "takeoff_frame": -1,
            "frac": float("nan"),
            "takeoff_gap": -1,
        }
        if traj:
            takeoff_fi = detect_takeoff_frame(traj)
            row["takeoff_frame"] = takeoff_fi
            row["frac"] = takeoff_fi / _last_sampled_idx(entry, stride)
            row["takeoff_gap"] = takeoff_gap(traj, takeoff_fi)
            ann = fences.get(clip_id)
            if ann is not None:
                fence_box = ann["box"]
                mpp = meters_per_pixel(fence_box)
                takeoff_horse = min(traj, key=lambda x: abs(x[0] - takeoff_fi))[1]
                row["mpp"] = mpp
                row["d_meters"] = takeoff_distance(takeoff_horse, fence_box, mpp)
                row["type"] = classify_fence(fence_box, pole_count=ann.get("pole_count"))
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def regression_guard(df: pd.DataFrame) -> None:
    sub = df[(df["takeoff_frame"] >= 0) & df["frac"].notna()]
    if sub.empty:
        print("[guard] no usable clips")
        return
    share_end = float((sub["frac"] >= 0.85).mean())
    print(f"clips checked: {len(sub)}")
    print(f"takeoff position (fraction into clip) - "
          f"mean {sub['frac'].mean():.2f}, median {sub['frac'].median():.2f}")
    print(f"share with takeoff in last 15%: {share_end:.0%}   (old buggy run: 100%)")
    print("PASS - takeoff frames spread through the clip"
          if share_end < 0.6 else
          "WARN - still clustering at clip end")


def score_against_truth(df: pd.DataFrame, takeoffs_path: Path,
                        tracks_path: Path | None = None) -> pd.DataFrame | None:
    """Score detected takeoff frames against hand-marked ground truth.

    Stratifies by whether the TRUE takeoff is observable in the horse track:
    clips whose lift-off falls in a camera-cut gap or past the track edge are a
    multi-camera-footage limitation, not a detector error, and dominate the
    pooled error. The "ok" stratum is the fair test of the detector.
    """
    if not takeoffs_path.exists():
        print(f"[score] no {takeoffs_path} yet - run TakeoffAnnotator to create ground truth")
        return None
    truth = pd.read_csv(takeoffs_path).rename(columns={"takeoff_frame": "true_frame"})
    truth["clip_id"] = truth["clip_id"].astype(str)
    merged = df.merge(truth[["clip_id", "true_frame"]], on="clip_id", how="inner")
    merged = merged[(merged["takeoff_frame"] >= 0) & merged["true_frame"].notna()].copy()
    if merged.empty:
        print("[score] no overlap between detections and ground truth")
        return None
    merged["err_frames"] = merged["takeoff_frame"] - merged["true_frame"]

    flags = {}
    if tracks_path is not None and tracks_path.exists():
        data = json.loads(tracks_path.read_text())
        clips = data["clips"]
        for cid, tf in zip(merged["clip_id"], merged["true_frame"]):
            entry = clips.get(cid)
            traj = _traj(entry) if entry else []
            flags[cid] = takeoff_observable(traj, int(tf))[1] if traj else "sparse"
    merged["flag"] = merged["clip_id"].map(flags).fillna("?")

    def _row(sub: pd.DataFrame, name: str) -> None:
        e = sub["err_frames"]
        print(f"[score] {name:24s} n={len(sub):2d}  bias {e.mean():+.2f}  "
              f"MAE {e.abs().mean():.2f}  within +-2 {(e.abs() <= 2).mean():.0%}")

    _row(merged, "all clips")
    ok = merged[merged["flag"] == "ok"]
    if not ok.empty and len(ok) < len(merged):
        _row(ok, "takeoff observable")
        _row(merged[merged["flag"] != "ok"], "takeoff in cut/at edge")
        print("[score] (observable = fair detector test; the rest are footage limits)")
    return merged


def plot_signals(tracks_path: Path, out_path: Path, max_clips: int = 12) -> None:
    data = json.loads(tracks_path.read_text())
    items = [(cid, e) for cid, e in data["clips"].items() if e["boxes"]][:max_clips]
    if not items:
        return
    cols = 3
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (cid, e) in zip(axes, items):
        traj = _traj(e)
        frames = np.array([fi for fi, _ in traj])
        ys = _interior_smooth(np.array([b.y2 for _, b in traj], dtype=float))
        resid = ys - _moving_average(ys, max(5, len(ys) // 4))
        tf = detect_takeoff_frame(traj)
        ax.plot(frames, resid, color="#4C72B0", lw=1.2)
        ax.axvline(tf, color="red", ls="--", lw=1.2)
        ax.set_title(f"{cid}\ntakeoff @ {tf}", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.suptitle("Detrended hoof-line residual (peak = detected takeoff)", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"[plot] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=Path, default=Path("data/annotations/tracks.json"))
    ap.add_argument("--fences", type=Path, default=Path("data/annotations/fences.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/annotations/auto.csv"))
    ap.add_argument("--plot", type=Path, default=Path("data/results/takeoff_local/signals.png"))
    ap.add_argument("--takeoffs", type=Path, default=Path("data/annotations/takeoffs.csv"))
    args = ap.parse_args()

    if not args.tracks.exists():
        print(f"[takeoff_local] missing {args.tracks} - run cache_tracks on Colab first")
        return
    df = recompute(args.tracks, args.fences, args.out)
    print(f"[takeoff_local] wrote {len(df)} rows -> {args.out}")
    regression_guard(df)
    score_against_truth(df, args.takeoffs, tracks_path=args.tracks)
    plot_signals(args.tracks, args.plot)


if __name__ == "__main__":
    main()

