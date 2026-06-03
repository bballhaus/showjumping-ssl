"""Replay find_takeoffs from cached scans — no YOLO, no video.

Loads `data/annotations/scan_cache.json` (written on Colab by `cache_scan`),
runs `find_takeoffs` on each video's track, prints the events, and saves a
per-video plot of the detrended hoof-line velocity with every detected takeoff
marked (green = single, red = combination element) and the candidate threshold
drawn in. This makes false positives ("too many clips") and missed combination
grouping visible at a glance, so thresholds can be tuned in seconds on a laptop.

Run locally:  python -m src.data.jumps_local
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.segment_jumps import _moving_average, _nan_smooth, find_takeoffs
from src.preprocess.detect import Box


def _samples(video_entry: dict) -> list[tuple[float, Box | None]]:
    out: list[tuple[float, Box | None]] = []
    for s in video_entry["samples"]:
        if len(s) == 2 and s[1] is None:
            out.append((float(s[0]), None))
        else:
            t, x1, y1, x2, y2, score = s
            out.append((float(t), Box(x1, y1, x2, y2, score=score, label="horse")))
    return out


def _detrended_velocity(samples, frame_h: float):
    times = np.array([t for t, _ in samples])
    y2 = np.array([(b.y2 / frame_h) if b is not None else np.nan for _, b in samples])
    ys = _nan_smooth(y2, k=3)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    win = max(5, int(round(0.8 / dt))) if dt > 0 else 5
    ys = ys - _moving_average(ys, win)
    vel = np.gradient(ys, times)
    return times, ys, vel


def replay(scan_path: Path, plot_path: Path, **find_kw) -> None:
    data = json.loads(scan_path.read_text())
    videos = data["videos"]
    n = len(videos)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * max(n, 1)))
    axes = np.atleast_1d(axes)
    vel_thresh = float(find_kw.get("vel_thresh", 0.15))
    land_thresh = float(find_kw.get("land_thresh", 0.05))

    for ax, (vid, entry) in zip(axes, videos.items()):
        samples = _samples(entry)
        times, ys, vel = _detrended_velocity(samples, entry["frame_h"])
        events = find_takeoffs(samples, entry["frame_h"], **find_kw)
        n_combo = sum(1 for e in events if e.kind == "combination")
        print(f"[jumps_local] {vid}: {len(events)} events ({n_combo} combinations)")
        for e in events:
            print(f"    t={e.time:6.2f}s  {e.kind:11s}  n_elements={e.n_elements}")

        ax.plot(times, vel, color="#4C72B0", lw=1.0, label="bottom-edge velocity")
        ax.axhline(-vel_thresh, color="green", ls=":", lw=0.9, label=f"-vel_thresh ({vel_thresh})")
        ax.axhline(land_thresh, color="orange", ls=":", lw=0.9, label=f"land_thresh ({land_thresh})")
        ax.axhline(0, color="grey", lw=0.5)
        for e in events:
            ax.axvline(e.time, color="red" if e.kind == "combination" else "green",
                       ls="--", lw=1.3)
        ax.set_title(f"{vid}  —  {len(events)} events ({n_combo} combos)", fontsize=9)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("detrended dy2/dt")
        ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=110)
    print(f"[jumps_local] wrote {plot_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, default=Path("data/annotations/scan_cache.json"))
    ap.add_argument("--plot", type=Path, default=Path("data/results/jumps_local/velocity.png"))
    ap.add_argument("--before", type=float, default=3.0)
    ap.add_argument("--after", type=float, default=1.0)
    ap.add_argument("--combo-gap", type=float, default=1.8)
    ap.add_argument("--land-thresh", type=float, default=0.05)
    ap.add_argument("--vel-thresh", type=float, default=0.15)
    ap.add_argument("--min-present-frac", type=float, default=0.6)
    args = ap.parse_args()

    if not args.scan.exists():
        print(f"[jumps_local] missing {args.scan} - run cache_scan on Colab first")
        return
    replay(args.scan, args.plot, before=args.before, after=args.after,
           combo_gap=args.combo_gap, land_thresh=args.land_thresh,
           vel_thresh=args.vel_thresh, min_present_frac=args.min_present_frac)


if __name__ == "__main__":
    main()
