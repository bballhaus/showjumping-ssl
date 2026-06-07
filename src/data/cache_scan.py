"""Cache full-video horse-track scans so find_takeoffs can be tuned offline.

`segment_jumps` scans each raw video at ~12 fps for horse boxes, then runs
`find_takeoffs` on that track. The scan is the only expensive, GPU-bound,
video-bound part. This dumps the scan (time + horse box per sample) for a few
videos to one JSON, so the takeoff-event logic and its thresholds can be tuned on
a laptop with no torch and no raw video via `jumps_local.py`.

Run on Colab:  python -m src.data.cache_scan --raw data/raw --limit-videos 3
Output:        data/annotations/scan_cache.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.segment_jumps import _scan_horse_track
from src.preprocess.detect import HorseDetector


def dump_scans(raw_dir: Path, detector: HorseDetector, out_path: Path,
               sample_fps: float = 12.0, limit_videos: int | None = None,
               max_seconds: float | None = None, start_seconds: float = 0.0) -> dict:
    vids = sorted(raw_dir.glob("*.mp4"))
    if limit_videos:
        vids = vids[:limit_videos]
    data: dict = {"sample_fps": sample_fps, "videos": {}}
    for i, v in enumerate(vids, 1):
        print(f"[cache_scan] ({i}/{len(vids)}) scanning {v.name} ...", flush=True)
        native_fps, frame_h, samples = _scan_horse_track(
            v, detector, sample_fps=sample_fps,
            max_seconds=max_seconds, start_seconds=start_seconds)
        last_t = samples[-1][0] if samples else 0.0
        print(f"[cache_scan]   -> {len(samples)} samples, {last_t:.1f}s", flush=True)
        packed = []
        for t, b in samples:
            if b is None:
                packed.append([float(t), None])
            else:
                packed.append([float(t), float(b.x1), float(b.y1),
                               float(b.x2), float(b.y2), float(b.score)])
        data["videos"][v.stem] = {
            "native_fps": float(native_fps),
            "frame_h": float(frame_h),
            "samples": packed,
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data))
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/annotations/scan_cache.json"))
    ap.add_argument("--weights", default="yolov8m.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample-fps", type=float, default=12.0)
    ap.add_argument("--limit-videos", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--start-seconds", type=float, default=0.0)
    args = ap.parse_args()

    detector = HorseDetector(weights=args.weights, device=args.device)
    data = dump_scans(args.raw, detector, args.out, sample_fps=args.sample_fps,
                      limit_videos=args.limit_videos, max_seconds=args.max_seconds,
                      start_seconds=args.start_seconds)
    n = sum(len(v["samples"]) for v in data["videos"].values())
    print(f"[cache_scan] {len(data['videos'])} videos, {n} samples -> {args.out}")


if __name__ == "__main__":
    main()

