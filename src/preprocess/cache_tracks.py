"""Cache YOLO horse-box trajectories so the takeoff detector can be iterated offline.

Detection is the only step that needs a GPU, the video files, and torch. It runs
once here and dumps each clip's horse-box trajectory — the single input
`detect_takeoff_frame` consumes — to a small JSON. Afterwards the takeoff math,
the `d` geometry, and the diagnostics replay on a laptop with no torch and no
video via `takeoff_local.py`, so tuning the detector no longer needs Colab.

Run on Colab:  python -m src.preprocess.cache_tracks --device cuda
Output:        data/annotations/tracks.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .detect import HorseDetector


def dump_tracks(clips_dir: Path, detector: HorseDetector, out_path: Path,
                stride: int = 2) -> dict:
    clips = sorted(clips_dir.glob("*.mp4"))
    data: dict = {"stride": stride, "clips": {}}
    for cp in clips:
        cap = cv2.VideoCapture(str(cp))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        cap.release()
        traj: list[list[float]] = []
        for fi, boxes in detector.detect_video(cp, stride=stride):
            if not boxes:
                continue
            b = max(boxes, key=lambda b: b.score)
            traj.append([int(fi), float(b.x1), float(b.y1),
                         float(b.x2), float(b.y2), float(b.score)])
        data["clips"][cp.stem] = {
            "frame_count": frame_count,
            "frame_h": frame_h,
            "frame_w": frame_w,
            "boxes": traj,
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data))
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, default=Path("data/clips"))
    ap.add_argument("--out", type=Path, default=Path("data/annotations/tracks.json"))
    ap.add_argument("--weights", type=str, default="yolov8n.pt")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    detector = HorseDetector(weights=args.weights, device=args.device)
    data = dump_tracks(args.clips, detector, args.out, stride=args.stride)
    n_boxes = sum(len(c["boxes"]) for c in data["clips"].values())
    print(f"[cache_tracks] {len(data['clips'])} clips, {n_boxes} boxes -> {args.out}")


if __name__ == "__main__":
    main()
