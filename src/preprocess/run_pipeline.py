"""Run the full preprocessing pipeline over a directory of clips.

Inputs:  data/clips/*.mp4, optional data/annotations/fences.csv with hand-drawn
         fence boxes (clip_id, x1, y1, x2, y2, pole_count).
Output:  CSV with one row per clip: clip_id, type, d_meters, takeoff_frame,
         mpp, n_horse_detections.

For clips without a fence annotation we still record horse-detection stats so
we can sort clips by how visible the action is.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import pandas as pd

from .detect import Box, HorseDetector
from .fence_type import classify_fence
from .geometry import detect_takeoff_frame, meters_per_pixel, takeoff_distance


def load_fence_annotations(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        out[str(row["clip_id"])] = {
            "box": Box(float(row["x1"]), float(row["y1"]),
                       float(row["x2"]), float(row["y2"]), label="fence"),
            "pole_count": int(row["pole_count"]) if "pole_count" in row and pd.notna(row["pole_count"]) else None,
        }
    return out


def _auto_fence_box(clip_path: Path, fence_detector, takeoff_fi: int) -> Box | None:
    """Detect the fence with the fine-tuned YOLO model near the takeoff frame."""
    import cv2
    cap = cv2.VideoCapture(str(clip_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(takeoff_fi, 0), n - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return fence_detector.best_box(frame)


def process_clip(clip_path: Path, detector: HorseDetector,
                 fence_ann: dict | None, fence_detector=None) -> dict:
    horses = detector.detect_video(clip_path, stride=2)
    horse_traj: list[tuple[int, Box]] = []
    for fi, boxes in horses:
        if not boxes:
            continue
        # Take the highest-confidence horse box per sampled frame.
        horse_traj.append((fi, max(boxes, key=lambda b: b.score)))

    row = {
        "clip_id": clip_path.stem,
        "n_horse_detections": len(horse_traj),
        "type": "",
        "d_meters": float("nan"),
        "mpp": float("nan"),
        "takeoff_frame": -1,
    }
    if not horse_traj:
        return row

    pole_count = None
    if fence_ann is not None:
        fence_box: Box = fence_ann["box"]
        pole_count = fence_ann.get("pole_count")
    elif fence_detector is not None:
        takeoff_fi = detect_takeoff_frame(horse_traj)
        fence_box = _auto_fence_box(clip_path, fence_detector, takeoff_fi)
        if fence_box is None:
            return row
    else:
        return row
    mpp = meters_per_pixel(fence_box)
    takeoff_fi = detect_takeoff_frame(horse_traj)
    # Pick horse box closest to takeoff frame.
    takeoff_horse = min(horse_traj, key=lambda x: abs(x[0] - takeoff_fi))[1]
    d = takeoff_distance(takeoff_horse, fence_box, mpp)
    ftype = classify_fence(fence_box, pole_count=pole_count)

    row.update({
        "type": ftype,
        "d_meters": d,
        "mpp": mpp,
        "takeoff_frame": takeoff_fi,
    })
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, default=Path("data/clips"))
    ap.add_argument("--fences", type=Path, default=Path("data/annotations/fences.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/annotations/auto.csv"))
    ap.add_argument("--weights", type=str, default="yolov8n.pt")
    ap.add_argument("--fence-weights", type=str, default=None,
                    help="Fine-tuned fence YOLO weights; enables automatic fence detection "
                         "for clips with no hand annotation.")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    detector = HorseDetector(weights=args.weights, device=args.device)
    fence_detector = None
    if args.fence_weights:
        from .fence_yolo import FenceDetector
        fence_detector = FenceDetector(args.fence_weights, device=args.device)
    fences = load_fence_annotations(args.fences)
    clips = sorted(args.clips.glob("*.mp4"))
    print(f"[pipeline] {len(clips)} clips, {len(fences)} fence annotations"
          f"{', auto-detect on' if fence_detector else ''}")

    rows = []
    for cp in clips:
        ann = fences.get(cp.stem)
        rows.append(process_clip(cp, detector, ann, fence_detector=fence_detector))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[pipeline] wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
