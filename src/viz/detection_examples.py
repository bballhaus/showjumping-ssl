"""Render annotated frames: horse box (from YOLO) + fence box (from CSV) +
takeoff-distance d overlaid on the takeoff frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.preprocess.detect import Box, HorseDetector
from src.preprocess.geometry import (
    detect_takeoff_frame,
    meters_per_pixel,
    takeoff_distance,
)


def _draw(frame, boxes: list[tuple[Box, tuple[int, int, int], str]]):
    for b, color, label in boxes:
        x1, y1, x2, y2 = [int(v) for v in b.to_xyxy()]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def render_clip(clip_path: Path, fence_box: Box, pole_count: int | None,
                detector: HorseDetector, out_path: Path) -> None:
    horses = detector.detect_video(clip_path, stride=2)
    horse_traj = [(fi, max(b, key=lambda x: x.score)) for fi, b in horses if b]
    if not horse_traj:
        print(f"[viz] no horse in {clip_path.name}")
        return
    mpp = meters_per_pixel(fence_box)
    takeoff_fi = detect_takeoff_frame(horse_traj)
    takeoff_horse = min(horse_traj, key=lambda x: abs(x[0] - takeoff_fi))[1]
    d = takeoff_distance(takeoff_horse, fence_box, mpp)

    cap = cv2.VideoCapture(str(clip_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, takeoff_fi)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return
    _draw(frame, [
        (takeoff_horse, (0, 200, 0), "horse"),
        (fence_box, (0, 100, 255), "fence"),
    ])
    cv2.putText(frame, f"d = {d:.2f} m  (mpp={mpp:.4f})",
                (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    print(f"[viz] wrote {out_path} (d={d:.2f} m)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, default=Path("data/clips"))
    ap.add_argument("--fences", type=Path, default=Path("data/annotations/fences.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("milestone/figures/det"))
    ap.add_argument("--weights", type=str, default="yolov8n.pt")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max", type=int, default=6)
    args = ap.parse_args()

    if not args.fences.exists():
        print(f"[viz] need {args.fences}; run src.preprocess.annotate first")
        return
    df = pd.read_csv(args.fences)
    detector = HorseDetector(weights=args.weights, device=args.device)
    rendered: list[Path] = []
    for _, row in df.head(args.max).iterrows():
        clip = args.clips / f"{row['clip_id']}.mp4"
        if not clip.exists():
            continue
        fence_box = Box(float(row["x1"]), float(row["y1"]),
                        float(row["x2"]), float(row["y2"]), label="fence")
        pole = int(row["pole_count"]) if pd.notna(row.get("pole_count")) else None
        out = args.out_dir / f"{row['clip_id']}.jpg"
        render_clip(clip, fence_box, pole, detector, out)
        if out.exists():
            rendered.append(out)

    if rendered:
        _save_grid(rendered, args.out_dir.parent / "det_grid.png", cols=2)


def _save_grid(images: list[Path], out_path: Path, cols: int = 2) -> None:
    imgs = [cv2.imread(str(p)) for p in images]
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        return
    h = max(i.shape[0] for i in imgs)
    w = max(i.shape[1] for i in imgs)
    imgs = [cv2.copyMakeBorder(i, 0, h - i.shape[0], 0, w - i.shape[1],
                               cv2.BORDER_CONSTANT, value=(0, 0, 0)) for i in imgs]
    rows = []
    for r in range(0, len(imgs), cols):
        chunk = imgs[r:r + cols]
        while len(chunk) < cols:
            chunk.append(np.zeros_like(imgs[0]))
        rows.append(np.hstack(chunk))
    grid = np.vstack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"[viz] wrote {out_path}")


if __name__ == "__main__":
    main()
