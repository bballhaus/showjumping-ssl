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


def render_truth_comparison(clips_dir: Path, tracks_path: Path, fences_path: Path,
                            takeoffs_path: Path, out_path: Path,
                            max_clips: int = 12) -> None:
    """Side-by-side detected-vs-hand-marked takeoff frames, from cached tracks.

    For every clip that has a fence box, a cached trajectory, and a ground-truth
    takeoff, draws the detector's takeoff frame (horse box + fence box + d) beside
    the human-marked frame, titled with both frame numbers and the error. Uses the
    cached boxes in tracks.json, so it never re-runs YOLO.
    """
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tracks = json.loads(Path(tracks_path).read_text())["clips"]
    fences = pd.read_csv(fences_path)
    fences["clip_id"] = fences["clip_id"].astype(str)
    truth = pd.read_csv(takeoffs_path)
    truth["clip_id"] = truth["clip_id"].astype(str)
    fence_map = {r["clip_id"]: r for _, r in fences.iterrows()}
    true_map = dict(zip(truth["clip_id"], truth["takeoff_frame"]))

    ids = [c for c in true_map if c in tracks and c in fence_map][:max_clips]
    if not ids:
        print("[viz] no clips with fence + track + ground truth")
        return

    def grab(clip_id: str, fi: int):
        cap = cv2.VideoCapture(str(Path(clips_dir) / f"{clip_id}.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(int(fi), 0), n - 1))
        ok, fr = cap.read()
        cap.release()
        return fr if ok else None

    fig, axes = plt.subplots(len(ids), 2, figsize=(10, 3.4 * len(ids)))
    axes = np.atleast_2d(axes)
    for r, cid in enumerate(ids):
        traj = [(int(b[0]), Box(b[1], b[2], b[3], b[4], score=b[5]))
                for b in tracks[cid]["boxes"]]
        row = fence_map[cid]
        fence_box = Box(float(row["x1"]), float(row["y1"]),
                        float(row["x2"]), float(row["y2"]), label="fence")
        mpp = meters_per_pixel(fence_box)
        det_fi = detect_takeoff_frame(traj)
        true_fi = int(true_map[cid])
        det_horse = min(traj, key=lambda x: abs(x[0] - det_fi))[1]
        true_horse = min(traj, key=lambda x: abs(x[0] - true_fi))[1]
        d = takeoff_distance(det_horse, fence_box, mpp)
        for c, (fi, horse, tag) in enumerate([
                (det_fi, det_horse, f"DETECTED f{det_fi}  d={d:.2f}m"),
                (true_fi, true_horse, f"TRUE f{true_fi}  (err {det_fi - true_fi:+d})")]):
            frame = grab(cid, fi)
            ax = axes[r, c]
            if frame is None:
                ax.axis("off")
                continue
            _draw(frame, [(horse, (0, 200, 0), "horse"),
                          (fence_box, (0, 100, 255), "fence")])
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{cid}\n{tag}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    print(f"[viz] wrote {out_path} ({len(ids)} clips)")


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
