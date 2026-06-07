"""Fine-tune YOLOv8 to detect showjumping fences.

Pretrained COCO YOLO has a `horse` class but no fence class, so the milestone
hand-drew fence boxes. This recovers the proposal's automated path: export the
hand-drawn boxes in `fences.csv` to YOLO format (one frame image + one label
file per box), then fine-tune YOLOv8 on that single-class ("fence") dataset.

The box was drawn on a scrubbed frame; we use the recorded `frame` index to grab
the exact image. Rows without a `frame` (the original milestone boxes) fall back
to the clip midpoint.

Usage:
    python -m src.preprocess.fence_yolo export --clips data/clips \
        --fences data/annotations/fences.csv --out data/fence_yolo
    python -m src.preprocess.fence_yolo train  --data data/fence_yolo/dataset.yaml \
        --weights yolov8n.pt --epochs 100
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import pandas as pd

from .detect import Box


def _frame_at(clip: Path, frame_idx: int | None):
    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fi = n // 2 if frame_idx is None or frame_idx < 0 else min(frame_idx, n - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _yolo_line(box: Box, w: int, h: int, cls: int = 0) -> str:
    cx = box.cx / w
    cy = box.cy / h
    bw = box.w / w
    bh = box.h / h
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def export(clips_dir: Path, fences_csv: Path, out_dir: Path, val_frac: float = 0.2,
           seed: int = 0) -> Path:
    df = pd.read_csv(fences_csv)
    if df.empty:
        raise SystemExit(f"[fence_yolo] {fences_csv} has no boxes")

    rng = random.Random(seed)
    rows = df.to_dict("records")
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_frac))
    splits = {"val": rows[:n_val], "train": rows[n_val:]}

    for split, recs in splits.items():
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        written = 0
        for r in recs:
            clip = clips_dir / f"{r['clip_id']}.mp4"
            if not clip.exists():
                print(f"[fence_yolo] missing clip {clip.name}, skipping")
                continue
            frame_idx = int(r["frame"]) if "frame" in r and pd.notna(r.get("frame")) else None
            frame = _frame_at(clip, frame_idx)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            box = Box(float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
            stem = r["clip_id"]
            cv2.imwrite(str(out_dir / "images" / split / f"{stem}.jpg"), frame)
            (out_dir / "labels" / split / f"{stem}.txt").write_text(_yolo_line(box, w, h) + "\n")
            written += 1
        print(f"[fence_yolo] {split}: wrote {written} images")

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: fence\n")
    print(f"[fence_yolo] dataset.yaml -> {yaml_path}")
    return yaml_path


def train(data_yaml: Path, weights: str = "yolov8n.pt", epochs: int = 100,
          imgsz: int = 320, device: str = "cpu", project: Path = Path("checkpoints/fence_yolo")):
    from ultralytics import YOLO
    model = YOLO(weights)
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, device=device,
                project=str(project), name="train", exist_ok=True)
    best = project / "train" / "weights" / "best.pt"
    print(f"[fence_yolo] best weights -> {best}")
    return best


class FenceDetector:
    """Inference wrapper around a fine-tuned fence YOLO model."""

    def __init__(self, weights: str, conf: float = 0.25, device: str = "cpu"):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device

    def detect_frame(self, frame) -> list[Box]:
        res = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False)[0]
        boxes: list[Box] = []
        if res.boxes is None:
            return boxes
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            boxes.append(Box(x1, y1, x2, y2, score=float(b.conf[0]), label="fence"))
        return boxes

    def best_box(self, frame) -> Box | None:
        boxes = self.detect_frame(frame)
        return max(boxes, key=lambda b: b.score) if boxes else None


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("export")
    pe.add_argument("--clips", type=Path, default=Path("data/clips"))
    pe.add_argument("--fences", type=Path, default=Path("data/annotations/fences.csv"))
    pe.add_argument("--out", type=Path, default=Path("data/fence_yolo"))
    pe.add_argument("--val-frac", type=float, default=0.2)

    pt = sub.add_parser("train")
    pt.add_argument("--data", type=Path, default=Path("data/fence_yolo/dataset.yaml"))
    pt.add_argument("--weights", type=str, default="yolov8n.pt")
    pt.add_argument("--epochs", type=int, default=100)
    pt.add_argument("--imgsz", type=int, default=320)
    pt.add_argument("--device", type=str, default="cpu")

    args = ap.parse_args()
    if args.cmd == "export":
        export(args.clips, args.fences, args.out, val_frac=args.val_frac)
    elif args.cmd == "train":
        train(args.data, weights=args.weights, epochs=args.epochs,
              imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()

