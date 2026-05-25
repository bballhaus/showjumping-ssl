"""YOLOv8 detection wrapper.

Pretrained COCO YOLO has a 'horse' class (id 17) but no showjumping-fence
class. For the milestone we:
  - run YOLO to get horse boxes per frame (free, no labels)
  - load fence boxes from a hand-annotation CSV (data/annotations/fences.csv)

Once we have a labeled fence set we can fine-tune YOLO; for now this lets us
make progress without waiting on a custom detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


HORSE_CLASS_ID = 17  # COCO


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    label: str = ""

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class HorseDetector:
    """Lightweight wrapper around Ultralytics YOLOv8 for horse detection."""

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35, device: str = "cpu"):
        # Import lazily so the rest of the codebase can be imported without ultralytics.
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device

    def detect_frame(self, frame: np.ndarray) -> list[Box]:
        result = self.model.predict(frame, conf=self.conf, classes=[HORSE_CLASS_ID],
                                    device=self.device, verbose=False)[0]
        boxes: list[Box] = []
        if result.boxes is None:
            return boxes
        for b in result.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            boxes.append(Box(x1, y1, x2, y2, score=float(b.conf[0]), label="horse"))
        return boxes

    def detect_video(self, video_path: Path, stride: int = 2) -> list[tuple[int, list[Box]]]:
        """Returns (frame_idx, boxes) for every `stride`-th frame."""
        cap = cv2.VideoCapture(str(video_path))
        out: list[tuple[int, list[Box]]] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                out.append((idx, self.detect_frame(frame)))
            idx += 1
        cap.release()
        return out
