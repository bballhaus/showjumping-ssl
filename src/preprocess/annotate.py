"""Tiny OpenCV annotation tool for fence boxes.

Walks through clips, lets you draw a single fence box per clip and tap a number
key (1 = vertical, 2 = oxer, or leave blank for "auto"). Saves to
data/annotations/fences.csv.

Controls: drag with mouse to draw the box; 1/2 = label and save; n = next clip
without saving; q = quit.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def grab_middle_frame(clip: Path):
    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, default=Path("data/clips"))
    ap.add_argument("--out", type=Path, default=Path("data/annotations/fences.csv"))
    args = ap.parse_args()

    if not args.clips.exists():
        raise SystemExit(f"[annotate] --clips path does not exist: {args.clips}")
    all_clips = sorted(args.clips.glob("*.mp4"))
    if not all_clips:
        raise SystemExit(
            f"[annotate] no .mp4 files found in {args.clips}.\n"
            f"           If this is a Google Drive path, make sure Drive Desktop "
            f"is installed and the folder has synced locally.")
    print(f"[annotate] found {len(all_clips)} clips in {args.clips}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if args.out.exists():
        with args.out.open() as f:
            for row in csv.DictReader(f):
                seen.add(row["clip_id"])
    if seen:
        print(f"[annotate] resuming — {len(seen)} clips already annotated")

    with args.out.open("a", newline="") as f:
        writer = csv.writer(f)
        if not seen:
            writer.writerow(["clip_id", "x1", "y1", "x2", "y2", "pole_count"])

        for clip in all_clips:
            if clip.stem in seen:
                continue
            frame = grab_middle_frame(clip)
            if frame is None:
                continue

            roi = cv2.selectROI(f"fence: {clip.name} (1=vert, 2=oxer, n=skip)",
                                frame, showCrosshair=False, fromCenter=False)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == ord("q"):
                break
            if key == ord("n") or roi == (0, 0, 0, 0):
                continue
            pole = {ord("1"): 1, ord("2"): 2}.get(key, "")
            x, y, w, h = roi
            writer.writerow([clip.stem, x, y, x + w, y + h, pole])
            f.flush()
            print(f"[annotate] {clip.stem} -> pole_count={pole}")


if __name__ == "__main__":
    main()

