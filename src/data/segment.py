"""Cut raw footage into fixed-length approach clips.

Two strategies:
  (a) Sliding window: emit a clip every `--stride` seconds. Cheap, no detection.
  (b) Detection-driven: keep windows where YOLO sees a horse for >= K consecutive
      frames. Catches the actual jump phase rather than crowd shots.

For the milestone we use (a) — it lets us start SSL pretraining immediately on
fully unlabeled clips. We rerun with (b) once YOLO is wired up to filter the
labeled subset.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
from tqdm.auto import tqdm


def video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(n / fps) if fps > 0 else 0.0


def cut_clip(src: Path, start_s: float, dur_s: float, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.2f}",
        "-i", str(src),
        "-t", f"{dur_s:.2f}",
        "-an",
        "-vf", "scale=320:-2",
        "-r", "16",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        str(out),
    ]
    res = subprocess.run(cmd, capture_output=True)
    return res.returncode == 0


def segment_video(src: Path, out_dir: Path, clip_len: float, stride: float,
                  progress: bool = True) -> int:
    dur = video_duration(src)
    if dur < clip_len:
        return 0
    starts = []
    t = 0.0
    while t + clip_len <= dur:
        starts.append(t)
        t += stride
    count = 0
    it = tqdm(starts, desc=src.name[:30], leave=False, unit="clip") if progress else starts
    for t in it:
        out = out_dir / f"{src.stem}_{int(t*1000):07d}.mp4"
        if cut_clip(src, t, clip_len, out):
            count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/clips"))
    ap.add_argument("--clip-len", type=float, default=2.0, help="seconds per clip")
    ap.add_argument("--stride", type=float, default=4.0, help="seconds between clip starts")
    args = ap.parse_args()

    vids = sorted([p for p in args.raw.glob("*.mp4")])
    if not vids:
        print(f"[segment] no videos in {args.raw}")
        return

    total = 0
    for v in tqdm(vids, desc="videos", unit="vid"):
        n = segment_video(v, args.out, args.clip_len, args.stride)
        tqdm.write(f"[segment] {v.name}: {n} clips")
        total += n
    print(f"[segment] total: {total} clips -> {args.out}")


if __name__ == "__main__":
    main()

