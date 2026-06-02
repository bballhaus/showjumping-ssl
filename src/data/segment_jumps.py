"""Detection-driven, takeoff-anchored clip segmentation (segment.py strategy "b").

The sliding-window cutter in segment.py is jump-agnostic: it emits a clip every
N seconds regardless of where the horse jumps, so most clips are not approaches
(the horse is far from the fence, or the clip lands just after landing). This
module instead locates takeoff events across the full video and cuts each clip
to END just after takeoff, so the approach fills the clip.

Takeoff signal: the horse-box bottom edge (y2, the hoof line) reverses down->up
at lift-off — the same cue geometry.detect_takeoff_frame uses, applied over the
whole video instead of within a pre-cut clip.

Window: `before` seconds before takeoff + `after` seconds after (default 3 + 1,
so takeoff sits ~75% through the clip, inside the back half the geometry pipeline
searches).

Subset-first workflow: pass limit_videos / max_seconds / max_clips_per_video to
preview a handful of clips and tune thresholds before committing to the full
corpus. Defaults write to data/clips_jumps so data/clips is left untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from src.data.segment import cut_clip, video_duration
from src.preprocess.detect import HorseDetector


def _scan_horse_track(video: Path, detector: HorseDetector, sample_fps: float = 8.0,
                      max_seconds: float | None = None, start_seconds: float = 0.0):
    """Sample the video at ~sample_fps; return (native_fps, frame_h, samples).

    samples: list of (time_s, Box|None) — highest-confidence horse box per sample.
    Scans the interval [start_seconds, max_seconds]; start_seconds lets you skip
    intro/course-walk footage (and preview a known-active window quickly).
    """
    cap = cv2.VideoCapture(str(video))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720.0
    stride = max(1, round(native_fps / sample_fps))
    start_idx = max(0, int(start_seconds * native_fps))
    max_idx = None if max_seconds is None else int(max_seconds * native_fps)
    if start_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    samples: list[tuple[float, object]] = []
    idx = start_idx
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_idx is not None and idx > max_idx:
            break
        if idx % stride == 0:
            boxes = detector.detect_frame(frame)
            box = max(boxes, key=lambda b: b.score) if boxes else None
            samples.append((idx / native_fps, box))
        idx += 1
    cap.release()
    return native_fps, float(frame_h), samples


def _nan_smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    """Interpolate over NaNs (for smoothing only) and box-filter."""
    idx = np.arange(len(x))
    good = ~np.isnan(x)
    if good.sum() < 2:
        return x
    x_interp = np.interp(idx, idx[good], x[good])
    kernel = np.ones(k) / k
    return np.convolve(x_interp, kernel, mode="same")


def find_takeoffs(samples, frame_h: float, before: float = 3.0, after: float = 1.0,
                  min_gap: float = 5.0, vel_thresh: float = 0.15,
                  min_present_frac: float = 0.6, require_growth: bool = False,
                  growth_ratio: float = 1.1) -> list[float]:
    """Return takeoff timestamps (seconds).

    A takeoff is a sharp upward reversal of the horse-box bottom edge: y2
    (normalized by frame height) moves UP fast at lift-off, so its velocity goes
    strongly negative. We also require the horse to be present through most of the
    [t-before, t+after] window. `require_growth` additionally demands the box grew
    over the approach (closer to camera/fence) — off by default so the first
    preview favours recall.
    """
    n = len(samples)
    if n < 5:
        return []
    times = np.array([t for t, _ in samples])
    have = np.array([b is not None for _, b in samples])
    y2 = np.array([(b.y2 / frame_h) if b is not None else np.nan for _, b in samples])
    area = np.array([(b.w * b.h) if b is not None else np.nan for _, b in samples])

    ys = _nan_smooth(y2, k=3)
    vel = np.gradient(ys, times)  # +ve = box bottom moving down; -ve = moving up

    candidates = [i for i in range(1, n - 1)
                  if not np.isnan(vel[i]) and vel[i] < -vel_thresh
                  and vel[i] <= vel[i - 1] and vel[i] <= vel[i + 1]]

    takeoffs: list[tuple[float, float]] = []
    last_t = -1e9
    for i in candidates:
        t = float(times[i])
        in_win = (times >= t - before) & (times <= t + after)
        if in_win.sum() == 0 or have[in_win].mean() < min_present_frac:
            continue
        if require_growth:
            a = area[(times >= t - before) & (times <= t)]
            a = a[~np.isnan(a)]
            if len(a) < 2 or a[-1] <= growth_ratio * a[0]:
                continue
        if t - last_t < min_gap:
            # Same jump picked up twice — keep the sharper (more negative) one.
            if takeoffs and vel[i] < takeoffs[-1][1]:
                takeoffs[-1] = (t, float(vel[i]))
                last_t = t
            continue
        takeoffs.append((t, float(vel[i])))
        last_t = t
    return [t for t, _ in takeoffs]


def segment_video_jumps(video: Path, out_dir: Path, detector: HorseDetector,
                        before: float = 3.0, after: float = 1.0, sample_fps: float = 8.0,
                        max_seconds: float | None = None, start_seconds: float = 0.0,
                        max_clips: int | None = None,
                        **find_kw) -> tuple[list[float], list[Path]]:
    """Scan one video, find takeoffs, cut [t-before, t+after] clips."""
    _, frame_h, samples = _scan_horse_track(video, detector, sample_fps=sample_fps,
                                            max_seconds=max_seconds, start_seconds=start_seconds)
    takeoffs = find_takeoffs(samples, frame_h, before=before, after=after, **find_kw)
    dur = video_duration(video)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for t in takeoffs:
        start, end = t - before, t + after
        if start < 0 or end > dur:
            continue
        # Keep the {video_id}_{startms} naming so clip_id parsing stays consistent.
        out = out_dir / f"{video.stem}_{int(start * 1000):07d}.mp4"
        if cut_clip(video, start, before + after, out):
            written.append(out)
        if max_clips and len(written) >= max_clips:
            break
    return takeoffs, written


def segment_jumps(raw_dir: Path, out_dir: Path, detector: HorseDetector | None = None,
                  weights: str = "yolov8m.pt", device: str = "cuda",
                  before: float = 3.0, after: float = 1.0, sample_fps: float = 8.0,
                  limit_videos: int | None = None, max_seconds: float | None = None,
                  start_seconds: float = 0.0,
                  max_clips_per_video: int | None = None, **find_kw) -> int:
    """Notebook entry point. Returns total clips written. See module docstring."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    vids = sorted(raw_dir.glob("*.mp4"))
    if limit_videos:
        vids = vids[:limit_videos]
    if not vids:
        print(f"[segment_jumps] no videos in {raw_dir}")
        return 0
    if detector is None:
        detector = HorseDetector(weights=weights, device=device)

    total = 0
    for v in tqdm(vids, desc="videos", unit="vid"):
        takeoffs, written = segment_video_jumps(
            v, out_dir, detector, before=before, after=after, sample_fps=sample_fps,
            max_seconds=max_seconds, start_seconds=start_seconds,
            max_clips=max_clips_per_video, **find_kw)
        tqdm.write(f"[segment_jumps] {v.name}: {len(takeoffs)} takeoffs -> {len(written)} clips")
        total += len(written)
    print(f"[segment_jumps] total: {total} clips -> {out_dir}")
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/clips_jumps"))
    ap.add_argument("--before", type=float, default=3.0, help="seconds before takeoff")
    ap.add_argument("--after", type=float, default=1.0, help="seconds after takeoff")
    ap.add_argument("--sample-fps", type=float, default=8.0,
                    help="detection sampling rate while scanning for takeoffs")
    ap.add_argument("--weights", default="yolov8m.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-videos", type=int, default=None, help="subset: first N videos")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="subset: only scan the first N seconds of each video")
    ap.add_argument("--start-seconds", type=float, default=0.0,
                    help="subset: skip the first N seconds (intro/course walk) before scanning")
    ap.add_argument("--max-clips-per-video", type=int, default=None)
    ap.add_argument("--min-gap", type=float, default=5.0,
                    help="minimum seconds between accepted takeoffs")
    ap.add_argument("--vel-thresh", type=float, default=0.15,
                    help="min upward bottom-edge velocity (frame-heights/sec) for a takeoff")
    ap.add_argument("--min-present-frac", type=float, default=0.6,
                    help="horse must be visible in >=this fraction of the clip window")
    ap.add_argument("--require-growth", action="store_true",
                    help="also require the horse box to grow over the approach")
    args = ap.parse_args()

    segment_jumps(args.raw, args.out, weights=args.weights, device=args.device,
                  before=args.before, after=args.after, sample_fps=args.sample_fps,
                  limit_videos=args.limit_videos, max_seconds=args.max_seconds,
                  start_seconds=args.start_seconds,
                  max_clips_per_video=args.max_clips_per_video, min_gap=args.min_gap,
                  vel_thresh=args.vel_thresh, min_present_frac=args.min_present_frac,
                  require_growth=args.require_growth)


if __name__ == "__main__":
    main()
