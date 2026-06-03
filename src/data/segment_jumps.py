"""Detection-driven, takeoff-anchored clip segmentation (segment.py strategy "b").

The sliding-window cutter in segment.py is jump-agnostic: it emits a clip every
N seconds regardless of where the horse jumps, so most clips are not approaches.
This module locates takeoff events across the full video and cuts each clip to
END just after takeoff, so the approach fills the clip.

Takeoff signal: the horse-box bottom edge (y2, the hoof line) reverses down->up
at lift-off — the same cue geometry.detect_takeoff_frame uses, applied over the
whole video instead of within a pre-cut clip.

Multi-effort obstacles:
  - A combination (double/treble) has elements one or two strides apart, so its
    takeoffs land ~0.5-1.8 s apart with a brief landing between them. These are
    grouped into a single event anchored at the FIRST element's takeoff and
    tagged "combination"; downstream measures d to that first element.
  - Fences in a line sit 3-6 strides (~2-4 s) apart and each has a real approach,
    so they are emitted as separate "single" events.
  - Two reversals closer than combo_gap with no landing between them are the same
    effort double-detected and are dropped.

Window: `before` seconds before the anchor takeoff + `after` seconds after.
Each event yields one fixed-length clip; kind / n_elements are recorded in a
manifest (jumps_manifest.csv) so the type-conditioned pipeline can flag
combinations.

Subset-first workflow: pass limit_videos / max_seconds / max_clips_per_video to
preview a handful of clips and tune thresholds before committing to the full
corpus. Defaults write to data/clips_jumps so data/clips is left untouched.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.data.segment import cut_clip, video_duration
from src.preprocess.detect import HorseDetector


@dataclass
class TakeoffEvent:
    """One jump effort. `time` anchors the first element; `kind` is single or
    combination; `n_elements` counts grouped efforts (1 for a single fence)."""

    time: float
    kind: str
    n_elements: int


def _scan_horse_track(video: Path, detector: HorseDetector, sample_fps: float = 12.0,
                      max_seconds: float | None = None, start_seconds: float = 0.0):
    """Sample the video at ~sample_fps; return (native_fps, frame_h, samples).

    samples: list of (time_s, Box|None) — highest-confidence horse box per sample.
    Scans the interval [start_seconds, max_seconds]; start_seconds lets you skip
    intro/course-walk footage. The default 12 fps resolves one-stride combination
    takeoffs (~0.5 s apart), which 8 fps smears together.
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
    """Interpolate over NaNs then apply an interior-only 3-tap average.

    Interior-only smoothing avoids the zero-padding boundary artifact of
    np.convolve(mode="same"), which would corrupt the endpoint velocities.
    """
    idx = np.arange(len(x))
    good = ~np.isnan(x)
    if good.sum() < 2:
        return x
    xi = np.interp(idx, idx[good], x[good])
    out = xi.copy()
    out[1:-1] = (xi[:-2] + xi[1:-1] + xi[2:]) / 3.0
    return out


def find_takeoffs(samples, frame_h: float, before: float = 3.0, after: float = 1.0,
                  combo_gap: float = 1.8, land_thresh: float = 0.05,
                  vel_thresh: float = 0.15, min_present_frac: float = 0.6,
                  require_growth: bool = False, growth_ratio: float = 1.1
                  ) -> list[TakeoffEvent]:
    """Return takeoff events, grouping combination elements onto their first effort.

    A takeoff is a sharp upward reversal of the horse-box bottom edge: y2
    (normalized by frame height) moves UP fast at lift-off, so its velocity goes
    strongly negative. We require the horse to be present through most of the
    [t-before, t+after] window. Consecutive reversals within `combo_gap` seconds
    that have a landing spike (velocity > land_thresh) between them are the same
    combination and collapse to one event anchored at the first; reversals farther
    apart are separate jumps (a line); closer reversals with no landing are the
    same effort and are dropped. `require_growth` additionally demands the box
    grew over the approach.
    """
    n = len(samples)
    if n < 5:
        return []
    times = np.array([t for t, _ in samples])
    have = np.array([b is not None for _, b in samples])
    y2 = np.array([(b.y2 / frame_h) if b is not None else np.nan for _, b in samples])
    area = np.array([(b.w * b.h) if b is not None else np.nan for _, b in samples])

    ys = _nan_smooth(y2, k=3)
    vel = np.gradient(ys, times)

    def _accept(i: int) -> bool:
        t = float(times[i])
        in_win = (times >= t - before) & (times <= t + after)
        if in_win.sum() == 0 or have[in_win].mean() < min_present_frac:
            return False
        if require_growth:
            a = area[(times >= t - before) & (times <= t)]
            a = a[~np.isnan(a)]
            if len(a) < 2 or a[-1] <= growth_ratio * a[0]:
                return False
        return True

    cand = [i for i in range(1, n - 1)
            if not np.isnan(vel[i]) and vel[i] < -vel_thresh
            and vel[i] <= vel[i - 1] and vel[i] <= vel[i + 1] and _accept(i)]
    if not cand:
        return []

    groups: list[list[int]] = [[cand[0]]]
    for i in cand[1:]:
        prev = groups[-1][-1]
        gap = float(times[i] - times[prev])
        landed = bool(np.nanmax(vel[prev:i + 1]) > land_thresh)
        if gap > combo_gap:
            groups.append([i])
        elif landed:
            groups[-1].append(i)

    return [TakeoffEvent(time=float(times[g[0]]),
                         kind="combination" if len(g) >= 2 else "single",
                         n_elements=len(g))
            for g in groups]


def segment_video_jumps(video: Path, out_dir: Path, detector: HorseDetector,
                        before: float = 3.0, after: float = 1.0, sample_fps: float = 12.0,
                        max_seconds: float | None = None, start_seconds: float = 0.0,
                        max_clips: int | None = None,
                        **find_kw) -> tuple[list[TakeoffEvent], list[Path], list[dict]]:
    """Scan one video, find takeoff events, cut one [t-before, t+after] clip each.

    Returns (events, written_paths, manifest_records).
    """
    _, frame_h, samples = _scan_horse_track(video, detector, sample_fps=sample_fps,
                                            max_seconds=max_seconds, start_seconds=start_seconds)
    events = find_takeoffs(samples, frame_h, before=before, after=after, **find_kw)
    dur = video_duration(video)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    records: list[dict] = []
    for ev in events:
        start, end = ev.time - before, ev.time + after
        if start < 0 or end > dur:
            continue
        out = out_dir / f"{video.stem}_{int(start * 1000):07d}.mp4"
        if cut_clip(video, start, before + after, out):
            written.append(out)
            records.append({"clip_id": out.stem, "video": video.stem,
                            "takeoff_s": round(ev.time, 3), "kind": ev.kind,
                            "n_elements": ev.n_elements})
        if max_clips and len(written) >= max_clips:
            break
    return events, written, records


def segment_jumps(raw_dir: Path, out_dir: Path, detector: HorseDetector | None = None,
                  weights: str = "yolov8m.pt", device: str = "cuda",
                  before: float = 3.0, after: float = 1.0, sample_fps: float = 12.0,
                  limit_videos: int | None = None, max_seconds: float | None = None,
                  start_seconds: float = 0.0,
                  max_clips_per_video: int | None = None, **find_kw) -> int:
    """Notebook entry point. Writes clips + jumps_manifest.csv; returns clip count."""
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
    all_records: list[dict] = []
    for v in tqdm(vids, desc="videos", unit="vid"):
        events, written, records = segment_video_jumps(
            v, out_dir, detector, before=before, after=after, sample_fps=sample_fps,
            max_seconds=max_seconds, start_seconds=start_seconds,
            max_clips=max_clips_per_video, **find_kw)
        all_records.extend(records)
        n_combo = sum(1 for e in events if e.kind == "combination")
        tqdm.write(f"[segment_jumps] {v.name}: {len(events)} events "
                   f"({n_combo} combinations) -> {len(written)} clips")
        total += len(written)

    if all_records:
        manifest = out_dir / "jumps_manifest.csv"
        pd.DataFrame(all_records).to_csv(manifest, index=False)
        print(f"[segment_jumps] manifest -> {manifest}")
    print(f"[segment_jumps] total: {total} clips -> {out_dir}")
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/clips_jumps"))
    ap.add_argument("--before", type=float, default=3.0, help="seconds before takeoff")
    ap.add_argument("--after", type=float, default=1.0, help="seconds after takeoff")
    ap.add_argument("--sample-fps", type=float, default=12.0,
                    help="detection sampling rate while scanning for takeoffs")
    ap.add_argument("--weights", default="yolov8m.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-videos", type=int, default=None, help="subset: first N videos")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="subset: only scan the first N seconds of each video")
    ap.add_argument("--start-seconds", type=float, default=0.0,
                    help="subset: skip the first N seconds (intro/course walk) before scanning")
    ap.add_argument("--max-clips-per-video", type=int, default=None)
    ap.add_argument("--combo-gap", type=float, default=1.8,
                    help="max seconds between efforts grouped as one combination")
    ap.add_argument("--land-thresh", type=float, default=0.05,
                    help="min downward bottom-edge velocity marking a landing between combination efforts")
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
                  max_clips_per_video=args.max_clips_per_video, combo_gap=args.combo_gap,
                  land_thresh=args.land_thresh, vel_thresh=args.vel_thresh,
                  min_present_frac=args.min_present_frac, require_growth=args.require_growth)


if __name__ == "__main__":
    main()
