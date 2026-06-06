"""Score-overlay knockdown mining: label jumps from the broadcast fault counter.

Knockdowns are ~8% of jumps, so uniform sampling starves the positive class. But
the broadcast tells us exactly which jump faulted: a rail down is precisely +4 on
the rider's running fault count, and it is the ONLY thing that adds exactly 4 mid
round (time penalties accrue in +1 steps; refusals also add 4 but read as a stop,
not a clear jump). The scoring delay means the counter updates ~1-3 s after the
rail actually falls, so the jump that caused a +4 step is the takeoff immediately
preceding it.

Pipeline: reuse segment_jumps' takeoff detection to locate every jump in a raw
video, then for each takeoff OCR the fault region in a short window before and
after the jump. A clean +4 between the two windows flags that jump as a knockdown
candidate. Candidates are written sorted by confidence to feed the OutcomeAnnotator
as a priority queue, turning a needle-in-haystack labeling job into a short
confirm-or-reject pass over high-precision guesses.

Two practical limits the broadcast imposes:
  - The fault region differs per broadcast (Wellington's "W" lower-third, Longines
    World Cup, Tryon all place it differently and some hide it mid-round), so
    SCORE_ROI is configured per source video in normalized coordinates. Run
    `suggest_roi` to dump overlay crops and eyeball the box before mining.
  - When the live counter is hidden, the windowed +4 signal is unavailable; the
    end-of-round total (mode='round') still tells you the round HAD a rail, so its
    jumps are all worth reviewing even if we can't say which one faulted.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.data.segment import cut_clip, video_duration
from src.data.segment_jumps import _scan_horse_track, find_takeoffs
from src.preprocess.detect import HorseDetector


SCORE_ROI: dict[str, tuple[float, float, float, float]] = {
    "2S-4eXbehr4": None,
    "jtEeeuyIpLA": None,
    "wCv2DYtxr5U": None,
    "h5_V5GYf7iM": None,
    "kZbGlTmo8Ds": None,
    "ApvPWiz8nBM": None,
}

PLAUSIBLE_FAULTS = {0, 4, 8, 12, 16, 20, 24}


@dataclass
class KnockdownCandidate:
    """One jump the fault counter implicates. `delta` is the OCR'd fault step;
    `confidence` blends step cleanliness with how stable the two reads were."""

    clip_id: str
    video: str
    takeoff_s: float
    fault_before: int
    fault_after: int
    delta: int
    confidence: float


class FaultReader:
    """Digit-only OCR over the score ROI, lazy-loading EasyOCR on first use.

    EasyOCR is robust to stylized broadcast fonts and runs on the Colab GPU; it is
    imported lazily so importing this module never requires it. The allowlist is
    restricted to digits because the only thing we read is a small fault integer.
    """

    def __init__(self, device: str = "cuda"):
        self._reader = None
        self._gpu = device != "cpu"

    def _ensure(self):
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(["en"], gpu=self._gpu, verbose=False)
        return self._reader

    def read_int(self, crop: np.ndarray) -> int | None:
        """Return the integer in the crop, or None if nothing digit-like is read.

        Upscales small crops so thin broadcast digits survive binarization, then
        keeps the highest-confidence all-digit token. Returns None rather than a
        guess when the read is empty so the caller can treat it as a dropout
        instead of a fault change."""
        if crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if h < 48:
            scale = 48.0 / max(h, 1)
            crop = cv2.resize(crop, (int(w * scale), 48), interpolation=cv2.INTER_CUBIC)
        out = self._ensure().readtext(crop, allowlist="0123456789", detail=1)
        best, best_conf = None, 0.0
        for _, text, conf in out:
            text = text.strip()
            if text.isdigit() and conf > best_conf:
                best, best_conf = int(text), conf
        return best


def _crop_roi(frame: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi
    return frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]


def _read_window(cap, reader: FaultReader, roi, t0: float, t1: float, fps: float,
                 step: float = 0.25) -> tuple[int | None, float]:
    """Mode fault value over [t0, t1] plus the fraction of reads matching it.

    Sampling several frames and taking the mode tolerates single-frame OCR
    misfires and brief overlay occlusion (a passing rail, motion blur). The
    agreement fraction becomes the per-window stability that feeds confidence."""
    vals: list[int] = []
    t = max(0.0, t0)
    while t <= t1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if ok:
            v = reader.read_int(_crop_roi(frame, roi))
            if v is not None:
                vals.append(v)
        t += step
    if not vals:
        return None, 0.0
    val, n = Counter(vals).most_common(1)[0]
    return val, n / len(vals)


def mine_video(video: Path, reader: FaultReader, detector: HorseDetector,
               roi: tuple[float, float, float, float],
               before: float = 3.0, after: float = 1.0, sample_fps: float = 12.0,
               pre_window: tuple[float, float] = (-2.0, -0.5),
               post_window: tuple[float, float] = (1.5, 5.0),
               min_stability: float = 0.5, restrict_plausible: bool = True,
               start_seconds: float = 0.0, max_seconds: float | None = None,
               **find_kw) -> list[KnockdownCandidate]:
    """Detect takeoffs in one video, then flag those followed by a +4 fault step.

    For each takeoff the fault counter is read in `pre_window` (just before the
    jump) and `post_window` (after the scoring delay), both relative to takeoff.
    A jump is a candidate when both reads are stable, the pre read is a plausible
    round total, and the step is a positive multiple of 4. Exactly +4 with high
    stability scores near 1.0; +8 (two rails, rare on one effort) and shakier
    reads score lower so the review queue sorts the surest knockdowns first.
    """
    fps, frame_h, samples = _scan_horse_track(
        video, detector, sample_fps=sample_fps,
        start_seconds=start_seconds, max_seconds=max_seconds)
    events = find_takeoffs(samples, frame_h, before=before, after=after, **find_kw)
    dur = video_duration(video)

    cap = cv2.VideoCapture(str(video))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 25.0
    out: list[KnockdownCandidate] = []
    for ev in tqdm(events, desc=f"ocr {video.stem[:18]}", unit="jump", leave=False):
        t = ev.time
        pre, pre_s = _read_window(cap, reader, roi, t + pre_window[0], t + pre_window[1], native_fps)
        post, post_s = _read_window(cap, reader, roi, t + post_window[0], t + post_window[1], native_fps)
        if pre is None or post is None:
            continue
        delta = post - pre
        if delta <= 0 or delta % 4 != 0:
            continue
        if restrict_plausible and pre not in PLAUSIBLE_FAULTS:
            continue
        if min(pre_s, post_s) < min_stability:
            continue
        clip_start = t - before
        if clip_start < 0 or t + after > dur:
            continue
        conf = min(pre_s, post_s) * (1.0 if delta == 4 else 0.6)
        out.append(KnockdownCandidate(
            clip_id=f"{video.stem}_{int(clip_start * 1000):07d}", video=video.stem,
            takeoff_s=round(t, 3), fault_before=pre, fault_after=post,
            delta=delta, confidence=round(conf, 3)))
    cap.release()
    return out


def mine(raw_dir: Path, out_csv: Path, clips_dir: Path | None = None,
         weights: str = "yolov8m.pt", device: str = "cuda",
         before: float = 3.0, after: float = 1.0, limit_videos: int | None = None,
         **kw) -> pd.DataFrame:
    """Notebook entry point: mine every raw video for knockdown candidates.

    Writes the candidate table sorted by descending confidence to `out_csv`. When
    `clips_dir` is given, also cuts the flagged [t-before, t+after] clips so the
    OutcomeAnnotator can be pointed straight at them with 'knockdown' pre-selected.
    Videos with no configured SCORE_ROI are skipped with a warning.
    """
    raw_dir, out_csv = Path(raw_dir), Path(out_csv)
    vids = sorted(raw_dir.glob("*.mp4"))
    if limit_videos:
        vids = vids[:limit_videos]
    detector = HorseDetector(weights=weights, device=device)
    reader = FaultReader(device=device)

    rows: list[dict] = []
    for v in tqdm(vids, desc="videos", unit="vid"):
        roi = SCORE_ROI.get(v.stem)
        if roi is None:
            tqdm.write(f"[mine_knockdowns] no SCORE_ROI for {v.stem}; skipping (run suggest_roi)")
            continue
        cands = mine_video(v, reader, detector, roi, before=before, after=after, **kw)
        tqdm.write(f"[mine_knockdowns] {v.name}: {len(cands)} knockdown candidates")
        for c in cands:
            rows.append(c.__dict__)
            if clips_dir is not None:
                cut_clip(v, c.takeoff_s - before, before + after,
                         Path(clips_dir) / f"{c.clip_id}.mp4")

    df = pd.DataFrame(rows).sort_values("confidence", ascending=False) if rows else pd.DataFrame(
        columns=[f.name for f in KnockdownCandidate.__dataclass_fields__.values()])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[mine_knockdowns] {len(df)} candidates -> {out_csv}")
    return df


def suggest_roi(video: Path, out_dir: Path, at_seconds: tuple[float, ...] = (60, 120, 240),
                roi: tuple[float, float, float, float] | None = None) -> list[Path]:
    """Dump full frames (and the candidate ROI crop) at a few times for ROI tuning.

    The fault box location is broadcast-specific, so configuring SCORE_ROI is a
    manual eyeball step. Saves frames at `at_seconds` to `out_dir`; pass a trial
    `roi` to also save the crop and confirm it isolates the fault digits before
    committing the box to SCORE_ROI."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    written: list[Path] = []
    for s in at_seconds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        p = out_dir / f"{video.stem}_t{int(s)}.png"
        cv2.imwrite(str(p), frame)
        written.append(p)
        if roi is not None:
            pc = out_dir / f"{video.stem}_t{int(s)}_roi.png"
            cv2.imwrite(str(pc), _crop_roi(frame, roi))
            written.append(pc)
    cap.release()
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/annotations/knockdown_candidates.csv"))
    ap.add_argument("--clips", type=Path, default=None,
                    help="if set, cut the flagged clips here for the annotator")
    ap.add_argument("--weights", default="yolov8m.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--before", type=float, default=3.0)
    ap.add_argument("--after", type=float, default=1.0)
    ap.add_argument("--limit-videos", type=int, default=None)
    ap.add_argument("--min-stability", type=float, default=0.5)
    ap.add_argument("--no-restrict-plausible", action="store_true",
                    help="accept any +4k step, not only plausible round totals")
    args = ap.parse_args()
    mine(args.raw, args.out, clips_dir=args.clips, weights=args.weights, device=args.device,
         before=args.before, after=args.after, limit_videos=args.limit_videos,
         min_stability=args.min_stability, restrict_plausible=not args.no_restrict_plausible)


if __name__ == "__main__":
    main()
