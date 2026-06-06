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
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.data.segment import cut_clip, video_duration
from src.data.segment_jumps import _scan_horse_track, find_takeoffs
from src.preprocess.annotate_colab import _build_frame_nav, _jpeg_bytes, _jpeg_data_url
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


def _existing_clip_for(from_clips: Path, video_stem: str, t: float,
                       clip_seconds: float, tol: float = 0.4) -> Path | None:
    """Existing clip in `from_clips` whose window contains takeoff time `t`.

    Keeps the mined positives identical to the rest of the labeled pool: rather
    than re-cutting (which would make cut-procedure a knockdown shortcut), the
    flagged takeoff is attached to the already-segmented clip that spans it. Clip
    ids encode the start in ms, so a clip starting at s covers [s, s+clip_seconds];
    the nearest-centered match within `tol` wins when windows overlap."""
    best, best_d = None, float("inf")
    for p in from_clips.glob(f"{video_stem}_*.mp4"):
        tail = p.stem.rsplit("_", 1)[-1]
        if not tail.isdigit():
            continue
        s = int(tail) / 1000.0
        if s - tol <= t <= s + clip_seconds + tol:
            d = abs(t - (s + clip_seconds / 2.0))
            if d < best_d:
                best, best_d = p, d
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
               from_clips: Path | None = None, clip_seconds: float = 2.0,
               **find_kw) -> list[KnockdownCandidate]:
    """Detect takeoffs in one video, then flag those followed by a +4 fault step.

    For each takeoff the fault counter is read in `pre_window` (just before the
    jump) and `post_window` (after the scoring delay), both relative to takeoff.
    A jump is a candidate when both reads are stable, the pre read is a plausible
    round total, and the step is a positive multiple of 4. Exactly +4 with high
    stability scores near 1.0; +8 (two rails, rare on one effort) and shakier
    reads score lower so the review queue sorts the surest knockdowns first.

    With `from_clips` set, each hit's clip_id is resolved to the existing
    `clip_seconds`-long clip that already spans the takeoff, so the mined
    positives are the same files the clean labels were drawn from. Hits with no
    spanning clip in the pool are dropped (they were never segmented). Without it,
    the id is the fresh [t-before, t+after] window the caller may cut.
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
        if from_clips is not None:
            match = _existing_clip_for(from_clips, video.stem, t, clip_seconds)
            if match is None:
                continue
            clip_id = match.stem
        else:
            clip_start = t - before
            if clip_start < 0 or t + after > dur:
                continue
            clip_id = f"{video.stem}_{int(clip_start * 1000):07d}"
        conf = min(pre_s, post_s) * (1.0 if delta == 4 else 0.6)
        out.append(KnockdownCandidate(
            clip_id=clip_id, video=video.stem,
            takeoff_s=round(t, 3), fault_before=pre, fault_after=post,
            delta=delta, confidence=round(conf, 3)))
    cap.release()
    return out


def mine(raw_dir: Path, out_csv: Path, out_clips: Path | None = None,
         from_clips: Path | None = Path("data/clips"),
         weights: str = "yolov8m.pt", device: str = "cuda",
         before: float = 3.0, after: float = 1.0, limit_videos: int | None = None,
         **kw) -> pd.DataFrame:
    """Notebook entry point: mine every raw video for knockdown candidates.

    Writes the candidate table sorted by descending confidence to `out_csv`. With
    `from_clips` (default data/clips) the positives are resolved to the existing
    labeled-pool clips that span each takeoff, and `out_clips` (e.g. data/knockdowns)
    receives COPIES of those exact files for review — same id, same cut as the
    clean clips, so the cut procedure can't become a knockdown shortcut and the
    training pool already contains them. Set from_clips=None to instead cut fresh
    [t-before, t+after] clips into out_clips. Videos with no SCORE_ROI are skipped.
    """
    raw_dir, out_csv = Path(raw_dir), Path(out_csv)
    vids = sorted(raw_dir.glob("*.mp4"))
    if limit_videos:
        vids = vids[:limit_videos]
    detector = HorseDetector(weights=weights, device=device)
    reader = FaultReader(device=device)
    if out_clips is not None:
        Path(out_clips).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for v in tqdm(vids, desc="videos", unit="vid"):
        roi = SCORE_ROI.get(v.stem)
        if roi is None:
            tqdm.write(f"[mine_knockdowns] no SCORE_ROI for {v.stem}; skipping (run suggest_roi)")
            continue
        cands = mine_video(v, reader, detector, roi, before=before, after=after,
                           from_clips=from_clips, **kw)
        tqdm.write(f"[mine_knockdowns] {v.name}: {len(cands)} knockdown candidates")
        for c in cands:
            rows.append(c.__dict__)
            if out_clips is None:
                continue
            dst = Path(out_clips) / f"{c.clip_id}.mp4"
            if from_clips is not None:
                shutil.copy(Path(from_clips) / f"{c.clip_id}.mp4", dst)
            else:
                cut_clip(v, c.takeoff_s - before, before + after, dst)

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


def _sample_frames_bgr(video: Path, n: int = 40, start_s: float = 0.0,
                       end_s: float | None = None):
    """Evenly sample `n` frames across [start_s, end_s] of a long broadcast.

    Full-class broadcasts run an hour or more, so decoding every frame (as the
    clip annotators do) would exhaust memory. Seeking to evenly spaced indices
    gives a scrubbable strip cheaply; narrow start_s/end_s to a single round to
    sample densely where the fault counter is actually on screen."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    start_i = max(0, int(start_s * fps))
    end_i = (total - 1) if end_s is None else min(total - 1, int(end_s * fps))
    if end_i <= start_i:
        end_i = max(start_i + 1, total - 1)
    idxs = np.linspace(start_i, end_i, num=max(1, min(n, end_i - start_i)), dtype=int)
    frames, times = [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
            times.append(round(int(i) / fps, 1))
    cap.release()
    return frames, times


class RoiEditor:
    """Drag-a-box editor to set SCORE_ROI[video] interactively in Colab.

    Scrub to a frame where the fault number is on screen, drag a tight box around
    just the digits, and click Save: the box is converted to normalized
    (x1, y1, x2, y2), written into the live SCORE_ROI dict, printed for pasting
    into the module, and shown as a zoomed crop so you can confirm it isolates the
    digits before mining. Built on the same BBoxWidget + button scrubber the fence
    annotator uses, since Colab's widget manager won't render an IntSlider.

    Usage:
        ed = RoiEditor('data/raw/2S-4eXbehr4.mp4'); ed.start()
        # drag, scrub, Save -> SCORE_ROI['2S-4eXbehr4'] is set
    """

    CLASSES = ["score"]

    def __init__(self, video, n: int = 40, start_s: float = 0.0,
                 end_s: float | None = None):
        self.video = Path(video)
        self.stem = self.video.stem
        self.frames, self.times = _sample_frames_bgr(self.video, n=n,
                                                      start_s=start_s, end_s=end_s)
        self.cur = 0
        self.roi = SCORE_ROI.get(self.stem)

    def _current_box(self):
        if not self.bbox.bboxes:
            return None
        b = self.bbox.bboxes[0]
        h, w = self.frames[self.cur].shape[:2]
        x1, y1 = b["x"] / w, b["y"] / h
        return (round(x1, 4), round(y1, 4),
                round(x1 + b["width"] / w, 4), round(y1 + b["height"] / h, 4))

    def _preview(self, roi) -> None:
        import ipywidgets as widgets

        crop = _crop_roi(self.frames[self.cur], roi)
        self.preview.value = _jpeg_bytes(crop) if crop.size else b""

    def _goto(self, i: int) -> None:
        i = max(0, min(int(i), len(self.frames) - 1))
        self.cur = i
        self.bbox.image = _jpeg_data_url(self.frames[i])
        if self.roi is not None:
            h, w = self.frames[i].shape[:2]
            x1, y1, x2, y2 = self.roi
            self.bbox.bboxes = [{"x": int(x1 * w), "y": int(y1 * h),
                                 "width": int((x2 - x1) * w), "height": int((y2 - y1) * h),
                                 "label": "score"}]
        self.readout.value = f"<b>{i}/{len(self.frames) - 1}</b> &nbsp; t={self.times[i]}s"

    def _step(self, d: int) -> None:
        self._goto(self.cur + d)

    def _save(self, *_) -> None:
        roi = self._current_box()
        if roi is None:
            self.status.value = "<b style='color:#c00'>draw a box first</b>"
            return
        SCORE_ROI[self.stem] = roi
        self.roi = roi
        self._preview(roi)
        self.status.value = (f"<b style='color:#080'>SCORE_ROI['{self.stem}'] = {roi}</b>"
                             "  &nbsp;(paste into mine_knockdowns.py to persist)")
        print(f"SCORE_ROI['{self.stem}'] = {roi}")

    def start(self) -> None:
        import ipywidgets as widgets
        from IPython.display import display
        from jupyter_bbox_widget import BBoxWidget

        if not self.frames:
            print(f"[RoiEditor] no frames decoded from {self.video}")
            return
        self.bbox = BBoxWidget(classes=self.CLASSES)
        self.readout = widgets.HTML()
        self.status = widgets.HTML()
        self.preview = widgets.Image(format="jpeg")
        btn_prev = widgets.Button(description="Preview crop", icon="search")
        btn_save = widgets.Button(description="Save ROI", button_style="success", icon="check")
        btn_prev.on_click(lambda _: (self._current_box() and self._preview(self._current_box())))
        btn_save.on_click(self._save)
        nav = _build_frame_nav(self.readout, self._goto, self._step, lambda: len(self.frames) - 1)

        self._goto(len(self.frames) // 2)
        display(widgets.HTML(f"<h4>ROI editor — {self.stem}</h4>"
                             "Scrub to a frame showing the fault number, box the digits, Save."),
                nav, self.bbox, widgets.HBox([btn_prev, btn_save]), self.status,
                widgets.HTML("crop preview:"), self.preview)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/annotations/knockdown_candidates.csv"))
    ap.add_argument("--out-clips", type=Path, default=Path("data/knockdowns"),
                    help="copy (or cut) the flagged clips here for the annotator")
    ap.add_argument("--from-clips", type=Path, default=Path("data/clips"),
                    help="existing labeled pool to draw positives from; pass '' to cut fresh")
    ap.add_argument("--weights", default="yolov8m.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--before", type=float, default=3.0)
    ap.add_argument("--after", type=float, default=1.0)
    ap.add_argument("--limit-videos", type=int, default=None)
    ap.add_argument("--min-stability", type=float, default=0.5)
    ap.add_argument("--no-restrict-plausible", action="store_true",
                    help="accept any +4k step, not only plausible round totals")
    args = ap.parse_args()
    from_clips = args.from_clips if str(args.from_clips) else None
    mine(args.raw, args.out, out_clips=args.out_clips, from_clips=from_clips,
         weights=args.weights, device=args.device, before=args.before, after=args.after,
         limit_videos=args.limit_videos, min_stability=args.min_stability,
         restrict_plausible=not args.no_restrict_plausible)


if __name__ == "__main__":
    main()
