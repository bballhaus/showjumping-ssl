"""Inline Colab annotators (fence boxes + jump outcomes).

Both annotators avoid the fragile ipympl / ``%matplotlib widget`` backend so the
final notebook runs top-to-bottom with no kernel restart:

* ``ColabAnnotator``  - drag a box with ``jupyter_bbox_widget`` (a Colab-native
  canvas widget), pick vertical/oxer, scrub frames with a slider.
* ``OutcomeAnnotator`` - a plain ``ipywidgets.Image`` + buttons, no matplotlib.

Usage (in a Colab cell):
    !pip install -q jupyter_bbox_widget ipywidgets
    from google.colab import output
    output.enable_custom_widget_manager()

    from src.preprocess.annotate_colab import ColabAnnotator
    ann = ColabAnnotator('data/clips', 'data/annotations/fences.csv', limit=120)
    ann.start()
"""

from __future__ import annotations

import base64
import csv
from pathlib import Path

import cv2


def _all_frames_bgr(clip: Path) -> list:
    """Decode every frame (BGR, cv2-native) so the slider can scrub."""
    cap = cv2.VideoCapture(str(clip))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def _jpeg_bytes(frame_bgr, quality: int = 70) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def _jpeg_data_url(frame_bgr) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(_jpeg_bytes(frame_bgr)).decode()


class ColabAnnotator:
    """Fence-box annotator built on jupyter_bbox_widget (no ipympl)."""

    HEADER = ["clip_id", "x1", "y1", "x2", "y2", "pole_count", "frame"]
    CLASSES = ["vertical", "oxer"]

    def __init__(self, clips_dir: str, out_csv: str = "data/annotations/fences.csv",
                 limit: int = 30, prefer_filtered: bool = True):
        self.clips_dir = Path(clips_dir)
        self.out_csv = Path(out_csv)
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        self.limit = limit

        all_clips = sorted(self.clips_dir.glob("*.mp4"))
        if not all_clips:
            raise SystemExit(f"[annotate] no .mp4 in {self.clips_dir}")

        self._init_csv()
        self.seen = self._load_seen()
        self.queue = [c for c in all_clips if c.stem not in self.seen][:limit]
        self.idx = 0
        self.session = 0
        self.frames: list = []

        if not self.queue:
            print(f"[annotate] all {len(all_clips)} clips already annotated in {self.out_csv}")
            return
        self._build_widgets()

    # -- CSV plumbing --------------------------------------------------------

    def _init_csv(self):
        if not self.out_csv.exists() or self.out_csv.stat().st_size == 0:
            with self.out_csv.open("w", newline="") as f:
                csv.writer(f).writerow(self.HEADER)

    def _load_seen(self) -> set:
        seen = set()
        with self.out_csv.open() as f:
            for row in csv.DictReader(f):
                seen.add(row["clip_id"])
        return seen

    def _append_row(self, clip_id, box, pole_count, frame) -> None:
        x1, y1, x2, y2 = box
        with self.out_csv.open("a", newline="") as f:
            csv.writer(f).writerow([clip_id, round(x1, 1), round(y1, 1),
                                    round(x2, 1), round(y2, 1), pole_count, frame])

    # -- UI ------------------------------------------------------------------

    def _build_widgets(self):
        import ipywidgets as widgets
        from jupyter_bbox_widget import BBoxWidget

        self.bbox = BBoxWidget(classes=self.CLASSES)
        self.btn_quit = widgets.Button(description="Quit", button_style="danger", icon="times")
        self.status = widgets.HTML(value="")
        self.frame_slider = widgets.IntSlider(
            value=0, min=0, max=0, step=1, description="Frame:",
            continuous_update=False, layout=widgets.Layout(width="500px"))

        self.bbox.on_submit(lambda *a: self._save())
        self.bbox.on_skip(lambda *a: self._skip())
        self.btn_quit.on_click(lambda _: self._quit())
        self.frame_slider.observe(self._on_frame_change, names="value")

        self.toolbar = widgets.HBox([self.btn_quit])

    def _set_image(self, i: int):
        self.bbox.image = _jpeg_data_url(self.frames[i])
        self.bbox.bboxes = []

    def _on_frame_change(self, change):
        if not self.frames:
            return
        i = int(change["new"])
        if 0 <= i < len(self.frames):
            self._set_image(i)

    def start(self):
        from IPython.display import display
        display(self.status, self.frame_slider, self.bbox, self.toolbar)
        self._show_clip()

    def _show_clip(self):
        if self.idx >= len(self.queue):
            self.status.value = (f"<b>Done.</b> {len(self.seen)} clips annotated in "
                                 f"<code>{self.out_csv}</code>.")
            return
        clip = self.queue[self.idx]
        self.frames = _all_frames_bgr(clip)
        if not self.frames:
            self.idx += 1
            self._show_clip()
            return

        n = len(self.frames)
        mid = n // 2
        self.frame_slider.unobserve(self._on_frame_change, names="value")
        self.frame_slider.max = n - 1
        self.frame_slider.value = mid
        self.frame_slider.observe(self._on_frame_change, names="value")
        self._set_image(mid)
        self._update_status()

    def _update_status(self):
        self.status.value = (
            f"<b>&#9989; {self.session} saved this session</b> &nbsp;|&nbsp; "
            f"{len(self.seen)} total in fences.csv &nbsp;|&nbsp; "
            f"clip {self.idx + 1}/{len(self.queue)} in queue &nbsp;|&nbsp; "
            f"<code>{self.queue[self.idx].name}</code><br>"
            f"scrub to the fence, pick vertical/oxer, drag a box, click Submit (or Skip)")

    def _save(self):
        boxes = list(self.bbox.bboxes or [])
        if not boxes:
            self.status.value = "<b style='color:red'>Draw a box first.</b>"
            return
        b = boxes[0]
        x1, y1 = float(b["x"]), float(b["y"])
        x2, y2 = x1 + float(b["width"]), y1 + float(b["height"])
        pole_count = 2 if b.get("label") == "oxer" else 1
        clip = self.queue[self.idx]
        self._append_row(clip.stem, (x1, y1, x2, y2), pole_count, int(self.frame_slider.value))
        self.seen.add(clip.stem)
        self.session += 1
        self.idx += 1
        self._show_clip()

    def _skip(self):
        self.idx += 1
        self._show_clip()

    def _quit(self):
        self.status.value = (f"<b>Stopped.</b> {len(self.seen)} clips annotated in "
                             f"<code>{self.out_csv}</code>.")
        self.idx = len(self.queue)


class OutcomeAnnotator:
    """Jump-outcome annotator {clean, knockdown, refusal} on plain ipywidgets."""

    HEADER = ["clip_id", "outcome"]
    OUTCOMES = ("clean", "knockdown", "refusal")

    def __init__(self, clips_dir: str, out_csv: str = "data/annotations/outcomes.csv",
                 limit: int = 150, only_clips: list | None = None):
        self.clips_dir = Path(clips_dir)
        self.out_csv = Path(out_csv)
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        self.limit = limit

        all_clips = sorted(self.clips_dir.glob("*.mp4"))
        if only_clips is not None:
            keep = set(only_clips)
            all_clips = [c for c in all_clips if c.stem in keep]
        if not all_clips:
            raise SystemExit(f"[outcome] no matching .mp4 in {self.clips_dir}")

        self._init_csv()
        self.seen = self._load_seen()
        self.queue = [c for c in all_clips if c.stem not in self.seen][:limit]
        self.idx = 0
        self.session = 0
        self.frames: list = []
        if not self.queue:
            print(f"[outcome] all {len(all_clips)} clips already labeled in {self.out_csv}")
            return
        self._build_widgets()

    def _init_csv(self):
        if not self.out_csv.exists() or self.out_csv.stat().st_size == 0:
            with self.out_csv.open("w", newline="") as f:
                csv.writer(f).writerow(self.HEADER)

    def _load_seen(self) -> set:
        seen = set()
        with self.out_csv.open() as f:
            for row in csv.DictReader(f):
                seen.add(row["clip_id"])
        return seen

    def _append_row(self, clip_id, outcome) -> None:
        with self.out_csv.open("a", newline="") as f:
            csv.writer(f).writerow([clip_id, outcome])

    def _build_widgets(self):
        import ipywidgets as widgets

        self.image = widgets.Image(format="jpeg", layout=widgets.Layout(width="600px"))
        self.btn_clean = widgets.Button(description="Clean", button_style="success")
        self.btn_knock = widgets.Button(description="Knockdown", button_style="warning")
        self.btn_refuse = widgets.Button(description="Refusal", button_style="danger")
        self.btn_skip = widgets.Button(description="Skip clip", icon="step-forward")
        self.btn_quit = widgets.Button(description="Quit", button_style="danger", icon="times")
        self.status = widgets.HTML(value="")
        self.frame_slider = widgets.IntSlider(
            value=0, min=0, max=0, step=1, description="Frame:",
            continuous_update=False, layout=widgets.Layout(width="500px"))

        self.btn_clean.on_click(lambda _: self._save("clean"))
        self.btn_knock.on_click(lambda _: self._save("knockdown"))
        self.btn_refuse.on_click(lambda _: self._save("refusal"))
        self.btn_skip.on_click(lambda _: self._skip())
        self.btn_quit.on_click(lambda _: self._quit())
        self.frame_slider.observe(self._on_frame_change, names="value")

        self.toolbar = widgets.HBox([
            self.btn_clean, self.btn_knock, self.btn_refuse, self.btn_skip, self.btn_quit])

    def _on_frame_change(self, change):
        if not self.frames:
            return
        i = int(change["new"])
        if 0 <= i < len(self.frames):
            self.image.value = _jpeg_bytes(self.frames[i])

    def start(self):
        from IPython.display import display
        display(self.status, self.image, self.frame_slider, self.toolbar)
        self._show_clip()

    def _show_clip(self):
        if self.idx >= len(self.queue):
            self.status.value = (f"<b>Done.</b> {len(self.seen)} outcomes in "
                                 f"<code>{self.out_csv}</code>.")
            return
        clip = self.queue[self.idx]
        self.frames = _all_frames_bgr(clip)
        if not self.frames:
            self.idx += 1
            self._show_clip()
            return

        n = len(self.frames)
        self.frame_slider.unobserve(self._on_frame_change, names="value")
        self.frame_slider.max = n - 1
        self.frame_slider.value = n // 2
        self.frame_slider.observe(self._on_frame_change, names="value")
        self.image.value = _jpeg_bytes(self.frames[n // 2])
        self._update_status()

    def _update_status(self):
        self.status.value = (
            f"<b>&#9989; {self.session} labeled this session</b> &nbsp;|&nbsp; "
            f"{len(self.seen)} total in outcomes.csv &nbsp;|&nbsp; "
            f"clip {self.idx + 1}/{len(self.queue)} in queue &nbsp;|&nbsp; "
            f"<code>{self.queue[self.idx].name}</code><br>"
            f"scrub to watch the jump, then pick the outcome")

    def _save(self, outcome: str):
        clip = self.queue[self.idx]
        self._append_row(clip.stem, outcome)
        self.seen.add(clip.stem)
        self.session += 1
        self.idx += 1
        self._show_clip()

    def _skip(self):
        self.idx += 1
        self._show_clip()

    def _quit(self):
        self.status.value = (f"<b>Stopped.</b> {len(self.seen)} outcomes in "
                             f"<code>{self.out_csv}</code>.")
        self.idx = len(self.queue)
