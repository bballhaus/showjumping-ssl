"""Inline Colab annotator for fence boxes.

Drag a rectangle on the displayed frame, then click one of the label buttons
(Vertical / Oxer / Skip). The annotation is appended to fences.csv and the
next clip loads automatically.

Usage (in a Colab cell):
    !pip install -q ipympl
    from google.colab import output
    output.enable_custom_widget_manager()
    %matplotlib widget

    from src.preprocess.annotate_colab import ColabAnnotator
    ann = ColabAnnotator('data/clips', 'data/annotations/fences.csv', limit=30)
    ann.start()
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.widgets import RectangleSelector


class ColabAnnotator:
    HEADER = ["clip_id", "x1", "y1", "x2", "y2", "pole_count", "frame"]

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
        if not self.queue:
            print(f"[annotate] all {len(all_clips)} clips already annotated in {self.out_csv}")
            return

        self.idx = 0
        self.current_box: tuple[float, float, float, float] | None = None
        self.fig = None
        self.ax = None
        self.selector = None

        self._build_widgets()

    # -- CSV plumbing --------------------------------------------------------

    def _init_csv(self):
        if not self.out_csv.exists() or self.out_csv.stat().st_size == 0:
            with self.out_csv.open("w", newline="") as f:
                csv.writer(f).writerow(self.HEADER)

    def _load_seen(self) -> set[str]:
        seen = set()
        with self.out_csv.open() as f:
            for row in csv.DictReader(f):
                seen.add(row["clip_id"])
        return seen

    def _append_row(self, clip_id: str, box, pole_count, frame) -> None:
        x1, y1, x2, y2 = box
        with self.out_csv.open("a", newline="") as f:
            csv.writer(f).writerow([clip_id,
                                    round(x1, 1), round(y1, 1),
                                    round(x2, 1), round(y2, 1),
                                    pole_count, frame])

    # -- Frame I/O -----------------------------------------------------------

    def _middle_frame(self, clip: Path):
        cap = cv2.VideoCapture(str(clip))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _all_frames(self, clip: Path) -> list:
        """Decode every frame so the slider can scrub. ~32 frames at 320px is cheap."""
        cap = cv2.VideoCapture(str(clip))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    # -- UI ------------------------------------------------------------------

    def _build_widgets(self):
        self.btn_vert = widgets.Button(description="Vertical (1 pole)",
                                       button_style="primary", icon="square-o")
        self.btn_oxer = widgets.Button(description="Oxer (2 poles)",
                                       button_style="warning", icon="th-large")
        self.btn_skip = widgets.Button(description="Skip clip", icon="step-forward")
        self.btn_undo = widgets.Button(description="Redraw box", icon="undo")
        self.btn_quit = widgets.Button(description="Quit", button_style="danger",
                                       icon="times")
        self.status = widgets.HTML(value="")
        # Frame scrubber. Camera tracks the horse, so the fence may only be
        # visible at one end of the 2-second clip. Slider value = frame index.
        self.frame_slider = widgets.IntSlider(
            value=0, min=0, max=0, step=1,
            description="Frame:", continuous_update=False,
            layout=widgets.Layout(width="500px"),
        )

        self.btn_vert.on_click(lambda _: self._save(pole_count=1))
        self.btn_oxer.on_click(lambda _: self._save(pole_count=2))
        self.btn_skip.on_click(lambda _: self._skip())
        self.btn_undo.on_click(lambda _: self._clear_box())
        self.btn_quit.on_click(lambda _: self._quit())
        self.frame_slider.observe(self._on_frame_change, names="value")

        self.toolbar = widgets.HBox([
            self.btn_vert, self.btn_oxer, self.btn_skip, self.btn_undo, self.btn_quit,
        ])

    def _on_frame_change(self, change):
        if not getattr(self, "frames", None):
            return
        i = int(change["new"])
        if 0 <= i < len(self.frames) and self.ax is not None and self.ax.images:
            self.ax.images[0].set_data(self.frames[i])
            self.fig.canvas.draw_idle()

    def _on_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return
        self.current_box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        self._update_status()

    def _update_status(self):
        n_done = len(self.seen)
        total = n_done + len(self.queue) - self.idx
        box_msg = (f"box: ({self.current_box[0]:.0f}, {self.current_box[1]:.0f}) "
                   f"to ({self.current_box[2]:.0f}, {self.current_box[3]:.0f})"
                   if self.current_box else "no box drawn yet — drag on the image")
        self.status.value = (
            f"<b>{n_done}/{total}</b> annotated &nbsp;|&nbsp; "
            f"current clip: <code>{self.queue[self.idx].name}</code> &nbsp;|&nbsp; "
            f"{box_msg}"
        )

    def _clear_box(self):
        self.current_box = None
        if self.selector is not None:
            try:
                self.selector.set_visible(False)
                self.selector.update()
            except Exception:
                pass
        self._update_status()

    # -- Main loop -----------------------------------------------------------

    def start(self):
        display(self.toolbar)
        display(self.frame_slider)
        display(self.status)
        self._show_clip()

    def _show_clip(self):
        if self.idx >= len(self.queue):
            self.status.value = (f"<b>Done.</b> {len(self.seen)} clips annotated in "
                                 f"<code>{self.out_csv}</code>.")
            if self.fig is not None:
                plt.close(self.fig)
            return

        clip = self.queue[self.idx]
        self.frames = self._all_frames(clip)
        if not self.frames:
            self.idx += 1
            self._show_clip()
            return

        # Slider spans the full clip; start at the middle (best guess for jump).
        n = len(self.frames)
        mid = n // 2
        # Suppress the observer callback during setup so changing the range doesn't fire.
        self.frame_slider.unobserve(self._on_frame_change, names="value")
        self.frame_slider.max = n - 1
        self.frame_slider.value = mid
        self.frame_slider.observe(self._on_frame_change, names="value")

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(9, 5))
            self.fig.canvas.toolbar_visible = False
            self.fig.canvas.header_visible = False
            self.fig.canvas.footer_visible = False
        self.ax.clear()
        self.ax.imshow(self.frames[mid])
        self.ax.set_title(
            f"Scrub the slider to find the fence, then drag a box — {clip.name}",
            fontsize=10)
        self.ax.set_xticks([]); self.ax.set_yticks([])

        # Replace the selector (creating a new one each clip is the safe path
        # — re-using one across axes that have been .clear()-ed is flaky).
        self.selector = RectangleSelector(
            self.ax, self._on_select,
            useblit=True, button=[1], interactive=True,
            minspanx=5, minspany=5, spancoords="pixels",
            props=dict(facecolor="none", edgecolor="orange", linewidth=2),
        )
        self.current_box = None
        self._update_status()
        self.fig.canvas.draw_idle()

    def _save(self, pole_count: int):
        if self.current_box is None:
            self.status.value = "<b style='color:red'>Draw a box first.</b>"
            return
        clip = self.queue[self.idx]
        self._append_row(clip.stem, self.current_box, pole_count, int(self.frame_slider.value))
        self.seen.add(clip.stem)
        self.idx += 1
        self._show_clip()

    def _skip(self):
        self.idx += 1
        self._show_clip()

    def _quit(self):
        self.status.value = (f"<b>Stopped.</b> {len(self.seen)} clips annotated in "
                             f"<code>{self.out_csv}</code>.")
        if self.fig is not None:
            plt.close(self.fig)
        self.idx = len(self.queue)


class OutcomeAnnotator:
    """Inline Colab annotator for jump outcomes {clean, knockdown, refusal}.

    No box drawing — scrub the slider to watch the jump, then click the outcome.
    Appends (clip_id, outcome) to outcomes.csv. Use this on the same clips you
    annotated fences for so the downstream label table can join type/d/outcome.

    Usage (in a Colab cell):
        from src.preprocess.annotate_colab import OutcomeAnnotator
        ann = OutcomeAnnotator('data/clips', 'data/annotations/outcomes.csv', limit=150)
        ann.start()
    """

    HEADER = ["clip_id", "outcome"]
    OUTCOMES = ("clean", "knockdown", "refusal")

    def __init__(self, clips_dir: str, out_csv: str = "data/annotations/outcomes.csv",
                 limit: int = 150, only_clips: list[str] | None = None):
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
        if not self.queue:
            print(f"[outcome] all {len(all_clips)} clips already labeled in {self.out_csv}")
            return

        self.idx = 0
        self.fig = None
        self.ax = None
        self._build_widgets()

    def _init_csv(self):
        if not self.out_csv.exists() or self.out_csv.stat().st_size == 0:
            with self.out_csv.open("w", newline="") as f:
                csv.writer(f).writerow(self.HEADER)

    def _load_seen(self) -> set[str]:
        seen = set()
        with self.out_csv.open() as f:
            for row in csv.DictReader(f):
                seen.add(row["clip_id"])
        return seen

    def _append_row(self, clip_id: str, outcome: str) -> None:
        with self.out_csv.open("a", newline="") as f:
            csv.writer(f).writerow([clip_id, outcome])

    def _all_frames(self, clip: Path) -> list:
        cap = cv2.VideoCapture(str(clip))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def _build_widgets(self):
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
        if not getattr(self, "frames", None):
            return
        i = int(change["new"])
        if 0 <= i < len(self.frames) and self.ax is not None and self.ax.images:
            self.ax.images[0].set_data(self.frames[i])
            self.fig.canvas.draw_idle()

    def _update_status(self):
        n_done = len(self.seen)
        total = n_done + len(self.queue) - self.idx
        self.status.value = (
            f"<b>{n_done}/{total}</b> labeled &nbsp;|&nbsp; "
            f"current clip: <code>{self.queue[self.idx].name}</code> &nbsp;|&nbsp; "
            f"scrub to watch the jump, then pick the outcome")

    def start(self):
        display(self.toolbar)
        display(self.frame_slider)
        display(self.status)
        self._show_clip()

    def _show_clip(self):
        if self.idx >= len(self.queue):
            self.status.value = (f"<b>Done.</b> {len(self.seen)} outcomes in "
                                 f"<code>{self.out_csv}</code>.")
            if self.fig is not None:
                plt.close(self.fig)
            return

        clip = self.queue[self.idx]
        self.frames = self._all_frames(clip)
        if not self.frames:
            self.idx += 1
            self._show_clip()
            return

        n = len(self.frames)
        self.frame_slider.unobserve(self._on_frame_change, names="value")
        self.frame_slider.max = n - 1
        self.frame_slider.value = n // 2
        self.frame_slider.observe(self._on_frame_change, names="value")

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(9, 5))
            self.fig.canvas.toolbar_visible = False
            self.fig.canvas.header_visible = False
            self.fig.canvas.footer_visible = False
        self.ax.clear()
        self.ax.imshow(self.frames[n // 2])
        self.ax.set_title(f"Did the rail stay up? — {clip.name}", fontsize=10)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self._update_status()
        self.fig.canvas.draw_idle()

    def _save(self, outcome: str):
        clip = self.queue[self.idx]
        self._append_row(clip.stem, outcome)
        self.seen.add(clip.stem)
        self.idx += 1
        self._show_clip()

    def _skip(self):
        self.idx += 1
        self._show_clip()

    def _quit(self):
        self.status.value = (f"<b>Stopped.</b> {len(self.seen)} outcomes in "
                             f"<code>{self.out_csv}</code>.")
        if self.fig is not None:
            plt.close(self.fig)
        self.idx = len(self.queue)
