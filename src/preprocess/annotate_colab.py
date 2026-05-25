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
    HEADER = ["clip_id", "x1", "y1", "x2", "y2", "pole_count"]

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

    def _append_row(self, clip_id: str, box, pole_count) -> None:
        x1, y1, x2, y2 = box
        with self.out_csv.open("a", newline="") as f:
            csv.writer(f).writerow([clip_id,
                                    round(x1, 1), round(y1, 1),
                                    round(x2, 1), round(y2, 1),
                                    pole_count])

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

        self.btn_vert.on_click(lambda _: self._save(pole_count=1))
        self.btn_oxer.on_click(lambda _: self._save(pole_count=2))
        self.btn_skip.on_click(lambda _: self._skip())
        self.btn_undo.on_click(lambda _: self._clear_box())
        self.btn_quit.on_click(lambda _: self._quit())

        self.toolbar = widgets.HBox([
            self.btn_vert, self.btn_oxer, self.btn_skip, self.btn_undo, self.btn_quit,
        ])

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
        frame = self._middle_frame(clip)
        if frame is None:
            self.idx += 1
            self._show_clip()
            return

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(9, 5))
            self.fig.canvas.toolbar_visible = False
            self.fig.canvas.header_visible = False
            self.fig.canvas.footer_visible = False
        self.ax.clear()
        self.ax.imshow(frame)
        self.ax.set_title(f"Drag a box around the fence — {clip.name}",
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
        self._append_row(clip.stem, self.current_box, pole_count)
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
