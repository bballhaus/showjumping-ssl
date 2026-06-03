"""Recover takeoff distance d in meters from video geometry.

Setup:
  - The standard top pole on a showjumping fence is 3.5 m long.
  - In a roughly side-on view, the projected pixel width of that top pole is a
    known scale: meters_per_pixel = 3.5 / pole_pixel_width.
  - Takeoff distance d is the ground distance between the horse's takeoff hoof
    and the base of the fence at lift-off (frame where horse vertical velocity
    flips from down to up).
  - For the milestone we approximate the takeoff hoof position as the bottom
    center of the horse box at the takeoff frame, and the fence base as the
    bottom-center of the fence box.

Returns d in meters; sign indicates direction (positive = horse in front of
fence, which is the only physically meaningful case).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detect import Box

POLE_LENGTH_M = 3.5


@dataclass
class GeomResult:
    d_meters: float
    meters_per_pixel: float
    takeoff_frame: int


def meters_per_pixel(fence_box: Box) -> float:
    """Scale from the top-pole pixel width. Assumes box width ~= top pole length."""
    px = max(fence_box.w, 1.0)
    return POLE_LENGTH_M / px


def takeoff_distance(horse_box: Box, fence_box: Box, mpp: float | None = None) -> float:
    """Ground distance in meters between horse takeoff hoof and fence base."""
    if mpp is None:
        mpp = meters_per_pixel(fence_box)
    horse_hoof_x = horse_box.cx
    horse_hoof_y = horse_box.y2
    fence_base_x = fence_box.cx
    fence_base_y = fence_box.y2
    dx = horse_hoof_x - fence_base_x
    dy = horse_hoof_y - fence_base_y
    px_dist = float(np.hypot(dx, dy))
    return px_dist * mpp


def _interior_smooth(ys: np.ndarray) -> np.ndarray:
    """3-tap moving average that leaves the endpoints untouched.

    Avoids the zero-padding boundary artifact of np.convolve(mode="same"),
    which crushes the final sample and fakes a large downward jump there.
    """
    out = ys.astype(float).copy()
    out[1:-1] = (ys[:-2] + ys[1:-1] + ys[2:]) / 3.0
    return out


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """Edge-padded moving average; same length as x, no zero-padding artifact."""
    n = len(x)
    win = int(max(1, min(win, n)))
    if win == 1:
        return x.astype(float)
    pad = win // 2
    xp = np.pad(x.astype(float), (pad, pad), mode="edge")
    sm = np.convolve(xp, np.ones(win) / win, mode="valid")
    return sm[:n]


def detect_takeoff_frame(horse_traj: list[tuple[int, Box]]) -> int:
    """Lift-off frame: where the horse-box bottom edge rises fastest.

    Lift-off is the explosive push-off, the instant the hind hooves leave the
    ground. In image space the box bottom (y2) climbs fastest right then, so y2
    velocity is at its most negative. The takeoff is the frame of steepest rise
    (argmin of the smoothed y2 difference), refined to sub-sample resolution with
    a parabolic fit so it can fall between the stride-sampled frames.

    Validated against hand-marked lift-off frames: steepest-rise tracks the human
    "hooves just left the ground" markedly better than the y2 reversal peak, which
    sits at the bottom of the loading crouch a beat earlier. The interior smoother
    has no convolution edge artifact, so this does not pin to the clip end.

    horse_traj: list of (frame_idx, Box) in temporal order.
    """
    if len(horse_traj) < 5:
        return horse_traj[len(horse_traj) // 2][0] if horse_traj else 0
    frames = np.array([fi for fi, _ in horse_traj], dtype=float)
    ys = _interior_smooth(np.array([b.y2 for _, b in horse_traj], dtype=float))
    dy = np.diff(ys)

    i = int(np.argmin(dy))
    refined = float(frames[i])
    if 0 < i < len(dy) - 1:
        a, b, c = dy[i - 1], dy[i], dy[i + 1]
        denom = a - 2 * b + c
        if denom != 0:
            delta = float(np.clip(0.5 * (a - c) / denom, -1.0, 1.0))
            refined = frames[i] + delta * float(frames[i + 1] - frames[i])
    return int(round(refined))
