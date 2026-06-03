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


def detect_takeoff_frame(horse_traj: list[tuple[int, Box]]) -> int:
    """Lift-off frame: the down->up reversal of the horse-box bottom edge.

    The hoof line (box y2) stops descending (velocity >= 0) and starts rising
    (velocity < 0); that sign-change frame is lift-off, returned in preference to
    the steepest-rise frame, which sits mid-ascent and biases d low. Searches the
    back half, where the approach pipeline places the jump. Falls back to the
    middle frame for short tracks and to the sharpest rise when no clean reversal
    is present.

    horse_traj: list of (frame_idx, Box) in temporal order.
    """
    if len(horse_traj) < 5:
        return horse_traj[len(horse_traj) // 2][0] if horse_traj else 0
    ys = np.array([b.y2 for _, b in horse_traj])
    dy = np.diff(_interior_smooth(ys))
    mid = len(dy) // 2
    reversals = [i for i in range(max(mid, 1), len(dy)) if dy[i] < 0 <= dy[i - 1]]
    if reversals:
        return horse_traj[min(reversals, key=lambda i: dy[i])][0]
    return horse_traj[mid + int(np.argmin(dy[mid:]))][0]
