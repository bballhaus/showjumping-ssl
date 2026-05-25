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
    # Pixel distance, biased toward horizontal (ground) component since we're in
    # a near-side view. We use Euclidean rather than purely horizontal because
    # the camera is rarely perfectly perpendicular to the line of travel.
    dx = horse_hoof_x - fence_base_x
    dy = horse_hoof_y - fence_base_y
    px_dist = float(np.hypot(dx, dy))
    return px_dist * mpp


def detect_takeoff_frame(horse_traj: list[tuple[int, Box]]) -> int:
    """Find the frame where horse vertical position reverses (down -> up).

    horse_traj: list of (frame_idx, Box) in temporal order.
    Returns frame_idx of takeoff; falls back to middle frame if no clear turn.
    """
    if len(horse_traj) < 5:
        return horse_traj[len(horse_traj) // 2][0] if horse_traj else 0
    ys = np.array([b.y2 for _, b in horse_traj])  # bottom of horse box: hoof line
    # Smooth slightly to suppress jitter.
    kernel = np.ones(3) / 3
    ys_s = np.convolve(ys, kernel, mode="same")
    dy = np.diff(ys_s)
    # Takeoff: y was increasing (horse going down/landing-flat) then decreases
    # sharply (horse going up). Argmin of dy after the midpoint is a robust pick.
    mid = len(dy) // 2
    rel = int(np.argmin(dy[mid:]))
    return horse_traj[mid + rel][0]
