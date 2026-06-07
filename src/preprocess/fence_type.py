"""Heuristic vertical-vs-oxer classifier from a fence bounding box.

A vertical is a single plane of poles, so its footprint is narrow front-to-back.
An oxer has front + back poles, so the box width-to-height ratio in a side view
is meaningfully larger.

This is a milestone-stage heuristic, not the final classifier. Once we have a
labeled set we replace it with a small CNN on the cropped fence region.
"""

from __future__ import annotations

from .detect import Box


def classify_fence(box: Box, pole_count: int | None = None) -> str:
    """Returns 'vertical' or 'oxer'.

    Decision rule:
      - pole_count, if known, dominates: 1 -> vertical, 2+ -> oxer
      - otherwise use aspect ratio of the fence box: w/h > 1.5 -> oxer
    """
    if pole_count is not None:
        return "oxer" if pole_count >= 2 else "vertical"
    ratio = box.w / max(box.h, 1.0)
    return "oxer" if ratio > 1.5 else "vertical"

