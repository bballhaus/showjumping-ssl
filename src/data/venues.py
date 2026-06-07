"""Source video -> competition venue mapping.

The six source videos span three physical venues; four of them are Wellington
International from distinct competitions (different day, course, and riders).
They share an arena and backdrop, so for cross-venue evaluation the domain is the
VENUE, not the video: a per-video leave-one-out split holds out one Wellington
video while three other Wellington videos remain in train, leaking the backdrop.
Group by venue to avoid that. URLs + notes live in clip_sources.txt.
"""

from __future__ import annotations

VIDEO_VENUE = {
    "2S-4eXbehr4": "tryon",
    "jtEeeuyIpLA": "madrid",
    "wCv2DYtxr5U": "wellington",
    "h5_V5GYf7iM": "wellington",
    "kZbGlTmo8Ds": "wellington",
    "ApvPWiz8nBM": "wellington",
}


def venue_of(video_id: str) -> str:
    return VIDEO_VENUE.get(str(video_id), "unknown")

