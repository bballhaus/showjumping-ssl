"""Drop clips where YOLO doesn't see a horse in enough frames.

Most 30-second-stride samples from a 2-hour Grand Prix broadcast land on
crowds, commentators, course walks, prize ceremonies, slow-mo replays, or
title cards. This filter keeps only clips with a horse visible in at least
`--min-frac` of the sampled frames.

Moves rejected clips into `data/clips_rejected/` rather than deleting, so you
can inspect what got cut.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tqdm.auto import tqdm

from src.preprocess.detect import HorseDetector


def filter_clips(clips_dir: Path, rejected_dir: Path, weights: str = "yolov8n.pt",
                 device: str = "cuda", stride: int = 4, min_frac: float = 0.4,
                 conf: float = 0.35) -> tuple[int, int]:
    """Returns (kept, rejected)."""
    detector = HorseDetector(weights=weights, conf=conf, device=device)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    clips = sorted(clips_dir.glob("*.mp4"))
    kept, rej = 0, 0
    pbar = tqdm(clips, desc="filter", unit="clip")
    for cp in pbar:
        dets = detector.detect_video(cp, stride=stride)
        n_sampled = len(dets)
        n_horse = sum(1 for _, boxes in dets if boxes)
        frac = n_horse / max(n_sampled, 1)
        if frac >= min_frac:
            kept += 1
        else:
            shutil.move(str(cp), str(rejected_dir / cp.name))
            rej += 1
        pbar.set_postfix(kept=kept, rejected=rej)
    return kept, rej


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, default=Path("data/clips"))
    ap.add_argument("--rejected", type=Path, default=Path("data/clips_rejected"))
    ap.add_argument("--weights", type=str, default="yolov8n.pt")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--stride", type=int, default=4,
                    help="Sample every Nth frame for detection (4 = ~8 samples per 2s clip).")
    ap.add_argument("--min-frac", type=float, default=0.4,
                    help="Keep clips where >=this fraction of sampled frames contain a horse.")
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()

    kept, rej = filter_clips(args.clips, args.rejected, weights=args.weights,
                             device=args.device, stride=args.stride,
                             min_frac=args.min_frac, conf=args.conf)
    print(f"[filter] kept {kept}, rejected {rej} -> {args.rejected}")


if __name__ == "__main__":
    main()
