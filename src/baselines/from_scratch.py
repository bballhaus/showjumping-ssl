"""From-scratch supervised baseline.

The same R(2+1)D-18 architecture as the SSL path, but randomly initialized and
trained end-to-end on the labeled set only — no pretraining. This is the headline
comparison for the sample-efficiency hypothesis: SSL should match this with far
fewer labels.

Thin wrapper over downstream.train.run with ckpt=None, kinetics_init=False,
finetune=True.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..downstream.train import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("data/annotations/labels.csv"))
    ap.add_argument("--clips", type=Path, default=Path("data/clips"))
    ap.add_argument("--no-type", action="store_true")
    ap.add_argument("--task", choices=["both", "outcome", "distance"], default="both")
    ap.add_argument("--n-labels", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--group-by-venue", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    metrics = run(args.labels, args.clips, ckpt=None, kinetics_init=False, finetune=True,
                  use_type=not args.no_type, task=args.task, n_labels=args.n_labels,
                  epochs=args.epochs, group_by_venue=args.group_by_venue,
                  device=args.device, seed=args.seed)
    print(json.dumps({"baseline": "from_scratch", **metrics}, indent=2))


if __name__ == "__main__":
    main()

