"""Generate placeholder PNGs so milestone.tex compiles before any data exists.

Run once after cloning so `pdflatex milestone.tex` has something to render.
The real figures overwrite these once the notebook has run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _placeholder(out: Path, title: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.text(0.5, 0.5, f"placeholder\n{title}", ha="center", va="center",
            fontsize=14, color="grey")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("lightgrey")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("milestone/figures"))
    args = ap.parse_args()
    for name, title in [
        ("training_curve.png", "SSL loss curves"),
        ("d_distribution.png", "Geometric d histogram"),
        ("tsne_type.png", "t-SNE of SSL embeddings"),
        ("det_grid.png", "Detection + d overlay"),
    ]:
        _placeholder(args.out / name, title)
    print(f"[placeholders] wrote 4 placeholder figures in {args.out}")


if __name__ == "__main__":
    main()
