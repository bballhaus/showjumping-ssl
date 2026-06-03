"""Outcome confusion matrix for the downstream classifier on the held-out split.

Runs one downstream evaluation with return_preds=True and renders the raw-count
confusion matrix over the outcome classes. With the heavy class imbalance this
shows directly whether the model is just predicting the majority "clean" class.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..downstream.dataset import OUTCOMES
from ..downstream.train import run


def plot(labels_csv: Path, clips_dir: Path, out_path: Path,
         ckpt: Path | None = None, finetune: bool = False,
         group_by_venue: bool = False, device: str = "cuda",
         seed: int = 0, epochs: int = 30) -> None:
    metrics = run(labels_csv, clips_dir, ckpt=ckpt, finetune=finetune,
                  task="outcome", group_by_venue=group_by_venue, device=device,
                  seed=seed, epochs=epochs, return_preds=True)
    y_true = metrics.get("y_true", [])
    y_pred = metrics.get("y_pred", [])
    if not y_true:
        print("[confusion] no predictions returned")
        return

    n = len(OUTCOMES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(OUTCOMES, rotation=20, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(OUTCOMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    f1 = metrics.get("macro_f1", float("nan"))
    ax.set_title(f"Outcome confusion (macro-F1={f1:.2f}, n_val={metrics.get('n_val')})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[confusion] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("data/annotations/labels.csv"))
    ap.add_argument("--clips", type=Path, default=Path("data/clips"))
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/encoder.pt"))
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--group-by-venue", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("milestone/figures/confusion.png"))
    args = ap.parse_args()
    plot(args.labels, args.clips, args.out, ckpt=args.ckpt, finetune=args.finetune,
         group_by_venue=args.group_by_venue, device=args.device, seed=args.seed)


if __name__ == "__main__":
    main()
