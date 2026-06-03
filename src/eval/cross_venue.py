"""Leave-one-out cross-venue / cross-competition generalization tests.

Test A (level="venue"): hold out an entire physical venue, train on the others.
The honest cross-venue test — the 4 Wellington videos count as ONE venue, so the
backdrop never appears in both train and val.

Test B (level="video", restrict_venue="wellington"): hold out one Wellington
video, train on the other three Wellington videos. Same venue/backdrop, different
competition (day, course, riders). If the model leans on venue shortcuts, B
scores higher than A — that gap is the whole point of the figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..downstream.split import leave_one_out_splits
from ..downstream.train import run


def run_lovo(labels_csv: Path, clips_dir: Path, ckpt: Path | None,
             level: str = "venue", restrict_venue: str | None = None,
             finetune: bool = False, epochs: int = 30, device: str = "cuda",
             seeds: tuple[int, ...] = (0, 1, 2)) -> pd.DataFrame:
    rows = []
    for holdout, train_ids, val_ids in leave_one_out_splits(
            labels_csv, level=level, restrict_venue=restrict_venue):
        if not train_ids or not val_ids:
            print(f"[cross_venue] skip {holdout}: "
                  f"train={len(train_ids)} val={len(val_ids)}")
            continue
        for seed in seeds:
            m = run(labels_csv, clips_dir, ckpt=ckpt, finetune=finetune,
                    task="outcome", train_ids=train_ids, val_ids=val_ids,
                    epochs=epochs, device=device, seed=seed)
            rows.append({"holdout": holdout, "seed": seed,
                         "n_train": m["n_train"], "n_val": m["n_val"],
                         "accuracy": m.get("accuracy"),
                         "macro_f1": m.get("macro_f1")})
            print(f"[cross_venue] holdout={holdout:12} seed={seed} "
                  f"n_tr={m['n_train']:3} n_val={m['n_val']:3} "
                  f"acc={m.get('accuracy'):.3f} f1={m.get('macro_f1'):.3f}")
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df.empty:
        print(f"[cross_venue] nothing to plot for {title}")
        return
    agg = df.groupby("holdout")["macro_f1"].agg(["mean", "min", "max"])
    x = range(len(agg))
    yerr = [agg["mean"] - agg["min"], agg["max"] - agg["mean"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, agg["mean"], color="#4c72b0", width=0.6)
    ax.errorbar(x, agg["mean"], yerr=yerr, fmt="none", ecolor="black", capsize=4)
    overall = df["macro_f1"].mean()
    ax.axhline(overall, ls="--", color="crimson", lw=1,
               label=f"mean over folds ({overall:.2f})")
    for i, v in enumerate(agg["mean"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(agg.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("macro-F1 (held-out group)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[cross_venue] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("data/annotations/labels.csv"))
    ap.add_argument("--clips", type=Path, default=Path("data/clips"))
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/encoder.pt"))
    ap.add_argument("--test", choices=["A", "B"], default="A",
                    help="A = leave-one-venue-out; B = within-Wellington cross-competition.")
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--csv-out", type=Path, default=None)
    ap.add_argument("--fig-out", type=Path, default=None)
    args = ap.parse_args()

    if args.test == "A":
        level, restrict, title = "venue", None, "Test A: leave-one-venue-out (cross-venue)"
        fig = args.fig_out or Path("milestone/figures/cross_venue_A.png")
        csv_out = args.csv_out or Path("data/results/cross_venue_A.csv")
    else:
        level, restrict = "video", "wellington"
        title = "Test B: leave-one-video-out within Wellington (cross-competition)"
        fig = args.fig_out or Path("milestone/figures/cross_venue_B.png")
        csv_out = args.csv_out or Path("data/results/cross_venue_B.csv")

    df = run_lovo(args.labels, args.clips, args.ckpt, level=level,
                  restrict_venue=restrict, finetune=args.finetune,
                  epochs=args.epochs, device=args.device, seeds=tuple(args.seeds))
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    print(f"[cross_venue] wrote {csv_out}")
    plot(df, fig, title)


if __name__ == "__main__":
    main()
