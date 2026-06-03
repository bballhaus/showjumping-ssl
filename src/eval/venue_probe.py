"""Linear venue probe: can a logistic regression recover the venue from frozen
embeddings? If the domain-adversarial objective worked, probe accuracy should sit
near the majority-class baseline rather than near 1.0 — the embeddings carry the
jump, not the backdrop. Reports cross-validated accuracy vs. baseline + a bar plot.

Clip ids are `{video_id}_{start_ms}` and some video ids contain underscores, so
the video id is everything before the LAST underscore.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from ..data.venues import venue_of


def _video_id(stem: str) -> str:
    return stem.rsplit("_", 1)[0]


def run(emb_npz: Path, out_path: Path | None = None) -> dict:
    data = np.load(emb_npz, allow_pickle=True)
    feats = data["features"]
    paths = [str(p) for p in data["paths"]]
    venues = np.array([venue_of(_video_id(Path(p).stem)) for p in paths])

    keep = venues != "unknown"
    feats, venues = feats[keep], venues[keep]
    classes, counts = np.unique(venues, return_counts=True)
    n = len(venues)
    if n < 10 or len(classes) < 2:
        print(f"[venue_probe] too few samples/venues (n={n}, venues={len(classes)})")
        return {}

    majority = counts.max() / n
    chance = 1.0 / len(classes)

    k = int(min(5, counts.min()))
    if k < 2:
        print(f"[venue_probe] smallest venue has <2 clips; cannot cross-validate")
        return {}
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    pipe_X = StandardScaler().fit_transform(feats)
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    scores = cross_val_score(clf, pipe_X, venues, cv=cv, scoring="accuracy")
    probe = float(scores.mean())

    metrics = {"probe_acc": probe, "probe_std": float(scores.std()),
               "majority_baseline": float(majority), "chance": float(chance),
               "n": int(n), "n_venues": int(len(classes))}
    print(f"[venue_probe] acc={probe:.3f}±{scores.std():.3f} "
          f"majority={majority:.3f} chance={chance:.3f} (n={n}, {len(classes)} venues)")

    if out_path is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ["linear probe", "majority\nbaseline"]
        vals = [probe, majority]
        ax.bar(bars, vals, color=["#4c72b0", "#bbbbbb"], width=0.6)
        ax.errorbar([0], [probe], yerr=[scores.std()], fmt="none",
                    ecolor="black", capsize=4)
        ax.axhline(chance, ls="--", color="crimson", lw=1,
                   label=f"uniform chance ({chance:.2f})")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("venue-prediction accuracy")
        ax.set_title("Linear venue probe on SSL embeddings")
        ax.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=180)
        plt.close()
        print(f"[venue_probe] wrote {out_path}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, default=Path("data/embeddings.npz"))
    ap.add_argument("--out", type=Path, default=Path("milestone/figures/venue_probe.png"))
    args = ap.parse_args()
    run(args.emb, args.out)


if __name__ == "__main__":
    main()
