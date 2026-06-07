"""Type-only logistic-regression baseline.

Predicts the outcome from fence type alone — no video, no encoder. This is the
floor: if a video model can't beat "guess from whether it's a vertical or oxer,"
it isn't learning approach quality. Reported alongside the encoder models on the
same train/val split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..downstream.dataset import OUTCOME_TO_IDX, OUTCOMES, TYPES
from ..downstream.split import make_split
from ..eval.metrics import classification_report


def _features(df: pd.DataFrame) -> np.ndarray:
    X = np.zeros((len(df), len(TYPES)), dtype=np.float32)
    for i, t in enumerate(df["type"].tolist()):
        if isinstance(t, str) and t in TYPES:
            X[i, TYPES.index(t)] = 1.0
    return X


def run(labels_csv: Path, val_frac: float = 0.25, group_by_venue: bool = False,
        seed: int = 0) -> dict:
    df = pd.read_csv(labels_csv)
    df = df[df["outcome"].isin(OUTCOMES)].reset_index(drop=True)
    train_ids, val_ids = make_split(labels_csv, val_frac=val_frac,
                                    group_by_venue=group_by_venue, seed=seed)
    tr = df[df["clip_id"].isin(set(train_ids))]
    va = df[df["clip_id"].isin(set(val_ids))]

    Xtr, ytr = _features(tr), tr["outcome"].map(OUTCOME_TO_IDX).to_numpy()
    Xva = _features(va)
    yva = va["outcome"].map(OUTCOME_TO_IDX).to_numpy()
    types = va["type"].tolist()

    if len(np.unique(ytr)) < 2:
        pred = np.full(len(va), ytr[0] if len(ytr) else 0)
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xva)

    return {"baseline": "type_logreg", "n_train": len(tr), "n_val": len(va),
            **classification_report(yva, pred, types=types)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("data/annotations/labels.csv"))
    ap.add_argument("--group-by-venue", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(json.dumps(run(args.labels, group_by_venue=args.group_by_venue, seed=args.seed),
                     indent=2))


if __name__ == "__main__":
    main()
