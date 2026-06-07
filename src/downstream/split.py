"""Train/val splits over the labeled clip set.

Two modes:
  - stratified: hold out a fraction of clips, stratified by outcome.
  - grouped (leave-venues-out): hold out entire source videos for val. This is
    the honest test of cross-venue generalization the milestone calls for — a
    model that learned venue shortcuts scores poorly here.

`subsample_train` then optionally trims the train set to `n` clips (class-
balanced where possible) for the sample-efficiency sweep.
"""

from __future__ import annotations

import random

import pandas as pd

from ..data.venues import venue_of
from .dataset import OUTCOMES


def _read(labels_csv) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    return df[df["outcome"].isin(OUTCOMES)].reset_index(drop=True)


def _video_id(clip_id: str) -> str:
    """Source video id = clip_id minus the trailing _{start_ms}. Uses the LAST
    underscore so video ids that contain underscores (e.g. h5_V5GYf7iM) survive,
    unlike the labels.csv 'video' column which splits on the first underscore."""
    return str(clip_id).rsplit("_", 1)[0]


def leave_one_out_splits(labels_csv, level: str = "venue",
                         restrict_venue: str | None = None):
    """Leave-one-group-out folds for the honest generalization tests.

    Groups are resolved from the clip id via src/data/venues.py, not the
    mislabeled 'video' column. Yields (holdout_name, train_ids, val_ids).

    level="venue": one fold per physical venue; the 4 Wellington videos are a
      single group, so no backdrop leaks into train (true cross-venue, Test A).
    level="video": one fold per source video. Combined with
      restrict_venue="wellington" this is the within-venue cross-competition
      test (same backdrop, different day/course/riders, Test B).
    """
    df = _read(labels_csv).copy()
    df["_vid"] = df["clip_id"].map(_video_id)
    df["_venue"] = df["_vid"].map(venue_of)
    if restrict_venue is not None:
        df = df[df["_venue"] == restrict_venue].reset_index(drop=True)
    group = df["_venue"] if level == "venue" else df["_vid"]

    folds = []
    for g in sorted(group.dropna().unique()):
        mask = group == g
        folds.append((str(g), df.loc[~mask, "clip_id"].tolist(),
                      df.loc[mask, "clip_id"].tolist()))
    return folds


def make_split(labels_csv, val_frac: float = 0.25, group_by_venue: bool = False,
               val_venues: list[str] | None = None, seed: int = 0):
    """Returns (train_ids, val_ids)."""
    df = _read(labels_csv)
    rng = random.Random(seed)

    if group_by_venue or val_venues is not None:
        venues = sorted(df["video"].astype(str).unique())
        if val_venues is None:
            rng.shuffle(venues)
            k = max(1, round(len(venues) * val_frac))
            val_venues = venues[:k]
        val_mask = df["video"].astype(str).isin(set(val_venues))
        return (df.loc[~val_mask, "clip_id"].tolist(),
                df.loc[val_mask, "clip_id"].tolist())

    train_ids, val_ids = [], []
    for _, grp in df.groupby("outcome"):
        ids = grp["clip_id"].tolist()
        rng.shuffle(ids)
        k = max(1, round(len(ids) * val_frac))
        val_ids += ids[:k]
        train_ids += ids[k:]
    return train_ids, val_ids


def subsample_train(labels_csv, train_ids: list[str], n: int, seed: int = 0) -> list[str]:
    """Trim train_ids to n, drawing class-balanced where the labels allow."""
    if n >= len(train_ids):
        return train_ids
    df = _read(labels_csv).set_index("clip_id")
    rng = random.Random(seed)
    by_cls: dict[str, list[str]] = {o: [] for o in OUTCOMES}
    for cid in train_ids:
        if cid in df.index:
            by_cls[df.loc[cid, "outcome"]].append(cid)
    for v in by_cls.values():
        rng.shuffle(v)

    picked: list[str] = []
    classes = [c for c in OUTCOMES if by_cls[c]]
    i = 0
    while len(picked) < n and classes:
        c = classes[i % len(classes)]
        if by_cls[c]:
            picked.append(by_cls[c].pop())
        else:
            classes.remove(c)
            continue
        i += 1
    return picked[:n]
