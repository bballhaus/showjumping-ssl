"""Labeled-clip dataset for the downstream heads.

Reads data/annotations/labels.csv (clip_id, video, type, outcome, d_meters) and
serves, per clip: the decoded video tensor, the outcome class, a type one-hot
(for type conditioning), and the geometric d target with a validity mask (some
clips have an outcome but no usable d).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..ssl.dataset import _read_video_frames, _to_chw

OUTCOMES = ("clean", "knockdown", "refusal")
TYPES = ("vertical", "oxer")
OUTCOME_TO_IDX = {o: i for i, o in enumerate(OUTCOMES)}


def type_onehot(t) -> torch.Tensor:
    """2-dim one-hot; all-zeros when type is unknown/missing."""
    v = torch.zeros(len(TYPES), dtype=torch.float32)
    if isinstance(t, str) and t in TYPES:
        v[TYPES.index(t)] = 1.0
    return v


class LabeledClipDataset(Dataset):
    def __init__(self, labels_csv: Path, clips_dir: Path, clip_ids: list[str] | None = None,
                 n_frames: int = 16, size: int = 112):
        df = pd.read_csv(labels_csv)
        df = df[df["outcome"].isin(OUTCOMES)].reset_index(drop=True)
        if clip_ids is not None:
            keep = set(clip_ids)
            df = df[df["clip_id"].isin(keep)].reset_index(drop=True)
        self.df = df
        self.clips_dir = Path(clips_dir)
        self.n_frames = n_frames
        self.size = size

    def __len__(self) -> int:
        return len(self.df)

    @property
    def videos(self) -> list[str]:
        return self.df["video"].astype(str).tolist()

    def class_counts(self) -> np.ndarray:
        idx = self.df["outcome"].map(OUTCOME_TO_IDX).to_numpy()
        return np.bincount(idx, minlength=len(OUTCOMES))

    def __getitem__(self, i: int) -> dict:
        r = self.df.iloc[i]
        clip = self.clips_dir / f"{r['clip_id']}.mp4"
        frames = _read_video_frames(clip, n_frames=self.n_frames, size=self.size)
        d = r.get("d_meters")
        d_val = float(d) if pd.notna(d) else 0.0
        return {
            "x": _to_chw(frames),
            "y_outcome": torch.tensor(OUTCOME_TO_IDX[r["outcome"]], dtype=torch.long),
            "type_onehot": type_onehot(r.get("type")),
            "d": torch.tensor(d_val, dtype=torch.float32),
            "d_valid": torch.tensor(1.0 if pd.notna(d) else 0.0, dtype=torch.float32),
            "clip_id": str(r["clip_id"]),
        }
