"""Clip dataset that yields two augmented views per sample (for contrastive) and
a shuffled-triple version (for temporal-order) on demand.
"""

from __future__ import annotations

import random
from pathlib import Path

import av
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_video_frames(path: Path, n_frames: int = 16, size: int = 112) -> np.ndarray:
    """Decode `n_frames` evenly-spaced frames at resolution `size`.

    Returns uint8 array (n_frames, H, W, 3).
    """
    container = av.open(str(path))
    stream = container.streams.video[0]
    total = stream.frames or 0
    frames = []
    if total > 0:
        idxs = np.linspace(0, total - 1, n_frames).astype(int).tolist()
    else:
        idxs = None

    decoded = []
    for i, frame in enumerate(container.decode(video=0)):
        decoded.append(frame.to_ndarray(format="rgb24"))
        if idxs is None and len(decoded) >= n_frames * 4:
            break

    if idxs is None:
        total = len(decoded)
        idxs = np.linspace(0, max(total - 1, 0), n_frames).astype(int).tolist()
    for i in idxs:
        frames.append(decoded[min(i, len(decoded) - 1)])

    arr = np.stack(frames, axis=0)
    # Center-crop / resize.
    out = np.empty((n_frames, size, size, 3), dtype=np.uint8)
    import cv2
    for i, f in enumerate(arr):
        h, w = f.shape[:2]
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        f = f[y0:y0 + s, x0:x0 + s]
        out[i] = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
    container.close()
    return out


def _to_chw(frames: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8 -> (3, T, H, W) float in [0,1]."""
    t = torch.from_numpy(frames).float() / 255.0
    return t.permute(3, 0, 1, 2).contiguous()


def _augment_clip(frames: np.ndarray, rng: random.Random) -> np.ndarray:
    """Light spatial + photometric augmentation for two contrastive views."""
    out = frames.copy()
    # Horizontal flip.
    if rng.random() < 0.5:
        out = out[:, :, ::-1, :]
    # Brightness jitter.
    g = rng.uniform(0.8, 1.2)
    out = np.clip(out.astype(np.float32) * g, 0, 255).astype(np.uint8)
    return out


class SSLClipDataset(Dataset):
    """Yields per sample:
        - 'view_a', 'view_b': two augmented 16-frame clips, shape (3, T, H, W)
        - 'shuffle_x':       (3, 3*sub_T, H, W) three shuffled sub-clips
        - 'shuffle_y':       int in [0, 5] for which permutation was applied
    """

    PERMUTATIONS = [
        (0, 1, 2), (0, 2, 1), (1, 0, 2),
        (1, 2, 0), (2, 0, 1), (2, 1, 0),
    ]

    def __init__(self, clips_dir: Path, n_frames: int = 16, size: int = 112, sub_T: int = 4):
        self.paths = sorted(Path(clips_dir).glob("*.mp4"))
        self.n_frames = n_frames
        self.size = size
        self.sub_T = sub_T

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(idx * 7919 + random.randint(0, 1 << 20))
        path = self.paths[idx]
        frames = _read_video_frames(path, n_frames=self.n_frames, size=self.size)

        view_a = _to_chw(_augment_clip(frames, rng))
        view_b = _to_chw(_augment_clip(frames, rng))

        # Temporal-order task: take 3 non-overlapping sub-clips and shuffle.
        # We re-read the clip at slightly different frame indices for variety.
        # For simplicity, we slice the already-decoded frames.
        idx_starts = [0, self.n_frames // 3, 2 * self.n_frames // 3]
        subs = [frames[s:s + self.sub_T] for s in idx_starts]
        # Pad if too short.
        subs = [s if len(s) == self.sub_T else np.pad(s, ((0, self.sub_T - len(s)), (0, 0), (0, 0), (0, 0)))
                for s in subs]
        perm_idx = rng.randrange(len(self.PERMUTATIONS))
        perm = self.PERMUTATIONS[perm_idx]
        shuffled = np.concatenate([subs[p] for p in perm], axis=0)
        shuffle_x = _to_chw(shuffled)

        return {
            "view_a": view_a,
            "view_b": view_b,
            "shuffle_x": shuffle_x,
            "shuffle_y": torch.tensor(perm_idx, dtype=torch.long),
            "path": str(path),
        }


class InferenceClipDataset(Dataset):
    """Plain dataset for embedding generation: single clip view, no augmentation."""

    def __init__(self, clips_dir: Path, n_frames: int = 16, size: int = 112):
        self.paths = sorted(Path(clips_dir).glob("*.mp4"))
        self.n_frames = n_frames
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        frames = _read_video_frames(path, n_frames=self.n_frames, size=self.size)
        return {"x": _to_chw(frames), "path": str(path)}
