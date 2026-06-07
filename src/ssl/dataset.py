"""Clip dataset that yields two augmented views per sample (for contrastive) and
a shuffled-triple version (for temporal-order) on demand.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def video_of(path: Path) -> str:
    """Source video id is the clip filename stem prefix before the underscore."""
    return path.stem.split("_", 1)[0]


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


def _augment_clip(frames: np.ndarray, rng: random.Random,
                  color_jitter: float = 0.3) -> np.ndarray:
    """Spatial + photometric augmentation for two contrastive views.

    Beyond a flip and brightness jitter, we apply per-channel color gains and a
    contrast shift. This is the "arena-color jitter" the milestone names as the
    venue-confound fallback: arenas differ in lighting, footing color, and crowd
    backdrop, so jittering color makes the contrastive objective stop using venue
    palette as a shortcut. `color_jitter` scales the strength (0 disables it).
    """
    out = frames.copy()
    if rng.random() < 0.5:
        out = out[:, :, ::-1, :]
    f = out.astype(np.float32)
    f *= rng.uniform(0.8, 1.2)
    if color_jitter > 0:
        gains = np.array([rng.uniform(1 - color_jitter, 1 + color_jitter)
                          for _ in range(3)], dtype=np.float32)
        f *= gains.reshape(1, 1, 1, 3)
        contrast = rng.uniform(1 - color_jitter, 1 + color_jitter)
        f = (f - 127.5) * contrast + 127.5
    return np.clip(f, 0, 255).astype(np.uint8)


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
        self.videos = [video_of(p) for p in self.paths]
        self.domain_ids = sorted(set(self.videos))
        self.video_to_idx = {v: i for i, v in enumerate(self.domain_ids)}

    @property
    def n_domains(self) -> int:
        return len(self.domain_ids)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(idx * 7919 + random.randint(0, 1 << 20))
        path = self.paths[idx]
        frames = _read_video_frames(path, n_frames=self.n_frames, size=self.size)

        view_a = _to_chw(_augment_clip(frames, rng))
        view_b = _to_chw(_augment_clip(frames, rng))

        idx_starts = [0, self.n_frames // 3, 2 * self.n_frames // 3]
        subs = [frames[s:s + self.sub_T] for s in idx_starts]
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
            "domain_y": torch.tensor(self.video_to_idx[self.videos[idx]], dtype=torch.long),
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


class VenueBalancedSampler(Sampler[int]):
    """Draws clips so every source video is represented roughly equally per epoch.

    Without this the batch composition mirrors the raw per-venue clip counts, which
    lets the contrastive objective exploit venue as a shortcut. We resample each
    under-represented video up to the size of the largest one, then shuffle.
    """

    def __init__(self, dataset: SSLClipDataset, seed: int = 0):
        self.seed = seed
        self.by_video: dict[str, list[int]] = defaultdict(list)
        for i, v in enumerate(dataset.videos):
            self.by_video[v].append(i)
        self.per_video = max((len(ix) for ix in self.by_video.values()), default=0)
        self.total = self.per_video * len(self.by_video)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self.total

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        order: list[int] = []
        for idxs in self.by_video.values():
            picks = list(idxs)
            while len(picks) < self.per_video:
                picks.append(rng.choice(idxs))
            rng.shuffle(picks)
            order.extend(picks[:self.per_video])
        rng.shuffle(order)
        self._epoch += 1
        return iter(order)
