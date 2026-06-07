"""Generate per-clip embeddings from a trained encoder for t-SNE / retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import InferenceClipDataset
from .encoder import R2Plus1DEncoder


def load_encoder(ckpt_path: Path, device: str, kinetics_init: bool = False) -> R2Plus1DEncoder:
    enc = R2Plus1DEncoder(kinetics_init=kinetics_init).to(device).eval()
    if ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location=device)
        enc.load_state_dict(sd["encoder"])
        print(f"[embed] loaded {ckpt_path}")
    else:
        print(f"[embed] no checkpoint at {ckpt_path}; using "
              f"{'Kinetics' if kinetics_init else 'random'} init")
    return enc


@torch.no_grad()
def embed_dir(clips_dir: Path, ckpt_path: Path, out_npz: Path,
              device: str = "cuda", batch_size: int = 8,
              kinetics_init: bool = False) -> None:
    enc = load_encoder(ckpt_path, device, kinetics_init=kinetics_init)
    ds = InferenceClipDataset(clips_dir)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=2)

    feats = []
    paths = []
    for batch in dl:
        x = batch["x"].to(device, non_blocking=True)
        f = enc.features(x).cpu().numpy()
        feats.append(f)
        paths.extend(batch["path"])
    feats = np.concatenate(feats, axis=0) if feats else np.zeros((0, enc.feat_dim))

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, features=feats, paths=np.array(paths))
    print(f"[embed] wrote {out_npz} ({feats.shape})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, default=Path("data/clips"))
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/encoder.pt"))
    ap.add_argument("--out", type=Path, default=Path("data/embeddings.npz"))
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--kinetics-init", action="store_true")
    args = ap.parse_args()
    embed_dir(args.clips, args.ckpt, args.out, device=args.device,
              kinetics_init=args.kinetics_init)


if __name__ == "__main__":
    main()

