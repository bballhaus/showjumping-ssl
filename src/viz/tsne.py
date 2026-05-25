"""t-SNE of clip embeddings, colored by an annotation column if available."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def _color_map(values):
    uniq = sorted(set(values))
    cmap = plt.get_cmap("tab10")
    return {v: cmap(i % 10) for i, v in enumerate(uniq)}


def plot_tsne(emb_npz: Path, annotations_csv: Path | None, color_by: str,
              out_path: Path, title: str = "t-SNE of clip embeddings") -> None:
    data = np.load(emb_npz, allow_pickle=True)
    feats = data["features"]
    paths = [str(p) for p in data["paths"]]
    if feats.shape[0] < 5:
        print(f"[tsne] only {feats.shape[0]} samples, skipping")
        return

    perp = max(5, min(30, feats.shape[0] // 3))
    emb2 = TSNE(n_components=2, perplexity=perp, init="pca",
                random_state=0).fit_transform(feats)

    labels = None
    if annotations_csv is not None and annotations_csv.exists():
        ann = pd.read_csv(annotations_csv)
        if color_by in ann.columns:
            stem_to_label = dict(zip(ann["clip_id"], ann[color_by]))
            labels = [stem_to_label.get(Path(p).stem, "unknown") for p in paths]

    plt.figure(figsize=(6, 5))
    if labels is None:
        plt.scatter(emb2[:, 0], emb2[:, 1], s=18, alpha=0.7)
    else:
        cmap = _color_map(labels)
        for lab in sorted(set(labels)):
            mask = np.array([l == lab for l in labels])
            plt.scatter(emb2[mask, 0], emb2[mask, 1], s=18, alpha=0.75,
                        label=str(lab), color=cmap[lab])
        plt.legend(loc="best", fontsize=8, frameon=False)
    plt.title(title)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[tsne] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--ann", type=Path, default=None)
    ap.add_argument("--color-by", type=str, default="type")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", type=str, default="t-SNE of clip embeddings")
    args = ap.parse_args()
    plot_tsne(args.emb, args.ann, args.color_by, args.out, args.title)


if __name__ == "__main__":
    main()
