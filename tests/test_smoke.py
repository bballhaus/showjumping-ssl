"""CPU-only smoke tests: no network, no GPU, no video files.

Committed annotation CSVs / tracks.json are read-only inputs; every output goes
to pytest's tmp_path. Run from the repo root: python -m pytest -q
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data" / "annotations"

MODULES = [
    "src.data.venues",
    "src.data.labels",
    "src.preprocess.detect",
    "src.preprocess.geometry",
    "src.preprocess.fence_type",
    "src.preprocess.takeoff_local",
    "src.ssl.encoder",
    "src.ssl.losses",
    "src.ssl.grl",
    "src.ssl.dataset",
    "src.downstream.dataset",
    "src.downstream.heads",
    "src.downstream.split",
    "src.eval.metrics",
]


@pytest.mark.parametrize("name", MODULES)
def test_imports(name):
    importlib.import_module(name)


def test_encoder_forward_and_ssl_heads():
    from src.ssl.encoder import R2Plus1DEncoder
    from src.ssl.grl import DomainHead, dann_lambda
    from src.ssl.losses import TemporalOrderHead, info_nce

    torch.manual_seed(0)
    enc = R2Plus1DEncoder(embed_dim=16, kinetics_init=False).eval()  # weights=None, no download
    x = torch.randn(2, 3, 4, 32, 32)  # (B, C, T, H, W)
    with torch.no_grad():
        f = enc.features(x)
        z = enc(x)
    assert f.shape == (2, enc.feat_dim)
    assert z.shape == (2, 16)
    assert torch.allclose(z.norm(dim=-1), torch.ones(2), atol=1e-5)

    assert torch.isfinite(info_nce(z, z.flip(0)))
    assert TemporalOrderHead(enc.feat_dim)(f).shape == (2, 6)
    assert DomainHead(enc.feat_dim, n_domains=3)(f, dann_lambda(5, 10)).shape == (2, 3)


def test_downstream_heads_and_metrics():
    from src.downstream.dataset import OUTCOMES, TYPES, type_onehot
    from src.downstream.heads import DistanceHead, OutcomeHead
    from src.eval.metrics import classification_report, d_mae

    feat = torch.randn(4, 512)
    oh = torch.stack([type_onehot(t) for t in ("vertical", "oxer", None, "bogus")])
    assert oh.sum().item() == 2
    for use_type in (True, False):
        logits = OutcomeHead(512, use_type=use_type).eval()(feat, oh)
        d = DistanceHead(512, use_type=use_type).eval()(feat, oh)
        assert logits.shape == (4, len(OUTCOMES))
        assert d.shape == (4,)

    rep = classification_report([0, 1, 2, 0], [0, 1, 1, 0], types=["vertical", "oxer"] * 2)
    assert rep["accuracy"] == pytest.approx(0.75)
    assert d_mae([3.0, 4.0], [3.5, 4.0], types=list(TYPES))["d_mae"] == pytest.approx(0.25)


def test_geometry_on_committed_fences():
    from src.preprocess.geometry import (
        POLE_LENGTH_M, detect_takeoff_frame, meters_per_pixel, takeoff_distance,
        takeoff_gap, takeoff_observable)
    from src.preprocess.run_pipeline import load_fence_annotations

    fences = load_fence_annotations(ANN / "fences.csv")
    assert len(fences) > 0
    clip_id, ann = next(iter(fences.items()))
    box = ann["box"]
    mpp = meters_per_pixel(box)
    assert mpp == pytest.approx(POLE_LENGTH_M / box.w)
    # A hoof exactly at the fence base is 0 m away; one pole length to the left is 3.5 m.
    from src.preprocess.detect import Box
    same = Box(box.x1, box.y1, box.x2, box.y2)
    assert takeoff_distance(same, box) == pytest.approx(0.0)
    shifted = Box(box.x1 - box.w, box.y1, box.x2 - box.w, box.y2)
    assert takeoff_distance(shifted, box) == pytest.approx(POLE_LENGTH_M)

    # Synthetic stride-2 track: an early small rise (stride), then the steep final
    # push-off over frames 20-28, then a flat tail. Detector keeps the LAST strong rise.
    ys = [100, 101, 102, 99, 100, 101, 102, 103, 104, 105] + [96, 87, 78, 69, 60] + [60] * 6
    traj = [(2 * i, Box(0, 0, 10, y)) for i, y in enumerate(ys)]
    tf = detect_takeoff_frame(traj)
    assert 20 <= tf <= 28
    assert takeoff_gap(traj, tf) <= 2
    assert takeoff_observable(traj, 0)[1] == "edge"


def test_takeoff_local_replay(tmp_path):
    from src.preprocess.takeoff_local import recompute, score_against_truth

    fences = pd.read_csv(ANN / "fences.csv")
    data = json.loads((ANN / "tracks.json").read_text())
    keep = [c for c in fences["clip_id"].astype(str) if c in data["clips"]][:5]
    assert keep, "no fence-annotated clips found in tracks.json"
    sub = {"stride": data.get("stride", 2), "clips": {c: data["clips"][c] for c in keep}}
    tracks = tmp_path / "tracks.json"
    tracks.write_text(json.dumps(sub))

    out = tmp_path / "auto.csv"
    df = recompute(tracks, ANN / "fences.csv", out)  # never touches data/annotations/auto.csv
    assert out.exists() and len(df) == len(keep)
    assert set(df["type"]) <= {"vertical", "oxer"}
    assert (df["d_meters"] >= 0).all()
    score_against_truth(df, ANN / "takeoffs.csv", tracks_path=tracks)


def _labels_csv(tmp_path) -> Path:
    from src.data.labels import build_labels

    bv = pd.read_csv(ANN / "by_video.csv")
    picks = bv.groupby("video").head(4)["clip_id"].tolist()  # 4 clips per source video
    outcomes = pd.DataFrame({
        "clip_id": picks,
        "outcome": [("clean", "knockdown", "refusal")[i % 3] for i in range(len(picks))],
    })
    oc = tmp_path / "outcomes.csv"
    outcomes.to_csv(oc, index=False)
    out = tmp_path / "labels.csv"
    df = build_labels(ANN / "auto.csv", oc, ANN / "by_video.csv", out)
    assert list(df.columns) == ["clip_id", "video", "type", "outcome", "d_meters"]
    assert len(df) == len(picks)
    assert set(df["video"]) == set(bv["video"])
    return out


def test_labels_and_splits(tmp_path):
    from src.data.venues import VIDEO_VENUE
    from src.downstream.split import leave_one_out_splits, make_split, subsample_train

    labels = _labels_csv(tmp_path)
    all_ids = set(pd.read_csv(labels)["clip_id"])

    tr, va = make_split(labels, seed=0)
    assert set(tr).isdisjoint(va) and set(tr) | set(va) == all_ids

    tr, va = make_split(labels, group_by_venue=True, seed=0)
    vid = pd.read_csv(labels).set_index("clip_id")["video"]
    assert set(vid[tr]).isdisjoint(set(vid[va]))

    folds = leave_one_out_splits(labels, level="venue")
    assert {f[0] for f in folds} == set(VIDEO_VENUE.values())
    for _, ftr, fva in folds:
        assert set(ftr).isdisjoint(fva) and set(ftr) | set(fva) == all_ids

    small = subsample_train(labels, sorted(all_ids), n=6, seed=0)
    assert len(small) == 6
