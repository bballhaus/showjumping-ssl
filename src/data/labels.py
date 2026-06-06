"""Build the merged downstream label table.

The downstream heads need one row per labeled clip with: source video (venue),
fence type, outcome, and geometric takeoff distance d. Those facts are spread
across three files produced earlier in the pipeline:

  - data/annotations/auto.csv      : type, d_meters, mpp, takeoff_frame (geometry)
  - data/annotations/outcomes.csv  : outcome in {clean, knockdown, refusal}
  - data/annotations/by_video.csv  : clip_id -> source video id (venue)

This module joins them on clip_id into a single labels.csv. A clip is included
in the labeled set only if it has an outcome label (the supervised target);
type/d are filled where available and left blank otherwise. Video is recovered
from the clip_id prefix when by_video.csv is absent. A geometric d above 10 m is
implausible (broken homography) and is nulled so it never enters the d target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

OUTCOMES = ("clean", "knockdown", "refusal")
LABEL_COLUMNS = ["clip_id", "video", "type", "outcome", "d_meters"]


def video_of(clip_id: str) -> str:
    """Source video id is the clip_id prefix before the first underscore."""
    return clip_id.split("_", 1)[0]


def build_labels(auto_csv: Path, outcomes_csv: Path, by_video_csv: Path | None,
                 out_csv: Path) -> pd.DataFrame:
    auto = pd.read_csv(auto_csv) if auto_csv.exists() else pd.DataFrame(columns=["clip_id"])
    if not outcomes_csv.exists():
        raise SystemExit(
            f"[labels] no outcomes at {outcomes_csv}; run the outcome annotator first "
            f"(src.preprocess.annotate_colab.OutcomeAnnotator)")
    outcomes = pd.read_csv(outcomes_csv)

    bad = set(outcomes["outcome"].dropna().unique()) - set(OUTCOMES)
    if bad:
        raise SystemExit(f"[labels] unexpected outcome values {bad}; allowed: {OUTCOMES}")

    df = outcomes.merge(
        auto[[c for c in ("clip_id", "type", "d_meters") if c in auto.columns]],
        on="clip_id", how="left",
    )

    if by_video_csv is not None and by_video_csv.exists():
        bv = pd.read_csv(by_video_csv)[["clip_id", "video"]]
        df = df.merge(bv, on="clip_id", how="left")
        df["video"] = df["video"].fillna(df["clip_id"].map(video_of))
    else:
        df["video"] = df["clip_id"].map(video_of)

    for col in LABEL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[LABEL_COLUMNS].drop_duplicates("clip_id").reset_index(drop=True)

    d = pd.to_numeric(df["d_meters"], errors="coerce")
    df["d_meters"] = d.where(d <= 10)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    n_typed = df["type"].isin(("vertical", "oxer")).sum()
    print(f"[labels] wrote {out_csv}: {len(df)} labeled clips "
          f"({df['video'].nunique()} venues, {n_typed} typed, "
          f"{df['d_meters'].notna().sum()} with d)")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", type=Path, default=Path("data/annotations/auto.csv"))
    ap.add_argument("--outcomes", type=Path, default=Path("data/annotations/outcomes.csv"))
    ap.add_argument("--by-video", type=Path, default=Path("data/annotations/by_video.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/annotations/labels.csv"))
    args = ap.parse_args()
    build_labels(args.auto, args.outcomes, args.by_video, args.out)


if __name__ == "__main__":
    main()
