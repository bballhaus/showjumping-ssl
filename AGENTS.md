# AGENTS.md

## Summary
CS 131 final project (Stanford, Spring 2026; sole author Brooke Ballhaus) that predicts
showjumping outcomes (clean / knockdown / refusal) and geometric takeoff distance `d` from
1.50 m competition broadcasts. Clips are cut around detected takeoffs, a R(2+1)D-18 video
encoder is pretrained self-supervised (InfoNCE + temporal order), and small heads are
fine-tuned and compared to three baselines. The central problem is the **venue confound**:
the six source videos span only three venues (four are Wellington), and clips from one
arena share a backdrop, so a naive encoder learns the venue instead of the jump. The fix is
a venue-balanced sampler, a domain-adversarial (gradient-reversal) venue head, and
evaluation that holds out whole venues (`--group-by-venue`, `src.eval.cross_venue`).
Any result reported on a stratified (non-venue-grouped) split is inflated.

## Repo map (actual tree)
```
src/
  data/        scrape.py (yt-dlp), clip_sources.txt (URLs + provenance), segment_jumps.py,
               segment.py, filter_clips.py, cache_scan.py, jumps_local.py, mine_knockdowns.py,
               venues.py (video id -> venue map), labels.py (label-table merge)
  preprocess/  detect.py, cache_tracks.py, jump_filter.py, takeoff_local.py, geometry.py,
               run_pipeline.py, fence_yolo.py, fence_type.py, fence_type_cnn.py,
               annotate.py (OpenCV), annotate_colab.py (Colab widgets)
  ssl/         encoder.py, dataset.py, losses.py, grl.py, train.py, embed.py
  downstream/  dataset.py, heads.py, split.py, train.py
  baselines/   from_scratch.py, imagenet2d.py, type_logreg.py
  eval/        metrics.py, sample_efficiency.py, ablations.py, cross_venue.py, venue_probe.py
  viz/         make_figures.py plus per-figure modules (tsne, confusion, training_curve, ...)
notebooks/     colab_final.ipynb (end-to-end), takeoff_validation.ipynb, colab_milestone.ipynb
data/
  annotations/ fences.csv, takeoffs.csv, tracks.json, auto.csv, auto_clean.csv, by_video.csv
  results/     ablations.csv, sample_efficiency.csv, train_log.csv, takeoff_local/, jumps_local/
final_deliverables/  proposal, milestone, final report PDFs and demo-day .pptx
requirements.txt
```
There are no tests, no packaging (setup.py/pyproject), and no CI. The README's "Layout"
block is out of date (it shows a `project/` root and `final_overleaf/`, `presentation/`,
`milestone_overleaf/` folders that are not in this repo); trust the tree above.

## Setup
- Python, run from the repo root; every module is invoked as `python -m src.<pkg>.<module>`.
- `pip install -r requirements.txt`. torch/torchvision are deliberately NOT listed: Colab
  ships CUDA builds. Locally, install them per the comment in `requirements.txt`. Never add
  torch to requirements.txt.
- The real workflow is Colab: open `notebooks/colab_final.ipynb`, set `GITHUB_REPO` and
  `DRIVE_DATA_ROOT` in the Setup cell. It clones to `/content/project` (hard-resetting to
  `origin/main`) and symlinks `data/` and `checkpoints/` to Google Drive.
- Recommended GPU: A100; T4 works with `--batch-size 4`.

## Pipeline (in order; full flags are in README "Running pieces locally")
1. Scrape: `python -m src.data.scrape --out data/raw` (reads `src/data/clip_sources.txt`).
2. Segment takeoff-anchored clips: `python -m src.data.segment_jumps --raw data/raw --out data/clips`.
3. Cache horse tracks and rank jumps: `src.preprocess.cache_tracks` -> `tracks.json`,
   then `src.preprocess.jump_filter`.
4. Annotate: Colab widgets in `colab_final.ipynb` / `src.preprocess.annotate_colab`
   (outcomes, takeoffs), `src.preprocess.annotate` for fence boxes.
5. Fence detection: `src.preprocess.fence_yolo export|train`, `src.preprocess.fence_type_cnn train`.
6. Horse YOLO + geometric `d`: `src.preprocess.run_pipeline` -> `auto.csv`.
7. Merge labels: `src.data.labels` -> `labels.csv` (needs `outcomes.csv`).
8. SSL pretrain + embed: `src.ssl.train --balance-venues --lambda-domain 0.3`, `src.ssl.embed`.
9. Downstream + baselines: `src.downstream.train`, `src.baselines.{from_scratch,imagenet2d,type_logreg}`.
10. Analysis: `src.eval.sample_efficiency`, `src.eval.ablations`, `src.eval.cross_venue`,
    then `src.viz.make_figures`.

Laptop-only (no GPU, no video), from cached inputs:
- `python -m src.preprocess.takeoff_local` replays takeoff detection from `tracks.json` +
  `fences.csv`. Note: by default it REWRITES `data/annotations/auto.csv` and the plot in
  `data/results/takeoff_local/`.
- `python -m src.data.jumps_local` needs `data/annotations/scan_cache.json`, which is not committed.

## Data
Committed:
- `fences.csv`, `takeoffs.csv`: hand-labeled (fence boxes, ground-truth takeoff frames).
  Do not regenerate, reformat, or overwrite.
- `tracks.json`: cached YOLO horse tracks from Colab; input to offline tuning. Do not regenerate.
- `auto.csv`, `auto_clean.csv`, `by_video.csv`: pipeline outputs kept as the reference
  tables for the report. Do not overwrite unless asked.
- `data/results/`: CSVs and plots behind the final report figures.
Not committed (gitignored, large or Drive-only): `data/raw/` (YouTube videos), `data/clips/`,
`checkpoints/`, `runs/`, `*.pt`, `outcomes.csv`, `labels.csv`, `embeddings.npz`,
`scan_cache.json`, generated figures. Code that needs these cannot run in a fresh clone.

## Conventions
- Modules expose an argparse `main` and use repo-relative default paths (`data/...`).
- Clip ids encode the source video; venue comes from `src/data/venues.py` (`VIDEO_VENUE`).
  If you add a source video, add it there and in `clip_sources.txt`.
- Any new split, baseline, or metric should support `--group-by-venue` and report it.
- Notebooks: treat `.ipynb` as JSON; keep edits minimal and do not commit outputs you
  did not produce.

## Guardrails
- Do not scrape YouTube (`src.data.scrape`, yt-dlp) or download videos without asking.
- Do not launch GPU training or YOLO/encoder inference (`ssl.train`, `downstream.train`,
  baselines, `fence_yolo train`, `run_pipeline`, `cache_tracks`) without asking.
- Do not edit hand annotations (`fences.csv`, `takeoffs.csv`) or `tracks.json`.
- `final_deliverables/` holds the submitted PDFs and .pptx: read-only.
- Do not add torch pins to requirements.txt.
