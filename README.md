# Predicting Showjumping Outcomes with Self-Supervised Video Representations

CS131 final project — Brooke Ballhaus.

This repo contains the milestone-stage implementation: data scraping, YOLOv8-based detection, geometric takeoff-distance extraction, SSL pretraining (contrastive + temporal-order), and the figures + 2-page PDF for the milestone deliverable.

## Layout

```
project/
├── src/
│   ├── data/        # scraping, segmentation
│   ├── preprocess/  # YOLO detection, fence type, takeoff-distance geometry
│   ├── ssl/         # encoder, contrastive + temporal-order objectives, training loop
│   └── viz/         # t-SNE, detection overlays, training curves
├── notebooks/
│   └── colab_milestone.ipynb  # end-to-end runnable on Colab Pro
├── milestone/
│   ├── milestone.tex          # 2-page PDF source
│   └── figures/               # generated figures
├── data/
│   ├── raw/         # downloaded YouTube videos
│   ├── clips/       # 2-second approach clips
│   ├── frames/      # extracted frames for annotation
│   └── annotations/ # CSV labels
└── checkpoints/     # SSL encoder weights
```

## Running on Colab

1. Upload the repo to Google Drive (or `git clone` if you push it).
2. Open `notebooks/colab_milestone.ipynb` in Colab Pro with a T4/L4 GPU.
3. Run cells top-to-bottom. Each section writes intermediate outputs to `data/` or `milestone/figures/`.

## Running locally

```bash
pip install -r requirements.txt

# 1. Scrape clips (edit src/data/clip_sources.txt first)
python -m src.data.scrape --out data/raw

# 2. Segment 2-second approach clips
python -m src.data.segment --raw data/raw --out data/clips

# 3. Run YOLO + geometry (saves per-clip CSV)
python -m src.preprocess.run_pipeline --clips data/clips --out data/annotations/auto.csv

# 4. SSL pretraining
python -m src.ssl.train --clips data/clips --epochs 30 --out checkpoints/

# 5. Generate figures
python -m src.viz.make_figures --ckpt checkpoints/encoder.pt --out milestone/figures/
```

## Milestone PDF

```bash
cd milestone && pdflatex milestone.tex
```
