"""Download YouTube showjumping footage with yt-dlp.

Reads URLs from src/data/clip_sources.txt (one per line, # for comments) and
saves 720p MP4s into data/raw/.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_SOURCES = Path(__file__).parent / "clip_sources.txt"


def read_sources(path: Path) -> list[str]:
    urls = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def download_one(url: str, out_dir: Path) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 720p mp4 keeps disk + decode cheap. yt-dlp picks the best matching format.
    # Don't capture output — yt-dlp's native progress bar streams to the parent
    # process so the user sees download progress live.
    cmd = [
        "yt-dlp",
        "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]",
        "--merge-output-format", "mp4",
        "--newline",  # play nicer with Jupyter line buffering
        "-o", str(out_dir / "%(id)s.%(ext)s"),
        url,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[scrape] failed: {url}")
        return None
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    args = ap.parse_args()

    urls = read_sources(args.sources)
    if not urls:
        print(f"[scrape] no URLs in {args.sources}. Add YouTube links and re-run.")
        return

    print(f"[scrape] {len(urls)} videos -> {args.out}")
    for url in urls:
        download_one(url, args.out)


if __name__ == "__main__":
    main()
