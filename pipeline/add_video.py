"""
Quick utility: add a specific YouTube URL to the catalog with a label and expected outcome.
Usage:
    python add_video.py <url> --label "2023 ALCS Game 4 close play" --expected safe
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw_videos"
CATALOG_FILE = Path(__file__).parent.parent / "data" / "catalog.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="YouTube URL of the clip")
    parser.add_argument("--label", default="", help="Human-readable description")
    parser.add_argument("--expected", choices=["safe", "out", "unknown"], default="unknown")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_template = str(RAW_DIR / "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=1080][ext=mp4]+bestaudio/best",
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--output", out_template,
        args.url,
    ]

    print(f"Downloading: {args.url}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("Download failed.")
        sys.exit(1)

    # Find what was downloaded
    downloaded = list(RAW_DIR.glob("*.mp4"))
    if not downloaded:
        print("No mp4 found after download.")
        sys.exit(1)

    latest = max(downloaded, key=lambda p: p.stat().st_mtime)
    catalog = json.loads(CATALOG_FILE.read_text()) if CATALOG_FILE.exists() else []

    entry = {
        "path": str(latest),
        "url": args.url,
        "label": args.label,
        "expected": args.expected,
        "downloaded_at": datetime.now().isoformat(),
        "stage": "raw",
    }
    catalog.append(entry)
    CATALOG_FILE.write_text(json.dumps(catalog, indent=2))
    print(f"Added to catalog: {latest.name}")
    print(f"Now run: python run_all.py --from 2")


if __name__ == "__main__":
    main()
