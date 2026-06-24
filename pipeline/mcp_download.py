"""
MCP-sourced video downloader.
URLs scraped directly from YouTube via browser — bypasses bot detection.
"""
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

RAW_DIR = Path(__file__).parent.parent / "data" / "raw_videos"
CATALOG_FILE = Path(__file__).parent.parent / "data" / "catalog.json"

# Curated list scraped via browser MCP — all verified as real, accessible videos
VIDEOS = [
    # --- CloseCallSports channel (frame-by-frame umpire analysis, best source) ---
    {
        "url": "https://www.youtube.com/watch?v=mZn2j8ea37w",
        "label": "Terry Francona Ejected - Replay Review Stands on Sal Stewart Pulled Foot at 1B",
        "expected": "out",
        "source": "CloseCallSports",
    },
    {
        "url": "https://www.youtube.com/watch?v=zBH58_0e1vc",
        "label": "Umpire Mechanics Teachable - Pulled Foot Play at 1B with Lance Barksdale",
        "expected": "unknown",
        "source": "CloseCallSports",
    },
    {
        "url": "https://www.youtube.com/watch?v=4uIqO-50Zvw",
        "label": "Rhys Hoskins Wide Throw - Umpire Tripp Gibson Position Adjustment at 1B",
        "expected": "unknown",
        "source": "CloseCallSports",
    },
    {
        "url": "https://www.youtube.com/watch?v=8lBfY9cXg2M",
        "label": "May 2022 Call of Month - Greg Gibson Multi-Faceted Out At First Base",
        "expected": "out",
        "source": "CloseCallSports",
    },
    {
        "url": "https://www.youtube.com/watch?v=yUE6stqPvqE",
        "label": "Teachable - Tripp Gibson Scramble to First Base",
        "expected": "unknown",
        "source": "CloseCallSports",
    },
    # --- Other MLB sources ---
    {
        "url": "https://www.youtube.com/watch?v=dfCfjT5BH9o",
        "label": "Armando Galarraga Almost Perfect Game - Jim Joyce blown call at 1B",
        "expected": "safe",   # Joyce called out, but runner was actually safe
        "source": "Detroit Tigers",
    },
    {
        "url": "https://www.youtube.com/watch?v=Lgzzki9JQj0",
        "label": "Runner Misses Base Touch But is SAFE - Replay Review Appeal Play Boston",
        "expected": "safe",
        "source": "MLB",
    },
    {
        "url": "https://www.youtube.com/watch?v=OtlfAszvbRs",
        "label": "MLB Replay Room EXCLUSIVE - What happens when a call is challenged",
        "expected": "unknown",
        "source": "MLB",
    },
    {
        "url": "https://www.youtube.com/watch?v=N4WqOwY_bko",
        "label": "MLB Close Call Compilation - Most Controversial Calls",
        "expected": "unknown",
        "source": "SportzCrow",
    },
]


def load_catalog():
    if CATALOG_FILE.exists():
        return json.loads(CATALOG_FILE.read_text())
    return []


def download(video: dict) -> Optional[dict]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_template = str(RAW_DIR / "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=1080][vcodec^=avc]+bestaudio/bestvideo[height<=1080]+bestaudio/best",
        "--cookies-from-browser", "chrome",
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--no-warnings",
        "--output", out_template,
        video["url"],
    ]

    print(f"  Downloading: {video['label'][:60]}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    for line in result.stdout.splitlines():
        if "Destination:" in line and ".mp4" in line:
            path = line.split("Destination:")[-1].strip()
            print(f"  ✓ Saved: {Path(path).name}")
            return {
                "path": path,
                "url": video["url"],
                "label": video["label"],
                "expected": video["expected"],
                "source": video["source"],
                "downloaded_at": datetime.now().isoformat(),
                "stage": "raw",
            }

    # Check stderr for errors
    if result.stderr:
        for line in result.stderr.splitlines():
            if "ERROR" in line:
                print(f"  ✗ {line.strip()}")

    return None


def main():
    catalog = load_catalog()
    existing_urls = {e.get("url") for e in catalog}

    new_entries = []
    for video in VIDEOS:
        if video["url"] in existing_urls:
            print(f"  SKIP (already in catalog): {video['label'][:50]}")
            continue

        print(f"\n[{VIDEOS.index(video)+1}/{len(VIDEOS)}]")
        entry = download(video)
        if entry:
            new_entries.append(entry)

    catalog.extend(new_entries)
    CATALOG_FILE.write_text(json.dumps(catalog, indent=2))

    print(f"\n{'='*50}")
    print(f"Downloaded: {len(new_entries)}/{len(VIDEOS)} videos")
    print(f"Catalog total: {len(catalog)}")
    print(f"\nNext: python run_all.py --from 2")


if __name__ == "__main__":
    main()
