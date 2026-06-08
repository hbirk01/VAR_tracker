"""
Stage 1: Download and catalog baseball close-play videos.
Uses yt-dlp to pull from YouTube (MLB highlights, umpire review clips).
"""
import subprocess
import json
import os
from pathlib import Path
from datetime import datetime

RAW_DIR = Path(__file__).parent.parent / "data" / "raw_videos"
CATALOG_FILE = Path(__file__).parent.parent / "data" / "catalog.json"

# Curated search queries for close first-base plays
SEARCH_QUERIES = [
    "MLB close play first base safe out replay 2023",
    "MLB replay review first base out safe 2024",
    "baseball bang bang play first base",
    "MLB expanded replay first base overturned",
    "baseball close call first base slow motion",
    "MLB first base close play umpire overturned 2022",
    "baseball first base replay challenge safe out",
]

# Direct known URLs of famous close plays (add more as found)
KNOWN_URLS = [
    # These are YouTube search results / highlight clips — add real URLs here
    # Format: {"url": "...", "label": "description", "expected": "safe|out|unknown"}
]


def search_and_download(query: str, max_results: int = 5) -> list[dict]:
    """Search YouTube and download top results."""
    search_url = f"ytsearch{max_results}:{query}"

    out_template = str(RAW_DIR / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--no-playlist",
        "--output", out_template,
        "--max-downloads", str(max_results),
        search_url,
    ]

    print(f"\n[SEARCH] {query}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    downloaded = []
    for line in result.stdout.splitlines():
        if "[download] Destination:" in line:
            path = line.split("Destination:")[-1].strip()
            downloaded.append({"path": path, "query": query})
            print(f"  Downloaded: {Path(path).name}")

    if result.returncode != 0 and result.stderr:
        print(f"  Warning: {result.stderr[:200]}")

    return downloaded


def download_direct(url: str, label: str, expected: str = "unknown") -> dict | None:
    """Download a specific known URL."""
    out_template = str(RAW_DIR / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=1080][ext=mp4]+bestaudio/best",
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--output", out_template,
        url,
    ]

    print(f"\n[DIRECT] {label}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    for line in result.stdout.splitlines():
        if "[download] Destination:" in line:
            path = line.split("Destination:")[-1].strip()
            print(f"  Downloaded: {Path(path).name}")
            return {"path": path, "url": url, "label": label, "expected": expected}

    return None


def load_catalog() -> list[dict]:
    if CATALOG_FILE.exists():
        return json.loads(CATALOG_FILE.read_text())
    return []


def save_catalog(entries: list[dict]):
    CATALOG_FILE.write_text(json.dumps(entries, indent=2))


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog()
    existing_paths = {e["path"] for e in catalog}

    new_entries = []

    # Download known specific clips first
    for item in KNOWN_URLS:
        entry = download_direct(item["url"], item["label"], item.get("expected", "unknown"))
        if entry and entry["path"] not in existing_paths:
            entry["downloaded_at"] = datetime.now().isoformat()
            entry["stage"] = "raw"
            new_entries.append(entry)

    # Search-based downloads
    for query in SEARCH_QUERIES:
        results = search_and_download(query, max_results=3)
        for r in results:
            if r["path"] not in existing_paths:
                r["downloaded_at"] = datetime.now().isoformat()
                r["stage"] = "raw"
                r["expected"] = "unknown"
                new_entries.append(r)

    catalog.extend(new_entries)
    save_catalog(catalog)

    print(f"\nTotal in catalog: {len(catalog)} videos")
    print(f"New downloads: {len(new_entries)}")
    print(f"Saved to: {CATALOG_FILE}")


if __name__ == "__main__":
    main()
