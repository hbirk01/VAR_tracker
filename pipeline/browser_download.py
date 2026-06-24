"""
Browser-assisted downloader.
Captures live stream URLs from Chrome via MCP, downloads all segments with
the same authenticated session, then merges with ffmpeg.

Usage: called by the MCP pipeline — not run standalone.
The MCP captures the base URL from network requests, then this script
downloads the full file by incrementing range parameters.
"""
import subprocess
import json
import re
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

RAW_DIR = Path(__file__).parent.parent / "data" / "raw_videos"
CATALOG_FILE = Path(__file__).parent.parent / "data" / "catalog.json"


def strip_range(url: str) -> str:
    """Remove range= and rn= params so we can download the whole stream."""
    # Replace range with full file range (no range param = full file)
    url = re.sub(r'&range=\d+-\d+', '', url)
    url = re.sub(r'&rn=\d+', '', url)
    url = re.sub(r'&rbuf=\d+', '', url)
    return url


def get_itag(url: str) -> str:
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    return params.get('itag', ['unknown'])[0]


def get_mime(url: str) -> str:
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    mime = params.get('mime', ['video/webm'])[0]
    return 'webm' if 'webm' in mime else 'mp4' if 'mp4' in mime else 'webm'


def download_stream(url: str, output_path: str) -> bool:
    """Download a stream URL using curl (inherits no cookies — URL is self-authenticated)."""
    clean_url = strip_range(url)
    cmd = [
        'curl', '-L', '-s', '--max-time', '120',
        '--retry', '3',
        '-H', 'Origin: https://www.youtube.com',
        '-H', 'Referer: https://www.youtube.com/',
        '-o', output_path,
        clean_url
    ]
    result = subprocess.run(cmd)
    if result.returncode == 0 and Path(output_path).stat().st_size > 1000:
        print(f"  ✓ Downloaded: {Path(output_path).name} ({Path(output_path).stat().st_size // 1024}KB)")
        return True
    print(f"  ✗ Failed: {Path(output_path).name}")
    return False


def merge_video_audio(video_path: str, audio_path: str, output_path: str) -> bool:
    """Merge separate video and audio streams into one mp4."""
    if not Path('ffmpeg').exists() and subprocess.run(['which', 'ffmpeg'], capture_output=True).returncode != 0:
        print("  ffmpeg not found — saving video-only file")
        import shutil
        shutil.copy(video_path, output_path)
        return True

    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy', '-c:a', 'aac',
        output_path
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


def process_captured_urls(video_url: str, audio_url: str, video_id: str, label: str = '', expected: str = 'unknown') -> Optional[dict]:
    """
    Given live video+audio stream URLs captured from the browser,
    download and merge them into a final mp4.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    video_ext = get_mime(video_url)
    audio_ext = get_mime(audio_url)

    video_tmp = str(RAW_DIR / f"{video_id}_video.{video_ext}")
    audio_tmp = str(RAW_DIR / f"{video_id}_audio.{audio_ext}")
    final_path = str(RAW_DIR / f"{video_id}.mp4")

    if Path(final_path).exists():
        print(f"  Already exists: {video_id}.mp4")
        return {"path": final_path, "video_id": video_id}

    print(f"  Downloading video stream (itag {get_itag(video_url)})...")
    if not download_stream(video_url, video_tmp):
        return None

    print(f"  Downloading audio stream (itag {get_itag(audio_url)})...")
    if not download_stream(audio_url, audio_tmp):
        return None

    print(f"  Merging...")
    if merge_video_audio(video_tmp, audio_tmp, final_path):
        # Clean up temp files
        for f in [video_tmp, audio_tmp]:
            if Path(f).exists():
                os.remove(f)
        size_mb = Path(final_path).stat().st_size / 1024 / 1024
        print(f"  ✓ Final: {video_id}.mp4 ({size_mb:.1f}MB)")
        return {
            "path": final_path,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "label": label,
            "expected": expected,
            "video_id": video_id,
            "stage": "raw",
        }
    return None


def add_to_catalog(entry: dict):
    catalog = json.loads(CATALOG_FILE.read_text()) if CATALOG_FILE.exists() else []
    existing = {e.get("url") for e in catalog}
    if entry.get("url") not in existing:
        catalog.append(entry)
        CATALOG_FILE.write_text(json.dumps(catalog, indent=2))
        print(f"  Added to catalog: {entry['video_id']}")


if __name__ == "__main__":
    # Example: paste captured URLs here from the MCP network inspector
    # then run: python pipeline/browser_download.py

    CAPTURES = [
        # {
        #   "video_id": "dfCfjT5BH9o",
        #   "label": "Armando Galarraga Almost Perfect Game",
        #   "expected": "safe",
        #   "video_url": "https://rr1---sn-cu-ajted.googlevideo.com/...",  # itag=302 (webm video)
        #   "audio_url": "https://rr1---sn-cu-ajted.googlevideo.com/...",  # itag=251 (webm audio)
        # },
    ]

    for capture in CAPTURES:
        print(f"\n[{capture['video_id']}] {capture['label']}")
        entry = process_captured_urls(
            capture["video_url"],
            capture["audio_url"],
            capture["video_id"],
            capture["label"],
            capture["expected"],
        )
        if entry:
            add_to_catalog(entry)
