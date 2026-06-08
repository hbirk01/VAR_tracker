"""
Stage 2: Extract frames from downloaded videos.
Focuses on the play window — trims to just the relevant ~5-10 seconds.
"""
import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path(__file__).parent.parent / "data" / "raw_videos"
FRAMES_DIR = Path(__file__).parent.parent / "data" / "frames"
CATALOG_FILE = Path(__file__).parent.parent / "data" / "catalog.json"

TARGET_FPS = 30  # normalize all to 30fps for consistent analysis
PLAY_WINDOW_SEC = 8  # extract this many seconds around detected play


def detect_play_window(video_path: str) -> tuple[float, float]:
    """
    Heuristic: find the most action-dense window.
    Uses frame difference (optical flow proxy) to find peak motion.
    Returns (start_sec, end_sec).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Sample every 0.5s to find motion peaks
    motion_scores = []
    prev_gray = None
    sample_interval = max(1, int(fps * 0.5))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90))
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray).mean()
                motion_scores.append((frame_idx / fps, diff))
            prev_gray = gray
        frame_idx += 1

    cap.release()

    if not motion_scores:
        return 0.0, min(PLAY_WINDOW_SEC, duration)

    # Find the peak motion moment
    times, scores = zip(*motion_scores)
    scores = np.array(scores)
    peak_idx = np.argmax(scores)
    peak_time = times[peak_idx]

    # Center window around peak
    half = PLAY_WINDOW_SEC / 2
    start = max(0, peak_time - half)
    end = min(duration, start + PLAY_WINDOW_SEC)

    return start, end


def extract_frames(video_path: str, output_dir: Path, start_sec: float, end_sec: float) -> list[str]:
    """Extract frames at TARGET_FPS between start and end seconds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    frame_interval = max(1, int(src_fps / TARGET_FPS))
    start_frame = int(start_sec * src_fps)
    end_frame = int(end_sec * src_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    saved = []
    frame_idx = start_frame
    extracted_count = 0

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % frame_interval == 0:
            timestamp_ms = int((frame_idx / src_fps) * 1000)
            out_path = output_dir / f"frame_{extracted_count:04d}_{timestamp_ms}ms.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved.append(str(out_path))
            extracted_count += 1
        frame_idx += 1

    cap.release()
    return saved


def process_video(entry: dict) -> dict:
    video_path = entry["path"]
    if not Path(video_path).exists():
        print(f"  SKIP (not found): {video_path}")
        return entry

    video_id = Path(video_path).stem
    output_dir = FRAMES_DIR / video_id

    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"  SKIP (already extracted): {video_id}")
        entry["frames_dir"] = str(output_dir)
        entry["stage"] = "frames"
        return entry

    print(f"  Detecting play window: {video_id}")
    start, end = detect_play_window(video_path)
    print(f"    Window: {start:.1f}s – {end:.1f}s")

    frames = extract_frames(video_path, output_dir, start, end)
    print(f"    Extracted {len(frames)} frames")

    entry["frames_dir"] = str(output_dir)
    entry["play_window"] = {"start_sec": start, "end_sec": end}
    entry["frame_count"] = len(frames)
    entry["stage"] = "frames"
    return entry


def main():
    catalog_path = CATALOG_FILE
    if not catalog_path.exists():
        print("No catalog found. Run 01_download.py first.")
        return

    catalog = json.loads(catalog_path.read_text())
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(catalog)} videos...")
    for i, entry in enumerate(tqdm(catalog)):
        print(f"\n[{i+1}/{len(catalog)}] {Path(entry['path']).name}")
        catalog[i] = process_video(entry)

    catalog_path.write_text(json.dumps(catalog, indent=2))
    print("\nDone. Catalog updated.")


if __name__ == "__main__":
    main()
