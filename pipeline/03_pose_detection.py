"""
Stage 3: Per-person pose estimation + player/ball detection.

Improvements over v1:
- Runs MediaPipe on each YOLO-cropped person (not the whole frame), giving
  far better keypoint accuracy for distant broadcast-view players.
- Tracks persons across frames and classifies runner vs fielder roles.
- Stores poses keyed by role so downstream stages don't mix them up.
"""
import cv2
import json
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from utils.player_tracker import PlayerTracker

FRAMES_DIR = Path(__file__).parent.parent / "data" / "frames"
CATALOG_FILE = Path(__file__).parent.parent / "data" / "catalog.json"

# Padding around YOLO crop so MediaPipe has context
CROP_PAD = 0.15  # 15% padding on each side

# MediaPipe keypoint indices relevant to contact detection
KEYPOINTS = {
    "left_ankle":      27, "right_ankle":      28,
    "left_heel":       29, "right_heel":       30,
    "left_foot_index": 31, "right_foot_index": 32,
    "left_wrist":      15, "right_wrist":      16,
    "left_elbow":      13, "right_elbow":      14,
    "left_hip":        23, "right_hip":        24,
    "left_shoulder":   11, "right_shoulder":   12,
}

# Visibility threshold for broadcast-range players
VIS_THRESHOLD = 0.05


def get_pose_detector():
    import mediapipe as mp
    return mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )


def get_yolo_detector():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")


def _padded_crop(img: np.ndarray, bbox: List[float]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Crop img to bbox with padding. Returns (crop, (x_offset, y_offset)).
    Offsets allow converting crop-space coords back to frame coords.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * CROP_PAD), int(bh * CROP_PAD)
    cx1 = max(0, int(x1) - pad_x)
    cy1 = max(0, int(y1) - pad_y)
    cx2 = min(w, int(x2) + pad_x)
    cy2 = min(h, int(y2) + pad_y)
    return img[cy1:cy2, cx1:cx2], (cx1, cy1)


def _run_mediapipe_on_crop(
    pose_detector,
    img: np.ndarray,
    bbox: List[float],
    frame_w: int,
    frame_h: int,
) -> Optional[Dict]:
    """Run MediaPipe on a padded crop; return keypoints in full-frame pixel coords."""
    crop, (ox, oy) = _padded_crop(img, bbox)
    if crop.size == 0:
        return None

    ch, cw = crop.shape[:2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = pose_detector.process(rgb)

    if not result.pose_landmarks:
        return None

    landmarks = result.pose_landmarks.landmark
    keypoints = {}
    for name, idx in KEYPOINTS.items():
        lm = landmarks[idx]
        # MediaPipe gives normalized coords within the crop → convert to full frame
        # Clamp to frame bounds (MediaPipe can extrapolate outside [0,1] for occluded points)
        kx = float(np.clip(lm.x * cw + ox, 0, frame_w - 1))
        ky = float(np.clip(lm.y * ch + oy, 0, frame_h - 1))
        keypoints[name] = {
            "x": kx,
            "y": ky,
            "z": float(lm.z),
            "visibility": float(lm.visibility),
        }
    return {"keypoints": keypoints, "source": "mediapipe_crop"}


def process_frame(
    frame_path: str,
    pose_detector,
    yolo_detector,
    role_bboxes: Optional[Dict[str, Optional[List[float]]]] = None,
) -> dict:
    """
    Run detection on a single frame.

    role_bboxes: {runner: bbox, fielder: bbox} from tracker (may be None per role).
    When provided, MediaPipe runs on each role's crop. Otherwise runs on whole frame.
    """
    img = cv2.imread(frame_path)
    if img is None:
        return {}

    h, w = img.shape[:2]

    # --- YOLO: detect all persons + ball ---
    yolo_results = yolo_detector(img, verbose=False)[0]
    detections = []
    for box in yolo_results.boxes:
        cls = int(box.cls[0])
        label = yolo_results.names[cls]
        if label in ("person", "sports ball"):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "label": label,
                "confidence": float(box.conf[0]),
                "bbox": [x1, y1, x2, y2],
                "center": [(x1 + x2) / 2, (y1 + y2) / 2],
            })

    # --- Per-role MediaPipe pose ---
    poses_by_role: Dict[str, Optional[Dict]] = {"runner": None, "fielder": None}

    if role_bboxes:
        for role, bbox in role_bboxes.items():
            if bbox is not None:
                pose = _run_mediapipe_on_crop(pose_detector, img, bbox, w, h)
                if pose:
                    pose["role"] = role
                    poses_by_role[role] = pose
    else:
        # Fallback: whole-frame MediaPipe (original behaviour)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose_detector.process(rgb)
        if result.pose_landmarks:
            lms = result.pose_landmarks.landmark
            keypoints = {}
            for name, idx in KEYPOINTS.items():
                lm = lms[idx]
                keypoints[name] = {
                    "x": float(lm.x * w), "y": float(lm.y * h),
                    "z": float(lm.z), "visibility": float(lm.visibility),
                }
            poses_by_role["runner"] = {"keypoints": keypoints, "source": "mediapipe_fullframe", "role": "runner"}

    # Build flat poses list (runner first) for backward compat with stage 4
    poses = [p for p in [poses_by_role.get("runner"), poses_by_role.get("fielder")] if p]

    return {
        "frame": frame_path,
        "resolution": [w, h],
        "poses": poses,
        "poses_by_role": {k: v for k, v in poses_by_role.items() if v},
        "detections": detections,
        "contact_analysis": {"foot_near_base": None, "glove_near_runner": None},
        "poses_3d": [],
    }


ANNOT_DIR = Path(__file__).parent.parent / "data" / "annotations"


def _build_tracker(
    frame_files: List[Path],
    yolo_detector,
    play_start: int = 0,
    play_end: Optional[int] = None,
) -> PlayerTracker:
    """
    First pass: run YOLO on play-window frames, feed bboxes to tracker.
    Restricting to the play window avoids cross-camera-cut track confusion.
    """
    end = play_end if play_end is not None else len(frame_files)
    tracker = PlayerTracker()
    for fi, fp in enumerate(frame_files[play_start:end], start=play_start):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        results = yolo_detector(img, verbose=False)[0]
        bboxes = []
        for box in results.boxes:
            if results.names[int(box.cls[0])] == "person":
                bboxes.append(box.xyxy[0].tolist())
        tracker.update(fi, bboxes)

    ref_img = frame_files[play_start] if play_start < len(frame_files) else frame_files[0]
    frame_w = cv2.imread(str(ref_img)).shape[1]
    tracker.classify_roles(frame_w)
    return tracker


def process_video_frames(entry: dict, pose_detector, yolo_detector) -> dict:
    frames_dir = entry.get("frames_dir")
    if not frames_dir or not Path(frames_dir).exists():
        return entry

    video_id = Path(entry["path"]).stem
    output_file = Path(frames_dir) / "detections.json"

    if output_file.exists():
        print(f"  SKIP (detections exist): {video_id}")
        entry["detections_file"] = str(output_file)
        entry["stage"] = "detected"
        return entry

    frame_files = sorted(Path(frames_dir).glob("frame_*.jpg"))
    if not frame_files:
        return entry

    # Read play window from annotation (handles camera-cut videos)
    ann_file = ANNOT_DIR / f"{video_id}.json"
    ann = json.loads(ann_file.read_text()) if ann_file.exists() else {}
    play_start = ann.get("play_start_frame", 0)
    play_end   = ann.get("play_end_frame", None)
    if play_start or play_end:
        print(f"  Tracking in play window: frames {play_start}–{play_end or 'end'} of {len(frame_files)}")
    else:
        print(f"  Tracking players across {len(frame_files)} frames...")
    tracker = _build_tracker(frame_files, yolo_detector, play_start, play_end)

    # Log role classification result
    runner_frames  = sum(1 for t in tracker.tracks if t.role == "runner")
    fielder_frames = sum(1 for t in tracker.tracks if t.role == "fielder")
    print(f"  Roles: {runner_frames} runner track(s), {fielder_frames} fielder track(s)")

    print(f"  Running per-person pose estimation...")
    results = []
    for fi, fp in enumerate(tqdm(frame_files, leave=False)):
        role_bboxes = tracker.get_role_at_frame(fi)
        frame_data = process_frame(str(fp), pose_detector, yolo_detector, role_bboxes)
        # Embed timestamp from filename (frame_NNNN_XXXXms.jpg)
        try:
            ts_str = fp.stem.split("_")[-1].replace("ms", "")
            frame_data["timestamp_ms"] = int(ts_str)
        except (ValueError, IndexError):
            frame_data["timestamp_ms"] = fi * 33
        results.append(frame_data)

    output_file.write_text(json.dumps(results, indent=2))
    entry["detections_file"] = str(output_file)
    entry["stage"] = "detected"
    return entry


def main():
    catalog_path = CATALOG_FILE
    if not catalog_path.exists():
        print("No catalog. Run previous stages first.")
        return

    catalog = json.loads(catalog_path.read_text())

    print("Loading models...")
    pose_detector = get_pose_detector()
    yolo_detector = get_yolo_detector()
    print("Models loaded.\n")

    for i, entry in enumerate(catalog):
        if entry.get("stage") not in ("frames", "detected"):
            continue
        print(f"[{i+1}/{len(catalog)}] {Path(entry['path']).name}")
        catalog[i] = process_video_frames(entry, pose_detector, yolo_detector)

    catalog_path.write_text(json.dumps(catalog, indent=2))
    print("\nPose detection complete.")


if __name__ == "__main__":
    main()
