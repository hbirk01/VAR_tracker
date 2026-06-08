"""
Stage 3: Run pose estimation + player/ball detection on extracted frames.
Outputs per-frame JSON with keypoints and bounding boxes.
"""
import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

FRAMES_DIR = Path(__file__).parent.parent / "data" / "frames"
CATALOG_FILE = Path(__file__).parent.parent / "data" / "catalog.json"

# MediaPipe keypoint indices relevant to contact detection
KEYPOINTS = {
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_elbow": 13,
    "right_elbow": 14,
}


def get_pose_detector():
    import mediapipe as mp
    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )
    return pose


def get_yolo_detector():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")  # nano for speed; swap to yolov8m for accuracy


def process_frame(frame_path: str, pose_detector, yolo_detector) -> dict:
    """Run detection on a single frame."""
    img = cv2.imread(frame_path)
    if img is None:
        return {}

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Pose estimation ---
    pose_results = pose_detector.process(rgb)
    poses = []
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        keypoints = {}
        for name, idx in KEYPOINTS.items():
            lm = landmarks[idx]
            keypoints[name] = {
                "x": lm.x * w,
                "y": lm.y * h,
                "z": lm.z,  # depth estimate (relative)
                "visibility": lm.visibility,
            }
        poses.append({"keypoints": keypoints, "source": "mediapipe"})

    # --- YOLO object detection (persons + sports ball) ---
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

    return {
        "frame": frame_path,
        "resolution": [w, h],
        "poses": poses,
        "detections": detections,
    }


def analyze_contact(frame_data: dict, base_region: dict | None = None) -> dict:
    """
    Compute contact-relevant metrics from a processed frame.
    Returns foot-to-base distance and glove proximity.
    """
    analysis = {"foot_near_base": None, "glove_near_runner": None}

    poses = frame_data.get("poses", [])
    if not poses:
        return analysis

    kp = poses[0]["keypoints"]
    detections = frame_data.get("detections", [])

    # Find persons (fielder vs runner — assume closest to base is fielder)
    persons = [d for d in detections if d["label"] == "person"]
    balls = [d for d in detections if d["label"] == "sports ball"]

    # Foot positions for primary pose
    feet = []
    for foot_key in ("left_ankle", "right_ankle", "left_heel", "right_heel"):
        if kp.get(foot_key, {}).get("visibility", 0) > 0.5:
            feet.append([kp[foot_key]["x"], kp[foot_key]["y"]])

    if feet and base_region:
        # Distance from closest foot to base center
        base_center = np.array(base_region["center"])
        min_dist = min(np.linalg.norm(np.array(f) - base_center) for f in feet)
        analysis["foot_to_base_px"] = float(min_dist)
        analysis["foot_near_base"] = min_dist < base_region.get("radius_px", 50)

    return analysis


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
    print(f"  Processing {len(frame_files)} frames...")

    results = []
    for fp in tqdm(frame_files, leave=False):
        frame_data = process_frame(str(fp), pose_detector, yolo_detector)
        frame_data["contact_analysis"] = analyze_contact(frame_data)
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
    print("Models loaded.")

    for i, entry in enumerate(catalog):
        if entry.get("stage") not in ("frames", "detected"):
            continue
        print(f"\n[{i+1}/{len(catalog)}] {Path(entry['path']).name}")
        catalog[i] = process_video_frames(entry, pose_detector, yolo_detector)

    catalog_path.write_text(json.dumps(catalog, indent=2))
    print("\nPose detection complete.")


if __name__ == "__main__":
    main()
