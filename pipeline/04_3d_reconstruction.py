"""
Stage 4: Project 2D detections into 3D field coordinates.
Uses baseball field homography (known dimensions) to establish world coordinates,
then triangulates keypoints across multiple camera angles when available.
"""
import cv2
import json
import numpy as np
from pathlib import Path

CATALOG_FILE = Path(__file__).parent.parent / "data" / "catalog.json"
MODELS_DIR = Path(__file__).parent.parent / "output" / "3d_models"

# Standard baseball field dimensions (feet, origin = home plate)
# First base is at (90, 0) in field coordinates
FIELD_LANDMARKS = {
    "home_plate":   np.array([0.0,   0.0,  0.0]),
    "first_base":   np.array([90.0,  0.0,  0.0]),
    "second_base":  np.array([90.0, 90.0,  0.0]),
    "third_base":   np.array([0.0,  90.0,  0.0]),
    "pitcher_mound": np.array([60.5, 42.78, 0.612]),  # mound height ~7.5 inches
}

# Pixels per foot at first base region (will be refined per video via homography)
DEFAULT_SCALE = 8.0  # rough estimate; calibrated per video


class FieldHomography:
    """
    Computes a planar homography mapping image pixels → field feet (ground plane).
    Requires the user to annotate 4+ field landmarks in the image, OR uses
    automatic detection of base markers (white squares).
    """

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        self.H = None  # 3x3 homography matrix
        self.H_inv = None

    def detect_bases_auto(self) -> dict[str, np.ndarray]:
        """
        Detect white base markers using color thresholding.
        Returns dict of {base_name: (x_px, y_px)} for detected bases.
        """
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        # White mask
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 40, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bases = {}
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 200 < area < 5000:  # filter by size typical for bases in video
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    candidates.append((cx, cy, area))

        # Sort by y (lower in image = closer to camera = home plate side)
        candidates.sort(key=lambda c: c[1], reverse=True)

        base_names = ["home_plate", "first_base", "second_base", "third_base"]
        for i, (cx, cy, _) in enumerate(candidates[:4]):
            bases[base_names[i]] = np.array([cx, cy])

        return bases

    def compute_homography(self, image_points: dict[str, np.ndarray]) -> bool:
        """
        Given at least 4 {base_name: (x_px, y_px)}, compute H.
        """
        src_pts = []  # image (pixel) points
        dst_pts = []  # world (feet) points — ground plane only (x, y)

        for name, img_pt in image_points.items():
            if name in FIELD_LANDMARKS:
                src_pts.append(img_pt[:2])
                dst_pts.append(FIELD_LANDMARKS[name][:2])  # ground plane

        if len(src_pts) < 4:
            return False

        src = np.array(src_pts, dtype=np.float32)
        dst = np.array(dst_pts, dtype=np.float32)
        self.H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
        if self.H is not None:
            self.H_inv = np.linalg.inv(self.H)
            return True
        return False

    def pixel_to_field(self, px: float, py: float) -> np.ndarray:
        """Map image pixel (px, py) → field feet (x, y, 0)."""
        if self.H is None:
            return np.array([px / DEFAULT_SCALE, py / DEFAULT_SCALE, 0.0])
        pt = np.array([px, py, 1.0])
        world = self.H @ pt
        world /= world[2]
        return np.array([world[0], world[1], 0.0])

    def field_to_pixel(self, fx: float, fy: float) -> np.ndarray:
        """Map field feet (fx, fy) → image pixel."""
        if self.H_inv is None:
            return np.array([fx * DEFAULT_SCALE, fy * DEFAULT_SCALE])
        pt = np.array([fx, fy, 1.0])
        px = self.H_inv @ pt
        px /= px[2]
        return px[:2]


def lift_pose_to_3d(frame_data: dict, homography: FieldHomography) -> dict:
    """
    Convert 2D keypoints to 3D field coordinates using homography + height prior.
    Feet/ankles → project to ground plane. Hands → estimate height from skeleton.
    """
    poses_3d = []
    for pose in frame_data.get("poses", []):
        kp2d = pose["keypoints"]
        kp3d = {}
        for name, pt in kp2d.items():
            if pt.get("visibility", 0) < 0.3:
                continue
            field_pos = homography.pixel_to_field(pt["x"], pt["y"])
            # Height: feet → 0, ankles → ~0.3ft, wrists → ~3-4ft (rough prior)
            height_prior = {
                "left_ankle": 0.3,   "right_ankle": 0.3,
                "left_heel": 0.1,    "right_heel": 0.1,
                "left_foot_index": 0.0, "right_foot_index": 0.0,
                "left_wrist": 3.5,   "right_wrist": 3.5,
                "left_elbow": 3.0,   "right_elbow": 3.0,
            }
            field_pos[2] = height_prior.get(name, 1.0)
            kp3d[name] = field_pos.tolist()
        poses_3d.append(kp3d)
    return poses_3d


def compute_contact_timing(detection_sequence: list[dict], first_base_world: np.ndarray) -> dict:
    """
    Walk through ordered frames and find:
    1. Frame when runner's foot first contacts base (distance < threshold)
    2. Frame when fielder's glove/hand reaches runner position (proximity < threshold)
    Returns timing dict with frame indices and milliseconds.
    """
    FOOT_CONTACT_THRESHOLD_FT = 1.5  # foot within 1.5 feet of base center
    GLOVE_CONTACT_THRESHOLD_FT = 2.0

    foot_contact_frame = None
    glove_contact_frame = None

    for i, frame_data in enumerate(detection_sequence):
        poses_3d = frame_data.get("poses_3d", [])
        if not poses_3d:
            continue

        # Check foot-to-base contact (first pose = primary player)
        for pose in poses_3d:
            for foot_key in ("left_foot_index", "right_foot_index", "left_heel", "right_heel"):
                if foot_key in pose:
                    foot_pos = np.array(pose[foot_key])
                    dist = np.linalg.norm(foot_pos[:2] - first_base_world[:2])
                    if dist < FOOT_CONTACT_THRESHOLD_FT and foot_contact_frame is None:
                        foot_contact_frame = {
                            "frame_idx": i,
                            "timestamp_ms": frame_data.get("timestamp_ms", i * 33),
                            "distance_ft": float(dist),
                            "keypoint": foot_key,
                        }

        # Check glove contact (wrist proximity to runner feet)
        if len(poses_3d) >= 2:
            runner_feet = []
            fielder_hands = []
            for pose_idx, pose in enumerate(poses_3d):
                for foot_key in ("left_foot_index", "right_foot_index"):
                    if foot_key in pose:
                        runner_feet.append(np.array(pose[foot_key]))
                for hand_key in ("left_wrist", "right_wrist"):
                    if hand_key in pose:
                        fielder_hands.append(np.array(pose[hand_key]))

            for hand in fielder_hands:
                for foot in runner_feet:
                    dist = np.linalg.norm(hand[:2] - foot[:2])
                    if dist < GLOVE_CONTACT_THRESHOLD_FT and glove_contact_frame is None:
                        glove_contact_frame = {
                            "frame_idx": i,
                            "timestamp_ms": frame_data.get("timestamp_ms", i * 33),
                            "distance_ft": float(dist),
                        }

    # Decision
    decision = "unknown"
    margin_ms = None
    if foot_contact_frame and glove_contact_frame:
        diff = foot_contact_frame["timestamp_ms"] - glove_contact_frame["timestamp_ms"]
        margin_ms = diff
        if diff < -50:
            decision = "SAFE"  # foot touched base significantly before glove
        elif diff > 50:
            decision = "OUT"   # glove tagged before foot touched base
        else:
            decision = "TOO_CLOSE"  # within 1 frame margin — inconclusive
    elif foot_contact_frame and not glove_contact_frame:
        decision = "SAFE"
    elif glove_contact_frame and not foot_contact_frame:
        decision = "OUT"

    return {
        "foot_contact": foot_contact_frame,
        "glove_contact": glove_contact_frame,
        "margin_ms": margin_ms,
        "decision": decision,
    }


def process_video(entry: dict) -> dict:
    detections_file = entry.get("detections_file")
    if not detections_file or not Path(detections_file).exists():
        return entry

    detection_data = json.loads(Path(detections_file).read_text())
    frames_dir = Path(entry["frames_dir"])

    # Use first frame for homography calibration
    first_frame = frames_dir / "frame_0000_0ms.jpg"
    if not first_frame.exists():
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        first_frame = frame_files[0] if frame_files else None

    if first_frame is None:
        return entry

    homo = FieldHomography(str(first_frame))
    detected_bases = homo.detect_bases_auto()
    if len(detected_bases) >= 4:
        homo.compute_homography(detected_bases)
        print(f"  Homography calibrated from {len(detected_bases)} bases")
    else:
        print(f"  Auto-detection found {len(detected_bases)} bases — using default scale")

    # Lift each frame's poses to 3D
    for frame_data in detection_data:
        frame_data["poses_3d"] = lift_pose_to_3d(frame_data, homo)

    # Compute contact timing
    first_base_world = FIELD_LANDMARKS["first_base"]
    timing = compute_contact_timing(detection_data, first_base_world)

    video_id = Path(entry["path"]).stem
    output = {
        "video_id": video_id,
        "expected": entry.get("expected", "unknown"),
        "decision": timing["decision"],
        "margin_ms": timing["margin_ms"],
        "foot_contact": timing["foot_contact"],
        "glove_contact": timing["glove_contact"],
        "calibrated_bases": list(detected_bases.keys()),
        "frame_count": len(detection_data),
    }

    output_path = MODELS_DIR / f"{video_id}_decision.json"
    output_path.write_text(json.dumps(output, indent=2))

    # Update detections file with 3D data
    Path(detections_file).write_text(json.dumps(detection_data, indent=2))

    entry["decision_file"] = str(output_path)
    entry["decision"] = timing["decision"]
    entry["margin_ms"] = timing["margin_ms"]
    entry["stage"] = "analyzed"
    return entry


def main():
    catalog_path = CATALOG_FILE
    if not catalog_path.exists():
        print("No catalog. Run previous stages first.")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(catalog_path.read_text())

    for i, entry in enumerate(catalog):
        if entry.get("stage") != "detected":
            continue
        print(f"\n[{i+1}/{len(catalog)}] {Path(entry['path']).name}")
        catalog[i] = process_video(entry)

    catalog_path.write_text(json.dumps(catalog, indent=2))
    print("\n3D reconstruction complete.")


if __name__ == "__main__":
    main()
