"""
Stage 4: Project 2D detections into 3D field coordinates + contact timing.

Improvements over v1:
- Base detection uses multi-frame consensus (majority vote across 20 frames)
  and restricts HSV search to a ROI where first base typically appears.
- Contact timing uses runner pose for foot-to-base, fielder pose for glove-to-runner.
- Visibility threshold lowered to 0.05 for broadcast-range players.
"""
import cv2
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple

CATALOG_FILE = Path(__file__).parent.parent / "data" / "catalog.json"
MODELS_DIR   = Path(__file__).parent.parent / "output" / "3d_models"
ANNOT_DIR    = Path(__file__).parent.parent / "data" / "annotations"

# Standard baseball field dimensions (feet, origin = home plate)
FIELD_LANDMARKS = {
    "home_plate":    np.array([  0.0,   0.0, 0.0]),
    "first_base":    np.array([ 90.0,   0.0, 0.0]),
    "second_base":   np.array([ 90.0,  90.0, 0.0]),
    "third_base":    np.array([  0.0,  90.0, 0.0]),
    "pitcher_mound": np.array([ 60.5,  42.78, 0.612]),
}

DEFAULT_SCALE = 8.0   # px/ft fallback
VIS_THRESHOLD = 0.05  # MediaPipe visibility cutoff for broadcast video

# Foot keypoints used for runner-base contact
FOOT_KEYS = ("left_foot_index", "right_foot_index", "left_heel", "right_heel",
             "left_ankle", "right_ankle")
# Hand keypoints used for fielder-glove contact
HAND_KEYS = ("left_wrist", "right_wrist")


# ---------------------------------------------------------------------------
# Field homography
# ---------------------------------------------------------------------------

class FieldHomography:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        self.H = None
        self.H_inv = None

    def detect_bases_auto(self, n_frames_sample: int = 1) -> Dict[str, np.ndarray]:
        """
        Detect white base markers via HSV thresholding on the frame.
        Focuses on plausible base regions and filters by aspect ratio.
        """
        if self.img is None:
            return {}

        h, w = self.img.shape[:2]
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        # White + near-white mask (bases are very bright)
        mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 50, 255]))

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100 or area > 8000:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = max(bw, bh) / (min(bw, bh) + 1e-3)
            if aspect > 4.0:  # too elongated — not a base
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            candidates.append((cx, cy, area))

        # Sort by y descending (lower in image = closer base in broadcast view)
        candidates.sort(key=lambda c: c[1], reverse=True)

        base_names = ["home_plate", "first_base", "second_base", "third_base"]
        bases = {}
        for i, (cx, cy, _) in enumerate(candidates[:4]):
            bases[base_names[i]] = np.array([cx, cy])
        return bases

    def compute_homography(self, image_points: Dict[str, np.ndarray]) -> bool:
        src_pts, dst_pts = [], []
        for name, img_pt in image_points.items():
            if name in FIELD_LANDMARKS:
                src_pts.append(img_pt[:2])
                dst_pts.append(FIELD_LANDMARKS[name][:2])
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
        if self.H is None:
            return np.array([px / DEFAULT_SCALE, py / DEFAULT_SCALE, 0.0])
        pt = np.array([px, py, 1.0])
        world = self.H @ pt
        world /= world[2]
        return np.array([world[0], world[1], 0.0])

    def field_to_pixel(self, fx: float, fy: float) -> np.ndarray:
        if self.H_inv is None:
            return np.array([fx * DEFAULT_SCALE, fy * DEFAULT_SCALE])
        pt = np.array([fx, fy, 1.0])
        px = self.H_inv @ pt
        px /= px[2]
        return px[:2]


# ---------------------------------------------------------------------------
# Base detection — multi-frame consensus
# ---------------------------------------------------------------------------

def detect_first_base_pixel(
    frames_dir: Path,
    sample_every: int = 5,
    annotation_file: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """
    Returns the pixel center of first base using:
    1. Manual annotation (highest priority, if file exists)
    2. Multi-frame HSV consensus from sampled frames
    3. None (caller falls back to foot-cluster heuristic)
    """
    # 1. Manual annotation
    if annotation_file and annotation_file.exists():
        ann = json.loads(annotation_file.read_text())
        val = ann.get("first_base_px")
        if val is not None and len(val) == 2:
            return np.array(val, dtype=float)

    # 2. HSV multi-frame consensus
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    sampled = frame_files[::sample_every][:20]

    all_candidates: List[np.ndarray] = []
    for fp in sampled:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h, w = img.shape[:2]

        # For 1B plays, first base is typically in the right half of the frame
        roi_x1 = w // 3
        roi = img[:, roi_x1:]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 50, 255]))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 80 < area < 6000:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = max(bw, bh) / (min(bw, bh) + 1e-3)
                if aspect <= 4.0:
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx = M["m10"] / M["m00"] + roi_x1
                        cy = M["m01"] / M["m00"]
                        # Skip scoreboard (top 20%) and watermark (bottom 8%)
                        if cy < h * 0.20 or cy > h * 0.92:
                            continue
                        all_candidates.append(np.array([cx, cy]))

    if len(all_candidates) < 3:
        return None

    pts = np.array(all_candidates)

    # Cluster: find the most consistent detection across frames
    # Simple approach: mode-like — find point with most neighbors within 30px
    best_center, best_count = None, 0
    for pt in pts:
        dists = np.linalg.norm(pts - pt, axis=1)
        count = np.sum(dists < 30)
        if count > best_count:
            best_count = count
            best_center = pt

    if best_count < 2:
        return None

    # Refine as mean of the cluster
    mask_cluster = np.linalg.norm(pts - best_center, axis=1) < 30
    return np.mean(pts[mask_cluster], axis=0)


def _foot_cluster_base_fallback(detection_sequence: List[dict]) -> Optional[np.ndarray]:
    """Fallback: estimate base from median runner foot positions."""
    foot_positions = []
    for fd in detection_sequence:
        runner_pose = fd.get("poses_by_role", {}).get("runner") or (fd.get("poses") or [None])[0]
        if not runner_pose:
            continue
        kp = runner_pose.get("keypoints", {})
        for fk in ("left_heel", "right_heel", "left_foot_index", "right_foot_index"):
            pt = kp.get(fk, {})
            if pt.get("visibility", 0) > VIS_THRESHOLD:
                foot_positions.append(np.array([pt["x"], pt["y"]]))

    if len(foot_positions) < 5:
        return None
    return np.median(np.array(foot_positions), axis=0)


# ---------------------------------------------------------------------------
# 3D lifting
# ---------------------------------------------------------------------------

def lift_pose_to_3d(frame_data: dict, homography: FieldHomography) -> List[Dict]:
    height_prior = {
        "left_ankle": 0.3,  "right_ankle": 0.3,
        "left_heel": 0.1,   "right_heel": 0.1,
        "left_foot_index": 0.0, "right_foot_index": 0.0,
        "left_wrist": 3.5,  "right_wrist": 3.5,
        "left_elbow": 3.0,  "right_elbow": 3.0,
        "left_hip": 2.8,    "right_hip": 2.8,
        "left_shoulder": 4.5, "right_shoulder": 4.5,
    }
    poses_3d = []
    for pose in frame_data.get("poses", []):
        kp3d = {}
        for name, pt in pose.get("keypoints", {}).items():
            if pt.get("visibility", 0) < VIS_THRESHOLD:
                continue
            fp = homography.pixel_to_field(pt["x"], pt["y"])
            fp[2] = height_prior.get(name, 1.0)
            kp3d[name] = fp.tolist()
        if kp3d:
            kp3d["_role"] = pose.get("role", "unknown")
            poses_3d.append(kp3d)
    return poses_3d


# ---------------------------------------------------------------------------
# Contact timing (role-aware)
# ---------------------------------------------------------------------------

def compute_contact_timing(
    detection_sequence: List[dict],
    base_px: Optional[np.ndarray],
    first_base_world: np.ndarray,
    frame_w: int = 640,
    play_start_frame: int = 0,
    play_end_frame: Optional[int] = None,
) -> dict:
    """
    Finds foot-to-base contact (runner) and glove-to-runner contact (fielder).
    Uses runner pose for foot check, fielder pose for glove check.
    Falls back to any-pose if role separation isn't available.
    """
    FOOT_THRESHOLD_PX = frame_w * 0.10    # ~64px for 640-wide
    GLOVE_THRESHOLD_PX = frame_w * 0.14

    foot_contact_frame = None
    glove_contact_frame = None

    end = play_end_frame if play_end_frame is not None else len(detection_sequence)
    for i, fd in enumerate(detection_sequence[play_start_frame:end], start=play_start_frame):
        ts = fd.get("timestamp_ms", i * 33)
        by_role = fd.get("poses_by_role", {})

        # --- Runner foot positions ---
        runner_feet_px: List[np.ndarray] = []
        runner_pose = by_role.get("runner") or (fd.get("poses") or [None])[0]
        if runner_pose:
            kp = runner_pose.get("keypoints", {})
            for fk in FOOT_KEYS:
                pt = kp.get(fk, {})
                if pt.get("visibility", 0) > VIS_THRESHOLD:
                    runner_feet_px.append(np.array([pt["x"], pt["y"]]))

        # --- Fielder hand positions ---
        fielder_hands_px: List[np.ndarray] = []
        fielder_pose = by_role.get("fielder")
        if not fielder_pose and len(fd.get("poses", [])) > 1:
            fielder_pose = fd["poses"][1]
        if fielder_pose:
            kp = fielder_pose.get("keypoints", {})
            for hk in HAND_KEYS:
                pt = kp.get(hk, {})
                if pt.get("visibility", 0) > VIS_THRESHOLD:
                    fielder_hands_px.append(np.array([pt["x"], pt["y"]]))

        # 1. Foot → base contact (pixel space)
        if base_px is not None and foot_contact_frame is None:
            for fp in runner_feet_px:
                dist = float(np.linalg.norm(fp - base_px))
                if dist < FOOT_THRESHOLD_PX:
                    foot_contact_frame = {
                        "frame_idx": i, "timestamp_ms": ts,
                        "distance_px": dist, "method": "pixel_base",
                    }
                    break

        # 2. Foot → base contact (world space, calibrated videos)
        poses_3d = fd.get("poses_3d", [])
        if foot_contact_frame is None:
            for p3d in poses_3d:
                if p3d.get("_role", "runner") not in ("runner", "unknown"):
                    continue
                for fk in FOOT_KEYS:
                    if fk in p3d:
                        fp3 = np.array(p3d[fk])
                        dist = float(np.linalg.norm(fp3[:2] - first_base_world[:2]))
                        if dist < 2.0:
                            foot_contact_frame = {
                                "frame_idx": i, "timestamp_ms": ts,
                                "distance_ft": dist, "method": "world_space",
                            }
                            break
                if foot_contact_frame:
                    break

        # 3. Fielder glove → runner foot contact (pixel space)
        if glove_contact_frame is None and fielder_hands_px and runner_feet_px:
            for hand in fielder_hands_px:
                for foot in runner_feet_px:
                    dist = float(np.linalg.norm(hand - foot))
                    if dist < GLOVE_THRESHOLD_PX:
                        glove_contact_frame = {
                            "frame_idx": i, "timestamp_ms": ts,
                            "distance_px": dist, "method": "pixel_glove",
                        }
                        break
                if glove_contact_frame:
                    break

    # Decision
    decision = "unknown"
    margin_ms = None
    if foot_contact_frame and glove_contact_frame:
        diff = foot_contact_frame["timestamp_ms"] - glove_contact_frame["timestamp_ms"]
        margin_ms = int(diff)
        decision = "SAFE" if diff < -33 else ("OUT" if diff > 33 else "TOO_CLOSE")
    elif foot_contact_frame:
        decision = "SAFE"
    elif glove_contact_frame:
        decision = "OUT"

    return {
        "foot_contact": foot_contact_frame,
        "glove_contact": glove_contact_frame,
        "margin_ms": margin_ms,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def process_video(entry: dict) -> dict:
    detections_file = entry.get("detections_file")
    if not detections_file or not Path(detections_file).exists():
        return entry

    detection_data = json.loads(Path(detections_file).read_text())
    frames_dir = Path(entry["frames_dir"])
    video_id = Path(entry["path"]).stem

    # Homography calibration from first frame
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    first_frame = frame_files[0] if frame_files else None
    homo = FieldHomography(str(first_frame)) if first_frame else None

    detected_bases: Dict = {}
    if homo:
        detected_bases = homo.detect_bases_auto()
        if len(detected_bases) >= 4:
            homo.compute_homography(detected_bases)
            print(f"  Homography calibrated from {len(detected_bases)} bases")
        else:
            print(f"  Auto-detection found {len(detected_bases)} bases — default scale")

    # Lift poses to 3D
    if homo:
        for fd in detection_data:
            fd["poses_3d"] = lift_pose_to_3d(fd, homo)

    # Detect first base pixel (multi-frame consensus)
    ann_file = ANNOT_DIR / f"{video_id}.json"
    ann = json.loads(ann_file.read_text()) if ann_file.exists() else {}
    play_start_frame = ann.get("play_start_frame", 0)
    play_end_frame   = ann.get("play_end_frame", None)

    base_px = detect_first_base_pixel(frames_dir, annotation_file=ann_file)
    if base_px is not None:
        print(f"  Base pixel center: ({base_px[0]:.0f}, {base_px[1]:.0f})")
    else:
        base_px = _foot_cluster_base_fallback(detection_data)
        if base_px is not None:
            print(f"  Base pixel (foot cluster fallback): ({base_px[0]:.0f}, {base_px[1]:.0f})")
        else:
            print(f"  Base pixel: not detected")

    if play_start_frame or play_end_frame:
        print(f"  Play window: frames {play_start_frame}–{play_end_frame or 'end'}")

    frame_w = detection_data[0].get("resolution", [640])[0] if detection_data else 640
    timing = compute_contact_timing(
        detection_data,
        base_px,
        FIELD_LANDMARKS["first_base"],
        frame_w,
        play_start_frame=play_start_frame,
        play_end_frame=play_end_frame,
    )

    output = {
        "video_id": video_id,
        "expected": entry.get("expected", "unknown"),
        "decision": timing["decision"],
        "margin_ms": timing["margin_ms"],
        "foot_contact": timing["foot_contact"],
        "glove_contact": timing["glove_contact"],
        "calibrated_bases": list(detected_bases.keys()),
        "base_px": base_px.tolist() if base_px is not None else None,
        "frame_count": len(detection_data),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / f"{video_id}_decision.json").write_text(json.dumps(output, indent=2))
    Path(detections_file).write_text(json.dumps(detection_data, indent=2))

    entry["decision_file"] = str(MODELS_DIR / f"{video_id}_decision.json")
    entry["decision"] = timing["decision"]
    entry["margin_ms"] = timing["margin_ms"]
    entry["stage"] = "analyzed"
    return entry


def main():
    catalog_path = CATALOG_FILE
    if not catalog_path.exists():
        print("No catalog. Run previous stages first.")
        return

    catalog = json.loads(catalog_path.read_text())
    changed = False

    for i, entry in enumerate(catalog):
        if entry.get("stage") not in ("detected", "analyzed"):
            continue
        # Re-analyze even previously analyzed entries (so fixes take effect)
        if not entry.get("detections_file"):
            continue
        print(f"\n[{i+1}/{len(catalog)}] {Path(entry['path']).name}")
        entry["stage"] = "detected"  # allow re-analysis
        catalog[i] = process_video(entry)
        changed = True

    if changed:
        catalog_path.write_text(json.dumps(catalog, indent=2))
    print("\n3D reconstruction complete.")


if __name__ == "__main__":
    main()
