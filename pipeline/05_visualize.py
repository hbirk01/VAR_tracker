"""
Stage 5: Generate annotated video output and decision overlay.
Draws keypoints, base proximity, timing bar, and final decision on frames.
"""
import cv2
import json
import numpy as np
from pathlib import Path

CATALOG_FILE = Path(__file__).parent.parent / "data" / "catalog.json"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "visualizations"

COLORS = {
    "SAFE": (50, 205, 50),
    "OUT": (0, 0, 220),
    "TOO_CLOSE": (0, 165, 255),
    "unknown": (128, 128, 128),
    "foot": (0, 255, 255),
    "hand": (255, 100, 0),
    "base": (255, 255, 255),
    "skeleton": (180, 180, 180),
}

MEDIAPIPE_CONNECTIONS = [
    ("left_ankle", "left_heel"), ("right_ankle", "right_heel"),
    ("left_heel", "left_foot_index"), ("right_heel", "right_foot_index"),
    ("left_wrist", "left_elbow"), ("right_wrist", "right_elbow"),
]


def draw_frame(frame_path: str, frame_data: dict, decision_info: dict, frame_idx: int, total_frames: int) -> np.ndarray:
    img = cv2.imread(frame_path)
    if img is None:
        return np.zeros((720, 1280, 3), dtype=np.uint8)

    h, w = img.shape[:2]
    overlay = img.copy()

    decision = decision_info.get("decision", "unknown")
    color = COLORS.get(decision, COLORS["unknown"])

    # --- Draw pose keypoints ---
    for pose in frame_data.get("poses", []):
        kp = pose["keypoints"]
        pts = {}
        for name, pt in kp.items():
            if pt.get("visibility", 0) > 0.4:
                x, y = int(pt["x"]), int(pt["y"])
                pts[name] = (x, y)
                dot_color = COLORS["foot"] if "ankle" in name or "foot" in name or "heel" in name else \
                            COLORS["hand"] if "wrist" in name else COLORS["skeleton"]
                cv2.circle(overlay, (x, y), 6, dot_color, -1)

        for a, b in MEDIAPIPE_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(overlay, pts[a], pts[b], COLORS["skeleton"], 2)

    # --- Draw YOLO detections ---
    for det in frame_data.get("detections", []):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = det["label"]
        conf = det["confidence"]
        box_color = (200, 200, 200) if label == "person" else (0, 255, 100)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 1)
        cv2.putText(overlay, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

    # Blend overlay
    img = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)

    # --- Decision banner ---
    banner_h = 60
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)

    decision_text = decision
    if decision_info.get("margin_ms") is not None:
        ms = decision_info["margin_ms"]
        sign = "+" if ms > 0 else ""
        decision_text += f"  (margin: {sign}{ms:.0f}ms)"

    cv2.putText(banner, decision_text, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

    # Progress bar
    progress = int((frame_idx / max(total_frames - 1, 1)) * (w - 40)) + 20
    cv2.rectangle(banner, (20, 50), (w - 20, 58), (60, 60, 60), -1)
    cv2.rectangle(banner, (20, 50), (progress, 58), color, -1)

    # Contact markers on progress bar
    foot_f = decision_info.get("foot_contact", {})
    glove_f = decision_info.get("glove_contact", {})
    if foot_f and foot_f.get("frame_idx") is not None:
        fx = int((foot_f["frame_idx"] / max(total_frames - 1, 1)) * (w - 40)) + 20
        cv2.circle(banner, (fx, 54), 5, COLORS["foot"], -1)
        cv2.putText(banner, "FOOT", (fx - 15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS["foot"], 1)
    if glove_f and glove_f.get("frame_idx") is not None:
        gx = int((glove_f["frame_idx"] / max(total_frames - 1, 1)) * (w - 40)) + 20
        cv2.circle(banner, (gx, 54), 5, COLORS["hand"], -1)
        cv2.putText(banner, "GLOVE", (gx - 20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS["hand"], 1)

    img = np.vstack([banner, img])
    return img


def render_video(entry: dict) -> str | None:
    frames_dir = entry.get("frames_dir")
    detections_file = entry.get("detections_file")
    decision_file = entry.get("decision_file")

    if not all([frames_dir, detections_file, decision_file]):
        return None

    frame_files = sorted(Path(frames_dir).glob("frame_*.jpg"))
    detections = json.loads(Path(detections_file).read_text())
    decision_info = json.loads(Path(decision_file).read_text())

    video_id = Path(entry["path"]).stem
    output_path = OUTPUT_DIR / f"{video_id}_analyzed.mp4"

    if output_path.exists():
        print(f"  SKIP (exists): {output_path.name}")
        return str(output_path)

    if not frame_files:
        return None

    sample = cv2.imread(str(frame_files[0]))
    if sample is None:
        return None

    h, w = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 30, (w, h + 60))

    total = len(frame_files)
    for i, (fp, fd) in enumerate(zip(frame_files, detections)):
        frame = draw_frame(str(fp), fd, decision_info, i, total)
        frame = cv2.resize(frame, (w, h + 60))
        writer.write(frame)

    writer.release()
    print(f"  Output: {output_path}")
    return str(output_path)


def print_summary(catalog: list[dict]):
    print("\n" + "=" * 60)
    print("PLAY DECISIONS SUMMARY")
    print("=" * 60)
    for entry in catalog:
        if entry.get("stage") == "analyzed":
            name = Path(entry["path"]).stem[:40]
            decision = entry.get("decision", "?")
            expected = entry.get("expected", "?")
            margin = entry.get("margin_ms")
            margin_str = f"{margin:+.0f}ms" if margin is not None else "N/A"
            match = "✓" if decision.lower() == expected.lower() else "✗" if expected != "unknown" else "-"
            print(f"  [{match}] {name[:35]:<35} → {decision:<12} (margin: {margin_str})")


def main():
    catalog_path = CATALOG_FILE
    if not catalog_path.exists():
        print("No catalog found.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(catalog_path.read_text())

    for i, entry in enumerate(catalog):
        if entry.get("stage") != "analyzed":
            continue
        print(f"\n[{i+1}] Rendering: {Path(entry['path']).name}")
        output = render_video(entry)
        if output:
            catalog[i]["output_video"] = output

    catalog_path.write_text(json.dumps(catalog, indent=2))
    print_summary(catalog)


if __name__ == "__main__":
    main()
