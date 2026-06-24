"""
Background pipeline runner.
Runs stages 1-5 for a single job, updating DB + broadcasting WS events at each step.
"""
import sys
import json
import traceback
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

from api import db, ws

STAGES = [
    ("downloading",  "Download video"),
    ("extracting",   "Extract frames"),
    ("detecting",    "Pose detection"),
    ("analyzing",    "3D reconstruction"),
    ("rendering",    "Render output"),
]


def _emit(job_id: str, status: str, stage: str, message: str = ""):
    db.update_job(job_id, status=status, stage=stage)
    ws.emit(job_id, {"status": status, "stage": stage, "message": message})


def run_job(job_id: str, url: str, expected: Optional[str]):
    try:
        _emit(job_id, "downloading", "download", f"Fetching {url}")
        _download(job_id, url)

        _emit(job_id, "extracting", "extract", "Extracting play frames")
        _extract(job_id)

        _emit(job_id, "detecting", "detect", "Running pose detection")
        _detect(job_id)

        _emit(job_id, "analyzing", "analyze", "Computing contact timing")
        decision_info = _analyze(job_id)

        _emit(job_id, "rendering", "render", "Rendering annotated video")
        video_path = _render(job_id)

        decision  = decision_info.get("decision", "unknown")
        margin_ms = decision_info.get("margin_ms")
        foot_ms   = decision_info.get("foot_contact", {}).get("timestamp_ms") if decision_info.get("foot_contact") else None
        glove_ms  = decision_info.get("glove_contact", {}).get("timestamp_ms") if decision_info.get("glove_contact") else None

        db.update_job(
            job_id,
            status="done",
            stage="done",
            decision=decision,
            margin_ms=margin_ms,
            foot_ms=foot_ms,
            glove_ms=glove_ms,
            video_path=str(video_path) if video_path else None,
        )
        ws.emit(job_id, {
            "status": "done",
            "stage": "done",
            "decision": decision,
            "margin_ms": margin_ms,
        })

    except Exception as exc:
        err = traceback.format_exc()
        db.update_job(job_id, status="error", error=err)
        ws.emit(job_id, {"status": "error", "message": str(exc)})


# ── Stage wrappers ────────────────────────────────────────────────────────────

def _catalog_entry(job_id: str) -> Optional[dict]:
    catalog_path = ROOT / "data" / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    for e in catalog:
        if Path(e["path"]).stem == job_id:
            return e
    return None


def _save_catalog(catalog: list):
    (ROOT / "data" / "catalog.json").write_text(json.dumps(catalog, indent=2))


def _download(job_id: str, url: str):
    import subprocess
    out_path = ROOT / "data" / "raw_videos" / f"{job_id}.mp4"
    if out_path.exists():
        return  # already downloaded

    result = subprocess.run(
        [
            str(ROOT / "venv" / "bin" / "yt-dlp"),
            "--client-side-ranges", "android_vr",
            "-f", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", str(out_path),
            url,
        ],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr[-500:]}")

    # Add to catalog
    catalog_path = ROOT / "data" / "catalog.json"
    catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else []
    # Remove any existing entry for this job_id
    catalog = [e for e in catalog if Path(e["path"]).stem != job_id]
    catalog.append({
        "path": str(ROOT / "data" / "raw_videos" / f"{job_id}.mp4"),
        "stage": "downloaded",
        "source_url": url,
        "expected": None,
        "notes": f"Submitted via API job {job_id}",
    })
    _save_catalog(catalog)


def _extract(job_id: str):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("stage2", ROOT / "pipeline" / "02_extract_frames.py")
    stage2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(stage2)
    catalog_path = ROOT / "data" / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    for i, e in enumerate(catalog):
        if Path(e["path"]).stem == job_id:
            catalog[i] = stage2.process_video(e)
            break
    _save_catalog(catalog)


def _detect(job_id: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage3", ROOT / "pipeline" / "03_pose_detection.py")
    stage3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(stage3)
    catalog_path = ROOT / "data" / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    pose_detector = stage3.get_pose_detector()
    yolo_detector = stage3.get_yolo_detector()
    for i, e in enumerate(catalog):
        if Path(e["path"]).stem == job_id:
            catalog[i] = stage3.process_video_frames(e, pose_detector, yolo_detector)
            break
    _save_catalog(catalog)


def _analyze(job_id: str) -> dict:
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage4", ROOT / "pipeline" / "04_3d_reconstruction.py")
    stage4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(stage4)
    catalog_path = ROOT / "data" / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    entry = {}
    for i, e in enumerate(catalog):
        if Path(e["path"]).stem == job_id:
            catalog[i] = stage4.process_video(e)
            entry = catalog[i]
            break
    _save_catalog(catalog)

    decision_file = entry.get("decision_file")
    if decision_file and Path(decision_file).exists():
        return json.loads(Path(decision_file).read_text())
    return {}


def _render(job_id: str) -> Optional[Path]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage5", ROOT / "pipeline" / "05_visualize.py")
    stage5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(stage5)
    catalog_path = ROOT / "data" / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    for e in catalog:
        if Path(e["path"]).stem == job_id:
            out = stage5.render_video(e)
            return Path(out) if out else None
    return None
