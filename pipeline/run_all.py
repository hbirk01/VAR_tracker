"""
Master runner — executes all pipeline stages in sequence.
Usage:
    python run_all.py               # full pipeline
    python run_all.py --from 3     # restart from stage 3
    python run_all.py --stage 1    # run only stage 1
"""
import argparse
import subprocess
import sys
from pathlib import Path

STAGES = [
    (1, "01_download.py",        "Download videos"),
    (2, "02_extract_frames.py",  "Extract frames"),
    (3, "03_pose_detection.py",  "Pose & object detection"),
    (4, "04_3d_reconstruction.py", "3D reconstruction & timing"),
    (5, "05_visualize.py",       "Visualize & render output"),
]

PIPELINE_DIR = Path(__file__).parent


def run_stage(script: str, label: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  STAGE: {label}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(PIPELINE_DIR / script)])
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_stage", type=int, default=1)
    parser.add_argument("--stage", type=int, default=None)
    args = parser.parse_args()

    for num, script, label in STAGES:
        if args.stage is not None and num != args.stage:
            continue
        if args.stage is None and num < args.from_stage:
            continue

        ok = run_stage(script, f"{num}. {label}")
        if not ok:
            print(f"\nStage {num} failed. Stopping.")
            sys.exit(1)

    print("\n\nAll stages complete.")


if __name__ == "__main__":
    main()
