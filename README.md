# BaseballCV — First Base Contact Analysis

Multi-angle video analysis pipeline to determine safe/out calls at first base using pose estimation, 3D reconstruction, and contact timing detection.

## Setup

```bash
cd /Users/Harvin/BaseballCV
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Also ensure `ffmpeg` is installed: `brew install ffmpeg`

## Pipeline

```
Stage 1 → Download videos (YouTube search + direct URLs)
Stage 2 → Extract frames, detect play window via motion analysis  
Stage 3 → YOLOv8 player/ball detection + MediaPipe pose estimation
Stage 4 → Homography calibration, 2D→3D lift, contact timing
Stage 5 → Annotated video output + decision overlay
```

## Run

```bash
# Full pipeline
python pipeline/run_all.py

# Add a specific clip (YouTube URL)
python pipeline/add_video.py "https://youtube.com/..." --label "ALCS 2023 close play" --expected safe

# Restart from a specific stage
python pipeline/run_all.py --from 3
```

## Output

- `output/visualizations/` — annotated MP4 with decision banner, timing bar, keypoints
- `output/3d_models/` — per-video JSON decision files with millisecond margins
- `data/catalog.json` — master record of all videos and their pipeline state

## Decision Logic

The system measures:
1. **Foot contact frame** — when runner's foot comes within 1.5ft of first base (field coords)
2. **Glove contact frame** — when fielder's hand comes within 2.0ft of runner's feet

If foot contact precedes glove contact by >50ms → **SAFE**  
If glove contact precedes foot contact by >50ms → **OUT**  
Within 50ms margin → **TOO_CLOSE** (inconclusive — human review needed)

## Known Limitations

- Single-angle broadcast video: foot occlusion is common on close plays
- 30fps source = 33ms minimum resolution (real plays often decided by <10ms)
- Auto base detection requires white base markers visible in frame
- Multi-angle triangulation requires synchronized feeds (manual timecode alignment)

## Adding Ground Truth

Edit `data/catalog.json` and set `"expected": "safe"` or `"expected": "out"` for any video where the official replay ruling is known. Stage 5 will score accuracy automatically.
