"""VAR Tracker API — FastAPI backend."""
import uuid
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl

from api import db, ws, runner

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="VAR Tracker", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=1)  # one job at a time


@app.on_event("startup")
async def startup():
    db.init_db()


# ── Request / response models ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    url: str
    expected: Optional[str] = None  # "safe" | "out" | None


class JobResponse(BaseModel):
    job_id: str
    status: str
    stage: Optional[str] = None
    decision: Optional[str] = None
    margin_ms: Optional[float] = None
    foot_ms: Optional[float] = None
    glove_ms: Optional[float] = None
    video_path: Optional[str] = None
    error: Optional[str] = None
    url: str
    expected: Optional[str] = None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: dict) -> "JobResponse":
        return cls(job_id=row["id"], **{k: v for k, v in row.items() if k != "id"})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/analyze", response_model=JobResponse)
async def submit_analysis(req: AnalyzeRequest):
    job_id = uuid.uuid4().hex[:12]
    db.create_job(job_id, req.url, req.expected)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, runner.run_job, job_id, req.url, req.expected)

    row = db.get_job(job_id)
    return JobResponse.from_row(row)


@app.get("/api/jobs", response_model=List[JobResponse])
def get_jobs():
    return [JobResponse.from_row(r) for r in db.list_jobs()]


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    row = db.get_job(job_id)
    if not row:
        raise HTTPException(404, "Job not found")
    return JobResponse.from_row(row)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    row = db.get_job(job_id)
    if not row:
        raise HTTPException(404, "Job not found")
    with db.connect() as conn:
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    return {"ok": True}


@app.get("/api/video/{job_id}")
def stream_video(job_id: str):
    row = db.get_job(job_id)
    if not row or not row.get("video_path"):
        raise HTTPException(404, "Video not available")
    path = Path(row["video_path"])
    if not path.exists():
        raise HTTPException(404, "Video file missing")
    return FileResponse(str(path), media_type="video/mp4")


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await ws.connect(job_id, websocket)
    # Send current state immediately on connect
    row = db.get_job(job_id)
    if row:
        await websocket.send_json({"status": row["status"], "stage": row["stage"]})
    try:
        while True:
            await websocket.receive_text()  # keep alive; client pings
    except WebSocketDisconnect:
        ws.disconnect(job_id, websocket)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/presets")
def get_presets():
    """Return catalog entries with source URLs as quick-launch presets."""
    ROOT = Path(__file__).parent.parent
    catalog_path = ROOT / "data" / "catalog.json"
    if not catalog_path.exists():
        return []
    catalog = json.loads(catalog_path.read_text())
    presets = []
    for e in catalog:
        url = e.get("source_url")
        if not url:
            continue
        vid = Path(e["path"]).stem
        presets.append({
            "id": vid,
            "url": url,
            "label": e.get("notes", vid)[:60],
            "expected": e.get("expected"),
            "decision": e.get("decision"),
            "margin_ms": e.get("margin_ms"),
        })
    return presets


@app.get("/health")
def health():
    return {"ok": True}
