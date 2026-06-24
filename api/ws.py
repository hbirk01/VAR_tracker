"""WebSocket connection manager — broadcast progress events to all listeners on a job."""
import asyncio
import json
from collections import defaultdict
from fastapi import WebSocket

# job_id -> list of connected websockets
_connections: dict[str, list[WebSocket]] = defaultdict(list)


async def connect(job_id: str, ws: WebSocket):
    await ws.accept()
    _connections[job_id].append(ws)


def disconnect(job_id: str, ws: WebSocket):
    _connections[job_id].remove(ws)
    if not _connections[job_id]:
        del _connections[job_id]


async def broadcast(job_id: str, event: dict):
    dead = []
    for ws in _connections.get(job_id, []):
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _connections[job_id].remove(ws)


def emit(job_id: str, event: dict):
    """Fire-and-forget from sync pipeline code via asyncio."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(broadcast(job_id, event))
    except RuntimeError:
        pass
