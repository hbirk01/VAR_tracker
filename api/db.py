"""SQLite job state — single file, no ORM needed."""
import sqlite3
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "data" / "jobs.db"


def init_db():
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id          TEXT PRIMARY KEY,
                url         TEXT NOT NULL,
                expected    TEXT,
                status      TEXT NOT NULL DEFAULT 'queued',
                stage       TEXT,
                error       TEXT,
                decision    TEXT,
                margin_ms   REAL,
                foot_ms     REAL,
                glove_ms    REAL,
                video_path  TEXT,
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)


@contextmanager
def connect():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_job(job_id: str, url: str, expected: Optional[str]):
    with connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, url, expected) VALUES (?, ?, ?)",
            (job_id, url, expected),
        )


def update_job(job_id: str, **kwargs):
    if not kwargs:
        return
    kwargs["updated_at"] = "datetime('now')"
    sets = ", ".join(
        f"{k} = datetime('now')" if v == "datetime('now')" else f"{k} = ?"
        for k, v in kwargs.items()
    )
    vals = [v for v in kwargs.values() if v != "datetime('now')"]
    with connect() as conn:
        conn.execute(f"UPDATE jobs SET {sets} WHERE id = ?", vals + [job_id])


def get_job(job_id: str) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None


def list_jobs() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
