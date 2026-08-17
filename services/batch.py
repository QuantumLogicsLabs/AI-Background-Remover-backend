"""
Batch Processing Service.

Manages batch background-removal jobs, persisted to MongoDB so job state
survives server restarts and horizontal scaling is possible.

Job states
──────────
  pending   → queued, not yet started
  running   → actively processing
  done      → all files complete (or errored individually)

Per-file states
───────────────
  queued    → waiting
  processing→ inference running
  done      → output PNG saved
  error     → inference failed for this file
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, Literal

# Add AI module to path (mirrors bg_removal.py pattern)
_AI_DIR = Path(__file__).resolve().parents[2] / "AI"
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

from inference import run_inference  # noqa: E402

from services.database import get_collection  # noqa: E402


# ── Type definitions ───────────────────────────────────────────────────────

FileStatus = Literal["queued", "processing", "done", "error"]
JobStatus  = Literal["pending", "running", "done"]


class FileEntry(TypedDict):
    original_name:   str
    upload_path:     str
    output_filename: str | None
    status:          FileStatus
    error:           str | None


class Job(TypedDict):
    job_id:     str
    user_id:    str
    quality:    str          # "fast" | "standard" | "quality"
    status:     JobStatus
    created_at: str
    files:      list[FileEntry]
    total:      int
    completed:  int
    failed:     int


OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Internal helpers ───────────────────────────────────────────────────────

def _collection():
    """Return the Motor collection for batch jobs."""
    return get_collection("batch_jobs")


async def _upsert_job(job: Job) -> None:
    """Persist the full job document to MongoDB (upsert by job_id)."""
    try:
        col = _collection()
        await col.replace_one(
            {"job_id": job["job_id"]},
            job,
            upsert=True,
        )
    except Exception:
        pass  # persistence failure should not crash the processing loop


# ── Public helpers ─────────────────────────────────────────────────────────

async def create_job(
    file_entries: list[dict],
    user_id: str,
    quality: str = "fast",
) -> str:
    """
    Register a new batch job, persist it to MongoDB, and return its job_id.

    Args:
        file_entries: list of {"original_name": str, "upload_path": str}
        user_id:      ID of the user who owns this job
        quality:      AI model quality — "fast" | "standard" | "quality"
    """
    job_id = str(uuid.uuid4())
    files: list[FileEntry] = [
        {
            "original_name":   e["original_name"],
            "upload_path":     e["upload_path"],
            "output_filename": None,
            "status":          "queued",
            "error":           None,
        }
        for e in file_entries
    ]
    job: Job = {
        "job_id":     job_id,
        "user_id":    user_id,
        "quality":    quality,
        "status":     "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files":      files,
        "total":      len(files),
        "completed":  0,
        "failed":     0,
    }
    await _upsert_job(job)
    return job_id


async def get_job(job_id: str, user_id: str | None = None) -> Job | None:
    """
    Retrieve job from MongoDB.

    Args:
        job_id:  The job identifier.
        user_id: When provided, the job must belong to this user (authorization).

    Returns:
        Job dict or None if not found / not owned by user.
    """
    try:
        query: dict = {"job_id": job_id}
        if user_id:
            query["user_id"] = user_id
        col = _collection()
        doc = await col.find_one(query, {"_id": 0})
        return dict(doc) if doc else None  # type: ignore[arg-type]
    except Exception:
        return None


def process_batch(job_id: str) -> None:
    """
    Background task entry point. Runs in FastAPI's thread-pool executor.

    Iterates over queued files, runs bg removal on each using the quality
    stored on the job, and persists per-file and overall job status to
    MongoDB after every file.

    Motor (async) collections are bound to the event loop that created the
    Motor client (FastAPI's main loop).  We must NOT call those collections
    from a *different* event loop — doing so raises
    ``RuntimeError: Task attached to a different loop``.

    Fix: spin up a brand-new Motor client (and therefore a fresh event-loop-
    independent connection) inside the dedicated asyncio.run() call.  The
    client is closed before the coroutine returns, so no connection is leaked.
    """
    asyncio.run(_process_batch_async(job_id))


async def _process_batch_async(job_id: str) -> None:
    """
    Async implementation of batch processing.

    Creates its own Motor client so it is fully decoupled from the Motor
    client that lives on FastAPI's main event loop.
    """
    import os as _os
    from motor.motor_asyncio import AsyncIOMotorClient

    mongo_uri   = _os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name     = _os.getenv("MONGO_DB_NAME", "ai_bg_remover")

    _uri_lower  = mongo_uri.lower()
    _use_tls    = (
        "ssl=true" in _uri_lower
        or "tls=true" in _uri_lower
        or _uri_lower.startswith("mongodb+srv://")
    )
    _tls_kwargs = {"tls": True, "tlsAllowInvalidCertificates": False} if _use_tls else {}

    _client = AsyncIOMotorClient(
        mongo_uri,
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
        socketTimeoutMS=15000,
        **_tls_kwargs,
    )
    col = _client[db_name]["batch_jobs"]

    async def _upsert(job: Job) -> None:
        try:
            await col.replace_one({"job_id": job["job_id"]}, job, upsert=True)
        except Exception:
            pass

    async def _fetch() -> Job | None:
        try:
            doc = await col.find_one({"job_id": job_id}, {"_id": 0})
            return dict(doc) if doc else None  # type: ignore[arg-type]
        except Exception:
            return None

    try:
        job = await _fetch()
        if job is None:
            return

        # Honour the quality chosen by the user; fall back to "fast" for
        # jobs created before the quality field was added.
        quality = job.get("quality", "fast")

        job["status"] = "running"
        await _upsert(job)

        for entry in job["files"]:
            entry["status"] = "processing"
            await _upsert(job)

            stem            = Path(entry["upload_path"]).stem
            output_filename = f"{stem}_result.png"
            output_path     = os.path.join(OUTPUT_DIR, output_filename)

            try:
                run_inference(entry["upload_path"], output_path, quality=quality)
                entry["output_filename"] = output_filename
                entry["status"]          = "done"
                job["completed"]        += 1
            except Exception as exc:
                entry["status"] = "error"
                entry["error"]  = str(exc)
                job["failed"]  += 1

            await _upsert(job)

        job["status"] = "done"
        await _upsert(job)

    finally:
        _client.close()


def build_zip(job: Job) -> tuple[bytes, str]:
    """
    Build an in-memory ZIP of all successfully processed output files.

    Args:
        job: The job dict (already retrieved from MongoDB).

    Returns:
        (zip_bytes, zip_filename)
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for entry in job["files"]:
            if entry["status"] == "done" and entry["output_filename"]:
                file_path = os.path.join(OUTPUT_DIR, entry["output_filename"])
                if os.path.isfile(file_path):
                    zf.write(file_path, arcname=entry["output_filename"])

    buf.seek(0)
    zip_filename = f"batch_{job['job_id'][:8]}_results.zip"
    return buf.read(), zip_filename
