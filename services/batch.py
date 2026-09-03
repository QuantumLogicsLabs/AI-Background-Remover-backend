"""
Batch Processing Service.

Manages batch background-removal jobs, persisted to MongoDB so job state
survives server restarts and horizontal scaling is possible.
Dispatches real-time SSE progress events via job_events.

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
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, TypedDict, Literal

logger = logging.getLogger(__name__)

# Add AI module to path (mirrors bg_removal.py pattern)
_AI_DIR = Path(__file__).resolve().parents[2] / "AI"
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

from inference import run_inference  # noqa: E402
from services.database import get_collection  # noqa: E402
from services.job_events import job_events  # noqa: E402


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
    except Exception as e:
        logger.warning(f"[Batch] Failed to persist job {job.get('job_id')} to DB: {e}")


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
    except Exception as e:
        logger.error(f"[Batch] Error fetching job {job_id}: {e}")
        raise HTTPException(status_code=503, detail=f"Database temporary error: {e}")


async def process_batch_job(job_id: str) -> None:
    """
    Async batch processing task executed by the job queue workers.
    Processes files sequentially, persists progress, and publishes SSE events.
    """
    job = await get_job(job_id)
    if job is None:
        logger.warning(f"[Batch] Job {job_id} not found for processing.")
        return

    quality = job.get("quality", "fast")
    job["status"] = "running"
    await _upsert_job(job)

    # Publish job started event
    await job_events.publish(job_id, "job_started", {
        "job_id": job_id,
        "status": "running",
        "quality": quality,
        "total": job["total"],
        "completed": 0,
        "failed": 0,
    })

    loop = asyncio.get_running_loop()

    for idx, entry in enumerate(job["files"]):
        entry["status"] = "processing"
        await _upsert_job(job)

        # Publish file processing event
        await job_events.publish(job_id, "file_processing", {
            "index": idx,
            "original_name": entry["original_name"],
            "status": "processing",
        })

        stem = Path(entry["upload_path"]).stem
        output_filename = f"{stem}_result.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        try:
            # Run CPU-bound AI inference in thread pool executor
            await loop.run_in_executor(
                None,
                run_inference,
                entry["upload_path"],
                output_path,
                quality,
            )
            entry["output_filename"] = output_filename
            entry["status"] = "done"
            job["completed"] += 1

            # Publish file done event
            await job_events.publish(job_id, "file_done", {
                "index": idx,
                "original_name": entry["original_name"],
                "output_filename": output_filename,
                "download_url": f"/api/download/{output_filename}",
                "status": "done",
                "completed": job["completed"],
                "failed": job["failed"],
                "total": job["total"],
            })

        except Exception as exc:
            logger.error(f"[Batch] Inference error on file '{entry['original_name']}': {exc}")
            entry["status"] = "error"
            entry["error"] = str(exc)
            job["failed"] += 1

            # Publish file error event
            await job_events.publish(job_id, "file_error", {
                "index": idx,
                "original_name": entry["original_name"],
                "status": "error",
                "error": str(exc),
                "completed": job["completed"],
                "failed": job["failed"],
                "total": job["total"],
            })

        await _upsert_job(job)

    job["status"] = "done"
    await _upsert_job(job)

    # Publish final job done event
    files_summary = [
        {
            "original_name": e["original_name"],
            "output_filename": e["output_filename"],
            "download_url": f"/api/download/{e['output_filename']}" if e["output_filename"] else None,
            "status": e["status"],
            "error": e["error"],
        }
        for e in job["files"]
    ]

    await job_events.publish(job_id, "job_done", {
        "job_id": job_id,
        "status": "done",
        "total": job["total"],
        "completed": job["completed"],
        "failed": job["failed"],
        "files": files_summary,
    })


def process_batch(job_id: str) -> None:
    """
    Synchronous fallback wrapper for FastAPI BackgroundTasks.
    """
    asyncio.run(process_batch_job(job_id))


# ── Format helpers (mirrors routes/download.py) ───────────────────────────

SUPPORTED_FORMATS = {"png", "jpeg", "webp"}

MEDIA_TYPES = {
    "png":  "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}

ZIP_EXTENSIONS = {
    "png":  ".png",
    "jpeg": ".jpg",
    "webp": ".webp",
}


def _convert_image(src_path: str, fmt: str, quality: int) -> bytes:
    """
    Convert *src_path* (always a PNG with possible alpha) to *fmt* at *quality*.
    Returns raw bytes of the converted image.
    """
    from PIL import Image  # lazy import — not needed unless downloading

    with Image.open(src_path) as img:
        if fmt == "jpeg" and img.mode in ("RGBA", "LA", "P"):
            # JPEG cannot carry transparency — flatten onto white
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
            img = bg
        elif img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA" if fmt in ("png", "webp") else "RGB")

        buf = io.BytesIO()
        pil_fmt = "JPEG" if fmt == "jpeg" else fmt.upper()
        save_kwargs: dict = {"format": pil_fmt}
        if fmt in ("jpeg", "webp"):
            save_kwargs["quality"] = quality
        if fmt == "webp":
            save_kwargs["method"] = 6  # best compression ratio
        img.save(buf, **save_kwargs)
        return buf.getvalue()


def _clean_arcname(original_name: str, fmt: str) -> str:
    """
    Build a human-readable filename for the ZIP entry, e.g.
    ``photo.jpg``  → ``photo_no_bg.png``
    """
    stem = Path(original_name).stem
    return f"{stem}_no_bg{ZIP_EXTENSIONS[fmt]}"


def stream_zip(
    job: "Job",
    fmt: str = "png",
    quality: int = 90,
) -> Generator[bytes, None, None]:
    """
    Stream a ZIP archive of all successfully processed output files.

    Yields raw bytes chunks suitable for a ``StreamingResponse``.
    Each output PNG is optionally converted to *fmt* at *quality* before
    being packed — keeping per-file memory proportional to one image, not
    the whole batch.

    Args:
        job:     The job dict (already retrieved from MongoDB).
        fmt:     Target image format — ``png`` (default), ``jpeg``, or ``webp``.
        quality: JPEG/WebP quality 1–100 (ignored for PNG).

    Yields:
        bytes chunks of the ZIP stream.
    """
    fmt = fmt.lower().strip()
    if fmt not in SUPPORTED_FORMATS:
        fmt = "png"
    quality = max(1, min(100, quality))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for entry in job["files"]:
            if entry["status"] != "done" or not entry["output_filename"]:
                continue

            src_path = os.path.join(OUTPUT_DIR, entry["output_filename"])
            if not os.path.isfile(src_path):
                logger.warning(f"[Batch] ZIP: missing output file '{src_path}', skipping.")
                continue

            arcname = _clean_arcname(entry["original_name"], fmt)

            try:
                if fmt == "png":
                    # Fast path: source is already PNG — no conversion
                    zf.write(src_path, arcname=arcname)
                else:
                    img_bytes = _convert_image(src_path, fmt, quality)
                    zf.writestr(arcname, img_bytes)
            except Exception as exc:
                logger.error(
                    f"[Batch] ZIP: failed to pack '{entry['original_name']}': {exc}"
                )

    # Yield the complete in-memory buffer as a single chunk.
    # For future true streaming this generator boundary makes it easy to
    # switch to a chunked zip writer without changing the caller.
    buf.seek(0)
    while chunk := buf.read(64 * 1024):  # 64 KB chunks
        yield chunk


def build_zip(job: "Job") -> tuple[bytes, str]:
    """
    Backwards-compatible helper — returns ``(zip_bytes, zip_filename)``.

    New callers should prefer :func:`stream_zip` for memory efficiency.
    """
    data = b"".join(stream_zip(job, fmt="png"))
    zip_filename = f"batch_{job['job_id'][:8]}_results.zip"
    return data, zip_filename
