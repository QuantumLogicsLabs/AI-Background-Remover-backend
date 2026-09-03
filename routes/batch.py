"""
Batch Processing Routes.

POST /api/batch/start
─────────────────────
Accepts up to 20 image files, saves them, creates a job record in MongoDB,
enqueues the job into the async worker pool, and returns the job_id immediately.

GET /api/batch/{job_id}/events
──────────────────────────────
Server-Sent Events (SSE) stream yielding real-time progress events:
  - snapshot: initial job state
  - job_started, file_processing, file_done, file_error, job_done

GET /api/batch/{job_id}/status
───────────────────────────────
Returns the full job state (per-file statuses, counts, overall status).
Fallback polling endpoint (every 1–2 seconds).

GET /api/batch/{job_id}/download-zip
─────────────────────────────────────
Streams a ZIP of all successfully processed output files, optionally converted
to a target format (PNG/JPEG/WebP) with configurable quality.
Only available when job status is "done".

Query params:
  ?format=png|jpeg|webp   (default: png)
  ?quality=1-100          (default: 90, ignored for PNG)
  ?token=<jwt>            (alternative to Authorization header — required
                           for browser fetch/anchor-tag downloads)

GET /api/batch/{job_id}/download   (legacy alias → redirects to download-zip)
"""

import os
import uuid

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Query
from fastapi.responses import Response, JSONResponse, StreamingResponse, RedirectResponse

from fastapi import Form
from models.user    import UserOut
from services.auth  import get_current_user, decode_token
from services.quota import check_and_increment_quota, refund_quota
from services.batch import create_job, get_job, stream_zip
from services.job_queue import job_queue
from services.job_events import job_events
from services.bg_removal import QUALITY_OPTIONS

router = APIRouter(tags=["Batch Processing"])

UPLOAD_DIR    = "uploads"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_MB   = 10
MAX_FILES     = 100

os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/batch/start")
async def batch_start(
    files:            list[UploadFile] = File(...),
    quality:          str              = Form("fast"),
    current_user:     UserOut          = Depends(get_current_user),
):
    """
    Start a batch background-removal job.

    - **files**   Up to 20 JPEG/PNG/WebP images (each ≤ 10 MB)
    - **quality** `fast` (default) | `standard` | `quality` — AI model to use

    Returns a **job_id** immediately. Connect to `/api/batch/{job_id}/events`
    or poll `/api/batch/{job_id}/status` for progress.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_FILES} files per batch.",
        )
    if quality not in QUALITY_OPTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"quality must be one of: {', '.join(QUALITY_OPTIONS)}.",
        )

    # ── Phase 1: validate and read every file before touching quota ──────
    validated: list[tuple[bytes, str]] = []

    for upload in files:
        if upload.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File '{upload.filename}' has unsupported type '{upload.content_type}'. "
                       "Use JPEG, PNG, or WebP.",
            )

        contents = await upload.read()
        if len(contents) > MAX_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File '{upload.filename}' exceeds {MAX_SIZE_MB} MB limit.",
            )

        safe_name = os.path.basename(upload.filename or "upload")
        validated.append((contents, safe_name))

    # ── Phase 2: charge quota for every validated file ───────────────────
    charged = 0
    try:
        for _ in validated:
            await check_and_increment_quota(current_user.user_id)
            charged += 1
    except Exception:
        if charged:
            await refund_quota(current_user.user_id, charged)
        raise

    # ── Phase 3: write files to disk ─────────────────────────────────────
    file_entries: list[dict] = []

    try:
        for contents, safe_name in validated:
            file_id     = str(uuid.uuid4())
            upload_path = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_name}")

            async with aiofiles.open(upload_path, "wb") as f:
                await f.write(contents)

            file_entries.append({
                "original_name": safe_name,
                "upload_path":   upload_path,
            })
    except Exception:
        await refund_quota(current_user.user_id, charged)
        raise

    job_id = await create_job(file_entries, user_id=current_user.user_id, quality=quality)

    # Offload to bounded async job queue worker pool
    await job_queue.enqueue(job_id)

    return JSONResponse({
        "job_id":      job_id,
        "total_files": len(file_entries),
        "quality":     quality,
        "status":      "pending",
    })


@router.get("/batch/{job_id}/events")
async def batch_events(
    job_id: str,
    token:  str | None = Query(None, description="Optional auth token for SSE EventSource"),
    current_user: UserOut = Depends(get_current_user),
):
    """
    Stream real-time progress events for a batch job via Server-Sent Events (SSE).
    """
    job = await get_job(job_id, user_id=current_user.user_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    snapshot = {
        "job_id": job["job_id"],
        "status": job["status"],
        "quality": job.get("quality", "fast"),
        "total": job["total"],
        "completed": job["completed"],
        "failed": job["failed"],
        "files": [
            {
                "original_name": e["original_name"],
                "output_filename": e["output_filename"],
                "download_url": f"/api/download/{e['output_filename']}" if e["output_filename"] else None,
                "status": e["status"],
                "error": e["error"],
            }
            for e in job["files"]
        ],
    }

    # If job is already done, return single snapshot event stream
    return StreamingResponse(
        job_events.event_generator(job_id, initial_data=snapshot),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/batch/{job_id}/status")
async def batch_status(
    job_id:       str,
    current_user: UserOut = Depends(get_current_user),
):
    """
    Get the current status of a batch job (fallback polling).

    Returns per-file statuses and overall progress counts.
    Users can only access their own jobs.
    """
    job = await get_job(job_id, user_id=current_user.user_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    files_out = [
        {
            "original_name":   entry["original_name"],
            "output_filename": entry["output_filename"],
            "download_url":    f"/api/download/{entry['output_filename']}"
                               if entry["output_filename"] else None,
            "status":          entry["status"],
            "error":           entry["error"],
        }
        for entry in job["files"]
    ]

    return JSONResponse({
        "job_id":     job["job_id"],
        "status":     job["status"],
        "quality":    job.get("quality", "fast"),
        "created_at": job["created_at"],
        "total":      job["total"],
        "completed":  job["completed"],
        "failed":     job["failed"],
        "files":      files_out,
    })


@router.get("/batch/{job_id}/download-zip")
async def batch_download_zip(
    job_id:  str,
    format:  str = Query(default="png",  description="Output format: png, jpeg, or webp"),
    quality: int = Query(default=90,     description="Quality 1–100 (JPEG/WebP; ignored for PNG)"),
    token:   str | None = Query(None,   description="JWT token (alternative to Authorization header)"),
    current_user: UserOut = Depends(get_current_user),
):
    """
    Stream all successfully processed images as a single ZIP archive,
    optionally converting each PNG to the requested *format* and *quality*.

    - **format** ``png`` (default, lossless, transparency preserved) |
                 ``jpeg`` (smaller, white background) |
                 ``webp`` (best compression, transparency preserved)
    - **quality** Compression quality 1–100 for JPEG/WebP (default 90).
      Silently ignored for PNG.
    - **token**   JWT bearer token as a query param — lets a browser
      ``fetch()`` call or ``<a download>`` tag authenticate without custom
      headers.

    Only available once the job **status** is ``done``.
    Users can only download their own jobs.
    """
    fmt = format.lower().strip()
    if fmt not in {"png", "jpeg", "webp"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{format}'. Choose from: jpeg, png, webp.",
        )
    quality = max(1, min(100, quality))

    job = await get_job(job_id, user_id=current_user.user_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Job is still '{job['status']}'. Wait until status is 'done'.",
        )

    completed_count = sum(1 for f in job["files"] if f["status"] == "done")
    if completed_count == 0:
        raise HTTPException(
            status_code=404,
            detail="No completed files available for download.",
        )

    ext_map  = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}
    zip_name = f"batch_{job['job_id'][:8]}_results_{fmt}{ext_map[fmt]}.zip"

    return StreamingResponse(
        stream_zip(job, fmt=fmt, quality=quality),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{zip_name}"',
            "X-Zip-Format":        fmt,
            "X-Zip-Files":         str(completed_count),
        },
    )


@router.get("/batch/{job_id}/download")
async def batch_download_legacy(
    job_id:       str,
    current_user: UserOut = Depends(get_current_user),
):
    """
    Legacy download alias — redirects to ``/download-zip`` (PNG, quality 90).

    Kept for backwards compatibility with older clients.
    """
    job = await get_job(job_id, user_id=current_user.user_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Job is still '{job['status']}'. Wait until status is 'done'.",
        )

    completed_count = sum(1 for f in job["files"] if f["status"] == "done")
    if completed_count == 0:
        raise HTTPException(
            status_code=404,
            detail="No completed files available for download.",
        )

    zip_name = f"batch_{job['job_id'][:8]}_results.zip"
    return StreamingResponse(
        stream_zip(job, fmt="png"),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{zip_name}"',
        },
    )
