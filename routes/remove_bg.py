from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse, Response
from services.bg_removal  import remove_background_bytes, QUALITY_OPTIONS
from services.database    import get_collection
from services.auth        import get_current_user
from services.quota       import check_and_increment_quota
from services.storage     import save_file
from models.user          import UserOut
import aiofiles
import os
import uuid
from datetime import datetime, timezone

router = APIRouter(tags=["Background Removal"])

# Output dir still needed — we write the result PNG so storage/download can serve it
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_MB   = 10


@router.post("/remove-background")
async def remove_bg_endpoint(
    file:         UploadFile = File(...),
    quality:      str        = Form("fast"),
    current_user: UserOut    = Depends(get_current_user),
):
    """
    Remove the background from an uploaded image.

    - **file**    JPEG, PNG, or WebP image (≤ 10 MB)
    - **quality** `fast` (default) — ISNet, quick results
                  `standard`       — U²-Net portrait model, best for people & faces
                  `quality`        — BiRefNet, superior edge detail (hair, fur, complex subjects)

    Speed note: the source image bytes are passed directly to the AI model
    in memory — no temporary upload file is written to disk, saving one
    extra I/O round-trip on every request.
    """
    await check_and_increment_quota(current_user.user_id)

    if quality not in QUALITY_OPTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"quality must be one of: {', '.join(QUALITY_OPTIONS)}.",
        )
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use JPEG, PNG, or WebP.")

    # Read the upload once into memory
    contents = await file.read()
    if len(contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_SIZE_MB} MB limit.")

    upload_id   = str(uuid.uuid4())
    safe_name   = os.path.basename(file.filename or "upload")

    # ── AI inference (fully in-memory — no temp file for the source) ────────
    try:
        result_bytes = await remove_background_bytes(contents, quality=quality)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    # ── Persist the result PNG so storage/download can serve it ────────────
    output_filename = f"{upload_id}_result.png"
    output_path     = os.path.join(OUTPUT_DIR, output_filename)

    async with aiofiles.open(output_path, "wb") as f:
        await f.write(result_bytes)

    download_url = await save_file(output_path, output_filename)

    # ── Save history record (best-effort) ───────────────────────────────────
    try:
        collection = get_collection("history")
        await collection.insert_one({
            "upload_id":       upload_id,
            "user_id":         current_user.user_id,
            "original_name":   safe_name,
            "output_filename": output_filename,
            "download_url":    download_url,
            "quality":         quality,
            "created_at":      datetime.now(timezone.utc),
        })
    except Exception:
        pass

    return JSONResponse({
        "output_filename": output_filename,
        "download_url":    download_url,
        "quality":         quality,
    })
