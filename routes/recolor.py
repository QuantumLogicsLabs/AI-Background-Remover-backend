"""
Magic Recolor Route.

POST /api/recolor
─────────────────
Accepts a source image and a stroke-mask PNG (painted by the user on the
frontend canvas), plus a target hex colour.  The mask encodes which pixels
the user painted over; white = repaint, black = leave alone.

Returns the recoloured image as a PNG saved to the output directory,
together with a download URL and a history record.
"""

import os
import uuid
import asyncio
from datetime import datetime, timezone

from fastapi              import APIRouter, File, Form, HTTPException, UploadFile, Depends
from fastapi.responses    import JSONResponse
import aiofiles

from services.auth        import get_current_user
from services.quota       import check_and_increment_quota
from services.storage     import save_file
from services.database    import get_collection
from services.recolor     import recolor_region
from models.user          import UserOut

router = APIRouter(tags=["Magic Recolor"])

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_MB   = 10


@router.post("/recolor")
async def recolor_endpoint(
    image:        UploadFile = File(...,  description="Source image (JPEG/PNG/WebP, ≤10 MB)"),
    mask:         UploadFile = File(...,  description="Stroke mask PNG — white = painted region"),
    target_color: str        = Form(...,  description="Target hex colour, e.g. #e83c6d"),
    strength:     float      = Form(1.0, description="Recolour intensity 0.0–1.0"),
    feather:      int        = Form(15,  description="Mask edge blur radius in pixels"),
    current_user: UserOut    = Depends(get_current_user),
):
    """
    Recolour a painted region of an image.

    - **image**        JPEG, PNG, or WebP (≤ 10 MB)
    - **mask**         Grayscale PNG where white marks strokes painted by the user
    - **target_color** CSS hex colour to shift the painted area to (e.g. `#e83c6d`)
    - **strength**     0.0 = no change · 1.0 = full recolour (default)
    - **feather**      Gaussian blur applied to mask edges for soft blending (default 15 px)
    """
    await check_and_increment_quota(current_user.user_id)

    # ── Validate source image ─────────────────────────────────────────────
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use JPEG, PNG, or WebP.",
        )

    image_bytes = await image.read()
    if len(image_bytes) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Source image exceeds {MAX_SIZE_MB} MB limit.",
        )

    # ── Validate mask ─────────────────────────────────────────────────────
    mask_bytes = await mask.read()
    if not mask_bytes:
        raise HTTPException(status_code=400, detail="Mask is empty.")

    # ── Validate target colour ────────────────────────────────────────────
    target_color = target_color.strip()
    if not target_color.startswith("#") or len(target_color.lstrip("#")) not in (3, 6):
        raise HTTPException(
            status_code=400,
            detail="target_color must be a CSS hex string, e.g. #e83c6d or #f00.",
        )

    # ── Clamp numeric params ──────────────────────────────────────────────
    strength = max(0.0, min(1.0, strength))
    feather  = max(0,   min(60, feather))

    # ── Run CPU-bound recolor in thread executor ──────────────────────────
    try:
        loop = asyncio.get_event_loop()
        result_bytes = await loop.run_in_executor(
            None,
            lambda: recolor_region(
                image_bytes=image_bytes,
                mask_bytes=mask_bytes,
                target_hex=target_color,
                strength=strength,
                feather=feather,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recolor failed: {exc}")

    # ── Persist result PNG ────────────────────────────────────────────────
    upload_id       = str(uuid.uuid4())
    safe_name       = os.path.basename(image.filename or "upload")
    output_filename = f"{upload_id}_recolored.png"
    output_path     = os.path.join(OUTPUT_DIR, output_filename)

    async with aiofiles.open(output_path, "wb") as f:
        await f.write(result_bytes)

    download_url = await save_file(output_path, output_filename)

    # ── Save history record ───────────────────────────────────────────────
    try:
        collection = get_collection("recolor_history")
        await collection.insert_one({
            "upload_id":       upload_id,
            "user_id":         current_user.user_id,
            "original_name":   safe_name,
            "output_filename": output_filename,
            "download_url":    download_url,
            "settings": {
                "target_color": target_color,
                "strength":     strength,
                "feather":      feather,
            },
            "created_at": datetime.now(timezone.utc),
        })
    except Exception:
        pass  # history is best-effort

    return JSONResponse({
        "upload_id":       upload_id,
        "output_filename": output_filename,
        "download_url":    download_url,
    })
