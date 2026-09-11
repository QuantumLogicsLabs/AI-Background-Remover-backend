import os, uuid, asyncio
from datetime import datetime, timezone
import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends
from fastapi.responses import JSONResponse
from services.database   import get_collection
from services.bg_removal import remove_background, QUALITY_OPTIONS
from services.smart_crop import smart_crop, get_aspect_ratio_keys
from services.auth       import get_current_user
from services.storage    import save_file
from services.tracking   import track_usage, track_action
from models.user         import UserOut

router = APIRouter(tags=["Smart Crop"])

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_MB   = 10


@router.post("/smart-crop")
async def smart_crop_endpoint(
    file:         UploadFile = File(...),
    padding_pct:  float      = Form(0.05),
    aspect_ratio: str        = Form("free"),
    min_size:     int        = Form(64),
    quality:      str        = Form("fast"),
    current_user: UserOut    = Depends(get_current_user),
):
    """
    Smart-crop an image around its subject.

    - **quality** `fast` (U2Net) or `quality` (BiRefNet) — controls the
      internal background-removal step used to detect the subject bbox.
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{file.content_type}'. Allowed: {ALLOWED_TYPES}")

    contents = await file.read()
    if len(contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_SIZE_MB} MB limit.")

    valid_ratios = get_aspect_ratio_keys()
    if aspect_ratio not in valid_ratios:
        raise HTTPException(status_code=400, detail=f"Invalid aspect_ratio '{aspect_ratio}'. Choose from: {valid_ratios}")
    if quality not in QUALITY_OPTIONS:
        raise HTTPException(status_code=400, detail=f"quality must be one of: {', '.join(QUALITY_OPTIONS)}.")

    padding_pct = max(0.0, min(0.5, padding_pct))
    min_size    = max(16, min(2048, min_size))

    upload_id   = str(uuid.uuid4())
    safe_name   = os.path.basename(file.filename or "upload")
    upload_path = os.path.join(UPLOAD_DIR, f"{upload_id}_{safe_name}")
    async with aiofiles.open(upload_path, "wb") as f:
        await f.write(contents)

    removed_filename = f"{upload_id}_removed.png"
    removed_path     = os.path.join(OUTPUT_DIR, removed_filename)
    try:
        await remove_background(upload_path, removed_path, quality=quality)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {exc}")

    cropped_filename = f"{upload_id}_cropped.png"
    cropped_path     = os.path.join(OUTPUT_DIR, cropped_filename)
    try:
        loop = asyncio.get_event_loop()
        crop_meta = await loop.run_in_executor(
            None,
            lambda: smart_crop(
                fg_path=removed_path, output_path=cropped_path,
                original_path=upload_path, padding_pct=padding_pct,
                aspect_ratio=aspect_ratio, min_size=min_size,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Smartcrop failed: {exc}")

    removed_url = await save_file(removed_path, removed_filename)
    cropped_url = await save_file(cropped_path, cropped_filename)

    try:
        collection = get_collection("smart_crop_history")
        await collection.insert_one({
            "upload_id": upload_id, "user_id": current_user.user_id,
            "original_name": safe_name,
            "removed_filename": removed_filename, "cropped_filename": cropped_filename,
            "settings": {
                "padding_pct": padding_pct, "aspect_ratio": aspect_ratio,
                "min_size": min_size, "quality": quality,
            },
            "crop_meta": crop_meta, "created_at": datetime.now(timezone.utc),
        })
    except Exception:
        pass

    await track_usage(
        user_id=current_user.user_id, feature="smart_crop", image_id=upload_id,
        metadata={"aspect_ratio": aspect_ratio, "quality": quality},
    )
    await track_action(
        user_id=current_user.user_id, image_id=upload_id, action_type="smart_crop",
        suggestion=f"Smart-cropped to aspect ratio '{aspect_ratio}'",
        applied=True,
    )

    return JSONResponse({
        "upload_id": upload_id,
        "removed_filename": removed_filename, "cropped_filename": cropped_filename,
        "removed_url": removed_url, "cropped_url": cropped_url,
        "crop_meta": crop_meta,
    })
