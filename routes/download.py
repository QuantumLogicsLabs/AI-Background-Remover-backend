"""
Download Route — supports on-the-fly format conversion and quality control.

GET /api/download/{filename}
  ?format=png|jpeg|webp   (default: png)
  ?quality=1-100          (default: 90, ignored for PNG which is lossless)

Returns the processed image converted to the requested format and quality.
Pillow handles in-memory conversion so no extra files are written to disk.
"""

import io
import logging
import os

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, FileResponse

router = APIRouter(tags=["Download"])

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output"

SUPPORTED_FORMATS = {"png", "jpeg", "webp"}

MEDIA_TYPES = {
    "png":  "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}

EXTENSIONS = {
    "png":  ".png",
    "jpeg": ".jpg",
    "webp": ".webp",
}


@router.get("/download/{filename}")
async def download_image(
    filename: str,
    format:  str = Query(default="png",  description="Output format: png, jpeg, or webp"),
    quality: int = Query(default=90,     description="Quality 1–100 (JPEG/WebP only; PNG is lossless)"),
):
    """
    Download a previously processed image, optionally converting it to a
    different format and/or adjusting compression quality.

    - **filename**: Name of the stored output file (e.g. ``uuid_result.png``)
    - **format**: Target format — ``png``, ``jpeg``, or ``webp`` (default: ``png``)
    - **quality**: Compression quality 1–100 for JPEG/WebP (default: 90).
      Silently ignored for PNG (lossless).
    """
    # ── Validate format ────────────────────────────────────────────────────
    fmt = format.lower().strip()
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{format}'. Choose from: {', '.join(sorted(SUPPORTED_FORMATS))}.",
        )

    # ── Clamp quality ──────────────────────────────────────────────────────
    quality = max(1, min(100, quality))

    # ── Resolve file path (prevent path traversal) ─────────────────────────
    safe_filename = os.path.basename(filename)
    file_path     = os.path.join(OUTPUT_DIR, safe_filename)

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found.")

    # ── Fast path: no conversion needed ───────────────────────────────────
    if fmt == "png":
        # PNG is always the stored format; stream directly without Pillow overhead
        return FileResponse(
            file_path,
            media_type=MEDIA_TYPES["png"],
            filename=_output_filename(safe_filename, "png"),
        )

    # ── Convert with Pillow ────────────────────────────────────────────────
    try:
        from PIL import Image

        with Image.open(file_path) as img:
            # JPEG does not support alpha — flatten onto white background
            if fmt == "jpeg" and img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                img = background
            elif img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

            buf = io.BytesIO()
            pil_fmt = "JPEG" if fmt == "jpeg" else fmt.upper()
            save_kwargs: dict = {"format": pil_fmt}
            if fmt in ("jpeg", "webp"):
                save_kwargs["quality"] = quality
            if fmt == "webp":
                save_kwargs["method"] = 6   # best compression ratio

            img.save(buf, **save_kwargs)
            buf.seek(0)

    except Exception:
        # Keep the details in the server log; do not leak internals to the client.
        logger.exception("Image conversion failed for %s (format=%s)", safe_filename, fmt)
        raise HTTPException(status_code=500, detail="Image conversion failed.")

    out_filename = _output_filename(safe_filename, fmt)

    return Response(
        content=buf.read(),
        media_type=MEDIA_TYPES[fmt],
        headers={
            "Content-Disposition": f'attachment; filename="{out_filename}"',
        },
    )


# ── Helpers ────────────────────────────────────────────────────────────────

def _output_filename(original: str, fmt: str) -> str:
    """Replace the extension of *original* with the target format's extension."""
    stem, _ = os.path.splitext(original)
    return f"{stem}{EXTENSIONS[fmt]}"