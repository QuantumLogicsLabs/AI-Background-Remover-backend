"""
Background removal service.

Bridges the FastAPI route layer with the AI inference pipeline.
"""

import sys
import asyncio
from pathlib import Path

# Add the AI module directory to sys.path so that `inference`, `preprocessing`,
# and `postprocessing` can be imported as top-level modules.
_AI_DIR = Path(__file__).resolve().parents[2] / "AI"
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

from inference import run_inference, run_inference_bytes, warm_up_models  # noqa: E402

# Valid quality values accepted by the API
QUALITY_OPTIONS = ("fast", "standard", "quality")


async def remove_background(
    input_path: str,
    output_path: str,
    quality: str = "fast",
) -> None:
    """
    Asynchronous wrapper — file-based interface.

    Runs `run_inference` in a thread-pool executor so it does not block
    FastAPI's event loop.

    Args:
        input_path:  Absolute or relative path to the source image.
        output_path: Destination path for the transparent PNG result.
        quality:     "fast" (isnet-general-use) or "quality" (BiRefNet).
    """
    if quality not in QUALITY_OPTIONS:
        quality = "fast"

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        run_inference,
        input_path,
        output_path,
        quality,
    )


async def remove_background_bytes(
    image_bytes: bytes,
    quality: str = "fast",
) -> bytes:
    """
    Asynchronous wrapper — fully in-memory interface (faster, no temp file).

    Passes raw image bytes directly to the rembg backend and returns the
    resulting transparent PNG as bytes — no disk I/O for the source image.

    Args:
        image_bytes: Raw JPEG / PNG / WebP bytes from the HTTP request.
        quality:     "fast" or "quality".

    Returns:
        Transparent PNG as raw bytes.
    """
    if quality not in QUALITY_OPTIONS:
        quality = "fast"

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        run_inference_bytes,
        image_bytes,
        quality,
    )


async def warm_up() -> None:
    """
    Pre-load all model sessions in a background thread so the first real
    request does not block waiting for model download / initialisation.

    Called once from the FastAPI lifespan startup hook.
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, warm_up_models)
