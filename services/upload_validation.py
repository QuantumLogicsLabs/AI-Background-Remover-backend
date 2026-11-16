"""Shared helpers for reading uploaded files safely."""

from __future__ import annotations

from fastapi import HTTPException, UploadFile, status

CHUNK_SIZE = 1024 * 1024  # read uploads 1 MB at a time
DEFAULT_MAX_MB = 10


async def read_upload_limited(file: UploadFile, max_mb: int = DEFAULT_MAX_MB) -> bytes:
    """
    Read an upload, but stop with HTTP 413 as soon as it grows past ``max_mb``.

    Unlike ``await file.read()`` followed by a size check, an oversized file is
    never fully loaded into memory.
    """
    limit = max_mb * 1024 * 1024
    data = bytearray()

    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > limit:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds {max_mb} MB limit.",
            )

    return bytes(data)