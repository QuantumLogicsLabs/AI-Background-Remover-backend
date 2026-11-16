"""Tests for services/upload_validation.py."""
from __future__ import annotations

import io

import pytest
from fastapi import HTTPException, UploadFile

from services.upload_validation import CHUNK_SIZE, read_upload_limited

MB = 1024 * 1024


def make_upload(data: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename="photo.png")


async def test_returns_the_whole_file_when_under_the_limit():
    assert await read_upload_limited(make_upload(b"abc"), max_mb=1) == b"abc"


async def test_empty_file_returns_empty_bytes():
    assert await read_upload_limited(make_upload(b""), max_mb=1) == b""


async def test_a_file_exactly_at_the_limit_is_accepted():
    data = b"x" * MB
    assert await read_upload_limited(make_upload(data), max_mb=1) == data


async def test_one_byte_over_the_limit_is_rejected_with_413():
    with pytest.raises(HTTPException) as exc:
        await read_upload_limited(make_upload(b"x" * (MB + 1)), max_mb=1)
    assert exc.value.status_code == 413
    assert "1 MB" in exc.value.detail


async def test_stops_reading_soon_after_the_limit():
    class CountingFile(io.BytesIO):
        total_read = 0

        def read(self, size=-1):
            chunk = super().read(size)
            CountingFile.total_read += len(chunk)
            return chunk

    upload = UploadFile(file=CountingFile(b"x" * (5 * MB)), filename="huge.png")
    with pytest.raises(HTTPException):
        await read_upload_limited(upload, max_mb=1)

    # limit + at most one extra chunk, never the whole 5 MB
    assert CountingFile.total_read <= MB + CHUNK_SIZE