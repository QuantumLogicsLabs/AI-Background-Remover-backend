"""Oversized uploads are rejected by the AI image endpoints and chat."""
from __future__ import annotations

import io

import httpx
import pytest

# Starts like a PNG but is 11 MB: over the 10 MB limit.
BIG = b"\x89PNG" + b"\x00" * (11 * 1024 * 1024)


class TestAiUploadLimits:
    @pytest.mark.parametrize("path", [
        "/api/image/analyze",
        "/api/image/caption",
        "/api/image/captions",
        "/api/image/suggestions",
        "/api/image/analyze-advanced",
    ])
    async def test_image_endpoints_reject_oversized_files(
        self, client: httpx.AsyncClient, auth_headers: dict, path: str
    ):
        resp = await client.post(
            path,
            headers=auth_headers,
            files={"file": ("huge.png", io.BytesIO(BIG), "image/png")},
        )
        assert resp.status_code == 413

    async def test_batch_reports_an_oversized_file_as_an_error(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        resp = await client.post(
            "/api/image/analyze-advanced-batch",
            headers=auth_headers,
            files=[("files", ("huge.png", io.BytesIO(BIG), "image/png"))],
        )
        assert resp.status_code == 200
        item = resp.json()["results"][0]
        assert item["status"] == "error"
        assert "10 MB" in item["error"]

    async def test_chat_rejects_an_oversized_image(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        resp = await client.post(
            "/api/chat",
            headers=auth_headers,
            data={"message": "hello"},
            files={"file": ("huge.png", io.BytesIO(BIG), "image/png")},
        )
        assert resp.status_code == 413