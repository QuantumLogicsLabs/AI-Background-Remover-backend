"""Tests for GET /api/download/{filename}."""
from __future__ import annotations

import io
from unittest.mock import patch

import httpx
import pytest
from PIL import Image


@pytest.fixture
def output_dir(tmp_path):
    """A temp output folder with one transparent PNG in it."""
    Image.new("RGBA", (8, 8), (255, 0, 0, 128)).save(tmp_path / "photo_result.png", format="PNG")
    with patch("routes.download.OUTPUT_DIR", str(tmp_path)):
        yield tmp_path


class TestDownload:
    async def test_png_is_served_as_is(self, client: httpx.AsyncClient, output_dir):
        resp = await client.get("/api/download/photo_result.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.content == (output_dir / "photo_result.png").read_bytes()

    async def test_converts_to_jpeg(self, client: httpx.AsyncClient, output_dir):
        resp = await client.get("/api/download/photo_result.png", params={"format": "jpeg"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert "photo_result.jpg" in resp.headers["content-disposition"]
        assert Image.open(io.BytesIO(resp.content)).format == "JPEG"

    async def test_converts_to_webp(self, client: httpx.AsyncClient, output_dir):
        resp = await client.get("/api/download/photo_result.png", params={"format": "webp", "quality": 80})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/webp"
        assert Image.open(io.BytesIO(resp.content)).format == "WEBP"

    async def test_unsupported_format_is_rejected(self, client: httpx.AsyncClient, output_dir):
        resp = await client.get("/api/download/photo_result.png", params={"format": "gif"})
        assert resp.status_code == 400

    async def test_missing_file_is_404(self, client: httpx.AsyncClient, output_dir):
        resp = await client.get("/api/download/nope.png")
        assert resp.status_code == 404

    async def test_cannot_read_files_outside_the_output_folder(
        self, client: httpx.AsyncClient, tmp_path
    ):
        output = tmp_path / "out"
        output.mkdir()
        (tmp_path / "secret.png").write_bytes(b"secret")
        with patch("routes.download.OUTPUT_DIR", str(output)):
            resp = await client.get("/api/download/..%2Fsecret.png")
        assert resp.status_code == 404

    async def test_conversion_errors_do_not_leak_internal_details(
        self, client: httpx.AsyncClient, output_dir
    ):
        with patch("PIL.Image.open", side_effect=RuntimeError("secret-internal-detail")):
            resp = await client.get("/api/download/photo_result.png", params={"format": "jpeg"})
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Image conversion failed."
        assert "secret-internal-detail" not in resp.text