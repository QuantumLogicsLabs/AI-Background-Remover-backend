"""Tests for DELETE /api/image/{image_id} across all history collections."""
from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from tests.conftest import TEST_USER_ID, get_memory_collection


@pytest.fixture
def output_dir(tmp_path):
    """Point the route at a temp folder and the in-memory database."""
    with patch("routes.images.OUTPUT_DIR", str(tmp_path)), \
         patch("routes.images.get_collection", side_effect=get_memory_collection):
        yield tmp_path


class TestDeleteImage:
    @pytest.mark.parametrize("collection, id_field, suffixes", [
        ("history",            "upload_id", ["_result.png"]),
        ("enhance_history",    "upload_id", ["_enhanced.png"]),
        ("replace_bg_history", "result_id", ["_composited.png"]),
        ("smart_crop_history", "upload_id", ["_removed.png", "_cropped.png"]),
        ("recolor_history",    "upload_id", ["_recolored.png"]),
    ])
    async def test_deletes_record_and_files(
        self, client: httpx.AsyncClient, auth_headers: dict, output_dir,
        collection: str, id_field: str, suffixes: list[str],
    ):
        image_id = "img-1"
        files = []
        for suffix in suffixes:
            path = output_dir / f"{image_id}{suffix}"
            path.write_bytes(b"png")
            files.append(path)
        col = get_memory_collection(collection)
        await col.insert_one({id_field: image_id, "user_id": TEST_USER_ID})

        resp = await client.delete(f"/api/image/{image_id}", headers=auth_headers)

        assert resp.status_code == 200
        assert resp.json()["found"] is True
        assert all(not p.exists() for p in files)
        assert await col.find_one({id_field: image_id}) is None

    async def test_other_users_image_is_forbidden(
        self, client: httpx.AsyncClient, auth_headers: dict, output_dir
    ):
        path = output_dir / "img-2_recolored.png"
        path.write_bytes(b"png")
        col = get_memory_collection("recolor_history")
        await col.insert_one({"upload_id": "img-2", "user_id": "someone-else"})

        resp = await client.delete("/api/image/img-2", headers=auth_headers)

        assert resp.status_code == 403
        assert path.exists()
        assert await col.find_one({"upload_id": "img-2"}) is not None

    async def test_unknown_image_is_an_idempotent_success(
        self, client: httpx.AsyncClient, auth_headers: dict, output_dir
    ):
        resp = await client.delete("/api/image/does-not-exist", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["found"] is False

    async def test_requires_login(self, client: httpx.AsyncClient, output_dir):
        resp = await client.delete("/api/image/img-1")
        assert resp.status_code == 401