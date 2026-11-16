"""Tests for services/account_cleanup.py and DELETE /api/auth/account."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from services import account_cleanup
from tests.conftest import TEST_USER_ID, TEST_USER_PASS, get_memory_collection


def make_db(docs_by_collection=None, failing=()):
    """Fake get_collection: find() returns the given docs, delete_many reports their count."""
    docs_by_collection = docs_by_collection or {}
    cols: dict[str, MagicMock] = {}

    def get(name):
        if name in failing:
            raise RuntimeError("boom")
        if name not in cols:
            docs = docs_by_collection.get(name, [])
            cursor = MagicMock()
            cursor.to_list = AsyncMock(return_value=docs)
            col = MagicMock()
            col.find = MagicMock(return_value=cursor)
            col.delete_many = AsyncMock(return_value=MagicMock(deleted_count=len(docs)))
            cols[name] = col
        return cols[name]

    return get, cols


async def test_purges_every_user_collection():
    get, cols = make_db()
    with patch("services.account_cleanup.get_collection", side_effect=get):
        await account_cleanup.purge_user_data("u1")

    for name in account_cleanup.USER_COLLECTIONS:
        cols[name].delete_many.assert_awaited_once_with({"user_id": "u1"})


async def test_removes_output_files_named_in_history(tmp_path, monkeypatch):
    monkeypatch.setattr(account_cleanup, "OUTPUT_DIR", str(tmp_path))
    mine = [tmp_path / "a_result.png", tmp_path / "b_removed.png", tmp_path / "b_cropped.png"]
    someone_elses = tmp_path / "someone_elses.png"
    for path in [*mine, someone_elses]:
        path.write_bytes(b"x")

    get, _ = make_db({
        "history": [{"output_filename": "a_result.png"}],
        "smart_crop_history": [
            {"removed_filename": "b_removed.png", "cropped_filename": "b_cropped.png"}
        ],
    })
    with patch("services.account_cleanup.get_collection", side_effect=get):
        summary = await account_cleanup.purge_user_data("u1")

    assert all(not path.exists() for path in mine)
    assert someone_elses.exists()
    assert summary["files"] == 3
    assert summary["documents"] == 2


async def test_stored_filenames_cannot_escape_the_output_folder(tmp_path, monkeypatch):
    output = tmp_path / "out"
    output.mkdir()
    monkeypatch.setattr(account_cleanup, "OUTPUT_DIR", str(output))
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"x")

    get, _ = make_db({"history": [{"output_filename": "../outside.png"}]})
    with patch("services.account_cleanup.get_collection", side_effect=get):
        await account_cleanup.purge_user_data("u1")

    assert outside.exists()


async def test_one_failing_collection_does_not_stop_the_rest():
    get, cols = make_db(failing={"favorites"})
    with patch("services.account_cleanup.get_collection", side_effect=get):
        await account_cleanup.purge_user_data("u1")

    cols["history"].delete_many.assert_awaited_once()
    cols["conversations"].delete_many.assert_awaited_once()


async def test_deleting_an_account_removes_that_users_data_only(
    client: httpx.AsyncClient, auth_headers: dict
):
    history = get_memory_collection("history")
    await history.insert_one({"upload_id": "mine", "user_id": TEST_USER_ID})
    await history.insert_one({"upload_id": "theirs", "user_id": "someone-else"})

    with patch("services.account_cleanup.get_collection", side_effect=get_memory_collection):
        resp = await client.request(
            "DELETE", "/api/auth/account",
            headers=auth_headers, json={"password": TEST_USER_PASS},
        )

    assert resp.status_code == 204
    assert await history.find_one({"upload_id": "mine"}) is None
    assert await history.find_one({"upload_id": "theirs"}) is not None
    assert await get_memory_collection("users").find_one({"user_id": TEST_USER_ID}) is None