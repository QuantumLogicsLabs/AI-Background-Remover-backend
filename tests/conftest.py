"""
Shared pytest fixtures for backend API tests.

All external dependencies (MongoDB, AI inference, AI service, warm-up)
are mocked so tests are:
  - Fast: no model weights / GPU needed
  - Isolated: no real database mutations
  - Reliable: deterministic across CI environments
"""

from __future__ import annotations

import io
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
import httpx
from fastapi import FastAPI

# ── Ensure backend/ is on the Python path ──────────────────────────────────
_BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# ── Minimal 1×1 transparent PNG for upload fixtures ────────────────────────
MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)

# ── Stub user constants ──────────────────────────────────────────────────────
TEST_USER_ID    = str(uuid.uuid4())
TEST_USER_EMAIL = "testuser@example.com"
TEST_USER_NAME  = "Test User"
TEST_USER_PASS  = "Password123!"


def _make_user_doc():
    from services.auth import hash_password
    return {
        "user_id":         TEST_USER_ID,
        "name":            TEST_USER_NAME,
        "email":           TEST_USER_EMAIL,
        "hashed_password": hash_password(TEST_USER_PASS),
        "created_at":      datetime.now(timezone.utc),
        "email_verified":  True,
    }


def _make_access_token():
    from services.auth import create_access_token
    return create_access_token({"sub": TEST_USER_ID, "email": TEST_USER_EMAIL})


# ── In-memory async MongoDB collection mock ─────────────────────────────────

class MemoryCollection:
    """Simple in-memory async collection that quacks like Motor."""

    def __init__(self):
        self._docs: list[dict] = []

    async def find_one(self, query: dict, projection=None) -> dict | None:
        for doc in self._docs:
            if all(doc.get(k) == v for k, v in query.items()):
                result = dict(doc)
                if projection and "_id" in projection and projection["_id"] == 0:
                    result.pop("_id", None)
                return result
        return None

    async def insert_one(self, document: dict):
        self._docs.append(dict(document))
        mock = MagicMock()
        mock.inserted_id = "fake_id"
        return mock

    async def replace_one(self, query: dict, replacement: dict, upsert: bool = False):
        for i, doc in enumerate(self._docs):
            if all(doc.get(k) == v for k, v in query.items()):
                self._docs[i] = dict(replacement)
                return MagicMock(matched_count=1)
        if upsert:
            self._docs.append(dict(replacement))
        return MagicMock(matched_count=0)

    async def update_one(self, query: dict, update: dict, upsert: bool = False):
        for doc in self._docs:
            if all(doc.get(k) == v for k, v in query.items()):
                if "$set" in update:
                    doc.update(update["$set"])
                if "$inc" in update:
                    for field, val in update["$inc"].items():
                        doc[field] = doc.get(field, 0) + val
                return MagicMock(matched_count=1)
        if upsert:
            new_doc = {}
            if "$set" in update:
                new_doc.update(update["$set"])
            if "$inc" in update:
                for field, val in update["$inc"].items():
                    new_doc[field] = val
            self._docs.append(new_doc)
        return MagicMock(matched_count=0)

    async def find_one_and_update(
        self,
        query: dict,
        update: dict,
        upsert: bool = False,
        return_document: bool = False,
        projection: dict = None,
    ):
        matched_idx = -1
        for i, doc in enumerate(self._docs):
            if all(doc.get(k) == v for k, v in query.items()):
                matched_idx = i
                break

        old_doc = None
        if matched_idx >= 0:
            old_doc = dict(self._docs[matched_idx])
            target_doc = self._docs[matched_idx]
        elif upsert:
            target_doc = dict(query)
            if "$setOnInsert" in update:
                target_doc.update(update["$setOnInsert"])
            self._docs.append(target_doc)
        else:
            return None

        if "$set" in update:
            target_doc.update(update["$set"])
        if "$inc" in update:
            for field, val in update["$inc"].items():
                target_doc[field] = target_doc.get(field, 0) + val

        result_doc = target_doc if return_document else old_doc
        if result_doc is not None:
            result_doc = dict(result_doc)
            if projection and "_id" in projection and projection["_id"] == 0:
                result_doc.pop("_id", None)
        return result_doc

    async def delete_one(self, query: dict):
        for i, doc in enumerate(self._docs):
            if all(doc.get(k) == v for k, v in query.items()):
                del self._docs[i]
                return MagicMock(deleted_count=1)
        return MagicMock(deleted_count=0)

    async def delete_many(self, query: dict):
        initial = len(self._docs)
        self._docs = [doc for doc in self._docs if not all(doc.get(k) == v for k, v in query.items())]
        return MagicMock(deleted_count=initial - len(self._docs))

    async def insert_many(self, documents: list[dict]):
        for d in documents:
            self._docs.append(dict(d))
        mock = MagicMock()
        mock.inserted_ids = ["fake_id" for _ in documents]
        return mock

    async def create_index(self, *args, **kwargs):
        return "mock_index"

    async def create_indexes(self, *args, **kwargs):
        return ["mock_index"]

    def find(self, query=None, projection=None):
        results = []
        for doc in self._docs:
            if query is None or all(doc.get(k) == v for k, v in query.items()):
                result = dict(doc)
                if projection and "_id" in projection and projection["_id"] == 0:
                    result.pop("_id", None)
                results.append(result)
        mock_cursor = MagicMock()
        mock_cursor.sort = lambda *a, **kw: mock_cursor
        mock_cursor.limit = lambda *a, **kw: mock_cursor
        mock_cursor.skip = lambda *a, **kw: mock_cursor
        mock_cursor.__aiter__ = lambda self: aiter(results)
        mock_cursor.to_list = AsyncMock(return_value=results)
        return mock_cursor

    async def count_documents(self, query: dict) -> int:
        return sum(1 for doc in self._docs if all(doc.get(k) == v for k, v in query.items()))


async def aiter(items):
    for item in items:
        yield item


# ── Fixtures ─────────────────────────────────────────────────────────────────

_db_store: dict[str, MemoryCollection] = {}


def get_memory_collection(name: str) -> MemoryCollection:
    if name not in _db_store:
        _db_store[name] = MemoryCollection()
    return _db_store[name]


@pytest.fixture(autouse=True)
def clear_db():
    """Reset the in-memory DB before each test."""
    _db_store.clear()
    yield
    _db_store.clear()


@pytest_asyncio.fixture
async def app() -> FastAPI:
    """Build FastAPI test app with all mocks wired in."""

    # Patch DB connection checks so they always pass
    with patch("services.database.is_db_connected", return_value=True), \
         patch("services.database.get_collection", side_effect=get_memory_collection), \
         patch("services.bg_removal.warm_up", new_callable=AsyncMock), \
         patch("services.cleanup.start_cleanup_task"), \
         patch("services.cleanup.stop_cleanup_task"), \
         patch("services.job_queue.job_queue.start", new_callable=AsyncMock), \
         patch("services.job_queue.job_queue.stop", new_callable=AsyncMock):

        # Pre-seed a test user
        users_col = get_memory_collection("users")
        await users_col.insert_one(_make_user_doc())

        import app as app_module
        yield app_module.app


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Async HTTP test client."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Bearer token headers for a pre-registered test user."""
    token = _make_access_token()
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def png_file() -> bytes:
    """Minimal valid PNG bytes for upload tests."""
    return MINIMAL_PNG

# ── Email verification helpers ───────────────────────────────────────────────

@pytest.fixture(autouse=True)
def sent_codes():
    """
    Replace real email sending in EVERY test and record what would have been
    sent as (to_email, code) pairs. This also guarantees tests never send real
    emails, even if backend/.env contains SMTP credentials.
    """
    sent: list[tuple[str, str]] = []

    async def fake_send(to_email: str, code: str) -> None:
        sent.append((to_email, code))

    with patch("services.verification.send_verification_code_email", new=fake_send):
        yield sent


def code_sent_to(sent: list[tuple[str, str]], email: str) -> str:
    """Latest verification code emailed to `email` during this test."""
    for to_email, code in reversed(sent):
        if to_email == email:
            return code
    raise AssertionError(f"No verification code was sent to {email}")