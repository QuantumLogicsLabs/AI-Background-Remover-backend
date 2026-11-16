"""Tests for GET /api/health."""
from __future__ import annotations

from unittest.mock import patch

import httpx


class TestHealth:
    async def test_ok_when_the_database_is_connected(self, client: httpx.AsyncClient):
        with patch("routes.health.is_db_connected", return_value=True):
            resp = await client.get("/api/health")

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "database": "connected"}

    async def test_503_when_the_database_is_down(self, client: httpx.AsyncClient):
        with patch("routes.health.is_db_connected", return_value=False):
            resp = await client.get("/api/health")

        assert resp.status_code == 503
        assert resp.json() == {"status": "degraded", "database": "disconnected"}

    async def test_needs_no_login(self, client: httpx.AsyncClient):
        with patch("routes.health.is_db_connected", return_value=True):
            resp = await client.get("/api/health")  # no auth headers
        assert resp.status_code == 200