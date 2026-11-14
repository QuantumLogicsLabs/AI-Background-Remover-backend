"""The startup index setup creates the TTL indexes with the right options."""
from unittest.mock import AsyncMock, MagicMock, patch

import services.database as database


def _fake_client():
    collections: dict[str, MagicMock] = {}

    def get(name):
        if name not in collections:
            collections[name] = MagicMock(create_index=AsyncMock())
        return collections[name]

    db = MagicMock()
    db.__getitem__.side_effect = get
    client = MagicMock()
    client.__getitem__.return_value = db
    return client, collections


def _calls_named(collection, name):
    return [c for c in collection.create_index.await_args_list if c.kwargs.get("name") == name]


async def test_unverified_accounts_expire_after_configured_hours(monkeypatch):
    monkeypatch.setenv("UNVERIFIED_ACCOUNT_TTL_HOURS", "2")
    client, collections = _fake_client()
    with patch.object(database, "client", client):
        await database._setup_user_indexes()

    ttl = _calls_named(collections["users"], "users_unverified_ttl")
    assert len(ttl) == 1
    assert ttl[0].kwargs["expireAfterSeconds"] == 7200
    assert ttl[0].kwargs["partialFilterExpression"] == {"email_verified": False}


async def test_default_ttl_is_24_hours(monkeypatch):
    monkeypatch.delenv("UNVERIFIED_ACCOUNT_TTL_HOURS", raising=False)
    client, collections = _fake_client()
    with patch.object(database, "client", client):
        await database._setup_user_indexes()
    ttl = _calls_named(collections["users"], "users_unverified_ttl")
    assert ttl[0].kwargs["expireAfterSeconds"] == 86400


async def test_verification_codes_expire_on_their_own():
    client, collections = _fake_client()
    with patch.object(database, "client", client):
        await database._setup_user_indexes()
    ttl = _calls_named(collections["email_verifications"], "email_verifications_ttl")
    assert ttl[0].kwargs["expireAfterSeconds"] == 0