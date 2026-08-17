"""
Quota Service.

Enforces a configurable daily operation limit per user using a MongoDB
counter document with a TTL index that auto-expires at midnight UTC.

Design
──────
Each quota document has the shape:
    {
        "user_id":  "<uuid>",
        "date":     "2026-08-12",          # UTC date string (YYYY-MM-DD)
        "count":    42,                    # operations consumed today
        "expires_at": <datetime midnight>  # TTL field — MongoDB drops the doc at day rollover
    }

A sparse compound unique index on (user_id, date) ensures exactly one
counter per user per day. The TTL index on `expires_at` removes stale
documents automatically so the collection never grows unbounded.

Usage
─────
    from services.quota import check_and_increment_quota
    await check_and_increment_quota(user_id)   # raises HTTP 429 if over limit

    from services.quota import get_quota_status
    status = await get_quota_status(user_id)   # {"used": 5, "limit": 100, "remaining": 95}

Configuration
─────────────
    DAILY_QUOTA_LIMIT  — max operations per user per UTC day (default: 100)
                         Set to 0 to disable quota enforcement entirely.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

from fastapi import HTTPException, status

from services.database import get_collection

# ── Config ─────────────────────────────────────────────────────────────────

DAILY_QUOTA_LIMIT: int = int(os.getenv("DAILY_QUOTA_LIMIT", "100"))


# ── Helpers ────────────────────────────────────────────────────────────────

def _today_utc() -> str:
    """Return today's date as a YYYY-MM-DD string in UTC."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _midnight_utc() -> datetime:
    """Return the next midnight in UTC (start of tomorrow) — used as TTL anchor."""
    now   = datetime.now(timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return today + timedelta(days=1)


async def _ensure_indexes() -> None:
    """
    Create the required indexes once. Safe to call repeatedly (no-op if they exist).

    Indexes created:
      - Unique compound (user_id, date) — one counter per user per day
      - TTL on expires_at              — auto-cleanup at midnight
    """
    col = get_collection("quota")
    await col.create_index(
        [("user_id", 1), ("date", 1)],
        unique=True,
        name="quota_user_date_unique",
    )
    await col.create_index(
        "expires_at",
        expireAfterSeconds=0,
        name="quota_ttl",
    )


# ── Public API ──────────────────────────────────────────────────────────────

async def check_and_increment_quota(user_id: str) -> None:
    """
    Atomically check the user's daily quota and increment the counter.

    - If DAILY_QUOTA_LIMIT == 0, quota enforcement is disabled.
    - If the user has reached their limit, raises HTTP 429.
    - Uses findOneAndUpdate with upsert to guarantee atomicity; no race
      conditions even under concurrent requests.

    Args:
        user_id: The authenticated user's ID.

    Raises:
        HTTPException 429: Daily quota exceeded.
    """
    if DAILY_QUOTA_LIMIT == 0:
        return  # Quota disabled

    col   = get_collection("quota")
    today = _today_utc()

    # Atomically fetch-then-increment. We read the value *before* incrementing
    # so we can check whether the pre-increment count is already at the limit.
    doc = await col.find_one_and_update(
        {"user_id": user_id, "date": today},
        {
            "$inc":      {"count": 1},
            "$setOnInsert": {
                "user_id":    user_id,
                "date":       today,
                "expires_at": _midnight_utc(),
            },
        },
        upsert=True,
        return_document=False,  # return doc BEFORE update (pre-increment value)
        projection={"count": 1, "_id": 0},
    )

    # doc is None on the very first operation (upsert created the doc).
    # After upsert the count is 1, which is always within limit.
    pre_count = doc["count"] if doc else 0

    if pre_count >= DAILY_QUOTA_LIMIT:
        # Roll back the increment we just applied so the counter stays accurate
        await col.update_one(
            {"user_id": user_id, "date": today},
            {"$inc": {"count": -1}},
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Daily limit of {DAILY_QUOTA_LIMIT} operations reached. "
                "Your quota resets at midnight UTC."
            ),
            headers={"Retry-After": "86400"},
        )


async def refund_quota(user_id: str, count: int = 1) -> None:
    """
    Decrement the user's quota counter by ``count`` (default 1).

    Called when a batch file fails validation *after* quota was already charged,
    so the user is not penalised for files that were never processed.
    Non-fatal: if the document no longer exists (e.g. TTL expired) this is a no-op.

    Args:
        user_id: The authenticated user's ID.
        count:   Number of operations to refund (must be ≥ 1).
    """
    if DAILY_QUOTA_LIMIT == 0 or count < 1:
        return

    col   = get_collection("quota")
    today = _today_utc()

    try:
        await col.update_one(
            {"user_id": user_id, "date": today},
            {"$inc": {"count": -count}},
        )
    except Exception:
        pass  # Non-fatal — stale or missing doc is fine


async def get_quota_status(user_id: str) -> dict:
    """
    Return the user's current quota usage for today.

    Returns:
        {"used": int, "limit": int, "remaining": int, "resets_at": str}
    """
    col   = get_collection("quota")
    today = _today_utc()

    doc   = await col.find_one(
        {"user_id": user_id, "date": today},
        {"count": 1, "_id": 0},
    )
    used      = doc["count"] if doc else 0
    limit     = DAILY_QUOTA_LIMIT
    remaining = max(0, limit - used) if limit > 0 else None

    return {
        "used":      used,
        "limit":     limit,
        "remaining": remaining,
        "resets_at": _midnight_utc().isoformat(),
        "disabled":  limit == 0,
    }


async def setup_quota_indexes() -> None:
    """Call this once at application startup to create the required indexes."""
    try:
        await _ensure_indexes()
    except Exception as exc:
        # Non-fatal: app still starts, quota may not work until DB is reachable
        print(f"[QUOTA] Could not create quota indexes: {exc}")
