"""One-time email verification codes (sign-up and email change)."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from pymongo import ReturnDocument

from services import database as _database
from services.auth import SECRET_KEY
from services.email import send_verification_code_email

CODE_TTL_MIN        = 10
RESEND_COOLDOWN_SEC = 60
MAX_ATTEMPTS        = 5

PURPOSE_SIGNUP       = "signup"
PURPOSE_EMAIL_CHANGE = "email_change"

_COLLECTION = "email_verifications"


def _col():
    # Resolved at call time (not import time) so tests can swap in a fake database.
    return _database.get_collection(_COLLECTION)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _hash(user_id: str, purpose: str, code: str) -> str:
    msg = f"{user_id}:{purpose}:{code}".encode()
    return hmac.new(SECRET_KEY.encode(), msg, hashlib.sha256).hexdigest()


def _new_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


async def issue_code(user_id: str, email: str, purpose: str) -> bool:
    """
    Create and email a fresh code. Returns False (and sends nothing) if a code
    was already sent to this same address within the cooldown window.
    """
    col = _col()
    now = datetime.now(timezone.utc)

    existing = await col.find_one({"user_id": user_id, "purpose": purpose})
    if existing and existing.get("email") == email:
        if (now - _aware(existing["last_sent_at"])).total_seconds() < RESEND_COOLDOWN_SEC:
            return False

    code = _new_code()
    await col.replace_one(
        {"user_id": user_id, "purpose": purpose},
        {
            "user_id":      user_id,
            "purpose":      purpose,
            "email":        email,
            "code_hash":    _hash(user_id, purpose, code),
            "expires_at":   now + timedelta(minutes=CODE_TTL_MIN),
            "attempts":     0,
            "last_sent_at": now,
        },
        upsert=True,
    )

    try:
        await send_verification_code_email(email, code)
    except Exception as exc:
        print(f"[EMAIL] Failed to send verification code to {email}: {exc}")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "We couldn't send the verification email. Please try again in a moment.",
        ) from exc
    return True


async def consume_code(user_id: str, purpose: str, code: str) -> dict:
    """Validate a code (single use). Returns the stored record or raises HTTP 400."""
    bad = HTTPException(status.HTTP_400_BAD_REQUEST, "That code is incorrect or has expired.")
    col = _col()
    key = {"user_id": user_id, "purpose": purpose}

    # Atomic increment so parallel guesses cannot beat the attempt limit
    doc = await col.find_one_and_update(
        key,
        {"$inc": {"attempts": 1}},
        return_document=ReturnDocument.AFTER,
    )
    if doc is None:
        raise bad

    if doc["attempts"] > MAX_ATTEMPTS:
        await col.delete_one(key)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Too many incorrect attempts. Please request a new code.",
        )

    if datetime.now(timezone.utc) > _aware(doc["expires_at"]):
        await col.delete_one(key)
        raise bad

    if not hmac.compare_digest(doc["code_hash"], _hash(user_id, purpose, code)):
        raise bad

    await col.delete_one(key)
    return doc