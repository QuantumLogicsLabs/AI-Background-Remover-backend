"""Which email addresses the app accepts (Google accounts only by default)."""

from __future__ import annotations

import os
import re

from fastapi import HTTPException, status

_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]{2,}$")
_GMAIL_DOMAINS = {"gmail.com", "googlemail.com"}
# Gmail usernames: letters, numbers and single periods only (+tags are stripped first)
_GMAIL_LOCAL_RE = re.compile(r"^[a-z0-9]+(\.[a-z0-9]+)*$")


def allowed_domains() -> set[str]:
    raw = os.getenv("ALLOWED_EMAIL_DOMAINS", "gmail.com,googlemail.com")
    return {d.strip().lower() for d in raw.split(",") if d.strip()}


def enforce_allowed_email(email: str) -> str:
    """Return the normalised (trimmed, lowercase) email or raise HTTP 422."""
    e = (email or "").strip().lower()
    if len(e) > 254 or not _EMAIL_RE.match(e):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Enter a valid email address.")

    local, domain = e.rsplit("@", 1)
    if domain not in allowed_domains():
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Only Google (Gmail) accounts are supported. Please use your @gmail.com address.",
        )

    if domain in _GMAIL_DOMAINS:
        base = local.split("+", 1)[0]
        if not (3 <= len(base) <= 30) or not _GMAIL_LOCAL_RE.match(base):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "That is not a valid Gmail address.")

    return e