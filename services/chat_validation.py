"""Input checks for the chat endpoint."""

from __future__ import annotations

import json

from fastapi import HTTPException, status

MAX_MESSAGE_CHARS = 4000
MAX_HISTORY_TURNS = 20
VALID_ROLES = {"user", "assistant"}


def validate_message(message: str) -> str:
    """Return the trimmed message, or raise HTTP 422 if it is empty or too long."""
    text = (message or "").strip()
    if not text:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Message cannot be empty.")
    if len(text) > MAX_MESSAGE_CHARS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Message is too long (max {MAX_MESSAGE_CHARS} characters).",
        )
    return text


def parse_history(raw: str | None) -> list[dict[str, str]]:
    """
    Best-effort parse of the JSON ``history`` form field.

    Bad JSON, a non-list value and malformed turns are dropped instead of
    crashing the request. Only the most recent MAX_HISTORY_TURNS turns are
    kept, and each turn's text is capped at MAX_MESSAGE_CHARS.
    """
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []

    cleaned: list[dict[str, str]] = []
    for turn in data:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role not in VALID_ROLES or not isinstance(content, str) or not content.strip():
            continue
        cleaned.append({"role": role, "content": content[:MAX_MESSAGE_CHARS]})

    return cleaned[-MAX_HISTORY_TURNS:]