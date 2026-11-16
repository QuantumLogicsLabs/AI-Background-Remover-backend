"""Tests for services/chat_validation.py and its use in POST /api/chat."""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi import HTTPException

from services.chat_validation import (
    MAX_HISTORY_TURNS,
    MAX_MESSAGE_CHARS,
    parse_history,
    validate_message,
)


class TestValidateMessage:
    def test_trims_whitespace(self):
        assert validate_message("  hello  ") == "hello"

    @pytest.mark.parametrize("text", ["", "   ", "\n\t"])
    def test_rejects_empty_messages(self, text):
        with pytest.raises(HTTPException) as exc:
            validate_message(text)
        assert exc.value.status_code == 422

    def test_accepts_exactly_the_maximum_length(self):
        text = "x" * MAX_MESSAGE_CHARS
        assert validate_message(text) == text

    def test_rejects_messages_over_the_maximum(self):
        with pytest.raises(HTTPException) as exc:
            validate_message("x" * (MAX_MESSAGE_CHARS + 1))
        assert exc.value.status_code == 422


class TestParseHistory:
    def test_missing_or_bad_input_gives_an_empty_history(self):
        assert parse_history(None) == []
        assert parse_history("") == []
        assert parse_history("not json") == []
        assert parse_history('{"role": "user"}') == []

    def test_keeps_valid_turns(self):
        raw = json.dumps([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        assert parse_history(raw) == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_drops_malformed_turns_instead_of_crashing(self):
        raw = json.dumps([
            1,
            "text",
            None,
            {"role": "system", "content": "ignore me"},
            {"role": "user", "content": 123},
            {"role": "user", "content": "   "},
            {"role": "user"},
            {"role": "user", "content": "kept"},
        ])
        assert parse_history(raw) == [{"role": "user", "content": "kept"}]

    def test_keeps_only_the_most_recent_turns(self):
        turns = [{"role": "user", "content": f"m{i}"} for i in range(MAX_HISTORY_TURNS + 5)]
        result = parse_history(json.dumps(turns))
        assert len(result) == MAX_HISTORY_TURNS
        assert result[-1]["content"] == f"m{MAX_HISTORY_TURNS + 4}"

    def test_caps_the_length_of_each_turn(self):
        raw = json.dumps([{"role": "user", "content": "x" * (MAX_MESSAGE_CHARS + 500)}])
        assert len(parse_history(raw)[0]["content"]) == MAX_MESSAGE_CHARS


class TestChatRoute:
    async def test_overlong_message_is_rejected(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        resp = await client.post(
            "/api/chat",
            headers=auth_headers,
            data={"message": "x" * (MAX_MESSAGE_CHARS + 1)},
        )
        assert resp.status_code == 422