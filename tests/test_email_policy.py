"""Unit tests for services/email_policy.py (which email addresses are accepted)."""
import pytest
from fastapi import HTTPException

from services.email_policy import enforce_allowed_email


@pytest.fixture(autouse=True)
def default_domains(monkeypatch):
    monkeypatch.delenv("ALLOWED_EMAIL_DOMAINS", raising=False)


@pytest.mark.parametrize("raw, expected", [
    ("alice@gmail.com",            "alice@gmail.com"),
    ("  Alice.Smith@GMAIL.com ",   "alice.smith@gmail.com"),
    ("alice@googlemail.com",       "alice@googlemail.com"),
    ("alice+newsletter@gmail.com", "alice+newsletter@gmail.com"),
    ("a1b2c3@gmail.com",           "a1b2c3@gmail.com"),
])
def test_accepts_and_normalises(raw, expected):
    assert enforce_allowed_email(raw) == expected


@pytest.mark.parametrize("raw", [
    "",
    "not-an-email",
    "alice@",
    "@gmail.com",
    "alice@yahoo.com",
    "alice@outlook.com",
    "alice@gmail.co",
    "alice@gmail.com.evil.com",
    "x@gmail.com",               # too short for a Gmail username
    "ab@gmail.com",
    ".alice@gmail.com",          # leading dot
    "alice.@gmail.com",          # trailing dot
    "ali..ce@gmail.com",         # consecutive dots
    "ali_ce@gmail.com",          # Gmail has no underscores
    "ali ce@gmail.com",
    "a" * 31 + "@gmail.com",    # too long
])
def test_rejects(raw):
    with pytest.raises(HTTPException) as exc:
        enforce_allowed_email(raw)
    assert exc.value.status_code == 422


def test_rejection_message_mentions_gmail_for_other_domains():
    with pytest.raises(HTTPException) as exc:
        enforce_allowed_email("bob@yahoo.com")
    assert "gmail" in exc.value.detail.lower()


def test_extra_domains_can_be_allowed(monkeypatch):
    monkeypatch.setenv("ALLOWED_EMAIL_DOMAINS", "gmail.com, Acme.org")
    assert enforce_allowed_email("bob_x@acme.org") == "bob_x@acme.org"   # Gmail-only rules don't apply
    with pytest.raises(HTTPException):
        enforce_allowed_email("bob@yahoo.com")


def test_gmail_can_be_removed_from_the_list(monkeypatch):
    monkeypatch.setenv("ALLOWED_EMAIL_DOMAINS", "acme.org")
    with pytest.raises(HTTPException):
        enforce_allowed_email("alice@gmail.com")