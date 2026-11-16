"""Tests for the SECRET_KEY safety check in services/auth.py."""
import secrets

from services.auth import MIN_SECRET_LENGTH, secret_key_problem


def test_built_in_default_key_is_flagged():
    problem = secret_key_problem("change-me-in-production-use-a-long-random-string")
    assert problem is not None
    assert "SECRET_KEY" in problem


def test_env_example_placeholder_is_flagged():
    assert secret_key_problem("generate-with-python-secrets-token-hex-32") is not None


def test_short_key_is_flagged():
    problem = secret_key_problem("too-short")
    assert problem is not None
    assert str(MIN_SECRET_LENGTH) in problem


def test_empty_key_is_flagged():
    assert secret_key_problem("") is not None


def test_random_key_is_accepted():
    assert secret_key_problem(secrets.token_hex(32)) is None