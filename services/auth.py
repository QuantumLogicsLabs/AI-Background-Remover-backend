"""
Authentication Service.

Responsibilities
────────────────
  - Password hashing / verification (bcrypt)
  - JWT access token creation and verification (python-jose, short-lived)
  - Refresh token creation and verification (long-lived, httpOnly cookie)
  - FastAPI dependency: get_current_user — injects authenticated user into routes
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Cookie, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import bcrypt as _bcrypt
from jose import JWTError, jwt

from models.user import TokenData, UserOut
from services.database import get_collection

# ── Config from environment ────────────────────────────────────────────────

SECRET_KEY  = os.getenv("SECRET_KEY", "change-me-in-production-use-a-long-random-string")
ALGORITHM   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES  = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES",  "60"))
REFRESH_TOKEN_EXPIRE_DAYS    = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS",    "30"))

# Cookie name used for the refresh token
REFRESH_COOKIE_NAME = "bgr_refresh"

# ── Helpers ────────────────────────────────────────────────────────────────

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def hash_password(plain: str) -> str:
    return _bcrypt.hashpw(plain.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return _bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a short-lived access token (type: 'access')."""
    to_encode = data.copy()
    to_encode["type"] = "access"
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """Create a long-lived refresh token (type: 'refresh')."""
    to_encode = data.copy()
    to_encode["type"] = "refresh"
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str, expected_type: str = "access") -> TokenData:
    """
    Decode and validate a JWT.

    Args:
        token:         The raw JWT string.
        expected_type: "access" or "refresh" — the `type` claim must match.

    Raises:
        HTTPException 401 if the token is invalid, expired, or the wrong type.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        email:   str = payload.get("email")
        tok_type: str = payload.get("type", "access")

        if user_id is None:
            raise ValueError("missing sub")
        if tok_type != expected_type:
            raise ValueError(f"expected token type '{expected_type}', got '{tok_type}'")

        return TokenData(user_id=user_id, email=email)
    except (JWTError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI dependency ─────────────────────────────────────────────────────

def is_email_verified(doc: dict) -> bool:
    """Social-login accounts are verified by the provider; password accounts need the code."""
    return doc.get("email_verified") is True or bool(doc.get("oauth"))

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserOut:
    """
    Resolve a Bearer access JWT to a UserOut model.
    Raises 401 if the token is invalid or the user no longer exists.
    """
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token_data = decode_token(token, expected_type="access")

    collection = get_collection("users")
    doc = await collection.find_one({"user_id": token_data.user_id}, {"_id": 0})
    if doc is None:
        raise credentials_exc
    
    if not is_email_verified(doc):
        raise credentials_exc

    return UserOut(
        user_id    = doc["user_id"],
        name       = doc["name"],
        email      = doc["email"],
        created_at = doc["created_at"],
    )
