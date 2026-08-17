"""
Authentication Routes.

POST /api/auth/register        — create account, returns access token + sets refresh cookie
POST /api/auth/login           — exchange credentials for tokens
POST /api/auth/refresh         — exchange refresh cookie for a new access token
POST /api/auth/logout          — clear the refresh token cookie
GET  /api/auth/me              — return current user (requires access token)
GET  /api/auth/quota           — return daily quota usage for current user
POST /api/auth/forgot-password — send a password-reset email
POST /api/auth/reset-password  — consume reset token and set a new password
"""

from __future__ import annotations

import hashlib
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Cookie, HTTPException, Response, status, Depends
from pymongo.errors import (
    ServerSelectionTimeoutError,
    NetworkTimeout,
    ConnectionFailure,
    ConfigurationError,
    OperationFailure,
)

from models.user   import (
    UserCreate, UserLogin, UserOut, TokenResponse,
    UpdateProfileRequest, UpdatePasswordRequest, DeleteAccountRequest,
    ForgotPasswordRequest, ResetPasswordRequest,
)
from services.email import send_password_reset_email
from services.auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_token,
    get_current_user, REFRESH_COOKIE_NAME, REFRESH_TOKEN_EXPIRE_DAYS,
)
from services.database import get_collection, is_db_connected
from services.quota    import get_quota_status

router = APIRouter(prefix="/auth", tags=["Auth"])

# Derive cookie security flags from environment so dev (HTTP) works without
# warnings while production (HTTPS) enforces Secure + SameSite=None.
_COOKIE_SECURE   = os.getenv("COOKIE_SECURE",    "false").lower() == "true"
_COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE",  "lax")   # "none" in production behind HTTPS

_REFRESH_MAX_AGE = REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60  # seconds

_DB_ERRORS = (
    ServerSelectionTimeoutError,
    NetworkTimeout,
    ConnectionFailure,
    ConfigurationError,
    OperationFailure,
)


def _require_db() -> None:
    if not is_db_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is unreachable. Please try again later.",
        )


def _set_refresh_cookie(response: Response, token: str) -> None:
    """Attach the refresh token as an httpOnly cookie to the response."""
    response.set_cookie(
        key      = REFRESH_COOKIE_NAME,
        value    = token,
        httponly = True,
        secure   = _COOKIE_SECURE,
        samesite = _COOKIE_SAMESITE,
        max_age  = _REFRESH_MAX_AGE,
        path     = "/api/auth",   # cookie is only sent to auth endpoints
    )


def _clear_refresh_cookie(response: Response) -> None:
    """Remove the refresh token cookie."""
    response.delete_cookie(
        key      = REFRESH_COOKIE_NAME,
        httponly = True,
        secure   = _COOKIE_SECURE,
        samesite = _COOKIE_SAMESITE,
        path     = "/api/auth",
    )


# ── Register ───────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: UserCreate, response: Response):
    """
    Create a new user account.

    Returns a short-lived access token in the body and sets a long-lived
    refresh token in an httpOnly cookie.

    - **name**     Display name (2–80 chars)
    - **email**    Unique email address
    - **password** Min 8 characters
    """
    try:
        _require_db()
        collection = get_collection("users")

        existing = await collection.find_one({"email": body.email.lower()})
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with that email already exists.",
            )

        user_id = str(uuid.uuid4())
        now     = datetime.now(timezone.utc)

        user_doc = {
            "user_id":         user_id,
            "name":            body.name.strip(),
            "email":           body.email.lower(),
            "hashed_password": hash_password(body.password),
            "created_at":      now,
        }
        await collection.insert_one(user_doc)

        claims = {"sub": user_id, "email": body.email.lower()}
        access_token  = create_access_token(claims)
        refresh_token = create_refresh_token(claims)
        _set_refresh_cookie(response, refresh_token)

        user_out = UserOut(
            user_id    = user_id,
            name       = user_doc["name"],
            email      = user_doc["email"],
            created_at = now,
        )
        return TokenResponse(access_token=access_token, user=user_out)

    except HTTPException:
        raise
    except _DB_ERRORS as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is unreachable. Please try again later.",
        ) from exc


# ── Login ──────────────────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
async def login(body: UserLogin, response: Response):
    """
    Exchange email + password for a short-lived access token (body) and
    a long-lived refresh token (httpOnly cookie).
    """
    try:
        _require_db()
        collection = get_collection("users")
        doc = await collection.find_one({"email": body.email.lower()})

        if doc is None or not verify_password(body.password, doc["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        claims = {"sub": doc["user_id"], "email": doc["email"]}
        access_token  = create_access_token(claims)
        refresh_token = create_refresh_token(claims)
        _set_refresh_cookie(response, refresh_token)

        user_out = UserOut(
            user_id    = doc["user_id"],
            name       = doc["name"],
            email      = doc["email"],
            created_at = doc["created_at"],
        )
        return TokenResponse(access_token=access_token, user=user_out)

    except HTTPException:
        raise
    except _DB_ERRORS as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is unreachable. Please try again later.",
        ) from exc


# ── Refresh ────────────────────────────────────────────────────────────────

@router.post("/refresh")
async def refresh(
    response: Response,
    bgr_refresh: str | None = Cookie(default=None),
):
    """
    Exchange a valid refresh token cookie for a new short-lived access token.

    The refresh cookie is rotated on every call (new refresh token issued).
    Returns: {"access_token": str, "token_type": "bearer"}
    """
    if not bgr_refresh:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No refresh token provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate the refresh token — raises 401 on failure
    token_data = decode_token(bgr_refresh, expected_type="refresh")

    # Verify the user still exists in the database
    collection = get_collection("users")
    doc = await collection.find_one({"user_id": token_data.user_id}, {"_id": 0})
    if doc is None:
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    claims = {"sub": doc["user_id"], "email": doc["email"]}

    # Rotate: issue new access token + new refresh token
    new_access  = create_access_token(claims)
    new_refresh = create_refresh_token(claims)
    _set_refresh_cookie(response, new_refresh)

    return {"access_token": new_access, "token_type": "bearer"}


# ── Logout ─────────────────────────────────────────────────────────────────

@router.post("/logout", status_code=204)
async def logout(response: Response):
    """
    Clear the refresh token cookie. The client should discard its access token.
    """
    _clear_refresh_cookie(response)


# ── Me ─────────────────────────────────────────────────────────────────────

@router.get("/me", response_model=UserOut)
async def me(current_user: UserOut = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return current_user


# ── Quota ──────────────────────────────────────────────────────────────────

@router.get("/quota")
async def quota(current_user: UserOut = Depends(get_current_user)):
    """
    Return the authenticated user's current daily quota usage.

    Response fields:
    - **used**      Operations consumed today (UTC day)
    - **limit**     Daily limit (0 means disabled)
    - **remaining** Operations left today (null when disabled)
    - **resets_at** ISO-8601 timestamp of next quota reset (midnight UTC)
    - **disabled**  True when quota enforcement is turned off
    """
    return await get_quota_status(current_user.user_id)


# ── Update Profile ──────────────────────────────────────────────────────

@router.patch("/profile", response_model=UserOut)
async def update_profile(
    body: UpdateProfileRequest,
    current_user: UserOut = Depends(get_current_user),
):
    """
    Update the authenticated user's display name and/or email.

    - **name**  (optional) New display name (2–80 chars)
    - **email** (optional) New unique email address
    """
    collection = get_collection("users")

    updates: dict = {}
    if body.name is not None:
        updates["name"] = body.name.strip()
    if body.email is not None:
        new_email = body.email.lower()
        # Ensure the new email is not already taken by another account
        existing = await collection.find_one(
            {"email": new_email, "user_id": {"$ne": current_user.user_id}}
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="That email address is already in use.",
            )
        updates["email"] = new_email

    if not updates:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide at least one field to update.",
        )

    await collection.update_one(
        {"user_id": current_user.user_id},
        {"$set": updates},
    )

    # Return fresh user doc
    doc = await collection.find_one({"user_id": current_user.user_id}, {"_id": 0})
    return UserOut(
        user_id    = doc["user_id"],
        name       = doc["name"],
        email      = doc["email"],
        created_at = doc["created_at"],
    )


# ── Change Password ──────────────────────────────────────────────────

@router.patch("/password", status_code=204)
async def change_password(
    body: UpdatePasswordRequest,
    current_user: UserOut = Depends(get_current_user),
):
    """
    Change the authenticated user's password.

    Verifies the current password before applying the change.

    - **current_password** Existing password for verification
    - **new_password**     New password (min 8 chars)
    """
    from services.auth import hash_password, verify_password  # local import avoids circular

    collection = get_collection("users")
    doc = await collection.find_one({"user_id": current_user.user_id})
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    if not verify_password(body.current_password, doc["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect.",
        )

    if body.current_password == body.new_password:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="New password must differ from the current password.",
        )

    new_hash = hash_password(body.new_password)
    await collection.update_one(
        {"user_id": current_user.user_id},
        {"$set": {"hashed_password": new_hash}},
    )
    # 204 No Content — client should prompt user to re-login


# ── Delete Account ─────────────────────────────────────────────────────

@router.delete("/account", status_code=204)
async def delete_account(
    body: DeleteAccountRequest,
    response: Response,
    current_user: UserOut = Depends(get_current_user),
):
    """
    Permanently delete the authenticated user's account.

    Requires password confirmation. Clears the refresh cookie on success.

    - **password** Current password for confirmation
    """
    from services.auth import verify_password  # local import avoids circular

    collection = get_collection("users")
    doc = await collection.find_one({"user_id": current_user.user_id})
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    if not verify_password(body.password, doc["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Password is incorrect. Account not deleted.",
        )

    await collection.delete_one({"user_id": current_user.user_id})
    _clear_refresh_cookie(response)
    # 204 No Content


# ── Forgot Password ────────────────────────────────────────────────────────

_RESET_TOKEN_EXPIRE_HOURS = 1


@router.post("/forgot-password", status_code=202)
async def forgot_password(body: ForgotPasswordRequest):
    """
    Request a password-reset email.

    Always returns 202 Accepted regardless of whether the email exists,
    so that this endpoint cannot be used to enumerate registered accounts.

    - **email** The email address associated with the account
    """
    users_col  = get_collection("users")
    tokens_col = get_collection("password_reset_tokens")

    doc = await users_col.find_one({"email": body.email.lower()})
    if doc is None:
        # Silently succeed — do not reveal whether the email is registered
        return {"detail": "If that email is registered you will receive a reset link shortly."}

    # Generate a cryptographically secure random token
    raw_token   = secrets.token_urlsafe(48)          # 64-char URL-safe string
    token_hash  = hashlib.sha256(raw_token.encode()).hexdigest()  # store only the hash
    expires_at  = datetime.now(timezone.utc) + timedelta(hours=_RESET_TOKEN_EXPIRE_HOURS)

    # Invalidate any previous reset tokens for this user
    await tokens_col.delete_many({"user_id": doc["user_id"]})

    await tokens_col.insert_one({
        "user_id":    doc["user_id"],
        "token_hash": token_hash,
        "expires_at": expires_at,
        "used":       False,
    })

    try:
        await send_password_reset_email(doc["email"], raw_token)
    except Exception:
        # Do NOT expose SMTP errors to the client
        pass

    return {"detail": "If that email is registered you will receive a reset link shortly."}


# ── Reset Password ─────────────────────────────────────────────────────────

@router.post("/reset-password", status_code=200)
async def reset_password(body: ResetPasswordRequest):
    """
    Consume a valid password-reset token and set a new password.

    - **token**        The raw token received via email
    - **new_password** The new password (min 8 characters)
    """
    token_hash = hashlib.sha256(body.token.encode()).hexdigest()
    tokens_col = get_collection("password_reset_tokens")
    users_col  = get_collection("users")

    token_doc = await tokens_col.find_one({"token_hash": token_hash})

    invalid_exc = HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="This reset link is invalid or has expired. Please request a new one.",
    )

    if token_doc is None:
        raise invalid_exc

    if token_doc.get("used"):
        raise invalid_exc

    expires_at = token_doc["expires_at"]
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) > expires_at:
        await tokens_col.delete_one({"token_hash": token_hash})
        raise invalid_exc

    # Mark token as used immediately (single-use)
    await tokens_col.update_one(
        {"token_hash": token_hash},
        {"$set": {"used": True}},
    )

    # Update the user's password
    new_hash = hash_password(body.new_password)
    await users_col.update_one(
        {"user_id": token_doc["user_id"]},
        {"$set": {"hashed_password": new_hash}},
    )

    # Clean up used token
    await tokens_col.delete_one({"token_hash": token_hash})

    return {"detail": "Password has been reset successfully. You can now sign in."}
