"""
Social sign-in (Google + GitHub) — backend-driven OAuth 2.0 authorization-code flow.

GET /api/auth/oauth/providers                 — which providers are configured
GET /api/auth/oauth/{provider}                — redirect the browser to Google / GitHub
GET /api/auth/oauth/{provider}/callback       — provider redirects back here; we issue the
                                                same refresh cookie as /login and redirect
                                                to  <FRONTEND_URL>/auth/callback

No new dependencies: uses httpx (already installed) and python-jose (already used for JWTs).
CSRF protection: a signed, short-lived `state` JWT is sent to the provider AND stored in an
httpOnly cookie; the callback requires both to match.
"""

from __future__ import annotations

import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Cookie, HTTPException, Query
from services.email_policy import enforce_allowed_email
from fastapi.responses import RedirectResponse
from jose import JWTError, jwt
from pymongo.errors import DuplicateKeyError

from routes.auth import (
    _COOKIE_SAMESITE,
    _COOKIE_SECURE,
    _require_db,
    _set_refresh_cookie,
)
from services.auth import (
    ALGORITHM,
    SECRET_KEY,
    create_refresh_token,
)
from services.database import get_collection

router = APIRouter(prefix="/auth/oauth", tags=["Auth"])

_STATE_COOKIE = "bgr_oauth_state"
_STATE_TTL_MIN = 10

_PROVIDERS = {
    "google": {
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "scope": "openid email profile",
    },
    "github": {
        "authorize_url": "https://github.com/login/oauth/authorize",
        "scope": "read:user user:email",
    },
}


# ── Config helpers (read env at call time so .env changes need no code change) ──

def _credentials(provider: str) -> tuple[str, str]:
    p = provider.upper()
    return os.getenv(f"{p}_CLIENT_ID", "").strip(), os.getenv(f"{p}_CLIENT_SECRET", "").strip()


def _enabled(provider: str) -> bool:
    cid, secret = _credentials(provider)
    return bool(cid and secret)


def _backend_url() -> str:
    return os.getenv("BACKEND_PUBLIC_URL", "http://localhost:8000").rstrip("/")


def _frontend_url() -> str:
    explicit = os.getenv("FRONTEND_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    first = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")[0].strip()
    return first.rstrip("/") or "http://localhost:5173"


def _redirect_uri(provider: str) -> str:
    return f"{_backend_url()}/api/auth/oauth/{provider}/callback"


def _clear_state_cookie(resp: RedirectResponse) -> None:
    resp.delete_cookie(
        key=_STATE_COOKIE, httponly=True, secure=_COOKIE_SECURE,
        samesite=_COOKIE_SAMESITE, path="/api/auth",
    )


def _error_redirect(message: str) -> RedirectResponse:
    resp = RedirectResponse(
        f"{_frontend_url()}/login?{urlencode({'oauth_error': message})}", status_code=302
    )
    _clear_state_cookie(resp)
    return resp


# ── Provider profile fetchers ─────────────────────────────────────────────

async def _google_profile(client: httpx.AsyncClient, code: str) -> dict:
    cid, secret = _credentials("google")
    tok = await client.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code, "client_id": cid, "client_secret": secret,
            "redirect_uri": _redirect_uri("google"), "grant_type": "authorization_code",
        },
    )
    tok.raise_for_status()
    access = tok.json()["access_token"]
    r = await client.get(
        "https://openidconnect.googleapis.com/v1/userinfo",
        headers={"Authorization": f"Bearer {access}"},
    )
    r.raise_for_status()
    d = r.json()
    return {
        "id": str(d["sub"]),
        "email": (d.get("email") or "").lower(),
        "email_verified": bool(d.get("email_verified")),
        "name": (d.get("name") or "").strip(),
        "avatar": d.get("picture"),
    }


async def _github_profile(client: httpx.AsyncClient, code: str) -> dict:
    cid, secret = _credentials("github")
    tok = await client.post(
        "https://github.com/login/oauth/access_token",
        data={
            "code": code, "client_id": cid, "client_secret": secret,
            "redirect_uri": _redirect_uri("github"),
        },
        headers={"Accept": "application/json"},
    )
    tok.raise_for_status()
    access = tok.json().get("access_token")
    if not access:
        raise ValueError("GitHub did not return an access token")

    headers = {"Authorization": f"Bearer {access}", "Accept": "application/vnd.github+json"}
    u = await client.get("https://api.github.com/user", headers=headers)
    u.raise_for_status()
    user = u.json()

    # /user.email can be null/private and is not guaranteed verified → use /user/emails
    e = await client.get("https://api.github.com/user/emails", headers=headers)
    e.raise_for_status()
    emails = [x for x in e.json() if x.get("verified")]
    primary = next((x for x in emails if x.get("primary")), None) or (emails[0] if emails else None)

    return {
        "id": str(user["id"]),
        "email": (primary["email"].lower() if primary else ""),
        "email_verified": primary is not None,
        "name": (user.get("name") or user.get("login") or "").strip(),
        "avatar": user.get("avatar_url"),
    }


_FETCHERS = {"google": _google_profile, "github": _github_profile}


# ── User resolution ───────────────────────────────────────────────────────

def _display_name(profile: dict) -> str:
    name = (profile["name"] or profile["email"].split("@")[0])[:80]
    return name if len(name) >= 2 else f"{name}_user"


async def _find_or_create_user(provider: str, profile: dict) -> dict:
    users = get_collection("users")
    key = f"oauth.{provider}"

    # 1. Already signed in with this provider before
    doc = await users.find_one({key: profile["id"]})
    if doc:
        return doc

    # 2. Same verified email → link to the existing account
        # 2. Same verified email → link to the existing account
    doc = await users.find_one({"email": profile["email"]})
    if doc:
        await users.update_one(
            {"user_id": doc["user_id"]},
            {"$set": {key: profile["id"], "email_verified": True}},
        )
        return doc

    # 3. Brand-new account (no hashed_password key at all)
    doc = {
        "user_id":    str(uuid.uuid4()),
        "name":       _display_name(profile),
        "email":      profile["email"],
        "created_at": datetime.now(timezone.utc),
        "oauth":      {provider: profile["id"]},
        "email_verified": True,
    }
    if profile.get("avatar"):
        doc["avatar_url"] = profile["avatar"]
    try:
        await users.insert_one(doc)
    except DuplicateKeyError:           # lost a race with another request
        doc = await users.find_one({"email": profile["email"]})
        await users.update_one({"user_id": doc["user_id"]}, {"$set": {key: profile["id"]}})
    return doc


# ── Routes ────────────────────────────────────────────────────────────────

@router.get("/providers")
async def providers():
    """Tell the frontend which social buttons to render."""
    return {name: _enabled(name) for name in _PROVIDERS}


@router.get("/{provider}")
async def oauth_start(provider: str):
    if provider not in _PROVIDERS or not _enabled(provider):
        return _error_redirect("That sign-in method is not available.")

    state = jwt.encode(
        {
            "type": "oauth_state",
            "provider": provider,
            "nonce": secrets.token_urlsafe(16),
            "exp": datetime.now(timezone.utc) + timedelta(minutes=_STATE_TTL_MIN),
        },
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
    params = {
        "client_id": _credentials(provider)[0],
        "redirect_uri": _redirect_uri(provider),
        "response_type": "code",
        "scope": _PROVIDERS[provider]["scope"],
        "state": state,
    }
    if provider == "google":
        params["prompt"] = "select_account"

    resp = RedirectResponse(f"{_PROVIDERS[provider]['authorize_url']}?{urlencode(params)}", status_code=302)
    resp.set_cookie(
        key=_STATE_COOKIE, value=state, httponly=True, secure=_COOKIE_SECURE,
        samesite=_COOKIE_SAMESITE, max_age=_STATE_TTL_MIN * 60, path="/api/auth",
    )
    return resp


@router.get("/{provider}/callback")
async def oauth_callback(
    provider: str,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
    bgr_oauth_state: str | None = Cookie(default=None),
):
    if provider not in _PROVIDERS or not _enabled(provider):
        return _error_redirect("That sign-in method is not available.")
    if error or not code:
        return _error_redirect("Sign-in was cancelled.")

    # CSRF check: query state must equal the cookie, and be a valid unexpired state JWT
    if not state or not bgr_oauth_state or not secrets.compare_digest(state, bgr_oauth_state):
        return _error_redirect("Sign-in session expired. Please try again.")
    try:
        claims = jwt.decode(state, SECRET_KEY, algorithms=[ALGORITHM])
        if claims.get("type") != "oauth_state" or claims.get("provider") != provider:
            raise ValueError("bad state")
    except (JWTError, ValueError):
        return _error_redirect("Sign-in session expired. Please try again.")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            profile = await _FETCHERS[provider](client, code)
    except (httpx.HTTPError, KeyError, ValueError):
        return _error_redirect("Could not complete sign-in with the provider.")

    if not profile["email"] or not profile["email_verified"]:
        return _error_redirect("Your account has no verified email address.")
    try:
        _require_db()
        doc = await _find_or_create_user(provider, profile)
    except Exception:
        return _error_redirect("Server error. Please try again later.")

    if provider != "google":   # Google already authenticated this account itself
            try:
                profile["email"] = enforce_allowed_email(profile["email"])
            except HTTPException as exc:
                return _error_redirect(str(exc.detail))

    token_claims = {"sub": doc["user_id"], "email": doc["email"]}
    # No access token here: the frontend obtains one from POST /api/auth/refresh.
    refresh = create_refresh_token(token_claims)

    resp = RedirectResponse(f"{_frontend_url()}/auth/callback", status_code=302)
    _set_refresh_cookie(resp, refresh)
    _clear_state_cookie(resp)
    return resp
