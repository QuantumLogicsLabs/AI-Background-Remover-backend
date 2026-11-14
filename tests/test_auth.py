"""
Tests for authentication routes:
  POST  /api/auth/register            POST /api/auth/verify-email
  POST  /api/auth/resend-verification POST /api/auth/login
  POST  /api/auth/email-change        POST /api/auth/email-change/confirm
  PATCH /api/auth/profile             GET  /api/auth/me
  POST  /api/auth/logout
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import httpx

from services.verification import MAX_ATTEMPTS, RESEND_COOLDOWN_SEC
from tests.conftest import (
    TEST_USER_EMAIL, TEST_USER_NAME, TEST_USER_PASS,
    code_sent_to, get_memory_collection,
)

GOOD_PASSWORD = "SecurePass1!"


# ── Helpers ───────────────────────────────────────────────────────────────────

async def seed_user(
    email: str,
    *,
    verified: bool | None = True,
    password: str = GOOD_PASSWORD,
    name: str = "Seed User",
    **extra,
) -> dict:
    """
    Insert a user straight into the in-memory DB.
    verified=None leaves the field out entirely (a legacy account).
    """
    from services.auth import hash_password
    doc = {
        "user_id":         str(uuid.uuid4()),
        "name":            name,
        "email":           email,
        "hashed_password": hash_password(password),
        "created_at":      datetime.now(timezone.utc),
    }
    if verified is not None:
        doc["email_verified"] = verified
    doc.update(extra)
    await get_memory_collection("users").insert_one(doc)
    return doc


def bearer(doc: dict) -> dict[str, str]:
    from services.auth import create_access_token
    token = create_access_token({"sub": doc["user_id"], "email": doc["email"]})
    return {"Authorization": f"Bearer {token}"}


async def register(client, email="alice@gmail.com", password=GOOD_PASSWORD, name="Alice"):
    return await client.post("/api/auth/register", json={
        "name": name, "email": email, "password": password,
    })


async def signup(client, sent_codes, email="alice@gmail.com", password=GOOD_PASSWORD) -> str:
    """Register and return the emailed verification code."""
    resp = await register(client, email, password)
    assert resp.status_code == 201
    return code_sent_to(sent_codes, email)


async def verify(client, email: str, code: str):
    return await client.post("/api/auth/verify-email", json={"email": email, "code": code})


def wrong_code(real: str) -> str:
    return "000000" if real != "000000" else "999999"


# ── Register ──────────────────────────────────────────────────────────────────

class TestRegister:
    async def test_register_requires_verification(self, client: httpx.AsyncClient, sent_codes):
        """Sign-up creates an unverified account, emails a code and does NOT log in."""
        resp = await register(client, "Alice.Smith@Gmail.com")
        assert resp.status_code == 201
        data = resp.json()
        assert data == {"email": "alice.smith@gmail.com", "verification_required": True}
        assert "access_token" not in data
        assert "set-cookie" not in resp.headers

        user = await get_memory_collection("users").find_one({"email": "alice.smith@gmail.com"})
        assert user["email_verified"] is False

        code = code_sent_to(sent_codes, "alice.smith@gmail.com")
        assert len(code) == 6 and code.isdigit()

    async def test_register_non_gmail_rejected(self, client: httpx.AsyncClient, sent_codes):
        resp = await register(client, "bob@yahoo.com")
        assert resp.status_code == 422
        assert "gmail" in resp.json()["detail"].lower()
        assert sent_codes == []
        assert await get_memory_collection("users").find_one({"email": "bob@yahoo.com"}) is None

    async def test_register_bad_gmail_username_rejected(self, client: httpx.AsyncClient):
        resp = await register(client, "x@gmail.com")
        assert resp.status_code == 422

    async def test_register_duplicate_verified_email(self, client: httpx.AsyncClient):
        await seed_user("carol@gmail.com", verified=True)
        resp = await register(client, "carol@gmail.com")
        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"].lower()

    async def test_register_unverified_account_can_be_reclaimed(self, client: httpx.AsyncClient, sent_codes):
        """
        If someone signs up with an address they don't own, the real owner can
        register again and take over: only the inbox owner can verify.
        """
        await register(client, "dana@gmail.com", "FirstPass1!")
        code = code_sent_to(sent_codes, "dana@gmail.com")

        second = await register(client, "dana@gmail.com", "SecondPass2!", name="Dana Real")
        assert second.status_code == 201

        resp = await verify(client, "dana@gmail.com", code)
        assert resp.status_code == 200
        assert resp.json()["user"]["name"] == "Dana Real"

        old = await client.post("/api/auth/login", json={"email": "dana@gmail.com", "password": "FirstPass1!"})
        new = await client.post("/api/auth/login", json={"email": "dana@gmail.com", "password": "SecondPass2!"})
        assert old.status_code == 401
        assert new.status_code == 200

    async def test_register_never_overwrites_legacy_account(self, client: httpx.AsyncClient):
        """Accounts created before verification existed have no flag and must not be taken over."""
        await seed_user("legacy@gmail.com", verified=None)
        resp = await register(client, "legacy@gmail.com", "AttackerPass1!")
        assert resp.status_code == 409

    async def test_register_short_password(self, client: httpx.AsyncClient):
        resp = await register(client, "charlie@gmail.com", "abc")
        assert resp.status_code == 422

    async def test_register_invalid_email(self, client: httpx.AsyncClient):
        resp = await register(client, "not-an-email")
        assert resp.status_code == 422


# ── Verify email ──────────────────────────────────────────────────────────────

class TestVerifyEmail:
    async def test_verify_success_logs_in(self, client: httpx.AsyncClient, sent_codes):
        from services.auth import REFRESH_COOKIE_NAME
        code = await signup(client, sent_codes)

        resp = await verify(client, "alice@gmail.com", code)
        assert resp.status_code == 200
        data = resp.json()
        assert data["user"]["email"] == "alice@gmail.com"
        assert REFRESH_COOKIE_NAME in resp.headers.get("set-cookie", "")

        user = await get_memory_collection("users").find_one({"email": "alice@gmail.com"})
        assert user["email_verified"] is True

        me = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {data['access_token']}"})
        assert me.status_code == 200

    async def test_wrong_code_rejected(self, client: httpx.AsyncClient, sent_codes):
        code = await signup(client, sent_codes)
        resp = await verify(client, "alice@gmail.com", wrong_code(code))
        assert resp.status_code == 400
        assert "incorrect" in resp.json()["detail"].lower()
        user = await get_memory_collection("users").find_one({"email": "alice@gmail.com"})
        assert user["email_verified"] is False

    async def test_code_is_single_use(self, client: httpx.AsyncClient, sent_codes):
        code = await signup(client, sent_codes)
        assert (await verify(client, "alice@gmail.com", code)).status_code == 200
        assert (await verify(client, "alice@gmail.com", code)).status_code == 400

    async def test_locked_out_after_too_many_wrong_attempts(self, client: httpx.AsyncClient, sent_codes):
        code = await signup(client, sent_codes)
        for _ in range(MAX_ATTEMPTS):
            assert (await verify(client, "alice@gmail.com", wrong_code(code))).status_code == 400

        resp = await verify(client, "alice@gmail.com", code)   # even the right code now fails
        assert resp.status_code == 400
        assert "too many" in resp.json()["detail"].lower()
        user = await get_memory_collection("users").find_one({"email": "alice@gmail.com"})
        assert user["email_verified"] is False

    async def test_expired_code_rejected(self, client: httpx.AsyncClient, sent_codes):
        code = await signup(client, sent_codes)
        record = get_memory_collection("email_verifications")._docs[0]
        record["expires_at"] = datetime.now(timezone.utc) - timedelta(minutes=1)
        assert (await verify(client, "alice@gmail.com", code)).status_code == 400

    async def test_unknown_email_gets_same_error(self, client: httpx.AsyncClient):
        resp = await verify(client, "nobody@gmail.com", "123456")
        assert resp.status_code == 400
        assert "incorrect" in resp.json()["detail"].lower()

    async def test_code_must_be_six_characters(self, client: httpx.AsyncClient):
        resp = await verify(client, "alice@gmail.com", "123")
        assert resp.status_code == 422


# ── Login ────────────────────────────────────────────────────────────────────

class TestLogin:
    async def test_login_success(self, client: httpx.AsyncClient):
        resp = await client.post("/api/auth/login", json={
            "email": TEST_USER_EMAIL, "password": TEST_USER_PASS,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["user"]["email"] == TEST_USER_EMAIL

    async def test_login_wrong_password(self, client: httpx.AsyncClient):
        resp = await client.post("/api/auth/login", json={
            "email": TEST_USER_EMAIL, "password": "WrongPassword!",
        })
        assert resp.status_code == 401

    async def test_login_unknown_email(self, client: httpx.AsyncClient):
        resp = await client.post("/api/auth/login", json={
            "email": "unknown@example.com", "password": "Password1!",
        })
        assert resp.status_code == 401

    async def test_login_unverified_returns_403(self, client: httpx.AsyncClient):
        await register(client, "erin@gmail.com")
        resp = await client.post("/api/auth/login", json={
            "email": "erin@gmail.com", "password": GOOD_PASSWORD,
        })
        assert resp.status_code == 403
        detail = resp.json()["detail"]
        assert detail["code"] == "email_not_verified"
        assert detail["email"] == "erin@gmail.com"

    async def test_login_unverified_sends_a_fresh_code(self, client: httpx.AsyncClient, sent_codes):
        await seed_user("pending@gmail.com", verified=False)
        resp = await client.post("/api/auth/login", json={
            "email": "pending@gmail.com", "password": GOOD_PASSWORD,
        })
        assert resp.status_code == 403
        assert len(code_sent_to(sent_codes, "pending@gmail.com")) == 6

    async def test_wrong_password_does_not_reveal_unverified(self, client: httpx.AsyncClient, sent_codes):
        await seed_user("hidden@gmail.com", verified=False)
        resp = await client.post("/api/auth/login", json={
            "email": "hidden@gmail.com", "password": "WrongPassword!",
        })
        assert resp.status_code == 401
        assert sent_codes == []

    async def test_login_works_after_verifying(self, client: httpx.AsyncClient, sent_codes):
        code = await signup(client, sent_codes, "frank@gmail.com")
        await verify(client, "frank@gmail.com", code)
        resp = await client.post("/api/auth/login", json={
            "email": "frank@gmail.com", "password": GOOD_PASSWORD,
        })
        assert resp.status_code == 200

    async def test_legacy_non_google_account_is_refused(self, client: httpx.AsyncClient, sent_codes):
        await seed_user("old@yahoo.com", verified=None)
        resp = await client.post("/api/auth/login", json={
            "email": "old@yahoo.com", "password": GOOD_PASSWORD,
        })
        assert resp.status_code == 422
        assert sent_codes == []


# ── Get current user (me) ─────────────────────────────────────────────────────

class TestGetMe:
    async def test_get_me_authenticated(self, client: httpx.AsyncClient, auth_headers: dict):
        resp = await client.get("/api/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == TEST_USER_EMAIL
        assert data["name"] == TEST_USER_NAME

    async def test_get_me_unauthenticated(self, client: httpx.AsyncClient):
        assert (await client.get("/api/auth/me")).status_code == 401

    async def test_get_me_invalid_token(self, client: httpx.AsyncClient):
        resp = await client.get("/api/auth/me", headers={"Authorization": "Bearer not.a.valid.jwt"})
        assert resp.status_code == 401

    async def test_unverified_account_token_is_rejected(self, client: httpx.AsyncClient):
        user = await seed_user("ghost@gmail.com", verified=False)
        resp = await client.get("/api/auth/me", headers=bearer(user))
        assert resp.status_code == 401

    async def test_social_login_account_counts_as_verified(self, client: httpx.AsyncClient):
        user = await seed_user("social@gmail.com", verified=None, oauth={"google": "12345"})
        resp = await client.get("/api/auth/me", headers=bearer(user))
        assert resp.status_code == 200


# ── Resend verification ───────────────────────────────────────────────────────

class TestResendVerification:
    async def _resend(self, client, email):
        return await client.post("/api/auth/resend-verification", json={"email": email})

    async def test_unknown_email_looks_identical_and_sends_nothing(self, client: httpx.AsyncClient, sent_codes):
        resp = await self._resend(client, "nobody@gmail.com")
        assert resp.status_code == 202
        assert sent_codes == []

    async def test_unverified_account_gets_a_code(self, client: httpx.AsyncClient, sent_codes):
        await seed_user("gina@gmail.com", verified=False)
        resp = await self._resend(client, "gina@gmail.com")
        assert resp.status_code == 202
        assert len(code_sent_to(sent_codes, "gina@gmail.com")) == 6

    async def test_verified_account_gets_nothing(self, client: httpx.AsyncClient, sent_codes):
        await seed_user("hank@gmail.com", verified=True)
        assert (await self._resend(client, "hank@gmail.com")).status_code == 202
        assert sent_codes == []

    async def test_cooldown_blocks_rapid_resends(self, client: httpx.AsyncClient, sent_codes):
        await seed_user("ivy@gmail.com", verified=False)
        await self._resend(client, "ivy@gmail.com")
        await self._resend(client, "ivy@gmail.com")
        assert len(sent_codes) == 1

        record = get_memory_collection("email_verifications")._docs[0]
        record["last_sent_at"] = datetime.now(timezone.utc) - timedelta(seconds=RESEND_COOLDOWN_SEC + 1)
        await self._resend(client, "ivy@gmail.com")
        assert len(sent_codes) == 2


# ── Change email + profile ────────────────────────────────────────────────────

class TestEmailChange:
    async def test_request_sends_code_to_new_address(self, client: httpx.AsyncClient, sent_codes):
        owner = await seed_user("owner@gmail.com")
        resp = await client.post("/api/auth/email-change", json={"email": "owner.new@gmail.com"}, headers=bearer(owner))
        assert resp.status_code == 202
        assert len(code_sent_to(sent_codes, "owner.new@gmail.com")) == 6

    async def test_request_requires_login(self, client: httpx.AsyncClient):
        resp = await client.post("/api/auth/email-change", json={"email": "x.new@gmail.com"})
        assert resp.status_code == 401

    async def test_request_rejects_non_gmail(self, client: httpx.AsyncClient):
        owner = await seed_user("owner@gmail.com")
        resp = await client.post("/api/auth/email-change", json={"email": "me@yahoo.com"}, headers=bearer(owner))
        assert resp.status_code == 422

    async def test_request_rejects_current_address(self, client: httpx.AsyncClient):
        owner = await seed_user("owner@gmail.com")
        resp = await client.post("/api/auth/email-change", json={"email": "owner@gmail.com"}, headers=bearer(owner))
        assert resp.status_code == 422

    async def test_request_rejects_address_in_use(self, client: httpx.AsyncClient):
        owner = await seed_user("owner@gmail.com")
        await seed_user("taken@gmail.com")
        resp = await client.post("/api/auth/email-change", json={"email": "taken@gmail.com"}, headers=bearer(owner))
        assert resp.status_code == 409

    async def test_confirm_changes_the_email(self, client: httpx.AsyncClient, sent_codes):
        owner = await seed_user("owner@gmail.com")
        h = bearer(owner)
        await client.post("/api/auth/email-change", json={"email": "owner.new@gmail.com"}, headers=h)
        code = code_sent_to(sent_codes, "owner.new@gmail.com")

        resp = await client.post("/api/auth/email-change/confirm", json={"code": code}, headers=h)
        assert resp.status_code == 200
        assert resp.json()["email"] == "owner.new@gmail.com"

        me = await client.get("/api/auth/me", headers=h)
        assert me.json()["email"] == "owner.new@gmail.com"

    async def test_confirm_with_wrong_code_keeps_old_email(self, client: httpx.AsyncClient, sent_codes):
        owner = await seed_user("owner@gmail.com")
        h = bearer(owner)
        await client.post("/api/auth/email-change", json={"email": "owner.new@gmail.com"}, headers=h)
        code = code_sent_to(sent_codes, "owner.new@gmail.com")

        resp = await client.post("/api/auth/email-change/confirm", json={"code": wrong_code(code)}, headers=h)
        assert resp.status_code == 400
        me = await client.get("/api/auth/me", headers=h)
        assert me.json()["email"] == "owner@gmail.com"

    async def test_confirm_without_request_fails(self, client: httpx.AsyncClient):
        owner = await seed_user("owner@gmail.com")
        resp = await client.post("/api/auth/email-change/confirm", json={"code": "123456"}, headers=bearer(owner))
        assert resp.status_code == 400

    async def test_profile_updates_name_but_ignores_email(self, client: httpx.AsyncClient):
        owner = await seed_user("owner@gmail.com")
        resp = await client.patch(
            "/api/auth/profile",
            json={"name": "New Name", "email": "sneaky@gmail.com"},
            headers=bearer(owner),
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "New Name"
        assert resp.json()["email"] == "owner@gmail.com"

    async def test_profile_requires_a_name(self, client: httpx.AsyncClient):
        owner = await seed_user("owner@gmail.com")
        resp = await client.patch("/api/auth/profile", json={}, headers=bearer(owner))
        assert resp.status_code == 422


# ── Logout ────────────────────────────────────────────────────────────────────

class TestLogout:
    async def test_logout(self, client: httpx.AsyncClient, auth_headers: dict):
        resp = await client.post("/api/auth/logout", headers=auth_headers)
        assert resp.status_code in (200, 204)