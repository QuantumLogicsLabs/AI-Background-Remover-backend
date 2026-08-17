"""
Email Service.

Sends transactional emails (e.g. password-reset) via SMTP.

Required environment variables (set in .env):
  SMTP_HOST     — SMTP server hostname  (e.g. smtp.gmail.com)
  SMTP_PORT     — SMTP port             (default 587)
  SMTP_USER     — SMTP login username
  SMTP_PASSWORD — SMTP login password
  SMTP_FROM     — Sender "From" address (defaults to SMTP_USER)
  FRONTEND_URL  — Base URL of the frontend (e.g. http://localhost:5173)

If SMTP_HOST / SMTP_USER / SMTP_PASSWORD are not set, the service prints
the reset link to stdout so development works without an email provider.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText


# ── Config ────────────────────────────────────────────────────────────────────

_SMTP_HOST     = os.getenv("SMTP_HOST", "")
_SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
_SMTP_USER     = os.getenv("SMTP_USER", "")
_SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
_SMTP_FROM     = os.getenv("SMTP_FROM", _SMTP_USER)
_FRONTEND_URL  = os.getenv("FRONTEND_URL", "http://localhost:5173")


# ── Internal sender ───────────────────────────────────────────────────────────

def _send_smtp(to: str, subject: str, html: str, plain: str) -> None:
    """Send an email via SMTP with TLS. Raises on failure."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = _SMTP_FROM
    msg["To"]      = to

    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html,  "html"))

    ctx = ssl.create_default_context()
    with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT) as server:
        server.ehlo()
        server.starttls(context=ctx)
        server.login(_SMTP_USER, _SMTP_PASSWORD)
        server.sendmail(_SMTP_FROM, to, msg.as_string())


# ── Public API ────────────────────────────────────────────────────────────────

async def send_password_reset_email(to_email: str, reset_token: str) -> None:
    """
    Send a password-reset email to *to_email* containing a link with
    *reset_token* embedded as a query parameter.

    Falls back to printing the link to stdout when SMTP is not configured.
    """
    reset_url = f"{_FRONTEND_URL}/reset-password?token={reset_token}"
    subject   = "Reset your AI Background Remover password"

    plain = (
        f"Hi,\n\n"
        f"We received a request to reset your password.\n\n"
        f"Click the link below (valid for 1 hour):\n{reset_url}\n\n"
        f"If you did not request a password reset, you can safely ignore this email.\n\n"
        f"— AI Background Remover Team"
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    body  {{ font-family: Arial, sans-serif; background: #0f0f11; color: #e2e2e2; margin: 0; padding: 0; }}
    .wrap {{ max-width: 480px; margin: 40px auto; background: #1a1a1f; border-radius: 12px;
             padding: 40px; border: 1px solid #2a2a35; }}
    h2    {{ color: #e040fb; margin-top: 0; }}
    p     {{ line-height: 1.6; color: #b0b0c0; }}
    .btn  {{ display: inline-block; margin-top: 24px; padding: 12px 28px;
             background: #e040fb; color: #fff; text-decoration: none;
             border-radius: 8px; font-weight: 700; font-size: 15px; }}
    .note {{ font-size: 12px; color: #666; margin-top: 32px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h2>Password Reset</h2>
    <p>We received a request to reset the password for your AI Background Remover account.</p>
    <p>Click the button below. The link is valid for <strong>1 hour</strong>.</p>
    <a href="{reset_url}" class="btn">Reset Password</a>
    <p class="note">
      If you didn't request a password reset, you can safely ignore this email.<br/>
      This link will expire automatically.
    </p>
  </div>
</body>
</html>"""

    smtp_configured = all([_SMTP_HOST, _SMTP_USER, _SMTP_PASSWORD])

    if not smtp_configured:
        # Dev fallback — log to console
        print(
            f"\n[EMAIL] ── Password Reset (DEV MODE) ──────────────────────────\n"
            f"  To:       {to_email}\n"
            f"  Subject:  {subject}\n"
            f"  Link:     {reset_url}\n"
            f"────────────────────────────────────────────────────────────────\n"
        )
        return

    try:
        _send_smtp(to_email, subject, html, plain)
        print(f"[EMAIL] Password reset email sent to {to_email}")
    except Exception as exc:
        # Log but do NOT expose SMTP errors to the caller (security)
        print(f"[EMAIL] Failed to send reset email to {to_email}: {exc}")
        raise
