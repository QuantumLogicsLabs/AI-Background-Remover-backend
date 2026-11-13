"""
User Pydantic schemas.

UserCreate             — payload for POST /api/auth/register
UserLogin              — payload for POST /api/auth/login
UserOut                — safe public representation (no password hash)
UserInDB               — full internal representation (includes hashed_password)
TokenResponse          — JWT response body
TokenData              — decoded JWT claims
UpdateProfileRequest   — payload for PATCH /api/auth/profile
UpdatePasswordRequest  — payload for PATCH /api/auth/password
DeleteAccountRequest   — payload for DELETE /api/auth/account
ForgotPasswordRequest  — payload for POST /api/auth/forgot-password
ResetPasswordRequest   — payload for POST /api/auth/reset-password
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    name:     str       = Field(..., min_length=2,  max_length=80)
    email:    EmailStr
    password: str       = Field(..., min_length=8,  max_length=128)


class UserLogin(BaseModel):
    email:    EmailStr
    password: str       = Field(..., min_length=1)


class UserOut(BaseModel):
    user_id:    str
    name:       str
    email:      str
    created_at: datetime


class UserInDB(UserOut):
    hashed_password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user:         UserOut


class TokenData(BaseModel):
    user_id:  Optional[str] = None
    email:    Optional[str] = None


class UpdateProfileRequest(BaseModel):
    """Payload for PATCH /api/auth/profile (name only; email changes need verification)."""
    name: Optional[str] = Field(None, min_length=2, max_length=80)


class UpdatePasswordRequest(BaseModel):
    """Payload for PATCH /api/auth/password."""
    current_password: str = Field(..., min_length=1)
    new_password:     str = Field(..., min_length=8, max_length=128)


class DeleteAccountRequest(BaseModel):
    """Payload for DELETE /api/auth/account — password required unless the account
    was created via Google/GitHub and has no password."""
    password: Optional[str] = Field(None, min_length=1)


class ForgotPasswordRequest(BaseModel):
    """Payload for POST /api/auth/forgot-password."""
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    """Payload for POST /api/auth/reset-password."""
    token:        str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)

class RegisterResponse(BaseModel):
    """Sign-up does not log the user in; they must verify their email first."""
    email: str
    verification_required: bool = True


class VerifyEmailRequest(BaseModel):
    email: EmailStr
    code:  str = Field(..., min_length=6, max_length=6)


class ResendVerificationRequest(BaseModel):
    email: EmailStr


class EmailChangeRequest(BaseModel):
    email: EmailStr


class ConfirmEmailChangeRequest(BaseModel):
    code: str = Field(..., min_length=6, max_length=6)
