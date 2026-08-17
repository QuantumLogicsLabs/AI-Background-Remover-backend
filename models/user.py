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
    """Payload for PATCH /api/auth/profile — all fields optional."""
    name:  Optional[str]      = Field(None, min_length=2, max_length=80)
    email: Optional[EmailStr] = None


class UpdatePasswordRequest(BaseModel):
    """Payload for PATCH /api/auth/password."""
    current_password: str = Field(..., min_length=1)
    new_password:     str = Field(..., min_length=8, max_length=128)


class DeleteAccountRequest(BaseModel):
    """Payload for DELETE /api/auth/account — requires password confirmation."""
    password: str = Field(..., min_length=1)


class ForgotPasswordRequest(BaseModel):
    """Payload for POST /api/auth/forgot-password."""
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    """Payload for POST /api/auth/reset-password."""
    token:        str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)
