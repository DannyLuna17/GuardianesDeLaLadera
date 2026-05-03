from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SchemaBase(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        protected_namespaces=(),
    )


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(alias="currentPassword", min_length=8, max_length=256)
    new_password: str = Field(alias="newPassword", min_length=8, max_length=256)


class TokenResponse(SchemaBase):
    access_token: str = Field(alias="accessToken")
    token_type: str = Field(alias="tokenType")
    expires_at: datetime = Field(alias="expiresAt")
    role: str
    username: str


class UserRead(SchemaBase):
    id: int
    username: str
    role: str
    is_active: bool = Field(alias="isActive")


class ChangePasswordResponse(SchemaBase):
    username: str
    changed_at: datetime = Field(alias="changedAt")
