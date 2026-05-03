from __future__ import annotations

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.exceptions import ApiError
from app.core.security import decode_access_token
from app.db.session import get_db
from app.models import UserAccount
from app.services.auth import AuthService

bearer_scheme = HTTPBearer(auto_error=False)


def get_auth_service(session: Session = Depends(get_db)) -> AuthService:
    return AuthService(session)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserAccount:
    if credentials is None:
        raise ApiError(401, "missing_token", "Authorization bearer token is required.")
    try:
        payload = decode_access_token(credentials.credentials)
    except Exception as exc:
        raise ApiError(401, "invalid_token", "Bearer token is invalid or expired.") from exc
    user = auth_service.get_user_by_username(payload["sub"])
    if not user.is_active:
        raise ApiError(403, "user_inactive", "This user account is inactive.")
    return user


def require_admin(user: UserAccount = Depends(get_current_user)) -> UserAccount:
    if user.role != "admin":
        raise ApiError(403, "forbidden", "Admin role is required for this operation.")
    return user
