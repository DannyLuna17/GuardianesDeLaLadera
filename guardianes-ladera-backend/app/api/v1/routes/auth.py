from fastapi import APIRouter, Depends, Request

from app.api.v1.dependencies.auth import get_auth_service, get_current_user
from app.core.rate_limit import get_client_ip, login_limiter
from app.schemas.auth import (
    ChangePasswordRequest,
    ChangePasswordResponse,
    LoginRequest,
    TokenResponse,
    UserRead,
)
from app.services.auth import AuthService

router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(
    request: Request,
    payload: LoginRequest,
    service: AuthService = Depends(get_auth_service),
):
    login_limiter.check(get_client_ip(request))
    return service.authenticate(payload.username, payload.password)


@router.get("/me", response_model=UserRead)
def me(
    current_user=Depends(get_current_user),
    service: AuthService = Depends(get_auth_service),
):
    return service.current_user_read(current_user)


@router.post("/change-password", response_model=ChangePasswordResponse)
def change_password(
    payload: ChangePasswordRequest,
    current_user=Depends(get_current_user),
    service: AuthService = Depends(get_auth_service),
):
    changed_at = service.change_password(
        username=current_user.username,
        current_password=payload.current_password,
        new_password=payload.new_password,
    )
    return ChangePasswordResponse(username=current_user.username, changedAt=changed_at)
