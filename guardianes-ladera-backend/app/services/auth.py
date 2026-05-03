from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.exceptions import ApiError
from app.core.security import create_access_token, hash_password, verify_password
from app.models import JobExecution, UserAccount
from app.schemas.auth import TokenResponse, UserRead


class AuthService:
    def __init__(self, session: Session) -> None:
        self.session = session

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc).replace(microsecond=0)

    def _record_job(self, *, job_type: str, details: dict, when: datetime) -> None:
        self.session.add(
            JobExecution(
                job_type=job_type,
                status="completed",
                started_at=when,
                completed_at=when,
                details=details,
            )
        )

    @staticmethod
    def _normalize_role(role: str) -> str:
        normalized = role.strip().lower()
        if not normalized:
            raise ApiError(400, "invalid_user_role", "User role must not be blank.")
        return normalized

    def _get_user_by_username_or_404(self, username: str) -> UserAccount:
        statement = select(UserAccount).where(UserAccount.username == username)
        user = self.session.scalar(statement)
        if user is None:
            raise ApiError(
                404,
                "user_not_found",
                f"User account '{username}' was not found.",
            )
        return user

    def _active_admin_count(self) -> int:
        statement = select(UserAccount).where(
            UserAccount.role == "admin", UserAccount.is_active.is_(True)
        )
        return len(list(self.session.scalars(statement).all()))

    def authenticate(self, username: str, password: str) -> TokenResponse:
        statement = select(UserAccount).where(UserAccount.username == username)
        user = self.session.scalar(statement)
        if user is None or not verify_password(password, user.password_hash):
            raise ApiError(401, "invalid_credentials", "Invalid username or password.")
        if not user.is_active:
            raise ApiError(403, "user_inactive", "This user account is inactive.")
        token, expires_at = create_access_token(subject=user.username, role=user.role)
        return TokenResponse(
            accessToken=token,
            tokenType="bearer",
            expiresAt=expires_at,
            role=user.role,
            username=user.username,
        )

    def get_user_by_username(self, username: str) -> UserAccount:
        statement = select(UserAccount).where(UserAccount.username == username)
        user = self.session.scalar(statement)
        if user is None:
            raise ApiError(401, "invalid_token", "User referenced by token no longer exists.")
        return user

    def current_user_read(self, user: UserAccount) -> UserRead:
        return UserRead(
            id=user.id,
            username=user.username,
            role=user.role,
            isActive=user.is_active,
        )

    def list_users(self) -> list[UserAccount]:
        statement = select(UserAccount).order_by(UserAccount.username)
        return list(self.session.scalars(statement).all())

    def create_user(
        self,
        *,
        username: str,
        password: str,
        role: str,
        is_active: bool,
        created_by: str,
    ) -> UserAccount:
        normalized_username = username.strip()
        if not normalized_username:
            raise ApiError(
                400, "invalid_username", "Username must not be blank."
            )
        existing_user = self.session.scalar(
            select(UserAccount).where(UserAccount.username == normalized_username)
        )
        if existing_user is not None:
            raise ApiError(
                409,
                "user_already_exists",
                f"User account '{normalized_username}' already exists.",
            )

        now = self._now()
        user = UserAccount(
            username=normalized_username,
            password_hash=hash_password(password),
            role=self._normalize_role(role),
            is_active=is_active,
            created_at=now,
        )
        self.session.add(user)
        self.session.flush()
        self._record_job(
            job_type="auth_user_create",
            details={
                "actor": created_by,
                "target_username": normalized_username,
                "role": user.role,
                "is_active": is_active,
            },
            when=now,
        )
        self.session.commit()
        self.session.refresh(user)
        return user

    def update_user(
        self,
        *,
        username: str,
        updated_by: str,
        role: str | None,
        is_active: bool | None,
    ) -> UserAccount:
        user = self._get_user_by_username_or_404(username)
        previous_role = user.role
        previous_is_active = user.is_active

        if role is not None:
            normalized_role = self._normalize_role(role)
            if (
                previous_role == "admin"
                and normalized_role != "admin"
                and user.is_active
                and self._active_admin_count() <= 1
            ):
                raise ApiError(
                    400,
                    "last_admin_role_change_not_allowed",
                    "The last active admin account cannot lose the admin role.",
                )
            user.role = normalized_role

        if is_active is not None:
            if (
                user.role == "admin"
                and user.is_active
                and not is_active
                and self._active_admin_count() <= 1
            ):
                raise ApiError(
                    400,
                    "last_admin_deactivation_not_allowed",
                    "The last active admin account cannot be deactivated.",
                )
            user.is_active = is_active

        now = self._now()
        self._record_job(
            job_type="auth_user_update",
            details={
                "actor": updated_by,
                "target_username": user.username,
                "previous_role": previous_role,
                "next_role": user.role,
                "previous_is_active": previous_is_active,
                "next_is_active": user.is_active,
            },
            when=now,
        )
        self.session.commit()
        self.session.refresh(user)
        return user

    def reset_password(
        self,
        *,
        username: str,
        new_password: str,
        rotated_by: str,
    ) -> UserAccount:
        user = self._get_user_by_username_or_404(username)
        user.password_hash = hash_password(new_password)
        now = self._now()
        self._record_job(
            job_type="auth_password_reset",
            details={
                "actor": rotated_by,
                "target_username": user.username,
            },
            when=now,
        )
        self.session.commit()
        self.session.refresh(user)
        return user

    def change_password(
        self,
        *,
        username: str,
        current_password: str,
        new_password: str,
    ) -> datetime:
        user = self._get_user_by_username_or_404(username)
        if not verify_password(current_password, user.password_hash):
            raise ApiError(
                401,
                "invalid_current_password",
                "The current password is incorrect.",
            )
        if verify_password(new_password, user.password_hash):
            raise ApiError(
                400,
                "password_rotation_requires_new_secret",
                "The new password must differ from the current password.",
            )

        user.password_hash = hash_password(new_password)
        now = self._now()
        self._record_job(
            job_type="auth_password_change",
            details={"actor": user.username, "target_username": user.username},
            when=now,
        )
        self.session.commit()
        return now
