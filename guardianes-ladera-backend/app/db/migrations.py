from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from app.core.config import get_settings


def get_alembic_config(database_url: str | None = None) -> Config:
    settings = get_settings()
    backend_root = settings.backend_root
    config = Config(str(backend_root / "alembic.ini"))
    config.set_main_option("script_location", str(backend_root / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url or settings.database_url)
    config.set_main_option("prepend_sys_path", str(backend_root))
    return config


def migrations_directory() -> Path:
    return get_settings().backend_root / "alembic"


def upgrade_to_head(database_url: str | None = None) -> None:
    command.upgrade(get_alembic_config(database_url=database_url), "head")
