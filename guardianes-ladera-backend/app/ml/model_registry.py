from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

from app.core.config import get_settings
from app.core.exceptions import ApiError


class ModelRegistry:
    def __init__(
        self, artifacts_path: Path | None = None, manifest_path: Path | None = None
    ) -> None:
        settings = get_settings()
        self.artifacts_path = artifacts_path or settings.resolved_model_artifacts_path
        self.fallback_artifacts_path = (
            settings.backend_root / "app" / "ml" / "artifacts"
        )
        self.manifest_path = (
            manifest_path or settings.resolved_active_model_manifest_path
        )
        self.settings = settings

    @lru_cache
    def load(self, version: str) -> dict[str, Any]:
        artifact_path = self.artifacts_path / f"{version}.json"
        if not artifact_path.exists():
            fallback_artifact_path = self.fallback_artifacts_path / f"{version}.json"
            if fallback_artifact_path.exists():
                artifact_path = fallback_artifact_path
            else:
                raise ApiError(
                    500,
                    "model_artifact_missing",
                    f"Model artifact '{version}' was not found.",
                )
        with artifact_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def list_versions(self) -> list[str]:
        versions = {path.stem for path in self.artifacts_path.glob("*.json")}
        versions.update(
            path.stem for path in self.fallback_artifacts_path.glob("*.json")
        )
        return sorted(versions)

    def active_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        with self.manifest_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def active_version(self) -> str:
        manifest = self.active_manifest()
        if manifest and manifest.get("active_model_version"):
            return str(manifest["active_model_version"])
        return self.settings.model_version

    def previous_active_version(self) -> str | None:
        manifest = self.active_manifest()
        if manifest and manifest.get("previous_active_model_version"):
            return str(manifest["previous_active_model_version"])
        return None

    def load_active(self) -> dict[str, Any]:
        return self.load(self.active_version())

    def set_active_version(
        self,
        version: str,
        *,
        promoted_by: str | None = None,
        reason: str | None = None,
        source: str = "manual",
    ) -> dict[str, Any]:
        if version not in self.list_versions():
            raise ApiError(
                404, "model_not_found", f"Model artifact '{version}' was not found."
            )

        promoted_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        current_active_version = self.active_version()
        existing_manifest = self.active_manifest() or {}
        history = list(existing_manifest.get("history") or [])
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "active_model_version": version,
            "previous_active_model_version": current_active_version
            if current_active_version != version
            else existing_manifest.get("previous_active_model_version"),
            "promoted_at": promoted_at,
            "promoted_by": promoted_by,
            "reason": reason,
            "source": source,
            "history": history
            + [
                {
                    "version": version,
                    "previous_active_model_version": current_active_version,
                    "promoted_at": promoted_at,
                    "promoted_by": promoted_by,
                    "reason": reason,
                    "source": source,
                }
            ],
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def rollback_active_version(
        self,
        *,
        rolled_back_by: str | None = None,
        reason: str | None = None,
        source: str = "manual",
    ) -> dict[str, Any]:
        current_active_version = self.active_version()
        previous_active_version = self.previous_active_version()
        if previous_active_version is None:
            raise ApiError(
                400,
                "model_rollback_unavailable",
                "No previous active model is available for rollback.",
            )
        manifest = self.set_active_version(
            previous_active_version,
            promoted_by=rolled_back_by,
            reason=reason,
            source=f"{source}:rollback",
        )
        manifest["rolled_back_from_model_version"] = current_active_version
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
