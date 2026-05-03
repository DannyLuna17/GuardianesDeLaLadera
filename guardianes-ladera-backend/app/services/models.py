from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import ApiError
from app.ml.model_registry import ModelRegistry
from app.schemas.admin import ModelArtifactDetailRead, ModelArtifactSummaryRead


class ModelService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.registry = ModelRegistry()

    @staticmethod
    def _artifact_path(version: str, artifacts_path: Path) -> Path:
        return artifacts_path / f"{version}.json"

    def list_models(self) -> list[ModelArtifactSummaryRead]:
        models: list[ModelArtifactSummaryRead] = []
        active_version = self.registry.active_version()
        for version in self.registry.list_versions():
            artifact = self.registry.load(version)
            training = artifact.get("training", {})
            models.append(
                ModelArtifactSummaryRead(
                    version=version,
                    modelId=artifact.get("model_id", "unknown"),
                    artifactType=artifact.get("artifact_type", "weighted_sum"),
                    modelFamily=artifact.get("model_family", "baseline"),
                    description=artifact.get("description"),
                    active=version == active_version,
                    featureCount=len(artifact.get("feature_order", [])),
                    trainedAt=training.get("trained_at"),
                    dataset=training.get("dataset"),
                    rows=training.get("rows"),
                    metrics=training.get("metrics") or {},
                )
            )
        return sorted(models, key=lambda item: (not item.active, item.version))

    def get_model(self, version: str) -> ModelArtifactDetailRead:
        if version not in self.registry.list_versions():
            raise ApiError(
                404, "model_not_found", f"Model artifact '{version}' was not found."
            )
        artifact = self.registry.load(version)
        training = artifact.get("training", {})
        active_version = self.registry.active_version()
        return ModelArtifactDetailRead(
            version=version,
            modelId=artifact.get("model_id", "unknown"),
            artifactType=artifact.get("artifact_type", "weighted_sum"),
            modelFamily=artifact.get("model_family", "baseline"),
            description=artifact.get("description"),
            active=version == active_version,
            artifactPath=str(
                self._artifact_path(
                    version, self.settings.resolved_model_artifacts_path
                )
            ),
            featureOrder=artifact.get("feature_order", []),
            bounds=artifact.get("bounds", {}),
            freshnessPenalties=artifact.get("freshness_penalties", {}),
            training=training,
            calibration=artifact.get("calibration", {}),
            metrics=training.get("metrics") or {},
        )

    def promote_model(
        self,
        version: str,
        *,
        promoted_by: str,
        reason: str | None = None,
        source: str = "manual",
    ) -> dict:
        manifest = self.registry.set_active_version(
            version,
            promoted_by=promoted_by,
            reason=reason,
            source=source,
        )
        return {
            "modelVersion": version,
            "activeModelVersion": self.registry.active_version(),
            "previousActiveModelVersion": self.registry.previous_active_version(),
            "manifestPath": str(self.settings.resolved_active_model_manifest_path),
            "promotedAt": manifest["promoted_at"],
            "promotedBy": manifest.get("promoted_by"),
            "reason": manifest.get("reason"),
            "source": manifest.get("source"),
        }

    def rollback_model(
        self, *, rolled_back_by: str, reason: str | None = None, source: str = "manual"
    ) -> dict:
        manifest = self.registry.rollback_active_version(
            rolled_back_by=rolled_back_by,
            reason=reason,
            source=source,
        )
        return {
            "modelVersion": manifest["active_model_version"],
            "activeModelVersion": manifest["active_model_version"],
            "rolledBackFromModelVersion": manifest["rolled_back_from_model_version"],
            "previousActiveModelVersion": manifest.get("previous_active_model_version"),
            "manifestPath": str(self.settings.resolved_active_model_manifest_path),
            "promotedAt": manifest["promoted_at"],
            "promotedBy": manifest.get("promoted_by"),
            "reason": manifest.get("reason"),
            "source": manifest.get("source"),
        }
