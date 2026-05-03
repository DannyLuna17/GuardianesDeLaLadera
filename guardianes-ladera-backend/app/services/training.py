from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import ApiError
from app.ml.datasets import TrainingDatasetRegistry, rows_from_dataset
from app.ml.model_registry import ModelRegistry
from app.ml.training import (
    export_additive_spline_artifact,
    export_beta_regression_artifact,
    export_gradient_boosted_tree_artifact,
    export_seed_linear_artifact,
    export_xgboost_artifact,
)
from app.models import JobExecution
from app.schemas.admin import JobExecutionRead, RetrainModelResponse


class TrainingService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.settings = get_settings()
        self.dataset_registry = TrainingDatasetRegistry()
        self.model_registry = ModelRegistry()

    @staticmethod
    def _job_read(job: JobExecution) -> JobExecutionRead:
        return JobExecutionRead(
            id=job.id,
            jobType=job.job_type,
            status=job.status,
            startedAt=job.started_at,
            completedAt=job.completed_at,
            details=job.details or {},
        )

    def retrain_seed_model(
        self,
        *,
        version: str | None = None,
        alpha: float = 0.75,
        model_family: str = "linear_ridge",
        knot_count: int | None = None,
        learning_rate: float | None = None,
        estimator_count: int | None = None,
        max_depth: int | None = None,
        min_leaf_size: int | None = None,
        min_split_gain: float | None = None,
        early_stopping_rounds: int | None = None,
        dataset_version: str | None = None,
        origin: str = "manual",
    ) -> RetrainModelResponse:
        normalized_model_family = str(model_family or "linear_ridge").strip()
        if normalized_model_family not in {
            "linear_ridge",
            "beta_regression",
            "additive_spline",
            "gradient_boosted_tree",
            "xgboost",
        }:
            raise ApiError(
                400,
                "invalid_model_family",
                f"Unsupported model family: {model_family}",
            )
        if normalized_model_family == "additive_spline":
            if knot_count is not None and knot_count < 0:
                raise ApiError(
                    400,
                    "invalid_knot_count",
                    "knotCount must be zero or greater.",
                )
        if normalized_model_family in {"gradient_boosted_tree", "xgboost"}:
            if learning_rate is not None and (learning_rate <= 0 or learning_rate > 1):
                raise ApiError(
                    400,
                    "invalid_learning_rate",
                    "learningRate must be greater than zero and at most one.",
                )
            if estimator_count is not None and estimator_count <= 0:
                raise ApiError(
                    400,
                    "invalid_estimator_count",
                    "estimatorCount must be greater than zero.",
                )
            if max_depth is not None and max_depth <= 0:
                raise ApiError(
                    400,
                    "invalid_max_depth",
                    "maxDepth must be greater than zero.",
                )
            if min_leaf_size is not None and min_leaf_size <= 0:
                raise ApiError(
                    400,
                    "invalid_min_leaf_size",
                    "minLeafSize must be greater than zero.",
                )
            if min_split_gain is not None and min_split_gain < 0:
                raise ApiError(
                    400,
                    "invalid_min_split_gain",
                    "minSplitGain must be zero or greater.",
                )
        if self.settings.real_data_only and dataset_version is None:
            raise ApiError(
                409,
                "seed_model_retraining_disabled",
                "Model retraining now requires an explicit non-seed datasetVersion because REAL_DATA_ONLY is enabled.",
            )
        target_version = version or self.settings.model_version
        started_at = datetime.now(timezone.utc).replace(microsecond=0)
        job = JobExecution(
            job_type="model_retrain",
            status="running",
            started_at=started_at,
            details={
                "version": target_version,
                "alpha": alpha,
                "model_family": normalized_model_family,
                "hyperparameters": {
                    "alpha": (
                        alpha
                        if normalized_model_family
                        in {"additive_spline", "beta_regression"}
                        else None
                    ),
                    "knot_count": knot_count,
                    "learning_rate": learning_rate,
                    "estimator_count": estimator_count,
                    "max_depth": max_depth,
                    "min_leaf_size": min_leaf_size,
                    "min_split_gain": min_split_gain,
                    "early_stopping_rounds": early_stopping_rounds,
                },
                "origin": origin,
                "dataset_version": dataset_version or "frontend_seed_bootstrap",
            },
        )
        self.session.add(job)
        self.session.flush()

        try:
            training_rows = None
            dataset_name = "frontend_seed_bootstrap"
            if dataset_version is not None:
                dataset = self.dataset_registry.load(dataset_version)
                provenance = dataset.get("provenance") or {}
                dataset_mode = str(
                    provenance.get("dataset_mode")
                    or provenance.get("source")
                    or "unknown"
                )
                if self.settings.real_data_only and dataset_mode == "seed":
                    raise ApiError(
                        409,
                        "seed_dataset_disabled",
                        "Seed-backed model retraining is disabled because REAL_DATA_ONLY is enabled.",
                    )
                training_rows = rows_from_dataset(dataset)
                dataset_name = dataset_version
            if normalized_model_family == "linear_ridge":
                artifact_path, artifact = export_seed_linear_artifact(
                    version=target_version,
                    alpha=alpha,
                    rows=training_rows,
                    dataset_name=dataset_name,
                )
            elif normalized_model_family == "beta_regression":
                artifact_path, artifact = export_beta_regression_artifact(
                    version=target_version,
                    alpha=alpha,
                    rows=training_rows,
                    dataset_name=dataset_name,
                )
            elif normalized_model_family == "additive_spline":
                artifact_path, artifact = export_additive_spline_artifact(
                    version=target_version,
                    rows=training_rows,
                    dataset_name=dataset_name,
                    alpha=alpha,
                    knot_count=knot_count if knot_count is not None else 3,
                )
            elif normalized_model_family == "xgboost":
                artifact_path, artifact = export_xgboost_artifact(
                    version=target_version,
                    rows=training_rows,
                    dataset_name=dataset_name,
                    learning_rate=learning_rate if learning_rate is not None else 0.05,
                    estimator_count=estimator_count if estimator_count is not None else 64,
                    max_depth=max_depth if max_depth is not None else 4,
                    min_leaf_size=min_leaf_size if min_leaf_size is not None else 1,
                    min_split_gain=min_split_gain if min_split_gain is not None else 0.0,
                    early_stopping_rounds=(
                        early_stopping_rounds
                        if early_stopping_rounds is not None
                        else 8
                    ),
                )
            else:
                artifact_path, artifact = export_gradient_boosted_tree_artifact(
                    version=target_version,
                    rows=training_rows,
                    dataset_name=dataset_name,
                    learning_rate=learning_rate if learning_rate is not None else 0.1,
                    estimator_count=estimator_count if estimator_count is not None else 24,
                    max_depth=max_depth if max_depth is not None else 3,
                    min_leaf_size=min_leaf_size if min_leaf_size is not None else 2,
                    min_split_gain=min_split_gain if min_split_gain is not None else 0.0,
                    early_stopping_rounds=(
                        early_stopping_rounds
                        if early_stopping_rounds is not None
                        else 4
                    ),
                )
        except Exception as exc:
            completed_at = datetime.now(timezone.utc).replace(microsecond=0)
            job.status = "failed"
            job.completed_at = completed_at
            job.details = {
                "version": target_version,
                "alpha": alpha,
                "model_family": normalized_model_family,
                "hyperparameters": {
                    "alpha": (
                        alpha
                        if normalized_model_family
                        in {"additive_spline", "beta_regression"}
                        else None
                    ),
                    "knot_count": knot_count,
                    "learning_rate": learning_rate,
                    "estimator_count": estimator_count,
                    "max_depth": max_depth,
                    "min_leaf_size": min_leaf_size,
                    "min_split_gain": min_split_gain,
                    "early_stopping_rounds": early_stopping_rounds,
                },
                "origin": origin,
                "dataset_version": dataset_version or "frontend_seed_bootstrap",
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            self.session.commit()
            raise

        completed_at = datetime.now(timezone.utc).replace(microsecond=0)
        job.status = "completed"
        job.completed_at = completed_at
        job.details = {
            "version": target_version,
            "alpha": alpha,
            "model_family": normalized_model_family,
            "hyperparameters": (artifact.get("training") or {}).get("hyperparameters")
            or {},
            "origin": origin,
            "dataset_version": artifact["training"]["dataset"],
            "artifact_path": str(artifact_path),
            "rows": artifact["training"]["rows"],
            "metrics": artifact["training"]["metrics"],
        }
        self.session.commit()
        self.session.refresh(job)

        return RetrainModelResponse(
            job=self._job_read(job),
            modelVersion=target_version,
            modelFamily=str(artifact.get("model_family") or normalized_model_family),
            artifactPath=str(artifact_path),
            rows=artifact["training"]["rows"],
            alpha=(
                alpha
                if normalized_model_family
                in {"linear_ridge", "additive_spline", "beta_regression"}
                else None
            ),
            hyperparameters=(artifact.get("training") or {}).get("hyperparameters")
            or {},
            datasetVersion=artifact["training"]["dataset"],
            featureOrder=artifact["feature_order"],
            metrics=artifact["training"]["metrics"],
            activeModelVersion=self.model_registry.active_version(),
            overwroteActiveVersion=target_version
            == self.model_registry.active_version(),
        )
