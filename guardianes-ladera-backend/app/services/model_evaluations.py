from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import ApiError
from app.ml.datasets import TrainingDatasetRegistry
from app.ml.model_evaluations import (
    ModelEvaluationRegistry,
    build_model_evaluation,
    export_model_evaluation,
)
from app.ml.model_registry import ModelRegistry
from app.models import JobExecution
from app.schemas.admin import (
    EvaluateModelResponse,
    JobExecutionRead,
    ModelEvaluationDetailRead,
    ModelEvaluationSummaryRead,
)


class ModelEvaluationService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.settings = get_settings()
        self.model_registry = ModelRegistry()
        self.dataset_registry = TrainingDatasetRegistry()
        self.evaluation_registry = ModelEvaluationRegistry()

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

    def _evaluation_path(self, version: str) -> Path:
        return self.settings.resolved_model_evaluations_path / f"{version}.json"

    @staticmethod
    def _summary_read(version: str, evaluation: dict) -> ModelEvaluationSummaryRead:
        overall = evaluation["metrics"]["overall"]
        validation = evaluation["metrics"]["validation"]
        return ModelEvaluationSummaryRead(
            version=version,
            evaluationId=evaluation.get("evaluation_id", "unknown"),
            artifactType=evaluation.get("artifact_type", "model_evaluation"),
            modelVersion=evaluation["model_version"],
            datasetVersion=evaluation["dataset_version"],
            evaluatedAt=evaluation["evaluated_at"],
            rows=overall["rows"],
            overallRmse=overall["calibrated_metrics"]["rmse"],
            validationRmse=validation["calibrated_metrics"]["rmse"],
            overallRiskLevelAccuracy=overall["risk_level_accuracy"],
            validationRiskLevelAccuracy=validation["risk_level_accuracy"],
        )

    def list_evaluations(self) -> list[ModelEvaluationSummaryRead]:
        evaluations = []
        for version in self.evaluation_registry.list_versions():
            evaluation = self.evaluation_registry.load(version)
            evaluations.append(self._summary_read(version, evaluation))
        return sorted(evaluations, key=lambda item: item.evaluated_at, reverse=True)

    def get_evaluation(self, version: str) -> ModelEvaluationDetailRead:
        evaluation = self.evaluation_registry.load(version)
        return ModelEvaluationDetailRead(
            version=version,
            evaluationId=evaluation.get("evaluation_id", "unknown"),
            artifactType=evaluation.get("artifact_type", "model_evaluation"),
            artifactPath=str(self._evaluation_path(version)),
            modelVersion=evaluation["model_version"],
            datasetVersion=evaluation["dataset_version"],
            evaluatedAt=evaluation["evaluated_at"],
            modelSummary=evaluation.get("model_summary") or {},
            datasetSummary=evaluation.get("dataset_summary") or {},
            metrics=evaluation.get("metrics") or {},
            diagnostics=evaluation.get("diagnostics") or {},
            topErrors=evaluation.get("top_errors") or [],
        )

    def evaluate_model(
        self,
        *,
        model_version: str,
        dataset_version: str,
        version: str | None = None,
        top_error_count: int = 10,
        origin: str = "manual",
    ) -> EvaluateModelResponse:
        evaluation_version = version or f"eval-{model_version}-on-{dataset_version}"
        started_at = datetime.now(timezone.utc).replace(microsecond=0)
        job = JobExecution(
            job_type="model_evaluation",
            status="running",
            started_at=started_at,
            details={
                "origin": origin,
                "model_version": model_version,
                "dataset_version": dataset_version,
                "evaluation_version": evaluation_version,
                "top_error_count": top_error_count,
            },
        )
        self.session.add(job)
        self.session.flush()

        try:
            artifact = self.model_registry.load(model_version)
            dataset = self.dataset_registry.load(dataset_version)
            provenance = dataset.get("provenance") or {}
            dataset_mode = str(
                provenance.get("dataset_mode") or provenance.get("source") or "unknown"
            )
            if self.settings.real_data_only and dataset_mode == "seed":
                raise ApiError(
                    409,
                    "seed_dataset_disabled",
                    "Model evaluation on seed-backed datasets is disabled because REAL_DATA_ONLY is enabled.",
                )
            evaluation = build_model_evaluation(
                version=evaluation_version,
                artifact=artifact,
                dataset=dataset,
                top_error_count=top_error_count,
            )
            evaluation_path, _ = export_model_evaluation(evaluation)
        except Exception as exc:
            completed_at = datetime.now(timezone.utc).replace(microsecond=0)
            job.status = "failed"
            job.completed_at = completed_at
            job.details = {
                "origin": origin,
                "model_version": model_version,
                "dataset_version": dataset_version,
                "evaluation_version": evaluation_version,
                "top_error_count": top_error_count,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            self.session.commit()
            raise

        completed_at = datetime.now(timezone.utc).replace(microsecond=0)
        job.status = "completed"
        job.completed_at = completed_at
        job.details = {
            "origin": origin,
            "model_version": model_version,
            "dataset_version": dataset_version,
            "evaluation_version": evaluation_version,
            "top_error_count": top_error_count,
            "artifact_path": str(evaluation_path),
            "overall_rmse": evaluation["metrics"]["overall"]["calibrated_metrics"][
                "rmse"
            ],
            "validation_rmse": evaluation["metrics"]["validation"][
                "calibrated_metrics"
            ]["rmse"],
            "validation_risk_level_accuracy": evaluation["metrics"]["validation"][
                "risk_level_accuracy"
            ],
        }
        self.session.commit()
        self.session.refresh(job)

        return EvaluateModelResponse(
            job=self._job_read(job),
            evaluationVersion=evaluation_version,
            artifactPath=str(evaluation_path),
            modelVersion=model_version,
            datasetVersion=dataset_version,
            rows=evaluation["metrics"]["overall"]["rows"],
            metrics=evaluation["metrics"]["overall"],
            validationMetrics=evaluation["metrics"]["validation"],
            diagnostics=evaluation.get("diagnostics") or {},
            topErrors=evaluation["top_errors"],
        )
