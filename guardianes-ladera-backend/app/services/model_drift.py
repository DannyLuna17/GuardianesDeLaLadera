from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import ApiError
from app.ml.datasets import TrainingDatasetRegistry, normalize_dataset_context
from app.ml.model_drift import (
    ModelDriftReportRegistry,
    build_model_drift_report,
    export_model_drift_report,
)
from app.ml.model_evaluations import (
    ModelEvaluationRegistry,
    build_model_evaluation,
    export_model_evaluation,
)
from app.ml.model_registry import ModelRegistry
from app.ml.model_selection import ModelSelectionRunRegistry
from app.models import JobExecution
from app.schemas.admin import (
    JobExecutionRead,
    ModelDriftDetailRead,
    ModelDriftSummaryRead,
    ScanModelDriftResponse,
)


class ModelDriftService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.settings = get_settings()
        self.model_registry = ModelRegistry()
        self.dataset_registry = TrainingDatasetRegistry()
        self.evaluation_registry = ModelEvaluationRegistry()
        self.selection_registry = ModelSelectionRunRegistry()
        self.drift_registry = ModelDriftReportRegistry()

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

    def _drift_path(self, version: str) -> Path:
        return self.settings.resolved_model_drift_reports_path / f"{version}.json"

    def _evaluation_path(self, version: str) -> Path:
        return self.settings.resolved_model_evaluations_path / f"{version}.json"

    @staticmethod
    def _summary_read(version: str, report: dict) -> ModelDriftSummaryRead:
        baseline = report.get("baseline") or {}
        current = report.get("current") or {}
        summary = report.get("drift_summary") or {}
        current_context = normalize_dataset_context(current.get("dataset_context") or {})
        return ModelDriftSummaryRead(
            version=version,
            driftId=report.get("drift_id", "unknown"),
            artifactType=report.get("artifact_type", "model_drift_report"),
            createdAt=report.get("created_at", ""),
            modelVersion=report.get("model_version", "unknown"),
            datasetVersion=report.get("dataset_version", "unknown"),
            evaluationVersion=report.get("evaluation_version", "unknown"),
            severity=str(summary.get("severity") or "unavailable"),
            driftDetected=bool(summary.get("drift_detected")),
            baselineSource=baseline.get("source"),
            baselineReferenceVersion=baseline.get("reference_version"),
            baselineDatasetVersion=baseline.get("dataset_version"),
            validationRmse=current.get("validation_rmse"),
            baselineValidationRmse=baseline.get("validation_rmse"),
            validationRmseDelta=summary.get("validation_rmse_delta"),
            validationRiskLevelAccuracy=current.get("validation_risk_level_accuracy"),
            baselineValidationRiskLevelAccuracy=baseline.get(
                "validation_risk_level_accuracy"
            ),
            validationRiskLevelAccuracyDelta=summary.get(
                "validation_risk_level_accuracy_delta"
            ),
            validationRows=int(current.get("validation_rows") or 0),
            datasetFamily=str(current_context.get("dataset_family") or "unknown"),
            taxonomyGroup=(current_context.get("dataset_taxonomy") or {}).get(
                "taxonomy_group"
            ),
            evaluationCohortLabel=(current_context.get("evaluation_cohort") or {}).get(
                "bucket_label"
            ),
        )

    def list_drift_reports(self) -> list[ModelDriftSummaryRead]:
        reports = []
        for version in self.drift_registry.list_versions():
            report = self.drift_registry.load(version)
            reports.append(self._summary_read(version, report))
        return sorted(reports, key=lambda item: item.created_at, reverse=True)

    def get_drift_report(self, version: str) -> ModelDriftDetailRead:
        report = self.drift_registry.load(version)
        return ModelDriftDetailRead(
            version=version,
            driftId=report.get("drift_id", "unknown"),
            artifactType=report.get("artifact_type", "model_drift_report"),
            artifactPath=str(self._drift_path(version)),
            createdAt=report.get("created_at", ""),
            modelVersion=report.get("model_version", "unknown"),
            datasetVersion=report.get("dataset_version", "unknown"),
            evaluationVersion=report.get("evaluation_version", "unknown"),
            baseline=report.get("baseline") or {},
            current=report.get("current") or {},
            driftSummary=report.get("drift_summary") or {},
            diagnostics=report.get("diagnostics") or {},
            topErrors=report.get("top_errors") or [],
        )

    def _latest_labels_dataset_version(self) -> str:
        candidates: list[tuple[str, str]] = []
        for version in self.dataset_registry.list_versions():
            dataset = self.dataset_registry.load(version)
            provenance = dataset.get("provenance") or {}
            if str(provenance.get("dataset_mode") or "unknown") != "labels":
                continue
            exported_at = str(provenance.get("exported_at") or "")
            candidates.append((exported_at, version))
        if not candidates:
            raise ApiError(
                404,
                "model_drift_labels_dataset_not_found",
                "No label-backed training dataset is available for drift monitoring.",
            )
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _baseline_from_selection(self, model_version: str) -> dict | None:
        runs: list[dict] = []
        for version in self.selection_registry.list_versions():
            run = self.selection_registry.load(version)
            context = normalize_dataset_context(run.get("dataset_context") or {})
            if context.get("dataset_mode") != "labels":
                continue
            if str(run.get("best_model_version") or "") != model_version:
                continue
            runs.append(run)
        if not runs:
            return None

        runs.sort(
            key=lambda item: (
                not bool(item.get("promoted")),
                item.get("created_at", ""),
            ),
            reverse=True,
        )
        run = runs[0]
        candidate = next(
            candidate for candidate in run.get("candidates") or [] if candidate["rank"] == 1
        )
        context = normalize_dataset_context(run.get("dataset_context") or {})
        return {
            "source": "selection_run",
            "reference_kind": "promoted_selection_validation"
            if run.get("promoted")
            else "selection_validation",
            "reference_version": run["version"],
            "dataset_version": run["dataset_version"],
            "dataset_context": context,
            "validation_rmse": candidate.get("validation_rmse"),
            "validation_risk_level_accuracy": candidate.get(
                "validation_risk_level_accuracy"
            ),
            "validation_rows": candidate.get("validation_rows"),
            "promoted": bool(run.get("promoted")),
            "created_at": run.get("created_at"),
        }

    def _baseline_from_evaluation(self, model_version: str) -> dict | None:
        evaluations: list[dict] = []
        for version in self.evaluation_registry.list_versions():
            evaluation = self.evaluation_registry.load(version)
            if str(evaluation.get("model_version") or "") != model_version:
                continue
            context = normalize_dataset_context(
                ((evaluation.get("dataset_summary") or {}).get("provenance") or {})
            )
            if context.get("dataset_mode") != "labels":
                continue
            evaluations.append(evaluation)
        if not evaluations:
            return None

        evaluations.sort(key=lambda item: item.get("evaluated_at", ""), reverse=True)
        evaluation = evaluations[0]
        validation = ((evaluation.get("metrics") or {}).get("validation") or {})
        context = normalize_dataset_context(
            ((evaluation.get("dataset_summary") or {}).get("provenance") or {})
        )
        return {
            "source": "model_evaluation",
            "reference_kind": "labels_evaluation_validation",
            "reference_version": evaluation["version"],
            "dataset_version": evaluation["dataset_version"],
            "dataset_context": context,
            "validation_rmse": ((validation.get("calibrated_metrics") or {}).get("rmse")),
            "validation_risk_level_accuracy": validation.get("risk_level_accuracy"),
            "validation_rows": validation.get("rows"),
            "promoted": False,
            "created_at": evaluation.get("evaluated_at"),
        }

    def _resolve_baseline(self, model_version: str) -> dict:
        baseline = self._baseline_from_selection(model_version)
        if baseline:
            return baseline
        baseline = self._baseline_from_evaluation(model_version)
        if baseline:
            return baseline
        return {
            "source": "unavailable",
            "reference_kind": "unavailable",
            "reference_version": None,
            "dataset_version": None,
            "dataset_context": {},
            "validation_rmse": None,
            "validation_risk_level_accuracy": None,
            "validation_rows": None,
            "promoted": False,
            "created_at": None,
        }

    def scan_model_drift(
        self,
        *,
        model_version: str | None = None,
        dataset_version: str | None = None,
        version: str | None = None,
        evaluation_version: str | None = None,
        top_error_count: int = 10,
        warning_validation_rmse_increase: float | None = None,
        critical_validation_rmse_increase: float | None = None,
        warning_accuracy_drop: float | None = None,
        critical_accuracy_drop: float | None = None,
        origin: str = "manual",
    ) -> ScanModelDriftResponse:
        target_model_version = model_version or self.model_registry.active_version()
        target_dataset_version = dataset_version or self._latest_labels_dataset_version()
        drift_version = version or f"drift-{target_model_version}-on-{target_dataset_version}"
        target_evaluation_version = (
            evaluation_version
            or f"eval-drift-{target_model_version}-on-{target_dataset_version}"
        )
        started_at = datetime.now(timezone.utc).replace(microsecond=0)
        job = JobExecution(
            job_type="model_drift_scan",
            status="running",
            started_at=started_at,
            details={
                "origin": origin,
                "model_version": target_model_version,
                "dataset_version": target_dataset_version,
                "drift_version": drift_version,
                "evaluation_version": target_evaluation_version,
            },
        )
        self.session.add(job)
        self.session.flush()

        try:
            artifact = self.model_registry.load(target_model_version)
            dataset = self.dataset_registry.load(target_dataset_version)
            dataset_context = normalize_dataset_context(dataset.get("provenance") or {})
            if dataset_context.get("dataset_mode") != "labels":
                raise ApiError(
                    400,
                    "model_drift_requires_labels_dataset",
                    "Drift monitoring requires a label-backed training dataset.",
                )
            baseline = self._resolve_baseline(target_model_version)
            evaluation = build_model_evaluation(
                version=target_evaluation_version,
                artifact=artifact,
                dataset=dataset,
                top_error_count=top_error_count,
            )
            evaluation_path, _ = export_model_evaluation(evaluation)
            report = build_model_drift_report(
                version=drift_version,
                evaluation=evaluation,
                baseline=baseline,
                warning_validation_rmse_increase=warning_validation_rmse_increase
                if warning_validation_rmse_increase is not None
                else self.settings.model_drift_warning_validation_rmse_increase,
                critical_validation_rmse_increase=critical_validation_rmse_increase
                if critical_validation_rmse_increase is not None
                else self.settings.model_drift_critical_validation_rmse_increase,
                warning_accuracy_drop=warning_accuracy_drop
                if warning_accuracy_drop is not None
                else self.settings.model_drift_warning_accuracy_drop,
                critical_accuracy_drop=critical_accuracy_drop
                if critical_accuracy_drop is not None
                else self.settings.model_drift_critical_accuracy_drop,
            )
            report_path, saved_report = export_model_drift_report(report)
        except Exception as exc:
            completed_at = datetime.now(timezone.utc).replace(microsecond=0)
            job.status = "failed"
            job.completed_at = completed_at
            job.details = {
                "origin": origin,
                "model_version": target_model_version,
                "dataset_version": target_dataset_version,
                "drift_version": drift_version,
                "evaluation_version": target_evaluation_version,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            self.session.commit()
            raise

        completed_at = datetime.now(timezone.utc).replace(microsecond=0)
        summary = saved_report.get("drift_summary") or {}
        job.status = "completed"
        job.completed_at = completed_at
        job.details = {
            "origin": origin,
            "model_version": target_model_version,
            "dataset_version": target_dataset_version,
            "drift_version": drift_version,
            "evaluation_version": target_evaluation_version,
            "evaluation_path": str(evaluation_path),
            "artifact_path": str(report_path),
            "severity": summary.get("severity"),
            "drift_detected": bool(summary.get("drift_detected")),
            "validation_rmse_delta": summary.get("validation_rmse_delta"),
            "validation_risk_level_accuracy_delta": summary.get(
                "validation_risk_level_accuracy_delta"
            ),
        }
        self.session.commit()
        self.session.refresh(job)
        self.evaluation_registry.load.cache_clear()
        self.drift_registry.load.cache_clear()

        return ScanModelDriftResponse(
            job=self._job_read(job),
            driftVersion=drift_version,
            artifactPath=str(report_path),
            evaluationVersion=target_evaluation_version,
            evaluationArtifactPath=str(evaluation_path),
            modelVersion=target_model_version,
            datasetVersion=target_dataset_version,
            severity=str(summary.get("severity") or "unavailable"),
            driftDetected=bool(summary.get("drift_detected")),
            driftSummary=summary,
            baseline=saved_report.get("baseline") or {},
            current=saved_report.get("current") or {},
        )
