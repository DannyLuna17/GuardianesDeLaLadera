from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import ApiError
from app.ml.datasets import TrainingDatasetRegistry, normalize_dataset_context
from app.ml.model_diagnostics import (
    calibration_delta_summary,
    coefficient_drift_summary,
    feature_importance_change_summary,
)
from app.ml.model_evaluations import build_model_evaluation
from app.ml.model_registry import ModelRegistry
from app.ml.model_selection import ModelSelectionRunRegistry
from app.ml.model_shadow import (
    ModelShadowRunRegistry,
    build_model_shadow_run,
    export_model_shadow_run,
)
from app.models import JobExecution
from app.schemas.admin import (
    JobExecutionRead,
    ModelShadowCandidateRead,
    ModelShadowRunDetailRead,
    ModelShadowRunSummaryRead,
    ScanModelShadowResponse,
)


class ModelShadowService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.settings = get_settings()
        self.dataset_registry = TrainingDatasetRegistry()
        self.model_registry = ModelRegistry()
        self.selection_registry = ModelSelectionRunRegistry()
        self.shadow_registry = ModelShadowRunRegistry()

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

    def _shadow_path(self, version: str) -> Path:
        return self.settings.resolved_model_shadow_runs_path / f"{version}.json"

    def _latest_labels_dataset_version(self) -> str:
        candidates: list[tuple[str, str]] = []
        for version in self.dataset_registry.list_versions():
            dataset = self.dataset_registry.load(version)
            dataset_context = normalize_dataset_context(dataset.get("provenance") or {})
            if dataset_context.get("dataset_mode") != "labels":
                continue
            exported_at = str((dataset.get("provenance") or {}).get("exported_at") or "")
            candidates.append((exported_at, version))

        if not candidates:
            raise ApiError(
                404,
                "model_shadow_labels_dataset_not_found",
                "No label-backed training dataset is available for shadow evaluation.",
            )

        candidates.sort(reverse=True)
        return candidates[0][1]

    @staticmethod
    def _probability_metrics(bucket: dict) -> dict:
        return dict(bucket.get("probability_metrics") or {})

    @staticmethod
    def _metric_delta(
        active_value: float | None, challenger_value: float | None
    ) -> float | None:
        if active_value is None or challenger_value is None:
            return None
        return round(float(challenger_value) - float(active_value), 6)

    @staticmethod
    def _lower_is_better_improvement(
        active_value: float | None, challenger_value: float | None
    ) -> float | None:
        if active_value is None or challenger_value is None:
            return None
        return round(float(active_value) - float(challenger_value), 6)

    @staticmethod
    def _higher_is_better_gain(
        active_value: float | None, challenger_value: float | None
    ) -> float | None:
        if active_value is None or challenger_value is None:
            return None
        return round(float(challenger_value) - float(active_value), 6)

    @staticmethod
    def _slice_delta_summary(active_slices: dict, challenger_slices: dict) -> dict[str, dict]:
        slice_keys = sorted(set(active_slices) | set(challenger_slices))
        comparison: dict[str, dict] = {}
        for slice_key in slice_keys:
            active_slice = active_slices.get(slice_key) or {}
            challenger_slice = challenger_slices.get(slice_key) or {}
            active_rmse = ((active_slice.get("calibrated_metrics") or {}).get("rmse"))
            challenger_rmse = ((challenger_slice.get("calibrated_metrics") or {}).get("rmse"))
            active_accuracy = active_slice.get("risk_level_accuracy")
            challenger_accuracy = challenger_slice.get("risk_level_accuracy")
            active_probability = ModelShadowService._probability_metrics(active_slice)
            challenger_probability = ModelShadowService._probability_metrics(
                challenger_slice
            )

            rmse_delta = None
            rmse_improvement = None
            if active_rmse is not None and challenger_rmse is not None:
                rmse_delta = round(float(challenger_rmse) - float(active_rmse), 6)
                rmse_improvement = round(float(active_rmse) - float(challenger_rmse), 6)

            accuracy_delta = None
            if active_accuracy is not None and challenger_accuracy is not None:
                accuracy_delta = round(float(challenger_accuracy) - float(active_accuracy), 6)

            comparison[slice_key] = {
                "active_rows": int(active_slice.get("rows", 0)),
                "challenger_rows": int(challenger_slice.get("rows", 0)),
                "active_validation_rmse": active_rmse,
                "challenger_validation_rmse": challenger_rmse,
                "validation_rmse_delta": rmse_delta,
                "validation_rmse_improvement": rmse_improvement,
                "active_risk_level_accuracy": active_accuracy,
                "challenger_risk_level_accuracy": challenger_accuracy,
                "risk_level_accuracy_delta": accuracy_delta,
                "active_validation_brier_score": active_probability.get("brier_score"),
                "challenger_validation_brier_score": challenger_probability.get(
                    "brier_score"
                ),
                "validation_brier_score_delta": ModelShadowService._metric_delta(
                    active_probability.get("brier_score"),
                    challenger_probability.get("brier_score"),
                ),
                "validation_brier_score_improvement": ModelShadowService._lower_is_better_improvement(
                    active_probability.get("brier_score"),
                    challenger_probability.get("brier_score"),
                ),
                "active_validation_auroc": active_probability.get("auroc"),
                "challenger_validation_auroc": challenger_probability.get("auroc"),
                "validation_auroc_delta": ModelShadowService._higher_is_better_gain(
                    active_probability.get("auroc"),
                    challenger_probability.get("auroc"),
                ),
                "active_validation_auprc": active_probability.get("auprc"),
                "challenger_validation_auprc": challenger_probability.get("auprc"),
                "validation_auprc_delta": ModelShadowService._higher_is_better_gain(
                    active_probability.get("auprc"),
                    challenger_probability.get("auprc"),
                ),
                "active_validation_recall": active_probability.get("recall"),
                "challenger_validation_recall": challenger_probability.get("recall"),
                "validation_recall_delta": ModelShadowService._higher_is_better_gain(
                    active_probability.get("recall"),
                    challenger_probability.get("recall"),
                ),
                "active_validation_specificity": active_probability.get("specificity"),
                "challenger_validation_specificity": challenger_probability.get(
                    "specificity"
                ),
                "validation_specificity_delta": ModelShadowService._higher_is_better_gain(
                    active_probability.get("specificity"),
                    challenger_probability.get("specificity"),
                ),
                "active_validation_mcc": active_probability.get("mcc"),
                "challenger_validation_mcc": challenger_probability.get("mcc"),
                "validation_mcc_delta": ModelShadowService._higher_is_better_gain(
                    active_probability.get("mcc"),
                    challenger_probability.get("mcc"),
                ),
                "active_validation_ece": active_probability.get("ece"),
                "challenger_validation_ece": challenger_probability.get("ece"),
                "validation_ece_delta": ModelShadowService._metric_delta(
                    active_probability.get("ece"),
                    challenger_probability.get("ece"),
                ),
                "validation_ece_improvement": ModelShadowService._lower_is_better_improvement(
                    active_probability.get("ece"),
                    challenger_probability.get("ece"),
                ),
            }
        return comparison

    def _comparison_against_active(
        self,
        *,
        active_artifact: dict,
        challenger_artifact: dict,
        active_evaluation: dict,
        challenger_evaluation: dict,
    ) -> dict:
        active_validation = active_evaluation["metrics"]["validation"]
        challenger_validation = challenger_evaluation["metrics"]["validation"]
        active_overall = active_evaluation["metrics"]["overall"]
        challenger_overall = challenger_evaluation["metrics"]["overall"]
        active_validation_probability = self._probability_metrics(active_validation)
        challenger_validation_probability = self._probability_metrics(
            challenger_validation
        )
        active_overall_probability = self._probability_metrics(active_overall)
        challenger_overall_probability = self._probability_metrics(challenger_overall)
        active_slices = (
            (active_evaluation.get("diagnostics") or {}).get("validation_slices") or {}
        )
        challenger_slices = (
            (challenger_evaluation.get("diagnostics") or {}).get("validation_slices") or {}
        )
        return {
            "active_model_version": active_evaluation["model_version"],
            "active_model_family": active_artifact.get("model_family"),
            "challenger_model_version": challenger_evaluation["model_version"],
            "challenger_model_family": challenger_artifact.get("model_family"),
            "challenger_hyperparameters": (
                (challenger_artifact.get("training") or {}).get("hyperparameters") or {}
            ),
            "validation_rmse_delta": round(
                challenger_validation["calibrated_metrics"]["rmse"]
                - active_validation["calibrated_metrics"]["rmse"],
                6,
            ),
            "validation_rmse_improvement": round(
                active_validation["calibrated_metrics"]["rmse"]
                - challenger_validation["calibrated_metrics"]["rmse"],
                6,
            ),
            "validation_risk_level_accuracy_delta": round(
                challenger_validation["risk_level_accuracy"]
                - active_validation["risk_level_accuracy"],
                6,
            ),
            "overall_rmse_delta": round(
                challenger_overall["calibrated_metrics"]["rmse"]
                - active_overall["calibrated_metrics"]["rmse"],
                6,
            ),
            "overall_risk_level_accuracy_delta": round(
                challenger_overall["risk_level_accuracy"]
                - active_overall["risk_level_accuracy"],
                6,
            ),
            "active_validation_brier_score": active_validation_probability.get(
                "brier_score"
            ),
            "challenger_validation_brier_score": challenger_validation_probability.get(
                "brier_score"
            ),
            "validation_brier_score_delta": self._metric_delta(
                active_validation_probability.get("brier_score"),
                challenger_validation_probability.get("brier_score"),
            ),
            "validation_brier_score_improvement": self._lower_is_better_improvement(
                active_validation_probability.get("brier_score"),
                challenger_validation_probability.get("brier_score"),
            ),
            "active_validation_auroc": active_validation_probability.get("auroc"),
            "challenger_validation_auroc": challenger_validation_probability.get(
                "auroc"
            ),
            "validation_auroc_delta": self._higher_is_better_gain(
                active_validation_probability.get("auroc"),
                challenger_validation_probability.get("auroc"),
            ),
            "active_validation_auprc": active_validation_probability.get("auprc"),
            "challenger_validation_auprc": challenger_validation_probability.get(
                "auprc"
            ),
            "validation_auprc_delta": self._higher_is_better_gain(
                active_validation_probability.get("auprc"),
                challenger_validation_probability.get("auprc"),
            ),
            "active_validation_recall": active_validation_probability.get("recall"),
            "challenger_validation_recall": challenger_validation_probability.get(
                "recall"
            ),
            "validation_recall_delta": self._higher_is_better_gain(
                active_validation_probability.get("recall"),
                challenger_validation_probability.get("recall"),
            ),
            "active_validation_specificity": active_validation_probability.get(
                "specificity"
            ),
            "challenger_validation_specificity": challenger_validation_probability.get(
                "specificity"
            ),
            "validation_specificity_delta": self._higher_is_better_gain(
                active_validation_probability.get("specificity"),
                challenger_validation_probability.get("specificity"),
            ),
            "active_validation_mcc": active_validation_probability.get("mcc"),
            "challenger_validation_mcc": challenger_validation_probability.get("mcc"),
            "validation_mcc_delta": self._higher_is_better_gain(
                active_validation_probability.get("mcc"),
                challenger_validation_probability.get("mcc"),
            ),
            "active_validation_ece": active_validation_probability.get("ece"),
            "challenger_validation_ece": challenger_validation_probability.get("ece"),
            "validation_ece_delta": self._metric_delta(
                active_validation_probability.get("ece"),
                challenger_validation_probability.get("ece"),
            ),
            "validation_ece_improvement": self._lower_is_better_improvement(
                active_validation_probability.get("ece"),
                challenger_validation_probability.get("ece"),
            ),
            "overall_brier_score_delta": self._metric_delta(
                active_overall_probability.get("brier_score"),
                challenger_overall_probability.get("brier_score"),
            ),
            "overall_auroc_delta": self._higher_is_better_gain(
                active_overall_probability.get("auroc"),
                challenger_overall_probability.get("auroc"),
            ),
            "overall_auprc_delta": self._higher_is_better_gain(
                active_overall_probability.get("auprc"),
                challenger_overall_probability.get("auprc"),
            ),
            "coefficient_drift": coefficient_drift_summary(
                active_artifact, challenger_artifact
            ),
            "feature_importance_change": feature_importance_change_summary(
                active_artifact, challenger_artifact
            ),
            "calibration_delta": calibration_delta_summary(
                active_artifact, challenger_artifact
            ),
            "validation_slice_deltas": {
                "by_phase": self._slice_delta_summary(
                    active_slices.get("by_phase") or {},
                    challenger_slices.get("by_phase") or {},
                ),
                "by_target_risk_level": self._slice_delta_summary(
                    active_slices.get("by_target_risk_level") or {},
                    challenger_slices.get("by_target_risk_level") or {},
                ),
                "by_spatial_block": self._slice_delta_summary(
                    active_slices.get("by_spatial_block") or {},
                    challenger_slices.get("by_spatial_block") or {},
                ),
                "by_temporal_holdout_tag": self._slice_delta_summary(
                    active_slices.get("by_temporal_holdout_tag") or {},
                    challenger_slices.get("by_temporal_holdout_tag") or {},
                ),
            },
        }

    def _default_candidate_versions(
        self, *, active_model_version: str, max_candidates: int
    ) -> tuple[list[str], dict]:
        model_versions = [active_model_version]
        recent_best_candidates: list[str] = []
        for version in sorted(
            self.selection_registry.list_versions(),
            reverse=True,
        ):
            run = self.selection_registry.load(version)
            best_model_version = str(run.get("best_model_version") or "")
            if not best_model_version or best_model_version == active_model_version:
                continue
            if best_model_version in recent_best_candidates:
                continue
            if best_model_version not in self.model_registry.list_versions():
                continue
            recent_best_candidates.append(best_model_version)
            if len(recent_best_candidates) >= max(max_candidates - 1, 0):
                break

        model_versions.extend(recent_best_candidates)
        return model_versions[:max_candidates], {
            "mode": "recent_best_candidates",
            "max_candidates": max_candidates,
            "recent_best_candidates": recent_best_candidates,
        }

    @staticmethod
    def _candidate_read(candidate: dict) -> ModelShadowCandidateRead:
        return ModelShadowCandidateRead(
            rank=int(candidate["rank"]),
            role=str(candidate["role"]),
            modelVersion=str(candidate["model_version"]),
            modelFamily=str(candidate.get("model_family") or "baseline"),
            artifactPath=str(candidate["artifact_path"]),
            evaluationVersion=str(candidate["evaluation_version"]),
            overallRmse=float(candidate["overall_rmse"]),
            validationRmse=float(candidate["validation_rmse"]),
            overallRiskLevelAccuracy=float(candidate["overall_risk_level_accuracy"]),
            validationRiskLevelAccuracy=float(
                candidate["validation_risk_level_accuracy"]
            ),
            overallBrierScore=candidate.get("overall_brier_score"),
            validationBrierScore=candidate.get("validation_brier_score"),
            overallAuroc=candidate.get("overall_auroc"),
            validationAuroc=candidate.get("validation_auroc"),
            overallAuprc=candidate.get("overall_auprc"),
            validationAuprc=candidate.get("validation_auprc"),
            validationRecall=candidate.get("validation_recall"),
            validationSpecificity=candidate.get("validation_specificity"),
            validationMcc=candidate.get("validation_mcc"),
            validationEce=candidate.get("validation_ece"),
            validationRows=int(candidate["validation_rows"]),
            hyperparameters=candidate.get("hyperparameters") or {},
            comparison=candidate.get("comparison") or {},
            topErrors=list(candidate.get("top_errors") or []),
        )

    def list_shadow_runs(self) -> list[ModelShadowRunSummaryRead]:
        runs: list[ModelShadowRunSummaryRead] = []
        for version in self.shadow_registry.list_versions():
            run = self.shadow_registry.load(version)
            runs.append(
                ModelShadowRunSummaryRead(
                    version=version,
                    shadowId=run.get("shadow_id", "unknown"),
                    artifactType=run.get("artifact_type", "model_shadow_run"),
                    datasetVersion=run["dataset_version"],
                    createdAt=run["created_at"],
                    activeModelVersion=run["active_model_version"],
                    bestModelVersion=run["best_model_version"],
                    activeStillBest=bool(run["active_still_best"]),
                    candidateCount=int(run["candidate_count"]),
                    recommendation=run.get("recommendation") or {},
                )
            )
        return sorted(runs, key=lambda item: item.created_at, reverse=True)

    def get_shadow_run(self, version: str) -> ModelShadowRunDetailRead:
        run = self.shadow_registry.load(version)
        return ModelShadowRunDetailRead(
            version=version,
            shadowId=run.get("shadow_id", "unknown"),
            artifactType=run.get("artifact_type", "model_shadow_run"),
            artifactPath=str(self._shadow_path(version)),
            datasetVersion=run["dataset_version"],
            createdAt=run["created_at"],
            datasetContext=run.get("dataset_context") or {},
            comparisonPolicy=run.get("comparison_policy") or {},
            activeModelVersion=run["active_model_version"],
            bestModelVersion=run["best_model_version"],
            activeStillBest=bool(run["active_still_best"]),
            activeCandidateRank=int(run["active_candidate_rank"]),
            candidateCount=int(run["candidate_count"]),
            candidateSelection=run.get("candidate_selection") or {},
            recommendation=run.get("recommendation") or {},
            candidates=[self._candidate_read(candidate) for candidate in run["candidates"]],
        )

    def scan_shadow_run(
        self,
        *,
        dataset_version: str | None = None,
        model_versions: list[str] | None = None,
        version: str | None = None,
        max_candidates: int = 4,
        top_error_count: int = 5,
        origin: str = "manual",
    ) -> ScanModelShadowResponse:
        resolved_dataset_version = dataset_version or self._latest_labels_dataset_version()
        target_version = version or f"shadow-{resolved_dataset_version}"
        started_at = datetime.now(timezone.utc).replace(microsecond=0)
        job = JobExecution(
            job_type="model_shadow_scan",
            status="running",
            started_at=started_at,
            details={
                "origin": origin,
                "dataset_version": resolved_dataset_version,
                "shadow_version": target_version,
                "requested_model_versions": model_versions or [],
                "max_candidates": max_candidates,
            },
        )
        self.session.add(job)
        self.session.flush()

        try:
            dataset = self.dataset_registry.load(resolved_dataset_version)
            dataset_context = normalize_dataset_context(dataset.get("provenance") or {})
            if dataset_context.get("dataset_mode") != "labels":
                raise ApiError(
                    400,
                    "model_shadow_requires_labels_dataset",
                    "Shadow evaluation requires a label-backed training dataset.",
                )

            active_model_version = self.model_registry.active_version()
            if model_versions:
                resolved_model_versions = []
                for item in model_versions:
                    normalized = str(item).strip()
                    if not normalized:
                        continue
                    if normalized not in resolved_model_versions:
                        resolved_model_versions.append(normalized)
                if active_model_version not in resolved_model_versions:
                    resolved_model_versions.insert(0, active_model_version)
                candidate_selection = {
                    "mode": "explicit",
                    "requested_model_versions": model_versions,
                    "resolved_model_versions": resolved_model_versions,
                    "max_candidates": max_candidates,
                }
            else:
                resolved_model_versions, candidate_selection = self._default_candidate_versions(
                    active_model_version=active_model_version,
                    max_candidates=max_candidates,
                )
                candidate_selection["resolved_model_versions"] = resolved_model_versions

            available_versions = set(self.model_registry.list_versions())
            missing_versions = [
                candidate_version
                for candidate_version in resolved_model_versions
                if candidate_version not in available_versions
            ]
            if missing_versions:
                raise ApiError(
                    404,
                    "model_shadow_model_not_found",
                    f"Shadow evaluation could not find model artifacts: {', '.join(missing_versions)}.",
                )

            active_artifact = self.model_registry.load(active_model_version)
            active_evaluation = build_model_evaluation(
                version=f"shadow-eval-{active_model_version}-on-{resolved_dataset_version}",
                artifact=active_artifact,
                dataset=dataset,
                top_error_count=top_error_count,
            )

            candidates: list[dict] = []
            for candidate_version in resolved_model_versions:
                artifact = self.model_registry.load(candidate_version)
                evaluation = (
                    active_evaluation
                    if candidate_version == active_model_version
                    else build_model_evaluation(
                        version=f"shadow-eval-{candidate_version}-on-{resolved_dataset_version}",
                        artifact=artifact,
                        dataset=dataset,
                        top_error_count=top_error_count,
                    )
                )
                overall_probability = self._probability_metrics(
                    evaluation["metrics"]["overall"]
                )
                validation_probability = self._probability_metrics(
                    evaluation["metrics"]["validation"]
                )
                comparison = (
                    {}
                    if candidate_version == active_model_version
                    else self._comparison_against_active(
                        active_artifact=active_artifact,
                        challenger_artifact=artifact,
                        active_evaluation=active_evaluation,
                        challenger_evaluation=evaluation,
                    )
                )
                candidates.append(
                    {
                        "rank": 0,
                        "role": "active"
                        if candidate_version == active_model_version
                        else "challenger",
                        "model_version": candidate_version,
                        "model_family": str(
                            artifact.get("model_family") or "baseline"
                        ),
                        "hyperparameters": (
                            (artifact.get("training") or {}).get("hyperparameters") or {}
                        ),
                        "artifact_path": str(
                            self.settings.resolved_model_artifacts_path
                            / f"{candidate_version}.json"
                        ),
                        "evaluation_version": evaluation["version"],
                        "overall_rmse": evaluation["metrics"]["overall"][
                            "calibrated_metrics"
                        ]["rmse"],
                        "validation_rmse": evaluation["metrics"]["validation"][
                            "calibrated_metrics"
                        ]["rmse"],
                        "overall_risk_level_accuracy": evaluation["metrics"]["overall"][
                            "risk_level_accuracy"
                        ],
                        "validation_risk_level_accuracy": evaluation["metrics"][
                            "validation"
                        ]["risk_level_accuracy"],
                        "overall_brier_score": overall_probability.get("brier_score"),
                        "validation_brier_score": validation_probability.get(
                            "brier_score"
                        ),
                        "overall_auroc": overall_probability.get("auroc"),
                        "validation_auroc": validation_probability.get("auroc"),
                        "overall_auprc": overall_probability.get("auprc"),
                        "validation_auprc": validation_probability.get("auprc"),
                        "validation_recall": validation_probability.get("recall"),
                        "validation_specificity": validation_probability.get(
                            "specificity"
                        ),
                        "validation_mcc": validation_probability.get("mcc"),
                        "validation_ece": validation_probability.get("ece"),
                        "validation_rows": evaluation["metrics"]["validation"]["rows"],
                        "comparison": comparison,
                        "top_errors": evaluation.get("top_errors") or [],
                    }
                )

            shadow_run = build_model_shadow_run(
                version=target_version,
                dataset_version=resolved_dataset_version,
                dataset_context=dataset_context,
                active_model_version=active_model_version,
                candidates=candidates,
                candidate_selection=candidate_selection,
            )
            shadow_path, saved_shadow = export_model_shadow_run(shadow_run)
        except Exception as exc:
            completed_at = datetime.now(timezone.utc).replace(microsecond=0)
            job.status = "failed"
            job.completed_at = completed_at
            job.details = {
                "origin": origin,
                "dataset_version": resolved_dataset_version,
                "shadow_version": target_version,
                "requested_model_versions": model_versions or [],
                "max_candidates": max_candidates,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            self.session.commit()
            raise

        completed_at = datetime.now(timezone.utc).replace(microsecond=0)
        best_candidate = next(
            candidate for candidate in saved_shadow["candidates"] if candidate["rank"] == 1
        )
        job.status = "completed"
        job.completed_at = completed_at
        job.details = {
            "origin": origin,
            "dataset_version": resolved_dataset_version,
            "shadow_version": target_version,
            "artifact_path": str(shadow_path),
            "active_model_version": saved_shadow["active_model_version"],
            "best_model_version": saved_shadow["best_model_version"],
            "active_still_best": bool(saved_shadow["active_still_best"]),
            "candidate_count": int(saved_shadow["candidate_count"]),
            "best_validation_rmse": best_candidate["validation_rmse"],
            "best_validation_risk_level_accuracy": best_candidate[
                "validation_risk_level_accuracy"
            ],
            "recommendation": saved_shadow.get("recommendation") or {},
        }
        self.session.commit()
        self.session.refresh(job)
        self.shadow_registry.load.cache_clear()

        return ScanModelShadowResponse(
            job=self._job_read(job),
            shadowVersion=target_version,
            artifactPath=str(shadow_path),
            datasetVersion=resolved_dataset_version,
            activeModelVersion=saved_shadow["active_model_version"],
            bestModelVersion=saved_shadow["best_model_version"],
            activeStillBest=bool(saved_shadow["active_still_best"]),
            candidateCount=int(saved_shadow["candidate_count"]),
            recommendation=saved_shadow.get("recommendation") or {},
            candidates=[self._candidate_read(candidate) for candidate in saved_shadow["candidates"]],
        )
