from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

import app.services.model_selection_helpers as selection_helpers
import app.services.model_selection_candidates as selection_candidates
import app.services.model_selection_comparison as selection_comparison
import app.services.model_selection_validation as selection_validation
import app.services.model_selection_stability as selection_stability
from app.core.config import get_settings
from app.core.exceptions import ApiError
from app.ml.datasets import (
    TrainingDatasetRegistry,
    normalize_dataset_context,
    rows_from_dataset,
)
from app.ml.model_evaluations import (
    build_model_evaluation,
    build_prediction_records,
    evaluate_prediction_records,
)
from app.ml.model_registry import ModelRegistry
from app.ml.model_selection import (
    ModelSelectionRunRegistry,
    alpha_slug,
    build_model_selection_run,
    candidate_rank_key,
    export_model_selection_run,
)
from app.models import JobExecution
from app.schemas.admin import (
    ExportTrainingDatasetResponse,
    JobExecutionRead,
    ModelReviewTaskRead,
    ModelSelectionCandidateRead,
    ModelSelectionRunDetailRead,
    ModelSelectionRunSummaryRead,
    NotificationEventRead,
    RunModernLabelsBenchmarkResponse,
    RunModernLabelsBenchmarkReviewResponse,
    TuneModelResponse,
)
from app.services.datasets import TrainingDatasetService
from app.services.model_selection_constants import (
    MODEL_MODERN_LABELS_BENCHMARK_ALERT_EVENT_TYPE,
    MODERN_LABELS_BENCHMARK_ALPHAS,
    MODERN_LABELS_BENCHMARK_EARLY_STOPPING_ROUNDS,
    MODERN_LABELS_BENCHMARK_ESTIMATOR_COUNTS,
    MODERN_LABELS_BENCHMARK_KNOT_COUNTS,
    MODERN_LABELS_BENCHMARK_LEARNING_RATES,
    MODERN_LABELS_BENCHMARK_MAX_DEPTHS,
    MODERN_LABELS_BENCHMARK_MIN_LEAF_SIZES,
    MODERN_LABELS_BENCHMARK_MIN_SPLIT_GAINS,
    MODERN_LABELS_BENCHMARK_MODEL_FAMILIES,
    MODERN_LABELS_BENCHMARK_PRESET,
)
from app.services.model_review_tasks import ModelReviewTaskService
from app.services.notifications import NotificationService


class ModelSelectionService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.settings = get_settings()
        self.dataset_registry = TrainingDatasetRegistry()
        self.training_dataset_service = TrainingDatasetService(session)
        self.model_registry = ModelRegistry()
        self.selection_registry = ModelSelectionRunRegistry()
        self.notification_service = NotificationService(session)
        self.model_review_task_service = ModelReviewTaskService(session)

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

    def _selection_path(self, version: str) -> Path:
        return self.settings.resolved_model_selection_runs_path / f"{version}.json"

    @staticmethod
    def _dataset_mode(dataset: dict) -> str:
        provenance = dataset.get("provenance") or {}
        return str(
            provenance.get("dataset_mode") or provenance.get("source") or "unknown"
        )

    @staticmethod
    def _dataset_context(dataset: dict) -> dict:
        provenance = dataset.get("provenance") or {}
        summary = dataset.get("summary") or {}
        return normalize_dataset_context(
            {
                "dataset_mode": provenance.get("dataset_mode")
                or provenance.get("source")
                or "unknown",
                "dataset_family": provenance.get("dataset_family")
                or summary.get("dataset_family")
                or "unknown",
                "time_window": provenance.get("time_window")
                or summary.get("time_window")
                or {},
                "source": provenance.get("source"),
                "label_source_families": provenance.get("label_source_families")
                or summary.get("label_source_families")
                or [],
                "dataset_taxonomy": provenance.get("dataset_taxonomy")
                or summary.get("dataset_taxonomy")
                or {},
                "evaluation_cohort": provenance.get("evaluation_cohort")
                or summary.get("evaluation_cohort")
                or {},
            }
        )

    @staticmethod
    def _parse_iso_datetime(value: str | None) -> datetime | None:
        return selection_helpers.parse_iso_datetime(value)

    def _time_window_gap_days(
        self, current_time_window: dict, historical_time_window: dict
    ) -> float | None:
        return selection_helpers.time_window_gap_days(
            current_time_window, historical_time_window
        )

    @staticmethod
    def _taxonomy_group(dataset_context: dict) -> str | None:
        return selection_helpers.taxonomy_group(dataset_context)

    @staticmethod
    def _cohort_distance(
        current_cohort: dict, historical_cohort: dict
    ) -> int | None:
        return selection_helpers.cohort_distance(current_cohort, historical_cohort)

    @staticmethod
    def _candidate_signature(candidate: dict) -> str:
        return selection_helpers.candidate_signature(candidate)

    @staticmethod
    def _unique_floats(values: list[float] | None, *, default: list[float]) -> list[float]:
        return selection_helpers.unique_floats(values, default=default)

    @staticmethod
    def _unique_ints(values: list[int] | None, *, default: list[int]) -> list[int]:
        return selection_helpers.unique_ints(values, default=default)

    @staticmethod
    def _resolve_model_families(
        *,
        model_family: str,
        model_families: list[str] | None,
    ) -> list[str]:
        return selection_helpers.resolve_model_families(
            model_family=model_family,
            model_families=model_families,
        )

    @staticmethod
    def _benchmark_timestamp_slug() -> str:
        return selection_helpers.benchmark_timestamp_slug()

    def _latest_labels_dataset(self) -> tuple[str, dict] | None:
        latest: tuple[datetime, str, dict] | None = None
        for version in self.dataset_registry.list_versions():
            dataset = self.dataset_registry.load(version)
            if self._dataset_mode(dataset) != "labels":
                continue
            exported_at = self._parse_iso_datetime(
                (dataset.get("provenance") or {}).get("exported_at")
            ) or datetime.min.replace(tzinfo=timezone.utc)
            candidate = (exported_at, version, dataset)
            if latest is None or candidate[:2] > latest[:2]:
                latest = candidate
        if latest is None:
            return None
        return latest[1], latest[2]

    def _benchmark_dataset_resolution(
        self,
        *,
        source: str,
        dataset_version: str,
        dataset: dict,
        export_response: ExportTrainingDatasetResponse | None,
    ) -> dict:
        summary = dict(dataset.get("summary") or {})
        provenance = dict(dataset.get("provenance") or {})
        dataset_context = self._dataset_context(dataset)
        return {
            "source": source,
            "dataset_version": dataset_version,
            "dataset_mode": self._dataset_mode(dataset),
            "dataset_family": dataset_context.get("dataset_family"),
            "rows": int(summary.get("rows", len(dataset.get("rows", [])))),
            "labels": summary.get("labels"),
            "exported_at": provenance.get("exported_at"),
            "time_window": dataset_context.get("time_window") or {},
            "dataset_taxonomy": dataset_context.get("dataset_taxonomy") or {},
            "evaluation_cohort": dataset_context.get("evaluation_cohort") or {},
            "dataset_exported": export_response is not None,
        }

    def _resolve_modern_labels_validation_policy(
        self,
        *,
        dataset: dict,
        validation_strategy: str | None,
        validation_fold_count: int | None,
        nested_outer_fold_count: int | None,
    ) -> dict:
        attempted_strategies: list[dict] = []
        requested_validation_fold_count = validation_fold_count or 2
        requested_strategies = (
            [validation_strategy]
            if validation_strategy
            else [
                "temporal_holdout_backtest",
                "spatial_block_kfold",
                "dataset_holdout",
            ]
        )

        for strategy in requested_strategies:
            try:
                validation_plan = self._resolve_validation_plan(
                    dataset=dataset,
                    validation_strategy=strategy,
                    validation_fold_count=requested_validation_fold_count,
                )
            except ApiError as exc:
                attempted_strategies.append(
                    {
                        "strategy": strategy,
                        "available": False,
                        "error_code": exc.code,
                        "error_message": exc.message,
                    }
                )
                if validation_strategy:
                    raise
                continue

            selection_mode = (
                "single_stage"
                if strategy == "dataset_holdout"
                else "nested_outer_estimate"
            )
            resolved_nested_outer_fold_count: int | None = None
            nested_feasibility: list[dict] = []
            if selection_mode == "nested_outer_estimate":
                outer_plan = self._resolve_validation_plan(
                    dataset=dataset,
                    validation_strategy=strategy,
                    validation_fold_count=(
                        nested_outer_fold_count
                        or validation_plan.get("fold_count")
                        or requested_validation_fold_count
                    ),
                )
                resolved_nested_outer_fold_count = int(outer_plan["fold_count"])
                nested_ready, nested_feasibility = (
                    self._benchmark_nested_selection_feasibility(
                        dataset=dataset,
                        validation_strategy=strategy,
                        validation_fold_count=int(validation_plan["fold_count"]),
                        nested_outer_fold_count=resolved_nested_outer_fold_count,
                    )
                )
                if not nested_ready:
                    selection_mode = "single_stage"
                    resolved_nested_outer_fold_count = None

            attempted_strategies.append(
                {
                    "strategy": strategy,
                    "available": True,
                    "resolved_fold_count": int(validation_plan["fold_count"]),
                    "validation_unit": validation_plan["validation_unit"],
                }
            )
            return {
                "validation_strategy": strategy,
                "validation_fold_count": int(validation_plan["fold_count"]),
                "validation_unit": validation_plan["validation_unit"],
                "selection_mode": selection_mode,
                "nested_outer_fold_count": resolved_nested_outer_fold_count,
                "nested_feasibility": nested_feasibility,
                "attempted_strategies": attempted_strategies,
            }

        raise ApiError(
            400,
            "modern_labels_benchmark_validation_unavailable",
            "The modern labels benchmark could not resolve a supported validation strategy for the selected dataset.",
        )

    def _benchmark_nested_selection_feasibility(
        self,
        *,
        dataset: dict,
        validation_strategy: str,
        validation_fold_count: int,
        nested_outer_fold_count: int,
    ) -> tuple[bool, list[dict]]:
        if validation_strategy == "dataset_holdout":
            return False, []

        rows = rows_from_dataset(dataset)
        row_contexts = self._dataset_row_contexts(dataset)
        outer_plan = self._resolve_validation_plan(
            dataset=dataset,
            validation_strategy=validation_strategy,
            validation_fold_count=nested_outer_fold_count,
        )
        assessments: list[dict] = []
        for outer_fold in outer_plan["folds"]:
            outer_train_rows = [rows[index] for index in outer_fold["train_indices"]]
            outer_train_contexts = [
                dict(row_contexts[index] or {}) for index in outer_fold["train_indices"]
            ]
            outer_train_dataset = self._build_subset_dataset(
                dataset=dataset,
                rows=outer_train_rows,
                row_contexts=outer_train_contexts,
                version_suffix=f"benchmark-inner-{outer_fold['fold_id']}",
                description_suffix=f"benchmark-inner:{outer_fold['fold_id']}",
            )
            try:
                self._resolve_validation_plan(
                    dataset=outer_train_dataset,
                    validation_strategy=validation_strategy,
                    validation_fold_count=validation_fold_count,
                )
                assessments.append(
                    {
                        "fold_id": outer_fold["fold_id"],
                        "available": True,
                        "inner_strategy": validation_strategy,
                        "fallback_used": False,
                    }
                )
                continue
            except ApiError as exc:
                fallback_reason = exc.code

            try:
                self._resolve_validation_plan(
                    dataset=outer_train_dataset,
                    validation_strategy="dataset_holdout",
                    validation_fold_count=None,
                )
                assessments.append(
                    {
                        "fold_id": outer_fold["fold_id"],
                        "available": True,
                        "inner_strategy": "dataset_holdout",
                        "fallback_used": True,
                        "fallback_reason": fallback_reason,
                    }
                )
            except ApiError as fallback_exc:
                assessments.append(
                    {
                        "fold_id": outer_fold["fold_id"],
                        "available": False,
                        "inner_strategy": "dataset_holdout",
                        "fallback_used": True,
                        "fallback_reason": fallback_reason,
                        "error_code": fallback_exc.code,
                        "error_message": fallback_exc.message,
                    }
                )
                return False, assessments

        return True, assessments

    @staticmethod
    def _family_rollups(candidates: list[dict]) -> list[dict]:
        return selection_helpers.family_rollups(candidates)

    def _resolve_boosted_tree_candidate_configs(
        self,
        *,
        learning_rates: list[float] | None,
        estimator_counts: list[int] | None,
        max_depths: list[int] | None,
        min_leaf_sizes: list[int] | None,
        min_split_gains: list[float] | None,
        early_stopping_rounds: int | None,
    ) -> list[dict[str, int | float]]:
        return selection_helpers.resolve_boosted_tree_candidate_configs(
            learning_rates=learning_rates,
            estimator_counts=estimator_counts,
            max_depths=max_depths,
            min_leaf_sizes=min_leaf_sizes,
            min_split_gains=min_split_gains,
            early_stopping_rounds=early_stopping_rounds,
        )

    def _resolve_additive_spline_candidate_configs(
        self,
        *,
        alphas: list[float] | None,
        knot_counts: list[int] | None,
    ) -> list[dict[str, int | float]]:
        return selection_helpers.resolve_additive_spline_candidate_configs(
            alphas=alphas,
            knot_counts=knot_counts,
        )

    @staticmethod
    def _boosted_tree_candidate_suffix(config: dict[str, int | float]) -> str:
        return selection_helpers.boosted_tree_candidate_suffix(config)

    @staticmethod
    def _xgboost_candidate_suffix(config: dict[str, int | float]) -> str:
        return selection_helpers.xgboost_candidate_suffix(config)

    @staticmethod
    def _additive_spline_candidate_suffix(config: dict[str, int | float]) -> str:
        return selection_helpers.additive_spline_candidate_suffix(config)

    @staticmethod
    def _probability_metrics(bucket: dict) -> dict:
        return selection_helpers.probability_metrics(bucket)

    @staticmethod
    def _dataset_row_contexts(dataset: dict) -> list[dict]:
        return selection_helpers.dataset_row_contexts(dataset)

    def _temporal_bucket_reference(
        self, context: dict[str, object], fallback_label: str
    ) -> datetime | None:
        return selection_helpers.temporal_bucket_reference(context, fallback_label)

    def _resolve_validation_plan(
        self,
        *,
        dataset: dict,
        validation_strategy: str,
        validation_fold_count: int | None,
    ) -> dict:
        return selection_validation.resolve_validation_plan(
            dataset=dataset,
            validation_strategy=validation_strategy,
            validation_fold_count=validation_fold_count,
            row_contexts=self._dataset_row_contexts(dataset),
            temporal_bucket_reference=self._temporal_bucket_reference,
        )

    def _build_validation_fold_dataset(
        self,
        *,
        dataset: dict,
        rows: list,
        row_contexts: list[dict],
        plan: dict,
        fold: dict,
    ) -> dict:
        return selection_validation.build_validation_fold_dataset(
            dataset=dataset,
            rows=rows,
            row_contexts=row_contexts,
            plan=plan,
            fold=fold,
        )

    def _validation_summary_from_records(
        self,
        *,
        plan: dict,
        validation_records: list[dict],
        fold_summaries: list[dict],
        top_error_count: int,
    ) -> dict:
        return selection_validation.validation_summary_from_records(
            plan=plan,
            validation_records=validation_records,
            fold_summaries=fold_summaries,
            top_error_count=top_error_count,
        )

    def _build_subset_dataset(
        self,
        *,
        dataset: dict,
        rows: list,
        row_contexts: list[dict],
        version_suffix: str,
        description_suffix: str,
    ) -> dict:
        return selection_validation.build_subset_dataset(
            dataset=dataset,
            rows=rows,
            row_contexts=row_contexts,
            version_suffix=version_suffix,
            description_suffix=description_suffix,
        )

    def _ranking_candidate_from_validation_summary(
        self,
        *,
        candidate_spec: dict,
        validation_summary: dict,
    ) -> dict:
        return selection_validation.ranking_candidate_from_validation_summary(
            candidate_spec=candidate_spec,
            validation_summary=validation_summary,
        )

    def _nested_selection_estimation(
        self,
        *,
        active_artifact: dict,
        dataset: dict,
        rows: list,
        row_contexts: list[dict],
        candidate_specs: list[dict],
        validation_strategy: str,
        validation_fold_count: int | None,
        nested_outer_fold_count: int | None,
        top_error_count: int,
    ) -> dict:
        outer_plan = self._resolve_validation_plan(
            dataset=dataset,
            validation_strategy=validation_strategy,
            validation_fold_count=nested_outer_fold_count,
        )
        nested_validation_records: list[dict] = []
        active_validation_records: list[dict] = []
        outer_fold_summaries: list[dict] = []
        selected_candidate_counts: dict[str, int] = {}
        selected_family_counts: dict[str, int] = {}

        for outer_fold in outer_plan["folds"]:
            outer_train_rows = [rows[index] for index in outer_fold["train_indices"]]
            outer_train_contexts = [
                dict(row_contexts[index] or {}) for index in outer_fold["train_indices"]
            ]
            outer_train_dataset = self._build_subset_dataset(
                dataset=dataset,
                rows=outer_train_rows,
                row_contexts=outer_train_contexts,
                version_suffix=f"nested-inner-{outer_fold['fold_id']}",
                description_suffix=f"nested-inner:{outer_fold['fold_id']}",
            )
            inner_plan_fallback_reason: str | None = None
            try:
                inner_plan = self._resolve_validation_plan(
                    dataset=outer_train_dataset,
                    validation_strategy=validation_strategy,
                    validation_fold_count=validation_fold_count,
                )
            except ApiError as exc:
                if validation_strategy == "dataset_holdout":
                    raise
                inner_plan = self._resolve_validation_plan(
                    dataset=outer_train_dataset,
                    validation_strategy="dataset_holdout",
                    validation_fold_count=None,
                )
                inner_plan_fallback_reason = exc.code
            inner_candidate_rankings: list[dict] = []
            inner_validation_summaries: dict[str, dict] = {}
            for candidate_spec in candidate_specs:
                inner_summary = self._evaluate_candidate_under_validation_plan(
                    model_family=str(candidate_spec["model_family"]),
                    version=str(candidate_spec["candidate_version"]),
                    dataset=outer_train_dataset,
                    rows=outer_train_rows,
                    row_contexts=outer_train_contexts,
                    plan=inner_plan,
                    hyperparameters=dict(candidate_spec["hyperparameters"]),
                    top_error_count=top_error_count,
                )
                inner_validation_summaries[str(candidate_spec["candidate_version"])] = (
                    inner_summary
                )
                inner_candidate_rankings.append(
                    self._ranking_candidate_from_validation_summary(
                        candidate_spec=candidate_spec,
                        validation_summary=inner_summary,
                    )
                )

            best_inner_candidate = sorted(
                inner_candidate_rankings, key=candidate_rank_key
            )[0]
            best_candidate_signature = self._candidate_signature(best_inner_candidate)
            selected_candidate_counts[best_candidate_signature] = (
                selected_candidate_counts.get(best_candidate_signature, 0) + 1
            )
            selected_family = str(best_inner_candidate["model_family"])
            selected_family_counts[selected_family] = (
                selected_family_counts.get(selected_family, 0) + 1
            )
            selected_spec = next(
                candidate_spec
                for candidate_spec in candidate_specs
                if str(candidate_spec["candidate_version"])
                == str(best_inner_candidate["model_version"])
            )
            outer_artifact = self._train_candidate_artifact(
                model_family=str(selected_spec["model_family"]),
                version=f"{selected_spec['candidate_version']}--nested-{outer_fold['fold_id']}",
                rows=outer_train_rows,
                dataset_name=f"{dataset['version']}:nested:{outer_fold['fold_id']}",
                hyperparameters=dict(selected_spec["hyperparameters"]),
                train_rows=outer_train_rows,
                validation_rows=[],
            )
            outer_fold_dataset = self._build_validation_fold_dataset(
                dataset=dataset,
                rows=rows,
                row_contexts=row_contexts,
                plan=outer_plan,
                fold=outer_fold,
            )
            outer_prediction_records = build_prediction_records(
                outer_artifact, outer_fold_dataset
            )
            outer_validation_records = [
                record
                for record in outer_prediction_records
                if record["split"] == "validation"
            ]
            active_prediction_records = build_prediction_records(
                active_artifact, outer_fold_dataset
            )
            active_outer_validation_records = [
                record
                for record in active_prediction_records
                if record["split"] == "validation"
            ]
            nested_validation_records.extend(outer_validation_records)
            active_validation_records.extend(active_outer_validation_records)
            outer_metrics = evaluate_prediction_records(outer_validation_records)
            outer_probability = self._probability_metrics(outer_metrics)
            active_outer_metrics = evaluate_prediction_records(
                active_outer_validation_records
            )
            active_outer_probability = self._probability_metrics(active_outer_metrics)
            inner_summary = inner_validation_summaries[
                str(selected_spec["candidate_version"])
            ]
            outer_fold_summaries.append(
                {
                    "fold_id": outer_fold["fold_id"],
                    "selected_model_family": selected_family,
                    "selected_model_version": str(best_inner_candidate["model_version"]),
                    "selected_candidate_signature": best_candidate_signature,
                    "selected_hyperparameters": dict(
                        selected_spec["hyperparameters"]
                    ),
                    "inner_requested_strategy": validation_strategy,
                    "inner_strategy": inner_plan["strategy"],
                    "inner_fold_count": inner_plan["fold_count"],
                    "inner_plan_fallback_reason": inner_plan_fallback_reason,
                    "inner_validation_rmse": best_inner_candidate["validation_rmse"],
                    "inner_validation_brier_score": best_inner_candidate.get(
                        "validation_brier_score"
                    ),
                    "outer_validation_rmse": outer_metrics["calibrated_metrics"][
                        "rmse"
                    ],
                    "active_outer_validation_rmse": active_outer_metrics[
                        "calibrated_metrics"
                    ]["rmse"],
                    "outer_validation_rmse_improvement_vs_active": round(
                        active_outer_metrics["calibrated_metrics"]["rmse"]
                        - outer_metrics["calibrated_metrics"]["rmse"],
                        6,
                    ),
                    "outer_validation_risk_level_accuracy": outer_metrics[
                        "risk_level_accuracy"
                    ],
                    "active_outer_validation_risk_level_accuracy": active_outer_metrics[
                        "risk_level_accuracy"
                    ],
                    "outer_validation_brier_score": outer_probability.get(
                        "brier_score"
                    ),
                    "active_outer_validation_brier_score": active_outer_probability.get(
                        "brier_score"
                    ),
                    "outer_validation_auprc": outer_probability.get("auprc"),
                    "active_outer_validation_auprc": active_outer_probability.get(
                        "auprc"
                    ),
                    "outer_validation_recall": outer_probability.get("recall"),
                    "active_outer_validation_recall": active_outer_probability.get(
                        "recall"
                    ),
                    "outer_validation_mcc": outer_probability.get("mcc"),
                    "active_outer_validation_mcc": active_outer_probability.get(
                        "mcc"
                    ),
                    "validation_rows": outer_metrics["rows"],
                    "metadata": dict(outer_fold.get("metadata") or {}),
                    "inner_validation_summary": {
                        key: value
                        for key, value in inner_summary.items()
                        if key != "metrics"
                    },
                }
            )

        nested_summary = self._validation_summary_from_records(
            plan={
                "requested_strategy": validation_strategy,
                "strategy": outer_plan["strategy"],
                "validation_unit": outer_plan["validation_unit"],
                "requested_fold_count": nested_outer_fold_count,
            },
            validation_records=nested_validation_records,
            fold_summaries=outer_fold_summaries,
            top_error_count=top_error_count,
        )
        active_summary = self._validation_summary_from_records(
            plan={
                "requested_strategy": validation_strategy,
                "strategy": outer_plan["strategy"],
                "validation_unit": outer_plan["validation_unit"],
                "requested_fold_count": nested_outer_fold_count,
            },
            validation_records=active_validation_records,
            fold_summaries=[],
            top_error_count=top_error_count,
        )
        nested_probability = self._probability_metrics(nested_summary["metrics"])
        active_probability = self._probability_metrics(active_summary["metrics"])
        nested_summary["selection_mode"] = "nested_outer_estimate"
        nested_summary["inner_validation_strategy"] = validation_strategy
        nested_summary["inner_requested_fold_count"] = validation_fold_count
        nested_summary["outer_validation_strategy"] = outer_plan["strategy"]
        nested_summary["outer_requested_fold_count"] = nested_outer_fold_count
        nested_summary["outer_fold_count"] = outer_plan["fold_count"]
        nested_summary["active_metrics"] = active_summary["metrics"]
        nested_summary["comparison_vs_active"] = {
            "active_validation_rmse": active_summary["metrics"]["calibrated_metrics"][
                "rmse"
            ],
            "selected_validation_rmse": nested_summary["metrics"][
                "calibrated_metrics"
            ]["rmse"],
            "validation_rmse_improvement": round(
                active_summary["metrics"]["calibrated_metrics"]["rmse"]
                - nested_summary["metrics"]["calibrated_metrics"]["rmse"],
                6,
            ),
            "active_validation_risk_level_accuracy": active_summary["metrics"][
                "risk_level_accuracy"
            ],
            "selected_validation_risk_level_accuracy": nested_summary["metrics"][
                "risk_level_accuracy"
            ],
            "validation_risk_level_accuracy_delta": round(
                nested_summary["metrics"]["risk_level_accuracy"]
                - active_summary["metrics"]["risk_level_accuracy"],
                6,
            ),
            "active_validation_brier_score": active_probability.get("brier_score"),
            "selected_validation_brier_score": nested_probability.get("brier_score"),
            "validation_brier_score_improvement": self._lower_is_better_improvement(
                active_probability.get("brier_score"),
                nested_probability.get("brier_score"),
            ),
            "active_validation_auprc": active_probability.get("auprc"),
            "selected_validation_auprc": nested_probability.get("auprc"),
            "validation_auprc_delta": self._higher_is_better_gain(
                active_probability.get("auprc"),
                nested_probability.get("auprc"),
            ),
            "active_validation_recall": active_probability.get("recall"),
            "selected_validation_recall": nested_probability.get("recall"),
            "validation_recall_delta": self._higher_is_better_gain(
                active_probability.get("recall"),
                nested_probability.get("recall"),
            ),
            "active_validation_mcc": active_probability.get("mcc"),
            "selected_validation_mcc": nested_probability.get("mcc"),
            "validation_mcc_delta": self._higher_is_better_gain(
                active_probability.get("mcc"),
                nested_probability.get("mcc"),
            ),
            "active_validation_ece": active_probability.get("ece"),
            "selected_validation_ece": nested_probability.get("ece"),
            "validation_ece_improvement": self._lower_is_better_improvement(
                active_probability.get("ece"),
                nested_probability.get("ece"),
            ),
            "validation_rows": nested_summary["metrics"]["rows"],
        }
        nested_summary["selected_candidate_frequencies"] = [
            {"candidate_signature": signature, "count": count}
            for signature, count in sorted(
                selected_candidate_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        nested_summary["selected_family_frequencies"] = [
            {"model_family": family, "count": count}
            for family, count in sorted(
                selected_family_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        temporal_outer_folds = [
            fold
            for fold in outer_fold_summaries
            if (fold.get("metadata") or {}).get("validation_bucket") is not None
        ]
        ordered_temporal_outer_folds = sorted(
            temporal_outer_folds,
            key=lambda fold: (
                self._parse_iso_datetime(
                    str(
                        (fold.get("metadata") or {}).get(
                            "validation_bucket_reference_at"
                        )
                    )
                    if (fold.get("metadata") or {}).get(
                        "validation_bucket_reference_at"
                    )
                    is not None
                    else None
                )
                or datetime.min.replace(tzinfo=timezone.utc),
                str((fold.get("metadata") or {}).get("validation_bucket") or ""),
            ),
        )
        latest_temporal_fold = (
            ordered_temporal_outer_folds[-1] if ordered_temporal_outer_folds else None
        )
        temporal_bucket_labels = [
            str((fold.get("metadata") or {}).get("validation_bucket"))
            for fold in ordered_temporal_outer_folds
        ]
        nested_summary["temporal_cohort_coverage"] = {
            "available": bool(ordered_temporal_outer_folds),
            "bucket_count": len({label for label in temporal_bucket_labels if label}),
            "bucket_labels": list(dict.fromkeys(label for label in temporal_bucket_labels if label)),
            "latest_bucket": (
                (latest_temporal_fold.get("metadata") or {}).get("validation_bucket")
                if latest_temporal_fold
                else None
            ),
            "latest_bucket_reference_at": (
                (latest_temporal_fold.get("metadata") or {}).get(
                    "validation_bucket_reference_at"
                )
                if latest_temporal_fold
                else None
            ),
            "latest_fold_id": (
                latest_temporal_fold.get("fold_id") if latest_temporal_fold else None
            ),
            "latest_fold_validation_rmse_improvement_vs_active": (
                latest_temporal_fold.get("outer_validation_rmse_improvement_vs_active")
                if latest_temporal_fold
                else None
            ),
            "latest_fold_wins_vs_active": bool(
                (latest_temporal_fold or {}).get(
                    "outer_validation_rmse_improvement_vs_active", 0.0
                )
                > 0
            ),
        }
        return nested_summary

    def _nested_temporal_recent_window_summary(
        self,
        nested_estimation: dict,
        *,
        recent_window_size: int | None,
    ) -> dict:
        temporal_outer_folds = [
            fold
            for fold in list(nested_estimation.get("folds") or [])
            if (fold.get("metadata") or {}).get("validation_bucket") is not None
        ]
        ordered_temporal_outer_folds = sorted(
            temporal_outer_folds,
            key=lambda fold: (
                self._parse_iso_datetime(
                    str(
                        (fold.get("metadata") or {}).get(
                            "validation_bucket_reference_at"
                        )
                    )
                    if (fold.get("metadata") or {}).get(
                        "validation_bucket_reference_at"
                    )
                    is not None
                    else None
                )
                or datetime.min.replace(tzinfo=timezone.utc),
                str((fold.get("metadata") or {}).get("validation_bucket") or ""),
            ),
        )
        resolved_requested_window_size = (
            int(recent_window_size) if recent_window_size and recent_window_size > 0 else None
        )
        recent_temporal_outer_folds = (
            ordered_temporal_outer_folds[-resolved_requested_window_size:]
            if resolved_requested_window_size is not None
            else ordered_temporal_outer_folds
        )
        bucket_labels = [
            str((fold.get("metadata") or {}).get("validation_bucket"))
            for fold in recent_temporal_outer_folds
            if (fold.get("metadata") or {}).get("validation_bucket") is not None
        ]
        improvements = [
            float(fold.get("outer_validation_rmse_improvement_vs_active"))
            for fold in recent_temporal_outer_folds
            if fold.get("outer_validation_rmse_improvement_vs_active") is not None
        ]
        win_count = sum(1 for value in improvements if value > 0)
        latest_recent_fold = (
            recent_temporal_outer_folds[-1] if recent_temporal_outer_folds else None
        )
        return {
            "available": bool(recent_temporal_outer_folds),
            "requested_window_size": resolved_requested_window_size,
            "resolved_window_size": len(recent_temporal_outer_folds),
            "bucket_count": len({label for label in bucket_labels if label}),
            "bucket_labels": list(dict.fromkeys(label for label in bucket_labels if label)),
            "win_count_vs_active": win_count,
            "loss_or_tie_count_vs_active": max(
                len(recent_temporal_outer_folds) - win_count, 0
            ),
            "win_rate_vs_active": (
                round(win_count / len(recent_temporal_outer_folds), 6)
                if recent_temporal_outer_folds
                else None
            ),
            "average_validation_rmse_improvement_vs_active": (
                round(sum(improvements) / len(improvements), 6)
                if improvements
                else None
            ),
            "minimum_validation_rmse_improvement_vs_active": (
                round(min(improvements), 6) if improvements else None
            ),
            "maximum_validation_rmse_improvement_vs_active": (
                round(max(improvements), 6) if improvements else None
            ),
            "latest_bucket": (
                (latest_recent_fold.get("metadata") or {}).get("validation_bucket")
                if latest_recent_fold
                else None
            ),
            "latest_bucket_reference_at": (
                (latest_recent_fold.get("metadata") or {}).get(
                    "validation_bucket_reference_at"
                )
                if latest_recent_fold
                else None
            ),
        }

    def _evaluate_artifact_under_validation_plan(
        self,
        *,
        artifact: dict,
        dataset: dict,
        rows: list,
        row_contexts: list[dict],
        plan: dict,
        top_error_count: int,
    ) -> dict:
        return selection_validation.evaluate_artifact_under_validation_plan(
            artifact=artifact,
            dataset=dataset,
            rows=rows,
            row_contexts=row_contexts,
            plan=plan,
            top_error_count=top_error_count,
        )

    def _evaluate_candidate_under_validation_plan(
        self,
        *,
        model_family: str,
        version: str,
        dataset: dict,
        rows: list,
        row_contexts: list[dict],
        plan: dict,
        hyperparameters: dict,
        top_error_count: int,
    ) -> dict:
        return selection_validation.evaluate_candidate_under_validation_plan(
            model_family=model_family,
            version=version,
            dataset=dataset,
            rows=rows,
            row_contexts=row_contexts,
            plan=plan,
            hyperparameters=hyperparameters,
            top_error_count=top_error_count,
            train_candidate_artifact=self._train_candidate_artifact,
        )

    def _train_candidate_artifact(
        self,
        *,
        model_family: str,
        version: str,
        rows: list,
        dataset_name: str,
        hyperparameters: dict,
        train_rows: list | None = None,
        validation_rows: list | None = None,
    ) -> dict:
        return selection_candidates.train_candidate_artifact(
            model_family=model_family,
            version=version,
            rows=rows,
            dataset_name=dataset_name,
            hyperparameters=hyperparameters,
            train_rows=train_rows,
            validation_rows=validation_rows,
        )

    def _export_candidate_artifact(
        self,
        *,
        model_family: str,
        version: str,
        rows: list,
        dataset_name: str,
        hyperparameters: dict,
    ) -> tuple[Path, dict]:
        return selection_candidates.export_candidate_artifact(
            model_family=model_family,
            version=version,
            rows=rows,
            dataset_name=dataset_name,
            hyperparameters=hyperparameters,
        )

    def _candidate_payload(
        self,
        *,
        candidate_version: str,
        model_family: str,
        alpha: float | None,
        artifact_path: Path,
        artifact: dict,
        evaluation: dict,
        validation_summary: dict,
    ) -> dict:
        return selection_candidates.candidate_payload(
            candidate_version=candidate_version,
            model_family=model_family,
            alpha=alpha,
            artifact_path=artifact_path,
            artifact=artifact,
            evaluation=evaluation,
            validation_summary=validation_summary,
        )

    @staticmethod
    def _metric_delta(
        active_value: float | None, challenger_value: float | None
    ) -> float | None:
        return selection_helpers.metric_delta(active_value, challenger_value)

    @staticmethod
    def _lower_is_better_improvement(
        active_value: float | None, challenger_value: float | None
    ) -> float | None:
        return selection_helpers.lower_is_better_improvement(
            active_value, challenger_value
        )

    @staticmethod
    def _higher_is_better_gain(
        active_value: float | None, challenger_value: float | None
    ) -> float | None:
        return selection_helpers.higher_is_better_gain(active_value, challenger_value)

    def _recent_selection_runs_for_mode(
        self,
        *,
        dataset_mode: str,
        exclude_version: str | None = None,
    ) -> list[dict]:
        return selection_stability.recent_selection_runs_for_mode(
            selection_registry=self.selection_registry,
            dataset_mode=dataset_mode,
            exclude_version=exclude_version,
        )

    def _stability_assessment(
        self,
        *,
        best_candidate: dict,
        dataset_mode: str,
        dataset_context: dict,
        selection_version: str,
        stability_window_runs: int,
        required_consistent_wins: int,
        require_same_dataset_family: bool,
        require_same_dataset_taxonomy: bool,
        require_same_evaluation_cohort: bool,
        max_time_window_gap_days: int,
        max_cohort_distance: int,
        current_dataset_version: str,
    ) -> dict:
        historical_runs = self._recent_selection_runs_for_mode(
            dataset_mode=dataset_mode,
            exclude_version=selection_version,
        )
        return selection_stability.stability_assessment(
            best_candidate=best_candidate,
            historical_runs=historical_runs,
            dataset_context=dataset_context,
            selection_version=selection_version,
            stability_window_runs=stability_window_runs,
            required_consistent_wins=required_consistent_wins,
            require_same_dataset_family=require_same_dataset_family,
            require_same_dataset_taxonomy=require_same_dataset_taxonomy,
            require_same_evaluation_cohort=require_same_evaluation_cohort,
            max_time_window_gap_days=max_time_window_gap_days,
            max_cohort_distance=max_cohort_distance,
            current_dataset_version=current_dataset_version,
        )

    @staticmethod
    def _slice_delta_summary(
        active_slices: dict, challenger_slices: dict
    ) -> dict[str, dict]:
        return selection_comparison.slice_delta_summary(active_slices, challenger_slices)

    @staticmethod
    def _slice_gate_summary(
        slice_deltas: dict[str, dict],
        *,
        min_rows: int,
    ) -> dict:
        return selection_comparison.slice_gate_summary(
            slice_deltas,
            min_rows=min_rows,
        )

    def _best_vs_active_comparison(
        self,
        *,
        active_artifact: dict,
        challenger_artifact: dict,
        active_evaluation: dict,
        challenger_evaluation: dict,
        active_validation_summary: dict | None = None,
        challenger_validation_summary: dict | None = None,
    ) -> dict:
        return selection_comparison.best_vs_active_comparison(
            active_artifact=active_artifact,
            challenger_artifact=challenger_artifact,
            active_evaluation=active_evaluation,
            challenger_evaluation=challenger_evaluation,
            active_validation_summary=active_validation_summary,
            challenger_validation_summary=challenger_validation_summary,
        )

    @staticmethod
    def _candidate_read(candidate: dict) -> ModelSelectionCandidateRead:
        return ModelSelectionCandidateRead(
            rank=int(candidate["rank"]),
            modelVersion=str(candidate["model_version"]),
            modelFamily=str(candidate.get("model_family") or "linear_ridge"),
            alpha=(
                float(candidate["alpha"])
                if candidate.get("alpha") is not None
                else None
            ),
            artifactPath=str(candidate["artifact_path"]),
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
            validationSummary=candidate.get("validation_summary") or {},
            comparison=candidate.get("comparison") or {},
            topErrors=list(candidate.get("top_errors") or []),
        )

    def run_modern_labels_benchmark(
        self,
        *,
        dataset_version: str | None = None,
        auto_export_dataset: bool = True,
        dataset_export_version: str | None = None,
        label_sources: list[str] | None = None,
        max_labels: int = 250,
        observed_after: datetime | None = None,
        observed_before: datetime | None = None,
        validation_strategy: str | None = None,
        validation_fold_count: int | None = None,
        nested_outer_fold_count: int | None = None,
        version: str | None = None,
        version_prefix: str | None = None,
        promote_best: bool = False,
        promotion_reason: str | None = None,
        promoted_by: str | None = None,
        origin: str = "manual",
    ) -> RunModernLabelsBenchmarkResponse:
        benchmark_origin = f"{origin}:modern_labels_benchmark"
        dataset_export_response: ExportTrainingDatasetResponse | None = None
        dataset_resolution_source = "provided"

        if dataset_version:
            dataset = self.dataset_registry.load(dataset_version)
        elif auto_export_dataset:
            export_version = (
                dataset_export_version
                or f"benchmark-labels-{self._benchmark_timestamp_slug()}"
            )
            dataset_export_response = self.training_dataset_service.export_dataset(
                version=export_version,
                source_mode="labels",
                label_sources=label_sources,
                max_labels=max_labels,
                observed_after=observed_after,
                observed_before=observed_before,
                origin=benchmark_origin,
            )
            dataset_version = dataset_export_response.dataset_version
            dataset = self.dataset_registry.load(dataset_version)
            dataset_resolution_source = "auto_exported"
        else:
            latest_labels_dataset = self._latest_labels_dataset()
            if latest_labels_dataset is None:
                raise ApiError(
                    404,
                    "modern_labels_benchmark_dataset_not_found",
                    "No labels-backed training dataset is available. Provide datasetVersion or enable autoExportDataset.",
                )
            dataset_version, dataset = latest_labels_dataset
            dataset_resolution_source = "latest_existing"

        assert dataset_version is not None
        if self._dataset_mode(dataset) != "labels":
            raise ApiError(
                400,
                "modern_labels_benchmark_requires_labels_dataset",
                "The modern labels benchmark only supports labels-backed training datasets.",
            )

        resolved_validation_policy = self._resolve_modern_labels_validation_policy(
            dataset=dataset,
            validation_strategy=validation_strategy,
            validation_fold_count=validation_fold_count,
            nested_outer_fold_count=nested_outer_fold_count,
        )

        selection_response = self.tune_model(
            dataset_version=dataset_version,
            alphas=MODERN_LABELS_BENCHMARK_ALPHAS,
            model_family=MODERN_LABELS_BENCHMARK_MODEL_FAMILIES[0],
            model_families=list(MODERN_LABELS_BENCHMARK_MODEL_FAMILIES),
            selection_mode=str(resolved_validation_policy["selection_mode"]),
            validation_strategy=str(resolved_validation_policy["validation_strategy"]),
            validation_fold_count=int(
                resolved_validation_policy["validation_fold_count"]
            ),
            nested_outer_fold_count=resolved_validation_policy[
                "nested_outer_fold_count"
            ],
            knot_counts=MODERN_LABELS_BENCHMARK_KNOT_COUNTS,
            learning_rates=MODERN_LABELS_BENCHMARK_LEARNING_RATES,
            estimator_counts=MODERN_LABELS_BENCHMARK_ESTIMATOR_COUNTS,
            max_depths=MODERN_LABELS_BENCHMARK_MAX_DEPTHS,
            min_leaf_sizes=MODERN_LABELS_BENCHMARK_MIN_LEAF_SIZES,
            min_split_gains=MODERN_LABELS_BENCHMARK_MIN_SPLIT_GAINS,
            early_stopping_rounds=MODERN_LABELS_BENCHMARK_EARLY_STOPPING_ROUNDS,
            version=version
            or f"selection-modern-labels-{self._benchmark_timestamp_slug()}",
            version_prefix=version_prefix or f"{dataset_version}-modern-benchmark",
            promote_best=promote_best,
            promotion_reason=promotion_reason,
            top_error_count=5,
            min_validation_rmse_improvement=0.0,
            min_validation_rows=1,
            require_labels_dataset_for_promotion=True,
            require_nested_estimation_for_promotion=(
                resolved_validation_policy["selection_mode"] == "nested_outer_estimate"
            ),
            min_calibration_gain=0.0,
            min_nested_outer_validation_rmse_improvement=0.0,
            min_nested_outer_selection_rate=(
                0.5
                if resolved_validation_policy["selection_mode"]
                == "nested_outer_estimate"
                else None
            ),
            require_nested_temporal_latest_win_for_promotion=(
                resolved_validation_policy["validation_strategy"]
                == "temporal_holdout_backtest"
            ),
            min_nested_temporal_outer_bucket_count=(
                2
                if resolved_validation_policy["validation_strategy"]
                == "temporal_holdout_backtest"
                else 0
            ),
            min_nested_temporal_latest_validation_rmse_improvement=(
                0.0
                if resolved_validation_policy["validation_strategy"]
                == "temporal_holdout_backtest"
                else None
            ),
            nested_temporal_recent_window_size=(
                2
                if resolved_validation_policy["validation_strategy"]
                == "temporal_holdout_backtest"
                else None
            ),
            min_nested_temporal_recent_win_rate=(
                0.5
                if resolved_validation_policy["validation_strategy"]
                == "temporal_holdout_backtest"
                else None
            ),
            min_nested_temporal_recent_average_validation_rmse_improvement=(
                0.0
                if resolved_validation_policy["validation_strategy"]
                == "temporal_holdout_backtest"
                else None
            ),
            max_spatial_slice_validation_rmse_regression=0.05,
            max_temporal_slice_validation_rmse_regression=(
                0.05
                if resolved_validation_policy["validation_strategy"]
                == "temporal_holdout_backtest"
                else None
            ),
            max_spatial_slice_regression_count=0,
            max_temporal_slice_regression_count=(
                0
                if resolved_validation_policy["validation_strategy"]
                == "temporal_holdout_backtest"
                else None
            ),
            slice_regression_min_rows=2,
            stability_window_runs=2,
            required_consistent_wins=2,
            stability_require_same_dataset_family=True,
            stability_require_same_dataset_taxonomy=True,
            stability_require_same_evaluation_cohort=False,
            stability_max_time_window_gap_days=90,
            stability_max_cohort_distance=1,
            promoted_by=promoted_by,
            origin=benchmark_origin,
        )

        dataset_resolution = self._benchmark_dataset_resolution(
            source=dataset_resolution_source,
            dataset_version=dataset_version,
            dataset=dataset,
            export_response=dataset_export_response,
        )
        resolved_policy = {
            "preset": MODERN_LABELS_BENCHMARK_PRESET,
            "model_families": list(MODERN_LABELS_BENCHMARK_MODEL_FAMILIES),
            "alphas": list(MODERN_LABELS_BENCHMARK_ALPHAS),
            "knot_counts": list(MODERN_LABELS_BENCHMARK_KNOT_COUNTS),
            "learning_rates": list(MODERN_LABELS_BENCHMARK_LEARNING_RATES),
            "estimator_counts": list(MODERN_LABELS_BENCHMARK_ESTIMATOR_COUNTS),
            "max_depths": list(MODERN_LABELS_BENCHMARK_MAX_DEPTHS),
            "min_leaf_sizes": list(MODERN_LABELS_BENCHMARK_MIN_LEAF_SIZES),
            "min_split_gains": list(MODERN_LABELS_BENCHMARK_MIN_SPLIT_GAINS),
            "early_stopping_rounds": MODERN_LABELS_BENCHMARK_EARLY_STOPPING_ROUNDS,
            "selection_mode": resolved_validation_policy["selection_mode"],
            "validation_strategy": resolved_validation_policy["validation_strategy"],
            "validation_fold_count": resolved_validation_policy[
                "validation_fold_count"
            ],
            "nested_outer_fold_count": resolved_validation_policy[
                "nested_outer_fold_count"
            ],
            "validation_unit": resolved_validation_policy["validation_unit"],
            "nested_feasibility": resolved_validation_policy["nested_feasibility"],
            "attempted_strategies": resolved_validation_policy[
                "attempted_strategies"
            ],
            "promotion_policy": {
                "require_labels_dataset_for_promotion": True,
                "require_nested_estimation_for_promotion": (
                    resolved_validation_policy["selection_mode"]
                    == "nested_outer_estimate"
                ),
                "min_nested_outer_selection_rate": (
                    0.5
                    if resolved_validation_policy["selection_mode"]
                    == "nested_outer_estimate"
                    else None
                ),
                "require_nested_temporal_latest_win_for_promotion": (
                    resolved_validation_policy["validation_strategy"]
                    == "temporal_holdout_backtest"
                ),
                "nested_temporal_recent_window_size": (
                    2
                    if resolved_validation_policy["validation_strategy"]
                    == "temporal_holdout_backtest"
                    else None
                ),
                "max_spatial_slice_validation_rmse_regression": 0.05,
                "max_temporal_slice_validation_rmse_regression": (
                    0.05
                    if resolved_validation_policy["validation_strategy"]
                    == "temporal_holdout_backtest"
                    else None
                ),
                "max_spatial_slice_regression_count": 0,
                "max_temporal_slice_regression_count": (
                    0
                    if resolved_validation_policy["validation_strategy"]
                    == "temporal_holdout_backtest"
                    else None
                ),
                "slice_regression_min_rows": 2,
                "stability_window_runs": 2,
                "required_consistent_wins": 2,
            },
        }

        return RunModernLabelsBenchmarkResponse(
            preset=MODERN_LABELS_BENCHMARK_PRESET,
            datasetResolution=dataset_resolution,
            datasetExport=dataset_export_response,
            resolvedPolicy=resolved_policy,
            selection=selection_response,
        )

    @staticmethod
    def build_modern_labels_benchmark_recommendation(
        benchmark: RunModernLabelsBenchmarkResponse,
    ) -> dict:
        selection = benchmark.selection
        promotion_decision = dict(selection.promotion_decision or {})
        best_vs_active = dict(selection.best_candidate_comparison or {})
        best_model_version = str(selection.best_model_version or "")
        active_model_version = str(selection.active_model_version or "")

        has_best_candidate = bool(best_model_version)
        challenger_differs_from_active = (
            has_best_candidate
            and best_model_version != active_model_version
        )
        promotion_eligible = bool(promotion_decision.get("eligible"))

        skipped_reasons: list[str] = []
        if not has_best_candidate:
            skipped_reasons.append("missing_best_candidate")
        if has_best_candidate and not challenger_differs_from_active:
            skipped_reasons.append("best_candidate_matches_active_model")
        if has_best_candidate and not promotion_eligible:
            skipped_reasons.append("promotion_decision_not_eligible")

        actionable = (
            has_best_candidate
            and challenger_differs_from_active
            and promotion_eligible
        )

        if actionable:
            summary = (
                f"Modern labels benchmark promotes challenger {best_model_version} "
                f"over active model {active_model_version} on dataset "
                f"{selection.dataset_version}. "
                "Action: open a governed promotion review."
            )
            recommended_action = (
                "open_promotion_review_for_modern_labels_benchmark"
            )
        else:
            reason_display = ", ".join(skipped_reasons) or "no_evidence"
            summary = (
                f"Modern labels benchmark on dataset {selection.dataset_version} "
                f"did not produce an actionable promotion recommendation "
                f"(reason: {reason_display}). "
                "No review task will be opened."
            )
            recommended_action = "no_action_required"

        return {
            "preset": benchmark.preset,
            "actionable": actionable,
            "recommended_action": recommended_action,
            "summary": summary,
            "active_model_version": active_model_version,
            "best_model_version": best_model_version,
            "dataset_version": selection.dataset_version,
            "selection_version": selection.selection_version,
            "has_best_candidate": has_best_candidate,
            "best_candidate_differs_from_active": (
                challenger_differs_from_active
            ),
            "promotion_decision_eligible": promotion_eligible,
            "skipped_reasons": skipped_reasons,
            "promotion_decision": promotion_decision,
            "best_candidate_comparison": best_vs_active,
            "blocking_reasons": list(
                promotion_decision.get("blocking_reasons") or []
            ),
        }

    def _benchmark_alert_targets(self) -> list[dict]:
        usernames = (
            list(self.settings.notification_model_monitoring_usernames)
            or list(self.settings.notification_release_ops_usernames)
            or (
                [self.settings.seed_admin_username]
                if self.settings.seed_admin_username
                else []
            )
        )
        normalized: list[dict] = []
        seen: set[str] = set()
        for index, username in enumerate(usernames):
            normalized_username = str(username).strip()
            if not normalized_username or normalized_username in seen:
                continue
            seen.add(normalized_username)
            normalized.append(
                {
                    "username": normalized_username,
                    "routing_audience": "model_monitoring_watch",
                    "routing_reason": "modern_labels_benchmark_review",
                    "is_primary": index == 0,
                }
            )
        return normalized

    @staticmethod
    def _build_modern_labels_benchmark_alert_details(
        *,
        benchmark: RunModernLabelsBenchmarkResponse,
        recommendation: dict,
    ) -> dict:
        selection = benchmark.selection
        promotion_decision = dict(selection.promotion_decision or {})
        dataset_export = (
            benchmark.dataset_export.model_dump(by_alias=True)
            if benchmark.dataset_export is not None
            else None
        )
        return {
            "preset": benchmark.preset,
            "model_version": selection.active_model_version,
            "active_model_version": selection.active_model_version,
            "best_model_version": selection.best_model_version,
            "dataset_version": selection.dataset_version,
            "selection_version": selection.selection_version,
            "selection_artifact_path": selection.artifact_path,
            "selection_job_id": selection.job.id,
            "recommended_action": recommendation["recommended_action"],
            "promotion_decision": promotion_decision,
            "best_candidate_comparison": dict(
                selection.best_candidate_comparison or {}
            ),
            "family_rollups": list(selection.family_rollups or []),
            "nested_estimation": dict(selection.nested_estimation or {}),
            "resolved_policy": dict(benchmark.resolved_policy or {}),
            "dataset_resolution": dict(benchmark.dataset_resolution or {}),
            "dataset_export": dataset_export,
            "recommendation": recommendation,
            "summary": recommendation["summary"],
        }

    def run_modern_labels_benchmark_with_review(
        self,
        *,
        dataset_version: str | None = None,
        auto_export_dataset: bool = True,
        dataset_export_version: str | None = None,
        label_sources: list[str] | None = None,
        max_labels: int = 250,
        observed_after: datetime | None = None,
        observed_before: datetime | None = None,
        validation_strategy: str | None = None,
        validation_fold_count: int | None = None,
        nested_outer_fold_count: int | None = None,
        version: str | None = None,
        version_prefix: str | None = None,
        promote_best: bool = False,
        promotion_reason: str | None = None,
        open_review_task: bool = True,
        assigned_reviewer: str | None = None,
        due_at: datetime | None = None,
        notes: str | None = None,
        opened_by: str,
        origin: str = "manual",
    ) -> RunModernLabelsBenchmarkReviewResponse:
        benchmark = self.run_modern_labels_benchmark(
            dataset_version=dataset_version,
            auto_export_dataset=auto_export_dataset,
            dataset_export_version=dataset_export_version,
            label_sources=label_sources,
            max_labels=max_labels,
            observed_after=observed_after,
            observed_before=observed_before,
            validation_strategy=validation_strategy,
            validation_fold_count=validation_fold_count,
            nested_outer_fold_count=nested_outer_fold_count,
            version=version,
            version_prefix=version_prefix,
            promote_best=promote_best,
            promotion_reason=promotion_reason,
            promoted_by=opened_by,
            origin=origin,
        )
        recommendation = self.build_modern_labels_benchmark_recommendation(
            benchmark
        )

        created_alerts: list[NotificationEventRead] = []
        acknowledged_alerts: list[NotificationEventRead] = []
        review_task: ModelReviewTaskRead | None = None

        if recommendation["actionable"] and open_review_task:
            targets = self._benchmark_alert_targets()
            if targets:
                alert_details = (
                    self._build_modern_labels_benchmark_alert_details(
                        benchmark=benchmark,
                        recommendation=recommendation,
                    )
                )
                alert_title = (
                    "Modern labels benchmark recommends promotion review for "
                    f"{benchmark.selection.best_model_version}"
                )
                alert_message = recommendation["summary"]
                alerts = self.notification_service.create_routed_events(
                    event_type=(
                        MODEL_MODERN_LABELS_BENCHMARK_ALERT_EVENT_TYPE
                    ),
                    severity="warning",
                    title=alert_title,
                    message=alert_message,
                    targets=targets,
                    details=alert_details,
                    template_key=(
                        MODEL_MODERN_LABELS_BENCHMARK_ALERT_EVENT_TYPE
                    ),
                )
                created_alerts = [
                    self.notification_service._read(alert) for alert in alerts
                ]
                if alerts:
                    ack_response = (
                        self.notification_service.acknowledge_notifications(
                            notification_ids=[alert.id for alert in alerts],
                            acknowledged_by=opened_by,
                        )
                    )
                    acknowledged_alerts = list(ack_response.notifications)
                    open_response = (
                        self.model_review_task_service
                        .open_review_tasks_from_alerts(
                            notification_ids=[alert.id for alert in alerts],
                            review_type="promotion_review",
                            opened_by=opened_by,
                            assigned_reviewer=assigned_reviewer,
                            due_at=due_at,
                            notes=notes,
                        )
                    )
                    if open_response.tasks:
                        review_task = open_response.tasks[0]
                    recommendation = {
                        **recommendation,
                        "created_alert_ids": [alert.id for alert in alerts],
                        "review_task_id": (
                            review_task.id if review_task is not None else None
                        ),
                    }

        return RunModernLabelsBenchmarkReviewResponse(
            benchmark=benchmark,
            recommendation=recommendation,
            createdAlerts=created_alerts,
            acknowledgedAlerts=acknowledged_alerts,
            reviewTask=review_task,
        )

    def list_selection_runs(self) -> list[ModelSelectionRunSummaryRead]:
        runs: list[ModelSelectionRunSummaryRead] = []
        for version in self.selection_registry.list_versions():
            run = self.selection_registry.load(version)
            best_candidate = next(
                candidate for candidate in run["candidates"] if candidate["rank"] == 1
            )
            runs.append(
                ModelSelectionRunSummaryRead(
                    version=version,
                    selectionId=run.get("selection_id", "unknown"),
                    artifactType=run.get("artifact_type", "model_selection_run"),
                    datasetVersion=run["dataset_version"],
                    createdAt=run["created_at"],
                    candidateCount=int(run["candidate_count"]),
                    bestModelVersion=run["best_model_version"],
                    promoted=bool(run["promoted"]),
                    activeModelVersion=run["active_model_version"],
                    bestValidationRmse=float(best_candidate["validation_rmse"]),
                    bestValidationRiskLevelAccuracy=float(
                        best_candidate["validation_risk_level_accuracy"]
                    ),
                    promotionDecision=run.get("promotion_decision") or {},
                )
            )
        return sorted(runs, key=lambda item: item.created_at, reverse=True)

    def get_selection_run(self, version: str) -> ModelSelectionRunDetailRead:
        run = self.selection_registry.load(version)
        return ModelSelectionRunDetailRead(
            version=version,
            selectionId=run.get("selection_id", "unknown"),
            artifactType=run.get("artifact_type", "model_selection_run"),
            artifactPath=str(self._selection_path(version)),
            datasetVersion=run["dataset_version"],
            createdAt=run["created_at"],
            comparisonPolicy=run.get("comparison_policy") or {},
            gatePolicy=run.get("gate_policy") or {},
            datasetContext=run.get("dataset_context") or {},
            candidateCount=int(run["candidate_count"]),
            bestModelVersion=run["best_model_version"],
            promoted=bool(run["promoted"]),
            promotion=run.get("promotion") or {},
            promotionDecision=run.get("promotion_decision") or {},
            activeModelVersion=run["active_model_version"],
            bestVsActiveComparison=run.get("best_vs_active_comparison") or {},
            familyRollups=list(run.get("family_rollups") or []),
            nestedEstimation=run.get("nested_estimation") or {},
            candidates=[
                self._candidate_read(candidate) for candidate in run["candidates"]
            ],
        )

    def tune_model(
        self,
        *,
        dataset_version: str,
        alphas: list[float] | None,
        model_family: str = "linear_ridge",
        model_families: list[str] | None = None,
        selection_mode: str = "single_stage",
        validation_strategy: str = "dataset_holdout",
        validation_fold_count: int | None = None,
        nested_outer_fold_count: int | None = None,
        knot_counts: list[int] | None = None,
        learning_rates: list[float] | None = None,
        estimator_counts: list[int] | None = None,
        max_depths: list[int] | None = None,
        min_leaf_sizes: list[int] | None = None,
        min_split_gains: list[float] | None = None,
        early_stopping_rounds: int | None = None,
        version: str | None = None,
        version_prefix: str | None = None,
        promote_best: bool = False,
        promotion_reason: str | None = None,
        top_error_count: int = 5,
        min_validation_rmse_improvement: float | None = None,
        min_validation_rows: int | None = None,
        require_labels_dataset_for_promotion: bool | None = None,
        require_nested_estimation_for_promotion: bool | None = None,
        min_calibration_gain: float | None = None,
        min_nested_outer_validation_rmse_improvement: float | None = None,
        min_nested_outer_selection_rate: float | None = None,
        require_nested_temporal_latest_win_for_promotion: bool | None = None,
        min_nested_temporal_outer_bucket_count: int | None = None,
        min_nested_temporal_latest_validation_rmse_improvement: float | None = None,
        nested_temporal_recent_window_size: int | None = None,
        min_nested_temporal_recent_win_rate: float | None = None,
        min_nested_temporal_recent_average_validation_rmse_improvement: float
        | None = None,
        max_spatial_slice_validation_rmse_regression: float | None = None,
        max_temporal_slice_validation_rmse_regression: float | None = None,
        max_spatial_slice_regression_count: int | None = None,
        max_temporal_slice_regression_count: int | None = None,
        slice_regression_min_rows: int | None = None,
        stability_window_runs: int | None = None,
        required_consistent_wins: int | None = None,
        stability_require_same_dataset_family: bool | None = None,
        stability_require_same_dataset_taxonomy: bool | None = None,
        stability_require_same_evaluation_cohort: bool | None = None,
        stability_max_time_window_gap_days: int | None = None,
        stability_max_cohort_distance: int | None = None,
        promoted_by: str | None = None,
        origin: str = "manual",
    ) -> TuneModelResponse:
        resolved_model_families = self._resolve_model_families(
            model_family=model_family,
            model_families=model_families,
        )
        normalized_selection_mode = str(selection_mode or "single_stage").strip()
        if normalized_selection_mode not in {"single_stage", "nested_outer_estimate"}:
            raise ApiError(
                400,
                "invalid_selection_mode",
                f"Unsupported selection mode: {selection_mode}",
            )
        normalized_model_family = (
            resolved_model_families[0]
            if len(resolved_model_families) == 1
            else "mixed"
        )

        unique_alphas: list[float] = []
        if set(resolved_model_families) & {"linear_ridge", "beta_regression"}:
            resolved_alphas = alphas or [0.25, 0.75, 1.5]
            if any(alpha <= 0 for alpha in resolved_alphas):
                raise ApiError(
                    400,
                    "invalid_alpha",
                    "All alpha candidates must be greater than zero.",
                )
            for alpha in resolved_alphas:
                if alpha not in unique_alphas:
                    unique_alphas.append(alpha)
        else:
            unique_alphas = []
        additive_spline_configs = (
            self._resolve_additive_spline_candidate_configs(
                alphas=alphas,
                knot_counts=knot_counts,
            )
            if "additive_spline" in resolved_model_families
            else []
        )
        boosted_tree_configs = (
            self._resolve_boosted_tree_candidate_configs(
                learning_rates=learning_rates,
                estimator_counts=estimator_counts,
                max_depths=max_depths,
                min_leaf_sizes=min_leaf_sizes,
                min_split_gains=min_split_gains,
                early_stopping_rounds=early_stopping_rounds,
            )
            if set(resolved_model_families) & {"gradient_boosted_tree", "xgboost"}
            else []
        )

        selection_version = version or f"selection-{dataset_version}"
        started_at = datetime.now(timezone.utc).replace(microsecond=0)
        job = JobExecution(
            job_type="model_tuning",
            status="running",
            started_at=started_at,
            details={
                "origin": origin,
                "dataset_version": dataset_version,
                "selection_version": selection_version,
                "model_family": normalized_model_family,
                "model_families": resolved_model_families,
                "selection_mode": normalized_selection_mode,
                "validation_strategy": validation_strategy,
                "validation_fold_count": validation_fold_count,
                "nested_outer_fold_count": nested_outer_fold_count,
                "require_nested_estimation_for_promotion": (
                    require_nested_estimation_for_promotion
                ),
                "min_nested_outer_validation_rmse_improvement": (
                    min_nested_outer_validation_rmse_improvement
                ),
                "min_nested_outer_selection_rate": min_nested_outer_selection_rate,
                "require_nested_temporal_latest_win_for_promotion": (
                    require_nested_temporal_latest_win_for_promotion
                ),
                "min_nested_temporal_outer_bucket_count": (
                    min_nested_temporal_outer_bucket_count
                ),
                "min_nested_temporal_latest_validation_rmse_improvement": (
                    min_nested_temporal_latest_validation_rmse_improvement
                ),
                "nested_temporal_recent_window_size": (
                    nested_temporal_recent_window_size
                ),
                "min_nested_temporal_recent_win_rate": (
                    min_nested_temporal_recent_win_rate
                ),
                "min_nested_temporal_recent_average_validation_rmse_improvement": (
                    min_nested_temporal_recent_average_validation_rmse_improvement
                ),
                "max_spatial_slice_validation_rmse_regression": (
                    max_spatial_slice_validation_rmse_regression
                ),
                "max_temporal_slice_validation_rmse_regression": (
                    max_temporal_slice_validation_rmse_regression
                ),
                "max_spatial_slice_regression_count": (
                    max_spatial_slice_regression_count
                ),
                "max_temporal_slice_regression_count": (
                    max_temporal_slice_regression_count
                ),
                "slice_regression_min_rows": slice_regression_min_rows,
                "alphas": unique_alphas,
                "additive_spline_configs": additive_spline_configs,
                "boosted_tree_configs": boosted_tree_configs,
                "promote_best": promote_best,
            },
        )
        self.session.add(job)
        self.session.flush()

        try:
            dataset = self.dataset_registry.load(dataset_version)
            rows = rows_from_dataset(dataset)
            dataset_mode = self._dataset_mode(dataset)
            if self.settings.real_data_only and dataset_mode == "seed":
                raise ApiError(
                    409,
                    "seed_dataset_disabled",
                    "Model tuning on seed-backed datasets is disabled because REAL_DATA_ONLY is enabled.",
                )
            dataset_context = self._dataset_context(dataset)
            candidate_prefix = version_prefix or f"{dataset_version}-candidate"
            candidates: list[dict] = []
            candidate_evaluations: dict[str, dict] = {}
            row_contexts = self._dataset_row_contexts(dataset)
            validation_plan = self._resolve_validation_plan(
                dataset=dataset,
                validation_strategy=validation_strategy,
                validation_fold_count=validation_fold_count,
            )
            candidate_specs: list[dict] = []

            for requested_family in resolved_model_families:
                if requested_family == "linear_ridge":
                    for alpha in unique_alphas:
                        candidate_specs.append(
                            {
                                "candidate_version": (
                                    f"{candidate_prefix}-alpha-{alpha_slug(alpha)}"
                                ),
                                "model_family": "linear_ridge",
                                "alpha": alpha,
                                "hyperparameters": {"alpha": alpha},
                            }
                        )
                elif requested_family == "beta_regression":
                    for alpha in unique_alphas:
                        candidate_specs.append(
                            {
                                "candidate_version": (
                                    f"{candidate_prefix}-beta-alpha-{alpha_slug(alpha)}"
                                ),
                                "model_family": "beta_regression",
                                "alpha": alpha,
                                "hyperparameters": {"alpha": alpha},
                            }
                        )
                elif requested_family == "additive_spline":
                    for config in additive_spline_configs:
                        candidate_specs.append(
                            {
                                "candidate_version": (
                                    f"{candidate_prefix}-"
                                    f"{self._additive_spline_candidate_suffix(config)}"
                                ),
                                "model_family": "additive_spline",
                                "alpha": float(config["alpha"]),
                                "hyperparameters": dict(config),
                            }
                        )
                elif requested_family == "xgboost":
                    for config in boosted_tree_configs:
                        candidate_specs.append(
                            {
                                "candidate_version": (
                                    f"{candidate_prefix}-"
                                    f"{self._xgboost_candidate_suffix(config)}"
                                ),
                                "model_family": "xgboost",
                                "alpha": None,
                                "hyperparameters": dict(config),
                            }
                        )
                else:
                    for config in boosted_tree_configs:
                        candidate_specs.append(
                            {
                                "candidate_version": (
                                    f"{candidate_prefix}-"
                                    f"{self._boosted_tree_candidate_suffix(config)}"
                                ),
                                "model_family": "gradient_boosted_tree",
                                "alpha": None,
                                "hyperparameters": dict(config),
                            }
                        )

            for candidate_spec in candidate_specs:
                candidate_version = str(candidate_spec["candidate_version"])
                artifact_path, artifact = self._export_candidate_artifact(
                    model_family=str(candidate_spec["model_family"]),
                    version=candidate_version,
                    rows=rows,
                    dataset_name=dataset_version,
                    hyperparameters=dict(candidate_spec["hyperparameters"]),
                )
                evaluation = build_model_evaluation(
                    version=f"eval-{candidate_version}-on-{dataset_version}",
                    artifact=artifact,
                    dataset=dataset,
                    top_error_count=top_error_count,
                )
                validation_summary = self._evaluate_candidate_under_validation_plan(
                    model_family=str(candidate_spec["model_family"]),
                    version=candidate_version,
                    dataset=dataset,
                    rows=rows,
                    row_contexts=row_contexts,
                    plan=validation_plan,
                    hyperparameters=dict(candidate_spec["hyperparameters"]),
                    top_error_count=top_error_count,
                )
                candidate_evaluations[candidate_version] = {
                    "evaluation": evaluation,
                    "validation_summary": validation_summary,
                }
                candidates.append(
                    self._candidate_payload(
                        candidate_version=candidate_version,
                        model_family=str(candidate_spec["model_family"]),
                        alpha=(
                            float(candidate_spec["alpha"])
                            if candidate_spec.get("alpha") is not None
                            else None
                        ),
                        artifact_path=artifact_path,
                        artifact=artifact,
                        evaluation=evaluation,
                        validation_summary=validation_summary,
                    )
                )

            active_model_version = self.model_registry.active_version()
            active_artifact = self.model_registry.load(active_model_version)
            active_evaluation = build_model_evaluation(
                version=f"eval-{active_model_version}-on-{dataset_version}",
                artifact=active_artifact,
                dataset=dataset,
                top_error_count=top_error_count,
            )
            active_validation_summary = self._evaluate_artifact_under_validation_plan(
                artifact=active_artifact,
                dataset=dataset,
                rows=rows,
                row_contexts=row_contexts,
                plan=validation_plan,
                top_error_count=top_error_count,
            )
            active_validation_probability = self._probability_metrics(
                active_validation_summary["metrics"]
            )

            gate_policy = {
                "requested_model_family": normalized_model_family,
                "requested_model_families": resolved_model_families,
                "selection_mode": normalized_selection_mode,
                "dataset_mode": dataset_mode,
                "validation_strategy": validation_plan["strategy"],
                "validation_unit": validation_plan["validation_unit"],
                "validation_fold_count": validation_plan["fold_count"],
                "nested_outer_fold_count": nested_outer_fold_count,
                "require_nested_estimation_for_promotion": (
                    require_nested_estimation_for_promotion
                    if require_nested_estimation_for_promotion is not None
                    else self.settings.model_promotion_require_nested_estimation
                ),
                "require_labels_dataset_for_promotion": (
                    require_labels_dataset_for_promotion
                    if require_labels_dataset_for_promotion is not None
                    else self.settings.model_promotion_require_labels_dataset
                ),
                "min_validation_rmse_improvement": (
                    min_validation_rmse_improvement
                    if min_validation_rmse_improvement is not None
                    else self.settings.model_promotion_min_validation_rmse_improvement
                ),
                "min_validation_rows": (
                    min_validation_rows
                    if min_validation_rows is not None
                    else self.settings.model_promotion_min_validation_rows
                ),
                "min_calibration_gain": (
                    min_calibration_gain
                    if min_calibration_gain is not None
                    else self.settings.model_promotion_min_calibration_gain
                ),
                "min_nested_outer_validation_rmse_improvement": (
                    min_nested_outer_validation_rmse_improvement
                    if min_nested_outer_validation_rmse_improvement is not None
                    else self.settings.model_promotion_min_nested_outer_validation_rmse_improvement
                ),
                "min_nested_outer_selection_rate": (
                    min_nested_outer_selection_rate
                    if min_nested_outer_selection_rate is not None
                    else self.settings.model_promotion_min_nested_outer_selection_rate
                ),
                "require_nested_temporal_latest_win_for_promotion": (
                    require_nested_temporal_latest_win_for_promotion
                    if require_nested_temporal_latest_win_for_promotion is not None
                    else self.settings.model_promotion_require_nested_temporal_latest_win
                ),
                "min_nested_temporal_outer_bucket_count": (
                    min_nested_temporal_outer_bucket_count
                    if min_nested_temporal_outer_bucket_count is not None
                    else self.settings.model_promotion_min_nested_temporal_outer_bucket_count
                ),
                "min_nested_temporal_latest_validation_rmse_improvement": (
                    min_nested_temporal_latest_validation_rmse_improvement
                    if min_nested_temporal_latest_validation_rmse_improvement
                    is not None
                    else self.settings.model_promotion_min_nested_temporal_latest_validation_rmse_improvement
                ),
                "nested_temporal_recent_window_size": (
                    nested_temporal_recent_window_size
                    if nested_temporal_recent_window_size is not None
                    else self.settings.model_promotion_nested_temporal_recent_window_size
                ),
                "min_nested_temporal_recent_win_rate": (
                    min_nested_temporal_recent_win_rate
                    if min_nested_temporal_recent_win_rate is not None
                    else self.settings.model_promotion_min_nested_temporal_recent_win_rate
                ),
                "min_nested_temporal_recent_average_validation_rmse_improvement": (
                    min_nested_temporal_recent_average_validation_rmse_improvement
                    if min_nested_temporal_recent_average_validation_rmse_improvement
                    is not None
                    else self.settings.model_promotion_min_nested_temporal_recent_average_validation_rmse_improvement
                ),
                "max_spatial_slice_validation_rmse_regression": (
                    max_spatial_slice_validation_rmse_regression
                    if max_spatial_slice_validation_rmse_regression is not None
                    else self.settings.model_promotion_max_spatial_slice_validation_rmse_regression
                ),
                "max_temporal_slice_validation_rmse_regression": (
                    max_temporal_slice_validation_rmse_regression
                    if max_temporal_slice_validation_rmse_regression is not None
                    else self.settings.model_promotion_max_temporal_slice_validation_rmse_regression
                ),
                "max_spatial_slice_regression_count": (
                    max_spatial_slice_regression_count
                    if max_spatial_slice_regression_count is not None
                    else self.settings.model_promotion_max_spatial_slice_regression_count
                ),
                "max_temporal_slice_regression_count": (
                    max_temporal_slice_regression_count
                    if max_temporal_slice_regression_count is not None
                    else self.settings.model_promotion_max_temporal_slice_regression_count
                ),
                "slice_regression_min_rows": (
                    slice_regression_min_rows
                    if slice_regression_min_rows is not None
                    else self.settings.model_promotion_slice_regression_min_rows
                ),
                "stability_window_runs": (
                    stability_window_runs
                    if stability_window_runs is not None
                    else self.settings.model_promotion_stability_window_runs
                ),
                "required_consistent_wins": (
                    required_consistent_wins
                    if required_consistent_wins is not None
                    else self.settings.model_promotion_required_consistent_wins
                ),
                "stability_require_same_dataset_family": (
                    stability_require_same_dataset_family
                    if stability_require_same_dataset_family is not None
                    else self.settings.model_promotion_stability_require_same_dataset_family
                ),
                "stability_require_same_dataset_taxonomy": (
                    stability_require_same_dataset_taxonomy
                    if stability_require_same_dataset_taxonomy is not None
                    else self.settings.model_promotion_stability_require_same_dataset_taxonomy
                ),
                "stability_require_same_evaluation_cohort": (
                    stability_require_same_evaluation_cohort
                    if stability_require_same_evaluation_cohort is not None
                    else self.settings.model_promotion_stability_require_same_evaluation_cohort
                ),
                "stability_max_time_window_gap_days": (
                    stability_max_time_window_gap_days
                    if stability_max_time_window_gap_days is not None
                    else self.settings.model_promotion_stability_max_time_window_gap_days
                ),
                "stability_max_cohort_distance": (
                    stability_max_cohort_distance
                    if stability_max_cohort_distance is not None
                    else self.settings.model_promotion_stability_max_cohort_distance
                ),
                "active_model_version": active_model_version,
                "active_model_family": str(
                    active_artifact.get("model_family") or "baseline"
                ),
                "active_validation_rmse": active_validation_summary["metrics"][
                    "calibrated_metrics"
                ]["rmse"],
                "active_validation_risk_level_accuracy": active_validation_summary[
                    "metrics"
                ]["risk_level_accuracy"],
                "active_validation_rows": active_validation_summary["metrics"]["rows"],
                "active_validation_brier_score": active_validation_probability.get(
                    "brier_score"
                ),
                "active_validation_auprc": active_validation_probability.get("auprc"),
                "active_validation_recall": active_validation_probability.get("recall"),
                "active_validation_mcc": active_validation_probability.get("mcc"),
                "active_validation_ece": active_validation_probability.get("ece"),
                "active_validation_summary": {
                    key: value
                    for key, value in active_validation_summary.items()
                    if key != "metrics"
                },
                "active_calibration_gain": (
                    (active_evaluation.get("diagnostics") or {})
                    .get("calibration_effect", {})
                    .get("validation_rmse_improvement")
                ),
            }
            comparison_policy = {
                "primary_metric": "validation_rmse",
                "tie_breakers": [
                    "validation_brier_score",
                    "validation_auprc_desc",
                    "validation_recall_desc",
                    "validation_mcc_desc",
                    "validation_risk_level_accuracy_desc",
                    "overall_rmse",
                    "alpha",
                    "model_version",
                ],
                "selection_mode": normalized_selection_mode,
                "requested_validation_strategy": validation_strategy,
                "validation_strategy": validation_plan["strategy"],
                "validation_unit": validation_plan["validation_unit"],
                "requested_fold_count": validation_fold_count,
                "folds_evaluated": validation_plan["fold_count"],
                "nested_outer_requested_fold_count": nested_outer_fold_count,
            }
            nested_estimation = (
                self._nested_selection_estimation(
                    active_artifact=active_artifact,
                    dataset=dataset,
                    rows=rows,
                    row_contexts=row_contexts,
                    candidate_specs=candidate_specs,
                    validation_strategy=validation_strategy,
                    validation_fold_count=validation_fold_count,
                    nested_outer_fold_count=nested_outer_fold_count,
                    top_error_count=top_error_count,
                )
                if normalized_selection_mode == "nested_outer_estimate"
                else {}
            )
            if nested_estimation:
                temporal_coverage = (
                    nested_estimation.setdefault("temporal_cohort_coverage", {})
                )
                temporal_coverage["recent_window"] = (
                    self._nested_temporal_recent_window_summary(
                        nested_estimation,
                        recent_window_size=gate_policy[
                            "nested_temporal_recent_window_size"
                        ],
                    )
                )
            selection_run = build_model_selection_run(
                version=selection_version,
                dataset_version=dataset_version,
                candidates=candidates,
                promoted=False,
                active_model_version=active_model_version,
                gate_policy=gate_policy,
                dataset_context=dataset_context,
                comparison_policy=comparison_policy,
                nested_estimation=nested_estimation,
            )
            selection_run["family_rollups"] = self._family_rollups(
                selection_run["candidates"]
            )
            for candidate in selection_run["candidates"]:
                candidate["comparison"] = {
                    "vs_active_validation_rmse_delta": round(
                        candidate["validation_rmse"]
                        - gate_policy["active_validation_rmse"],
                        6,
                    ),
                    "vs_active_validation_rmse_improvement": round(
                        gate_policy["active_validation_rmse"]
                        - candidate["validation_rmse"],
                        6,
                    ),
                    "vs_active_validation_risk_level_accuracy_delta": round(
                        candidate["validation_risk_level_accuracy"]
                        - gate_policy["active_validation_risk_level_accuracy"],
                        6,
                    ),
                    "vs_active_validation_brier_score_delta": self._metric_delta(
                        gate_policy["active_validation_brier_score"],
                        candidate.get("validation_brier_score"),
                    ),
                    "vs_active_validation_brier_score_improvement": self._lower_is_better_improvement(
                        gate_policy["active_validation_brier_score"],
                        candidate.get("validation_brier_score"),
                    ),
                    "vs_active_validation_auprc_delta": self._higher_is_better_gain(
                        gate_policy["active_validation_auprc"],
                        candidate.get("validation_auprc"),
                    ),
                    "vs_active_validation_recall_delta": self._higher_is_better_gain(
                        gate_policy["active_validation_recall"],
                        candidate.get("validation_recall"),
                    ),
                    "vs_active_validation_mcc_delta": self._higher_is_better_gain(
                        gate_policy["active_validation_mcc"],
                        candidate.get("validation_mcc"),
                    ),
                    "vs_active_validation_ece_delta": self._metric_delta(
                        gate_policy["active_validation_ece"],
                        candidate.get("validation_ece"),
                    ),
                    "vs_active_validation_ece_improvement": self._lower_is_better_improvement(
                        gate_policy["active_validation_ece"],
                        candidate.get("validation_ece"),
                    ),
                }
            best_candidate = next(
                candidate
                for candidate in selection_run["candidates"]
                if candidate["rank"] == 1
            )
            if selection_run.get("nested_estimation"):
                best_candidate_signature = self._candidate_signature(best_candidate)
                selected_candidate_frequencies = list(
                    selection_run["nested_estimation"].get(
                        "selected_candidate_frequencies"
                    )
                    or []
                )
                selected_count = next(
                    (
                        int(item.get("count") or 0)
                        for item in selected_candidate_frequencies
                        if str(item.get("candidate_signature")) == best_candidate_signature
                    ),
                    0,
                )
                outer_fold_count = int(
                    selection_run["nested_estimation"].get("outer_fold_count") or 0
                )
                selection_run["nested_estimation"][
                    "final_candidate_signature"
                ] = best_candidate_signature
                selection_run["nested_estimation"][
                    "final_candidate_outer_selection_count"
                ] = selected_count
                selection_run["nested_estimation"][
                    "final_candidate_outer_selection_rate"
                ] = round(selected_count / outer_fold_count, 6) if outer_fold_count > 0 else 0.0
            best_candidate_bundle = candidate_evaluations[best_candidate["model_version"]]
            best_candidate_evaluation = best_candidate_bundle["evaluation"]
            best_candidate_validation_summary = best_candidate_bundle[
                "validation_summary"
            ]
            best_candidate_validation_probability = self._probability_metrics(
                best_candidate_validation_summary["metrics"]
            )
            best_vs_active_comparison = self._best_vs_active_comparison(
                active_artifact=active_artifact,
                challenger_artifact=self.model_registry.load(
                    best_candidate["model_version"]
                ),
                active_evaluation=active_evaluation,
                challenger_evaluation=best_candidate_evaluation,
                active_validation_summary=active_validation_summary,
                challenger_validation_summary=best_candidate_validation_summary,
            )
            slice_gate_summary = {
                "spatial_block": self._slice_gate_summary(
                    (
                        best_vs_active_comparison.get("validation_slice_deltas") or {}
                    ).get("by_spatial_block")
                    or {},
                    min_rows=gate_policy["slice_regression_min_rows"],
                ),
                "temporal_holdout_tag": self._slice_gate_summary(
                    (
                        best_vs_active_comparison.get("validation_slice_deltas") or {}
                    ).get("by_temporal_holdout_tag")
                    or {},
                    min_rows=gate_policy["slice_regression_min_rows"],
                ),
            }
            stability_assessment = self._stability_assessment(
                best_candidate=best_candidate,
                dataset_mode=dataset_mode,
                dataset_context=dataset_context,
                selection_version=selection_version,
                stability_window_runs=gate_policy["stability_window_runs"],
                required_consistent_wins=gate_policy["required_consistent_wins"],
                require_same_dataset_family=gate_policy[
                    "stability_require_same_dataset_family"
                ],
                require_same_dataset_taxonomy=gate_policy[
                    "stability_require_same_dataset_taxonomy"
                ],
                require_same_evaluation_cohort=gate_policy[
                    "stability_require_same_evaluation_cohort"
                ],
                max_time_window_gap_days=gate_policy[
                    "stability_max_time_window_gap_days"
                ],
                max_cohort_distance=gate_policy["stability_max_cohort_distance"],
                current_dataset_version=dataset_version,
            )
            nested_estimation_summary = selection_run.get("nested_estimation") or {}
            nested_comparison = (
                nested_estimation_summary.get("comparison_vs_active") or {}
            )
            nested_outer_selection_rate = nested_estimation_summary.get(
                "final_candidate_outer_selection_rate"
            )
            nested_outer_rmse_improvement = nested_comparison.get(
                "validation_rmse_improvement"
            )
            nested_temporal_coverage = (
                nested_estimation_summary.get("temporal_cohort_coverage") or {}
            )
            nested_temporal_latest_rmse_improvement = nested_temporal_coverage.get(
                "latest_fold_validation_rmse_improvement_vs_active"
            )
            nested_temporal_recent_window = (
                nested_temporal_coverage.get("recent_window") or {}
            )
            nested_temporal_recent_win_rate = nested_temporal_recent_window.get(
                "win_rate_vs_active"
            )
            nested_temporal_recent_avg_rmse_improvement = (
                nested_temporal_recent_window.get(
                    "average_validation_rmse_improvement_vs_active"
                )
            )
            spatial_slice_summary = slice_gate_summary["spatial_block"]
            temporal_slice_summary = slice_gate_summary["temporal_holdout_tag"]

            blocking_reasons: list[str] = []
            rmse_improvement = round(
                gate_policy["active_validation_rmse"]
                - best_candidate["validation_rmse"],
                6,
            )
            challenger_calibration_gain = (
                (best_candidate_evaluation.get("diagnostics") or {})
                .get("calibration_effect", {})
                .get("validation_rmse_improvement")
            )
            if (
                gate_policy["require_labels_dataset_for_promotion"]
                and dataset_mode != "labels"
            ):
                blocking_reasons.append("labels_dataset_required_for_promotion")
            if best_candidate["validation_rows"] <= 0:
                blocking_reasons.append("validation_rows_required")
            if best_candidate["validation_rows"] < gate_policy["min_validation_rows"]:
                blocking_reasons.append("minimum_validation_rows_not_met")
            if best_candidate["model_version"] == active_model_version:
                blocking_reasons.append("best_candidate_already_active")
            if rmse_improvement < gate_policy["min_validation_rmse_improvement"]:
                blocking_reasons.append("minimum_validation_rmse_improvement_not_met")
            if gate_policy["require_nested_estimation_for_promotion"]:
                if not nested_estimation_summary:
                    blocking_reasons.append(
                        "nested_estimation_required_for_promotion"
                    )
                elif nested_estimation_summary.get("outer_fold_count", 0) <= 0:
                    blocking_reasons.append(
                        "nested_outer_estimation_folds_required"
                    )
            if gate_policy["min_nested_outer_selection_rate"] > 0:
                if nested_outer_selection_rate is None:
                    blocking_reasons.append(
                        "nested_outer_selection_rate_unavailable"
                    )
                elif (
                    nested_outer_selection_rate
                    < gate_policy["min_nested_outer_selection_rate"]
                ):
                    blocking_reasons.append(
                        "minimum_nested_outer_selection_rate_not_met"
                    )
            if gate_policy["min_nested_outer_validation_rmse_improvement"] > 0:
                if nested_outer_rmse_improvement is None:
                    blocking_reasons.append(
                        "nested_outer_validation_rmse_improvement_unavailable"
                    )
                elif (
                    nested_outer_rmse_improvement
                    < gate_policy["min_nested_outer_validation_rmse_improvement"]
                ):
                    blocking_reasons.append(
                        "minimum_nested_outer_validation_rmse_improvement_not_met"
                    )
            if gate_policy["min_nested_temporal_outer_bucket_count"] > 0:
                if (
                    int(nested_temporal_coverage.get("bucket_count") or 0)
                    < gate_policy["min_nested_temporal_outer_bucket_count"]
                ):
                    blocking_reasons.append(
                        "minimum_nested_temporal_outer_bucket_count_not_met"
                    )
            if gate_policy["require_nested_temporal_latest_win_for_promotion"]:
                if not nested_temporal_coverage.get("available"):
                    blocking_reasons.append(
                        "nested_temporal_latest_win_unavailable"
                    )
                elif not nested_temporal_coverage.get("latest_fold_wins_vs_active"):
                    blocking_reasons.append("nested_temporal_latest_win_not_met")
            if (
                gate_policy[
                    "min_nested_temporal_latest_validation_rmse_improvement"
                ]
                > 0
            ):
                if nested_temporal_latest_rmse_improvement is None:
                    blocking_reasons.append(
                        "nested_temporal_latest_validation_rmse_improvement_unavailable"
                    )
                elif (
                    nested_temporal_latest_rmse_improvement
                    < gate_policy[
                        "min_nested_temporal_latest_validation_rmse_improvement"
                    ]
                ):
                    blocking_reasons.append(
                        "minimum_nested_temporal_latest_validation_rmse_improvement_not_met"
                    )
            if gate_policy["min_nested_temporal_recent_win_rate"] > 0:
                if not nested_temporal_recent_window.get("available"):
                    blocking_reasons.append(
                        "nested_temporal_recent_win_rate_unavailable"
                    )
                elif nested_temporal_recent_win_rate is None:
                    blocking_reasons.append(
                        "nested_temporal_recent_win_rate_unavailable"
                    )
                elif (
                    nested_temporal_recent_win_rate
                    < gate_policy["min_nested_temporal_recent_win_rate"]
                ):
                    blocking_reasons.append(
                        "minimum_nested_temporal_recent_win_rate_not_met"
                    )
            if (
                gate_policy[
                    "min_nested_temporal_recent_average_validation_rmse_improvement"
                ]
                > 0
            ):
                if not nested_temporal_recent_window.get("available"):
                    blocking_reasons.append(
                        "nested_temporal_recent_average_validation_rmse_improvement_unavailable"
                    )
                elif nested_temporal_recent_avg_rmse_improvement is None:
                    blocking_reasons.append(
                        "nested_temporal_recent_average_validation_rmse_improvement_unavailable"
                    )
                elif (
                    nested_temporal_recent_avg_rmse_improvement
                    < gate_policy[
                        "min_nested_temporal_recent_average_validation_rmse_improvement"
                    ]
                ):
                    blocking_reasons.append(
                        "minimum_nested_temporal_recent_average_validation_rmse_improvement_not_met"
                    )
            if gate_policy["max_spatial_slice_validation_rmse_regression"] >= 0:
                if not spatial_slice_summary.get("available"):
                    blocking_reasons.append("spatial_slice_regression_unavailable")
                elif (
                    float(
                        spatial_slice_summary.get("worst_validation_rmse_regression")
                        or 0.0
                    )
                    > gate_policy["max_spatial_slice_validation_rmse_regression"]
                ):
                    blocking_reasons.append(
                        "maximum_spatial_slice_validation_rmse_regression_not_met"
                    )
            if gate_policy["max_temporal_slice_validation_rmse_regression"] >= 0:
                if not temporal_slice_summary.get("available"):
                    blocking_reasons.append("temporal_slice_regression_unavailable")
                elif (
                    float(
                        temporal_slice_summary.get("worst_validation_rmse_regression")
                        or 0.0
                    )
                    > gate_policy["max_temporal_slice_validation_rmse_regression"]
                ):
                    blocking_reasons.append(
                        "maximum_temporal_slice_validation_rmse_regression_not_met"
                    )
            if gate_policy["max_spatial_slice_regression_count"] >= 0:
                if not spatial_slice_summary.get("available"):
                    blocking_reasons.append(
                        "spatial_slice_regression_count_unavailable"
                    )
                elif (
                    int(spatial_slice_summary.get("regression_count") or 0)
                    > gate_policy["max_spatial_slice_regression_count"]
                ):
                    blocking_reasons.append(
                        "maximum_spatial_slice_regression_count_not_met"
                    )
            if gate_policy["max_temporal_slice_regression_count"] >= 0:
                if not temporal_slice_summary.get("available"):
                    blocking_reasons.append(
                        "temporal_slice_regression_count_unavailable"
                    )
                elif (
                    int(temporal_slice_summary.get("regression_count") or 0)
                    > gate_policy["max_temporal_slice_regression_count"]
                ):
                    blocking_reasons.append(
                        "maximum_temporal_slice_regression_count_not_met"
                    )
            if challenger_calibration_gain is None:
                blocking_reasons.append("calibration_gain_unavailable")
            elif challenger_calibration_gain < gate_policy["min_calibration_gain"]:
                blocking_reasons.append("minimum_calibration_gain_not_met")
            if (
                stability_assessment["window_runs_considered"]
                < gate_policy["required_consistent_wins"]
            ):
                blocking_reasons.append("insufficient_stability_window_runs")
            if not stability_assessment["consistent_enough"]:
                blocking_reasons.append("minimum_consistent_wins_not_met")
            stability_shortfall = (
                stability_assessment["window_runs_considered"]
                < gate_policy["required_consistent_wins"]
                or not stability_assessment["consistent_enough"]
            )
            if (
                stability_shortfall
                and stability_assessment["excluded_reason_counts"].get(
                    "dataset_family_mismatch", 0
                )
                > 0
            ):
                blocking_reasons.append("stability_dataset_family_requirement_not_met")
            if (
                stability_shortfall
                and stability_assessment["excluded_reason_counts"].get(
                    "dataset_taxonomy_mismatch", 0
                )
                > 0
            ):
                blocking_reasons.append("stability_dataset_taxonomy_requirement_not_met")
            if (
                stability_shortfall
                and (
                    stability_assessment["excluded_reason_counts"].get(
                        "time_window_gap_exceeded", 0
                    )
                    > 0
                    or stability_assessment["excluded_reason_counts"].get(
                        "time_window_unavailable", 0
                    )
                    > 0
                )
            ):
                blocking_reasons.append("stability_time_window_requirement_not_met")
            if (
                stability_shortfall
                and (
                    stability_assessment["excluded_reason_counts"].get(
                        "evaluation_cohort_group_mismatch", 0
                    )
                    > 0
                    or stability_assessment["excluded_reason_counts"].get(
                        "evaluation_cohort_distance_exceeded", 0
                    )
                    > 0
                    or stability_assessment["excluded_reason_counts"].get(
                        "evaluation_cohort_unavailable", 0
                    )
                    > 0
                )
            ):
                blocking_reasons.append(
                    "stability_evaluation_cohort_requirement_not_met"
                )
            blocking_reasons = list(dict.fromkeys(blocking_reasons))

            promotion_decision = {
                "requested_promotion": promote_best,
                "eligible": len(blocking_reasons) == 0,
                "promoted": False,
                "blocking_reasons": blocking_reasons,
                "dataset_mode": dataset_mode,
                "active_model_version": active_model_version,
                "active_model_family": gate_policy["active_model_family"],
                "active_validation_rmse": gate_policy["active_validation_rmse"],
                "active_validation_risk_level_accuracy": gate_policy[
                    "active_validation_risk_level_accuracy"
                ],
                "active_validation_rows": gate_policy["active_validation_rows"],
                "active_validation_brier_score": gate_policy[
                    "active_validation_brier_score"
                ],
                "active_validation_auprc": gate_policy["active_validation_auprc"],
                "active_validation_recall": gate_policy["active_validation_recall"],
                "active_validation_mcc": gate_policy["active_validation_mcc"],
                "active_validation_ece": gate_policy["active_validation_ece"],
                "active_calibration_gain": gate_policy["active_calibration_gain"],
                "challenger_model_version": best_candidate["model_version"],
                "challenger_model_family": best_candidate.get("model_family"),
                "challenger_hyperparameters": best_candidate.get("hyperparameters")
                or {},
                "challenger_validation_rmse": best_candidate["validation_rmse"],
                "challenger_validation_risk_level_accuracy": best_candidate[
                    "validation_risk_level_accuracy"
                ],
                "challenger_validation_rows": best_candidate["validation_rows"],
                "challenger_validation_brier_score": best_candidate_validation_probability.get(
                    "brier_score"
                ),
                "challenger_validation_auprc": best_candidate_validation_probability.get(
                    "auprc"
                ),
                "challenger_validation_recall": best_candidate_validation_probability.get(
                    "recall"
                ),
                "challenger_validation_mcc": best_candidate_validation_probability.get(
                    "mcc"
                ),
                "challenger_validation_ece": best_candidate_validation_probability.get(
                    "ece"
                ),
                "challenger_calibration_gain": challenger_calibration_gain,
                "minimum_validation_rows_required": gate_policy["min_validation_rows"],
                "minimum_calibration_gain_required": gate_policy[
                    "min_calibration_gain"
                ],
                "require_nested_estimation_for_promotion": gate_policy[
                    "require_nested_estimation_for_promotion"
                ],
                "minimum_nested_outer_validation_rmse_improvement_required": (
                    gate_policy["min_nested_outer_validation_rmse_improvement"]
                ),
                "minimum_nested_outer_selection_rate_required": gate_policy[
                    "min_nested_outer_selection_rate"
                ],
                "require_nested_temporal_latest_win_for_promotion": gate_policy[
                    "require_nested_temporal_latest_win_for_promotion"
                ],
                "minimum_nested_temporal_outer_bucket_count_required": gate_policy[
                    "min_nested_temporal_outer_bucket_count"
                ],
                "minimum_nested_temporal_latest_validation_rmse_improvement_required": gate_policy[
                    "min_nested_temporal_latest_validation_rmse_improvement"
                ],
                "nested_temporal_recent_window_size": gate_policy[
                    "nested_temporal_recent_window_size"
                ],
                "minimum_nested_temporal_recent_win_rate_required": gate_policy[
                    "min_nested_temporal_recent_win_rate"
                ],
                "minimum_nested_temporal_recent_average_validation_rmse_improvement_required": gate_policy[
                    "min_nested_temporal_recent_average_validation_rmse_improvement"
                ],
                "maximum_spatial_slice_validation_rmse_regression_allowed": gate_policy[
                    "max_spatial_slice_validation_rmse_regression"
                ],
                "maximum_temporal_slice_validation_rmse_regression_allowed": gate_policy[
                    "max_temporal_slice_validation_rmse_regression"
                ],
                "maximum_spatial_slice_regression_count_allowed": gate_policy[
                    "max_spatial_slice_regression_count"
                ],
                "maximum_temporal_slice_regression_count_allowed": gate_policy[
                    "max_temporal_slice_regression_count"
                ],
                "slice_regression_min_rows": gate_policy[
                    "slice_regression_min_rows"
                ],
                "stability_assessment": stability_assessment,
                "nested_estimation": nested_estimation_summary,
                "nested_outer_selection_rate": nested_outer_selection_rate,
                "nested_outer_validation_rmse_improvement": (
                    nested_outer_rmse_improvement
                ),
                "nested_temporal_cohort_coverage": nested_temporal_coverage,
                "nested_temporal_latest_validation_rmse_improvement": (
                    nested_temporal_latest_rmse_improvement
                ),
                "nested_temporal_recent_window": nested_temporal_recent_window,
                "nested_temporal_recent_win_rate": nested_temporal_recent_win_rate,
                "nested_temporal_recent_average_validation_rmse_improvement": (
                    nested_temporal_recent_avg_rmse_improvement
                ),
                "slice_gate_summary": slice_gate_summary,
                "validation_rmse_improvement": rmse_improvement,
                "validation_brier_score_improvement": self._lower_is_better_improvement(
                    gate_policy["active_validation_brier_score"],
                    best_candidate_validation_probability.get("brier_score"),
                ),
                "validation_auprc_delta": self._higher_is_better_gain(
                    gate_policy["active_validation_auprc"],
                    best_candidate_validation_probability.get("auprc"),
                ),
                "validation_recall_delta": self._higher_is_better_gain(
                    gate_policy["active_validation_recall"],
                    best_candidate_validation_probability.get("recall"),
                ),
                "validation_mcc_delta": self._higher_is_better_gain(
                    gate_policy["active_validation_mcc"],
                    best_candidate_validation_probability.get("mcc"),
                ),
                "validation_ece_improvement": self._lower_is_better_improvement(
                    gate_policy["active_validation_ece"],
                    best_candidate_validation_probability.get("ece"),
                ),
            }

            promotion = {
                "promoted": False,
                "reason": promotion_reason,
                "promoted_at": None,
                "promoted_by": promoted_by,
                "source": origin,
            }
            if promote_best and promotion_decision["eligible"]:
                manifest = self.model_registry.set_active_version(
                    selection_run["best_model_version"],
                    promoted_by=promoted_by,
                    reason=promotion_reason,
                    source=f"{origin}:model_tuning",
                )
                promotion = {
                    "promoted": True,
                    "reason": promotion_reason,
                    "promoted_at": manifest["promoted_at"],
                    "promoted_by": manifest.get("promoted_by"),
                    "source": manifest.get("source"),
                    "manifest_path": str(
                        self.settings.resolved_active_model_manifest_path
                    ),
                }
                selection_run["promoted"] = True
                selection_run["active_model_version"] = (
                    self.model_registry.active_version()
                )
                promotion_decision["promoted"] = True
            selection_run["promotion"] = promotion
            selection_run["promotion_decision"] = promotion_decision
            selection_run["best_vs_active_comparison"] = best_vs_active_comparison
            selection_path, saved_selection = export_model_selection_run(selection_run)
        except Exception as exc:
            completed_at = datetime.now(timezone.utc).replace(microsecond=0)
            job.status = "failed"
            job.completed_at = completed_at
            job.details = {
                "origin": origin,
                "dataset_version": dataset_version,
                "selection_version": selection_version,
                "model_family": normalized_model_family,
                "model_families": resolved_model_families,
                "selection_mode": normalized_selection_mode,
                "validation_strategy": validation_strategy,
                "validation_fold_count": validation_fold_count,
                "nested_outer_fold_count": nested_outer_fold_count,
                "alphas": unique_alphas,
                "additive_spline_configs": additive_spline_configs,
                "boosted_tree_configs": boosted_tree_configs,
                "require_nested_temporal_latest_win_for_promotion": (
                    require_nested_temporal_latest_win_for_promotion
                ),
                "min_nested_temporal_outer_bucket_count": (
                    min_nested_temporal_outer_bucket_count
                ),
                "min_nested_temporal_latest_validation_rmse_improvement": (
                    min_nested_temporal_latest_validation_rmse_improvement
                ),
                "nested_temporal_recent_window_size": (
                    nested_temporal_recent_window_size
                ),
                "min_nested_temporal_recent_win_rate": (
                    min_nested_temporal_recent_win_rate
                ),
                "min_nested_temporal_recent_average_validation_rmse_improvement": (
                    min_nested_temporal_recent_average_validation_rmse_improvement
                ),
                "max_spatial_slice_validation_rmse_regression": (
                    max_spatial_slice_validation_rmse_regression
                ),
                "max_temporal_slice_validation_rmse_regression": (
                    max_temporal_slice_validation_rmse_regression
                ),
                "max_spatial_slice_regression_count": (
                    max_spatial_slice_regression_count
                ),
                "max_temporal_slice_regression_count": (
                    max_temporal_slice_regression_count
                ),
                "slice_regression_min_rows": slice_regression_min_rows,
                "promote_best": promote_best,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            self.session.commit()
            raise

        completed_at = datetime.now(timezone.utc).replace(microsecond=0)
        job.status = "completed"
        job.completed_at = completed_at
        best_candidate = next(
            candidate
            for candidate in saved_selection["candidates"]
            if candidate["rank"] == 1
        )
        job.details = {
            "origin": origin,
            "dataset_version": dataset_version,
            "selection_version": selection_version,
            "model_family": normalized_model_family,
            "model_families": resolved_model_families,
            "selection_mode": normalized_selection_mode,
            "validation_strategy": validation_strategy,
            "validation_fold_count": validation_fold_count,
            "nested_outer_fold_count": nested_outer_fold_count,
            "alphas": unique_alphas,
            "additive_spline_configs": additive_spline_configs,
            "boosted_tree_configs": boosted_tree_configs,
            "require_nested_temporal_latest_win_for_promotion": (
                require_nested_temporal_latest_win_for_promotion
            ),
            "min_nested_temporal_outer_bucket_count": (
                min_nested_temporal_outer_bucket_count
            ),
            "min_nested_temporal_latest_validation_rmse_improvement": (
                min_nested_temporal_latest_validation_rmse_improvement
            ),
            "nested_temporal_recent_window_size": nested_temporal_recent_window_size,
            "min_nested_temporal_recent_win_rate": (
                min_nested_temporal_recent_win_rate
            ),
            "min_nested_temporal_recent_average_validation_rmse_improvement": (
                min_nested_temporal_recent_average_validation_rmse_improvement
            ),
            "max_spatial_slice_validation_rmse_regression": (
                max_spatial_slice_validation_rmse_regression
            ),
            "max_temporal_slice_validation_rmse_regression": (
                max_temporal_slice_validation_rmse_regression
            ),
            "max_spatial_slice_regression_count": (
                max_spatial_slice_regression_count
            ),
            "max_temporal_slice_regression_count": (
                max_temporal_slice_regression_count
            ),
            "slice_regression_min_rows": slice_regression_min_rows,
            "promote_best": promote_best,
            "artifact_path": str(selection_path),
            "best_model_version": saved_selection["best_model_version"],
            "best_model_family": best_candidate.get("model_family"),
            "best_validation_rmse": best_candidate["validation_rmse"],
            "best_validation_risk_level_accuracy": best_candidate[
                "validation_risk_level_accuracy"
            ],
            "promotion_decision": saved_selection.get("promotion_decision") or {},
        }
        self.session.commit()
        self.session.refresh(job)

        return TuneModelResponse(
            job=self._job_read(job),
            selectionVersion=selection_version,
            artifactPath=str(selection_path),
            datasetVersion=dataset_version,
            candidateCount=int(saved_selection["candidate_count"]),
            bestModelVersion=saved_selection["best_model_version"],
            promoted=bool(saved_selection["promoted"]),
            activeModelVersion=saved_selection["active_model_version"],
            promotionDecision=saved_selection.get("promotion_decision") or {},
            bestCandidateComparison=saved_selection.get("best_vs_active_comparison")
            or {},
            familyRollups=list(saved_selection.get("family_rollups") or []),
            nestedEstimation=saved_selection.get("nested_estimation") or {},
            candidates=[
                self._candidate_read(candidate)
                for candidate in saved_selection["candidates"]
            ],
        )
