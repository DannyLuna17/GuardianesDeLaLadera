"""Validation-plan helpers for model-selection workflows."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from app.core.exceptions import ApiError
from app.ml.datasets import build_training_dataset
from app.ml.model_evaluations import (
    build_prediction_records,
    build_validation_slice_metrics,
    evaluate_prediction_records,
)
from app.services import model_selection_helpers as selection_helpers

TemporalReferenceResolver = Callable[[dict[str, object], str], datetime | None]
CandidateArtifactTrainer = Callable[..., dict]


def resolve_validation_plan(
    *,
    dataset: dict,
    validation_strategy: str,
    validation_fold_count: int | None,
    row_contexts: list[dict],
    temporal_bucket_reference: TemporalReferenceResolver,
) -> dict:
    row_items = list(dataset.get("rows") or [])
    if not row_items:
        raise ApiError(
            400,
            "training_dataset_empty",
            "The selected training dataset has no rows to tune.",
        )

    if validation_strategy == "dataset_holdout":
        train_indices = [
            index
            for index, item in enumerate(row_items)
            if str(item.get("split") or "train") == "train"
        ]
        validation_indices = [
            index
            for index, item in enumerate(row_items)
            if str(item.get("split") or "train") == "validation"
        ]
        if not train_indices or not validation_indices:
            raise ApiError(
                400,
                "dataset_holdout_invalid",
                "The selected dataset does not contain both train and validation rows.",
            )
        policy = dict(
            (dataset.get("provenance") or {}).get("validation_policy")
            or (dataset.get("summary") or {}).get("validation_policy")
            or {}
        )
        return {
            "requested_strategy": validation_strategy,
            "strategy": "dataset_holdout",
            "validation_unit": str(policy.get("unit") or "dataset_split"),
            "requested_fold_count": validation_fold_count,
            "fold_count": 1,
            "policy": {
                **policy,
                "strategy": "dataset_holdout",
                "requested_strategy": validation_strategy,
                "resolved_fold_count": 1,
            },
            "folds": [
                {
                    "fold_id": "dataset-holdout",
                    "included_indices": train_indices + validation_indices,
                    "train_indices": train_indices,
                    "validation_indices": validation_indices,
                    "metadata": {
                        "strategy": "dataset_holdout",
                        "validation_unit": str(policy.get("unit") or "dataset_split"),
                        "validation_bucket": str(
                            policy.get("validation_bucket") or "validation"
                        ),
                    },
                }
            ],
        }

    if validation_strategy == "spatial_block_kfold":
        block_groups: dict[str, list[int]] = {}
        for index, item in enumerate(row_items):
            context = row_contexts[index]
            block_id = str(
                context.get("spatialBlockId") or item.get("zoneId") or f"row-{index}"
            )
            block_groups.setdefault(block_id, []).append(index)
        if len(block_groups) < 2:
            raise ApiError(
                400,
                "spatial_validation_unavailable",
                "Spatial block validation requires at least two distinct spatialBlockId groups.",
            )
        requested_fold_count = int(validation_fold_count or min(3, len(block_groups)))
        resolved_fold_count = max(2, min(requested_fold_count, len(block_groups)))
        ordered_blocks = sorted(
            block_groups,
            key=lambda value: (sum(ord(character) for character in value), value),
        )
        fold_blocks: list[list[str]] = [[] for _ in range(resolved_fold_count)]
        for index, block_id in enumerate(ordered_blocks):
            fold_blocks[index % resolved_fold_count].append(block_id)
        folds = []
        for index, validation_blocks in enumerate(fold_blocks, start=1):
            validation_index_set = {
                row_index
                for block_id in validation_blocks
                for row_index in block_groups[block_id]
            }
            train_indices = [
                row_index
                for row_index in range(len(row_items))
                if row_index not in validation_index_set
            ]
            validation_indices = sorted(validation_index_set)
            if not train_indices or not validation_indices:
                continue
            folds.append(
                {
                    "fold_id": f"spatial-fold-{index}",
                    "included_indices": list(range(len(row_items))),
                    "train_indices": train_indices,
                    "validation_indices": validation_indices,
                    "metadata": {
                        "strategy": "spatial_block_kfold",
                        "validation_unit": "spatialBlockId",
                        "validation_blocks": validation_blocks,
                        "validation_group_count": len(validation_blocks),
                    },
                }
            )
        if len(folds) < 2:
            raise ApiError(
                400,
                "spatial_validation_insufficient",
                "Spatial block validation did not produce enough non-empty folds.",
            )
        return {
            "requested_strategy": validation_strategy,
            "strategy": "spatial_block_kfold",
            "validation_unit": "spatialBlockId",
            "requested_fold_count": validation_fold_count,
            "fold_count": len(folds),
            "policy": {
                "strategy": "spatial_block_kfold",
                "requested_strategy": validation_strategy,
                "unit": "spatialBlockId",
                "requested_fold_count": validation_fold_count,
                "resolved_fold_count": len(folds),
                "group_count": len(block_groups),
                "leakage_guard": "spatial_block_group_holdout",
            },
            "folds": folds,
        }

    if validation_strategy != "temporal_holdout_backtest":
        raise ApiError(
            400,
            "invalid_validation_strategy",
            f"Unsupported validation strategy: {validation_strategy}",
        )

    temporal_buckets: dict[str, dict[str, object]] = {}
    for index, item in enumerate(row_items):
        context = row_contexts[index]
        bucket_label = str(
            context.get("temporalHoldoutTag")
            or context.get("observedAt")
            or item.get("phase")
            or f"row-{index}"
        )
        bucket = temporal_buckets.setdefault(
            bucket_label,
            {"indices": [], "reference_at": None},
        )
        bucket["indices"].append(index)
        reference_at = temporal_bucket_reference(context, bucket_label)
        current_reference = bucket.get("reference_at")
        if reference_at is not None and (
            current_reference is None or reference_at < current_reference
        ):
            bucket["reference_at"] = reference_at

    ordered_temporal_buckets = [
        (label, payload)
        for label, payload in temporal_buckets.items()
        if payload.get("reference_at") is not None
    ]
    ordered_temporal_buckets.sort(key=lambda item: (item[1]["reference_at"], item[0]))
    if len(ordered_temporal_buckets) < 2:
        raise ApiError(
            400,
            "temporal_validation_unavailable",
            "Temporal backtesting requires at least two ordered temporalHoldoutTag groups with parseable timestamps.",
        )
    max_folds = len(ordered_temporal_buckets) - 1
    requested_fold_count = int(validation_fold_count or min(3, max_folds))
    resolved_fold_count = max(1, min(requested_fold_count, max_folds))
    validation_positions = list(
        range(
            len(ordered_temporal_buckets) - resolved_fold_count,
            len(ordered_temporal_buckets),
        )
    )
    folds = []
    for fold_offset, bucket_position in enumerate(validation_positions, start=1):
        validation_label, validation_bucket = ordered_temporal_buckets[bucket_position]
        train_indices = [
            row_index
            for prior_label, prior_bucket in ordered_temporal_buckets[:bucket_position]
            for row_index in prior_bucket["indices"]
        ]
        validation_indices = list(validation_bucket["indices"])
        if not train_indices or not validation_indices:
            continue
        folds.append(
            {
                "fold_id": f"temporal-fold-{fold_offset}",
                "included_indices": train_indices + validation_indices,
                "train_indices": train_indices,
                "validation_indices": validation_indices,
                "metadata": {
                    "strategy": "temporal_holdout_backtest",
                    "validation_unit": "temporalHoldoutTag",
                    "validation_bucket": validation_label,
                    "validation_bucket_reference_at": validation_bucket[
                        "reference_at"
                    ].isoformat(),
                    "train_bucket_count": bucket_position,
                },
            }
        )
    if not folds:
        raise ApiError(
            400,
            "temporal_validation_insufficient",
            "Temporal backtesting did not produce any non-empty train/validation folds.",
        )
    return {
        "requested_strategy": validation_strategy,
        "strategy": "temporal_holdout_backtest",
        "validation_unit": "temporalHoldoutTag",
        "requested_fold_count": validation_fold_count,
        "fold_count": len(folds),
        "policy": {
            "strategy": "temporal_holdout_backtest",
            "requested_strategy": validation_strategy,
            "unit": "temporalHoldoutTag",
            "requested_fold_count": validation_fold_count,
            "resolved_fold_count": len(folds),
            "ordered_bucket_count": len(ordered_temporal_buckets),
            "leakage_guard": "forward_chaining_temporal_holdout",
        },
        "folds": folds,
    }


def build_validation_fold_dataset(
    *,
    dataset: dict,
    rows: list,
    row_contexts: list[dict],
    plan: dict,
    fold: dict,
) -> dict:
    included_indices = list(fold["included_indices"])
    validation_index_set = set(fold["validation_indices"])
    fold_rows = [rows[index] for index in included_indices]
    fold_contexts = []
    fold_splits = []
    for index in included_indices:
        split = "validation" if index in validation_index_set else "train"
        fold_splits.append(split)
        context = dict(row_contexts[index] or {})
        context["validationStrategy"] = plan["strategy"]
        context["validationFoldId"] = fold["fold_id"]
        fold_contexts.append(context)

    summary = dict(dataset.get("summary") or {})
    summary_extra = {
        key: summary[key]
        for key in (
            "dataset_family",
            "time_window",
            "dataset_taxonomy",
            "evaluation_cohort",
        )
        if key in summary
    }
    summary_extra["validation_plan"] = {
        "strategy": plan["strategy"],
        "fold_id": fold["fold_id"],
        "validation_unit": plan["validation_unit"],
        **(fold.get("metadata") or {}),
    }
    description = (
        f"{dataset.get('description') or 'Training dataset'} "
        f"[{plan['strategy']}:{fold['fold_id']}]"
    )
    return build_training_dataset(
        version=f"{dataset['version']}--{fold['fold_id']}",
        rows=fold_rows,
        description=description,
        provenance=dict(dataset.get("provenance") or {}),
        row_contexts=fold_contexts,
        row_splits=fold_splits,
        validation_policy_override={
            **dict(plan.get("policy") or {}),
            "fold_id": fold["fold_id"],
            **dict(fold.get("metadata") or {}),
        },
        summary_extra=summary_extra,
    )


def validation_summary_from_records(
    *,
    plan: dict,
    validation_records: list[dict],
    fold_summaries: list[dict],
    top_error_count: int,
) -> dict:
    aggregated_metrics = evaluate_prediction_records(validation_records)
    return {
        "requested_strategy": plan["requested_strategy"],
        "strategy": plan["strategy"],
        "validation_unit": plan["validation_unit"],
        "requested_fold_count": plan.get("requested_fold_count"),
        "fold_count": len(fold_summaries),
        "validation_rows": aggregated_metrics["rows"],
        "metrics": aggregated_metrics,
        "validation_slices": build_validation_slice_metrics(validation_records),
        "folds": fold_summaries,
        "top_errors": sorted(
            validation_records,
            key=lambda record: record["absError"],
            reverse=True,
        )[:top_error_count],
    }


def build_subset_dataset(
    *,
    dataset: dict,
    rows: list,
    row_contexts: list[dict],
    version_suffix: str,
    description_suffix: str,
) -> dict:
    summary = dict(dataset.get("summary") or {})
    summary_extra = {
        key: summary[key]
        for key in (
            "dataset_family",
            "time_window",
            "dataset_taxonomy",
            "evaluation_cohort",
        )
        if key in summary
    }
    description = (
        f"{dataset.get('description') or 'Training dataset'} [{description_suffix}]"
    )
    return build_training_dataset(
        version=f"{dataset['version']}--{version_suffix}",
        rows=rows,
        description=description,
        provenance=dict(dataset.get("provenance") or {}),
        row_contexts=row_contexts,
        summary_extra=summary_extra,
    )


def ranking_candidate_from_validation_summary(
    *,
    candidate_spec: dict,
    validation_summary: dict,
) -> dict:
    validation_metrics = validation_summary["metrics"]
    validation_probability = selection_helpers.probability_metrics(validation_metrics)
    return {
        "model_version": str(candidate_spec["candidate_version"]),
        "model_family": str(candidate_spec["model_family"]),
        "alpha": candidate_spec.get("alpha"),
        "hyperparameters": dict(candidate_spec["hyperparameters"]),
        "validation_rmse": validation_metrics["calibrated_metrics"]["rmse"],
        "validation_brier_score": validation_probability.get("brier_score"),
        "validation_auprc": validation_probability.get("auprc"),
        "validation_recall": validation_probability.get("recall"),
        "validation_mcc": validation_probability.get("mcc"),
        "validation_risk_level_accuracy": validation_metrics["risk_level_accuracy"],
        "overall_rmse": validation_metrics["calibrated_metrics"]["rmse"],
    }


def evaluate_artifact_under_validation_plan(
    *,
    artifact: dict,
    dataset: dict,
    rows: list,
    row_contexts: list[dict],
    plan: dict,
    top_error_count: int,
) -> dict:
    validation_records: list[dict] = []
    fold_summaries: list[dict] = []
    for fold in plan["folds"]:
        fold_dataset = build_validation_fold_dataset(
            dataset=dataset,
            rows=rows,
            row_contexts=row_contexts,
            plan=plan,
            fold=fold,
        )
        prediction_records = build_prediction_records(artifact, fold_dataset)
        fold_validation_records = [
            record for record in prediction_records if record["split"] == "validation"
        ]
        validation_records.extend(fold_validation_records)
        fold_metrics = evaluate_prediction_records(fold_validation_records)
        fold_probability = selection_helpers.probability_metrics(fold_metrics)
        fold_summaries.append(
            {
                "fold_id": fold["fold_id"],
                "train_rows": len(fold["train_indices"]),
                "validation_rows": fold_metrics["rows"],
                "validation_rmse": fold_metrics["calibrated_metrics"]["rmse"],
                "validation_risk_level_accuracy": fold_metrics[
                    "risk_level_accuracy"
                ],
                "validation_brier_score": fold_probability.get("brier_score"),
                "validation_auprc": fold_probability.get("auprc"),
                "validation_recall": fold_probability.get("recall"),
                "validation_mcc": fold_probability.get("mcc"),
                "metadata": dict(fold.get("metadata") or {}),
            }
        )
    return validation_summary_from_records(
        plan=plan,
        validation_records=validation_records,
        fold_summaries=fold_summaries,
        top_error_count=top_error_count,
    )


def evaluate_candidate_under_validation_plan(
    *,
    model_family: str,
    version: str,
    dataset: dict,
    rows: list,
    row_contexts: list[dict],
    plan: dict,
    hyperparameters: dict,
    top_error_count: int,
    train_candidate_artifact: CandidateArtifactTrainer,
) -> dict:
    validation_records: list[dict] = []
    fold_summaries: list[dict] = []
    for fold in plan["folds"]:
        included_rows = [rows[index] for index in fold["included_indices"]]
        fold_train_rows = [rows[index] for index in fold["train_indices"]]
        fold_validation_rows = [rows[index] for index in fold["validation_indices"]]
        fold_artifact = train_candidate_artifact(
            model_family=model_family,
            version=f"{version}--{fold['fold_id']}",
            rows=included_rows,
            dataset_name=f"{dataset['version']}:{plan['strategy']}:{fold['fold_id']}",
            hyperparameters=hyperparameters,
            train_rows=fold_train_rows,
            validation_rows=fold_validation_rows,
        )
        fold_dataset = build_validation_fold_dataset(
            dataset=dataset,
            rows=rows,
            row_contexts=row_contexts,
            plan=plan,
            fold=fold,
        )
        prediction_records = build_prediction_records(fold_artifact, fold_dataset)
        fold_validation_records = [
            record for record in prediction_records if record["split"] == "validation"
        ]
        validation_records.extend(fold_validation_records)
        fold_metrics = evaluate_prediction_records(fold_validation_records)
        fold_probability = selection_helpers.probability_metrics(fold_metrics)
        fold_summaries.append(
            {
                "fold_id": fold["fold_id"],
                "train_rows": len(fold["train_indices"]),
                "validation_rows": fold_metrics["rows"],
                "validation_rmse": fold_metrics["calibrated_metrics"]["rmse"],
                "validation_risk_level_accuracy": fold_metrics[
                    "risk_level_accuracy"
                ],
                "validation_brier_score": fold_probability.get("brier_score"),
                "validation_auprc": fold_probability.get("auprc"),
                "validation_recall": fold_probability.get("recall"),
                "validation_mcc": fold_probability.get("mcc"),
                "metadata": dict(fold.get("metadata") or {}),
            }
        )
    return validation_summary_from_records(
        plan=plan,
        validation_records=validation_records,
        fold_summaries=fold_summaries,
        top_error_count=top_error_count,
    )
