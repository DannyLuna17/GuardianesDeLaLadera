from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import ApiError
from app.db.bootstrap import clamp, risk_level_from_score
from app.ml.datasets import deserialize_training_row
from app.ml.model_diagnostics import (
    calibration_effect_summary,
    feature_importance_summary,
)
from app.ml.inference import BaselineInferenceService
from app.ml.training import apply_affine_calibration

MODEL_EVALUATION_ID = "spatial-risk-model-evaluation"
RISK_LEVEL_ORDER = ["Verde", "Amarillo", "Naranja", "Rojo"]
ELEVATED_RISK_THRESHOLD = 0.5
CALIBRATION_BIN_COUNT = 10


def _regression_metrics(
    predictions: list[float], targets: list[float]
) -> dict[str, float]:
    if not predictions:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "max_abs_error": 0.0,
            "r2": 0.0,
            "mean_signed_error": 0.0,
            "mean_prediction": 0.0,
            "mean_target": 0.0,
            "within_005_rate": 0.0,
            "within_010_rate": 0.0,
        }

    signed_errors = [
        prediction - target for prediction, target in zip(predictions, targets)
    ]
    absolute_errors = [abs(error) for error in signed_errors]
    squared_errors = [error**2 for error in signed_errors]
    target_mean = sum(targets) / len(targets)
    target_variance = sum((target - target_mean) ** 2 for target in targets)
    explained_error = sum(squared_errors)
    r2 = 1.0 if target_variance <= 1e-12 else 1 - (explained_error / target_variance)
    return {
        "mae": round(sum(absolute_errors) / len(absolute_errors), 6),
        "rmse": round((sum(squared_errors) / len(squared_errors)) ** 0.5, 6),
        "max_abs_error": round(max(absolute_errors), 6),
        "r2": round(r2, 6),
        "mean_signed_error": round(sum(signed_errors) / len(signed_errors), 6),
        "mean_prediction": round(sum(predictions) / len(predictions), 6),
        "mean_target": round(target_mean, 6),
        "within_005_rate": round(
            sum(1 for error in absolute_errors if error <= 0.05) / len(absolute_errors),
            6,
        ),
        "within_010_rate": round(
            sum(1 for error in absolute_errors if error <= 0.10) / len(absolute_errors),
            6,
        ),
    }


def _risk_level_counts(levels: list[str]) -> dict[str, int]:
    counts = {level: 0 for level in RISK_LEVEL_ORDER}
    for level in levels:
        counts[level] = counts.get(level, 0) + 1
    return counts


def _risk_level_confusion_matrix(
    target_levels: list[str], predicted_levels: list[str]
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {
        actual: {predicted: 0 for predicted in RISK_LEVEL_ORDER}
        for actual in RISK_LEVEL_ORDER
    }
    for actual, predicted in zip(target_levels, predicted_levels):
        if actual not in matrix:
            matrix[actual] = {level: 0 for level in RISK_LEVEL_ORDER}
        if predicted not in matrix[actual]:
            matrix[actual][predicted] = 0
        matrix[actual][predicted] += 1
    return matrix


def _safe_rate(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _auroc(scores: list[float], targets: list[int]) -> float | None:
    positive_count = sum(targets)
    negative_count = len(targets) - positive_count
    if positive_count <= 0 or negative_count <= 0:
        return None

    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda item: item[1])

    rank_sum = 0.0
    current_rank = 1
    index = 0
    while index < len(indexed_scores):
        tie_start = index
        tie_score = indexed_scores[index][1]
        while index < len(indexed_scores) and indexed_scores[index][1] == tie_score:
            index += 1
        tie_size = index - tie_start
        average_rank = current_rank + (tie_size - 1) / 2
        for tie_index in range(tie_start, index):
            original_index = indexed_scores[tie_index][0]
            if targets[original_index] == 1:
                rank_sum += average_rank
        current_rank += tie_size

    auc = (
        rank_sum - (positive_count * (positive_count + 1) / 2)
    ) / (positive_count * negative_count)
    return round(auc, 6)


def _average_precision(scores: list[float], targets: list[int]) -> float | None:
    positive_count = sum(targets)
    if positive_count <= 0:
        return None

    ranked = sorted(zip(scores, targets, strict=True), key=lambda item: item[0], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, target) in enumerate(ranked, start=1):
        if target == 1:
            true_positives += 1
            precision_sum += true_positives / rank
    return round(precision_sum / positive_count, 6)


def _expected_calibration_error(
    scores: list[float], targets: list[int], *, bins: int
) -> float:
    if not scores:
        return 0.0

    error = 0.0
    total = len(scores)
    for bucket in range(bins):
        lower = bucket / bins
        upper = (bucket + 1) / bins
        members = [
            index
            for index, score in enumerate(scores)
            if (lower <= score < upper) or (bucket == bins - 1 and score <= upper)
        ]
        if not members:
            continue
        confidence = sum(scores[index] for index in members) / len(members)
        accuracy = sum(targets[index] for index in members) / len(members)
        error += abs(confidence - accuracy) * (len(members) / total)
    return round(error, 6)


def _probability_metrics(
    predictions: list[float],
    targets: list[float],
    *,
    positive_threshold: float = ELEVATED_RISK_THRESHOLD,
    calibration_bins: int = CALIBRATION_BIN_COUNT,
) -> dict[str, Any]:
    if not predictions:
        return {
            "positive_threshold": positive_threshold,
            "positive_risk_levels": ["Naranja", "Rojo"],
            "positive_class": "elevated_risk",
            "prevalence": 0.0,
            "positive_prediction_rate": 0.0,
            "precision": None,
            "recall": None,
            "specificity": None,
            "f1": None,
            "balanced_accuracy": None,
            "mcc": None,
            "auroc": None,
            "auprc": None,
            "brier_score": 0.0,
            "ece": 0.0,
            "calibration_bins": calibration_bins,
        }

    normalized_scores = [clamp(float(prediction), 0.0, 1.0) for prediction in predictions]
    binary_targets = [
        1 if float(target) >= positive_threshold else 0 for target in targets
    ]
    binary_predictions = [
        1 if score >= positive_threshold else 0 for score in normalized_scores
    ]

    true_positives = sum(
        1
        for predicted, actual in zip(binary_predictions, binary_targets, strict=True)
        if predicted == 1 and actual == 1
    )
    false_positives = sum(
        1
        for predicted, actual in zip(binary_predictions, binary_targets, strict=True)
        if predicted == 1 and actual == 0
    )
    true_negatives = sum(
        1
        for predicted, actual in zip(binary_predictions, binary_targets, strict=True)
        if predicted == 0 and actual == 0
    )
    false_negatives = sum(
        1
        for predicted, actual in zip(binary_predictions, binary_targets, strict=True)
        if predicted == 0 and actual == 1
    )

    precision = _safe_rate(true_positives, true_positives + false_positives)
    recall = _safe_rate(true_positives, true_positives + false_negatives)
    specificity = _safe_rate(true_negatives, true_negatives + false_positives)
    f1 = (
        round(2 * precision * recall / (precision + recall), 6)
        if precision is not None
        and recall is not None
        and (precision + recall) > 0
        else None
    )
    balanced_accuracy = (
        round((recall + specificity) / 2, 6)
        if recall is not None and specificity is not None
        else None
    )
    mcc_denominator = math.sqrt(
        max(true_positives + false_positives, 0)
        * max(true_positives + false_negatives, 0)
        * max(true_negatives + false_positives, 0)
        * max(true_negatives + false_negatives, 0)
    )
    mcc = (
        round(
            (
                (true_positives * true_negatives)
                - (false_positives * false_negatives)
            )
            / mcc_denominator,
            6,
        )
        if mcc_denominator > 0
        else None
    )
    brier_score = round(
        sum(
            (score - actual) ** 2
            for score, actual in zip(normalized_scores, binary_targets, strict=True)
        )
        / len(normalized_scores),
        6,
    )

    return {
        "positive_threshold": positive_threshold,
        "positive_risk_levels": ["Naranja", "Rojo"],
        "positive_class": "elevated_risk",
        "prevalence": round(sum(binary_targets) / len(binary_targets), 6),
        "positive_prediction_rate": round(
            sum(binary_predictions) / len(binary_predictions), 6
        ),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "auroc": _auroc(normalized_scores, binary_targets),
        "auprc": _average_precision(normalized_scores, binary_targets),
        "brier_score": brier_score,
        "ece": _expected_calibration_error(
            normalized_scores, binary_targets, bins=calibration_bins
        ),
        "calibration_bins": calibration_bins,
    }


def _evaluate_prediction_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "rows": 0,
            "raw_metrics": _regression_metrics([], []),
            "calibrated_metrics": _regression_metrics([], []),
            "raw_probability_metrics": _probability_metrics([], []),
            "probability_metrics": _probability_metrics([], []),
            "risk_level_accuracy": 0.0,
            "target_risk_level_counts": _risk_level_counts([]),
            "predicted_risk_level_counts": _risk_level_counts([]),
            "risk_level_confusion_matrix": _risk_level_confusion_matrix([], []),
        }

    targets = [float(record["targetScore"]) for record in records]
    raw_predictions = [float(record["rawModelScore"]) for record in records]
    calibrated_predictions = [float(record["predictedScore"]) for record in records]
    target_levels = [str(record["targetRiskLevel"]) for record in records]
    predicted_levels = [str(record["predictedRiskLevel"]) for record in records]
    exact_risk_matches = sum(
        1
        for target_level, predicted_level in zip(target_levels, predicted_levels)
        if target_level == predicted_level
    )
    return {
        "rows": len(records),
        "raw_metrics": _regression_metrics(raw_predictions, targets),
        "calibrated_metrics": _regression_metrics(calibrated_predictions, targets),
        "raw_probability_metrics": _probability_metrics(raw_predictions, targets),
        "probability_metrics": _probability_metrics(calibrated_predictions, targets),
        "risk_level_accuracy": round(exact_risk_matches / len(records), 6),
        "target_risk_level_counts": _risk_level_counts(target_levels),
        "predicted_risk_level_counts": _risk_level_counts(predicted_levels),
        "risk_level_confusion_matrix": _risk_level_confusion_matrix(
            target_levels, predicted_levels
        ),
    }


def _slice_metric_view(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = _evaluate_prediction_records(records)
    return {
        "rows": metrics["rows"],
        "raw_metrics": metrics["raw_metrics"],
        "calibrated_metrics": metrics["calibrated_metrics"],
        "raw_probability_metrics": metrics["raw_probability_metrics"],
        "probability_metrics": metrics["probability_metrics"],
        "risk_level_accuracy": metrics["risk_level_accuracy"],
    }


def _slice_metrics(
    records: list[dict[str, Any]], *, key_name: str
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key_value = record.get(key_name)
        if key_value is None:
            key_value = (record.get("context") or {}).get(key_name)
        key = str(key_value or "unknown")
        buckets.setdefault(key, []).append(record)
    return {
        key: _slice_metric_view(bucket)
        for key, bucket in sorted(buckets.items(), key=lambda item: item[0])
    }


def score_training_row(
    artifact: dict[str, Any], item: dict[str, Any]
) -> dict[str, Any]:
    row = deserialize_training_row(item)
    if artifact.get("artifact_type") == "trained_linear_model":
        raw_score, component_scores, feature_vector = (
            BaselineInferenceService.score_trained_linear_artifact(
                row.drivers,
                row.feature_snapshot,
                artifact,
            )
        )
        calibrated_score = apply_affine_calibration(
            raw_score, artifact.get("calibration")
        )
    elif artifact.get("artifact_type") == "gradient_boosted_tree_model":
        raw_score, component_scores, feature_vector = (
            BaselineInferenceService.score_gradient_boosted_tree_artifact(
                row.drivers,
                row.feature_snapshot,
                artifact,
            )
        )
        calibrated_score = apply_affine_calibration(
            raw_score, artifact.get("calibration")
        )
    elif artifact.get("artifact_type") == "additive_spline_model":
        raw_score, component_scores, feature_vector = (
            BaselineInferenceService.score_additive_spline_artifact(
                row.drivers,
                row.feature_snapshot,
                artifact,
            )
        )
        calibrated_score = apply_affine_calibration(
            raw_score, artifact.get("calibration")
        )
    elif artifact.get("artifact_type") == "beta_regression_model":
        raw_score, component_scores, feature_vector = (
            BaselineInferenceService.score_beta_regression_artifact(
                row.drivers,
                row.feature_snapshot,
                artifact,
            )
        )
        calibrated_score = apply_affine_calibration(
            raw_score, artifact.get("calibration")
        )
    elif artifact.get("artifact_type") == "xgboost_model":
        raw_score, component_scores, feature_vector = (
            BaselineInferenceService.score_xgboost_artifact(
                row.drivers,
                row.feature_snapshot,
                artifact,
            )
        )
        calibrated_score = apply_affine_calibration(
            raw_score, artifact.get("calibration")
        )
    else:
        raw_score, component_scores = (
            BaselineInferenceService.score_weighted_sum_artifact(
                row.drivers,
                row.feature_snapshot,
                artifact,
            )
        )
        calibrated_score = raw_score
        feature_vector = row.scoring_features()

    predicted_score = clamp(
        calibrated_score,
        float(artifact["bounds"]["min"]),
        float(artifact["bounds"]["max"]),
    )
    target_score = float(row.target_score)
    abs_error = abs(predicted_score - target_score)
    return {
        "zoneId": row.zone_id,
        "phase": row.phase,
        "split": str(item.get("split") or "train"),
        "targetScore": round(target_score, 6),
        "rawModelScore": round(raw_score, 6),
        "predictedScore": round(predicted_score, 6),
        "absError": round(abs_error, 6),
        "signedError": round(predicted_score - target_score, 6),
        "targetRiskLevel": risk_level_from_score(target_score),
        "predictedRiskLevel": risk_level_from_score(predicted_score),
        "componentScores": {
            key: round(float(value), 6) for key, value in component_scores.items()
        },
        "featureVector": {
            key: round(float(value), 6) for key, value in feature_vector.items()
        },
        "context": dict(item.get("context") or {}),
    }


def build_prediction_records(
    artifact: dict[str, Any], dataset: dict[str, Any]
) -> list[dict[str, Any]]:
    if not dataset.get("rows"):
        raise ApiError(
            400,
            "model_evaluation_dataset_empty",
            "The selected training dataset has no rows to evaluate.",
        )
    return [score_training_row(artifact, item) for item in dataset["rows"]]


def evaluate_prediction_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return _evaluate_prediction_records(records)


def build_validation_slice_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "by_phase": _slice_metrics(records, key_name="phase"),
        "by_target_risk_level": _slice_metrics(records, key_name="targetRiskLevel"),
        "by_spatial_block": _slice_metrics(records, key_name="spatialBlockId"),
        "by_temporal_holdout_tag": _slice_metrics(
            records, key_name="temporalHoldoutTag"
        ),
    }


def build_model_evaluation(
    *,
    version: str,
    artifact: dict[str, Any],
    dataset: dict[str, Any],
    top_error_count: int = 10,
) -> dict[str, Any]:
    prediction_records = build_prediction_records(artifact, dataset)
    train_records = [
        record for record in prediction_records if record["split"] == "train"
    ]
    validation_records = [
        record for record in prediction_records if record["split"] == "validation"
    ]
    calibration = artifact.get("calibration") or {}

    return {
        "evaluation_id": MODEL_EVALUATION_ID,
        "version": version,
        "artifact_type": "model_evaluation",
        "model_version": artifact["version"],
        "dataset_version": dataset["version"],
        "evaluated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model_summary": {
            "model_id": artifact.get("model_id", "unknown"),
            "artifact_type": artifact.get("artifact_type", "weighted_sum"),
            "model_family": artifact.get("model_family", "baseline"),
            "training_dataset": (artifact.get("training") or {}).get("dataset"),
            "training_rows": (artifact.get("training") or {}).get("rows"),
            "training_metrics": (artifact.get("training") or {}).get("metrics") or {},
            "hyperparameters": (artifact.get("training") or {}).get("hyperparameters")
            or {},
            "calibration_method": calibration.get("method", "none"),
        },
        "dataset_summary": {
            "dataset_id": dataset.get("dataset_id", "unknown"),
            "artifact_type": dataset.get("artifact_type", "training_dataset"),
            "description": dataset.get("description"),
            "provenance": dataset.get("provenance") or {},
            "summary": dataset.get("summary") or {},
        },
        "metrics": {
            "overall": _evaluate_prediction_records(prediction_records),
            "train": _evaluate_prediction_records(train_records),
            "validation": _evaluate_prediction_records(validation_records),
        },
        "diagnostics": {
            "feature_importance": feature_importance_summary(artifact),
            "calibration_effect": calibration_effect_summary(artifact),
            "validation_slices": build_validation_slice_metrics(validation_records),
        },
        "top_errors": sorted(
            prediction_records, key=lambda record: record["absError"], reverse=True
        )[:top_error_count],
    }


def export_model_evaluation(
    evaluation: dict[str, Any],
    *,
    evaluations_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    settings = get_settings()
    target_directory = evaluations_path or settings.resolved_model_evaluations_path
    target_directory.mkdir(parents=True, exist_ok=True)
    evaluation_path = target_directory / f"{evaluation['version']}.json"
    evaluation_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    return evaluation_path, evaluation


class ModelEvaluationRegistry:
    def __init__(self, evaluations_path: Path | None = None) -> None:
        settings = get_settings()
        self.evaluations_path = (
            evaluations_path or settings.resolved_model_evaluations_path
        )

    @lru_cache
    def load(self, version: str) -> dict[str, Any]:
        evaluation_path = self.evaluations_path / f"{version}.json"
        if not evaluation_path.exists():
            raise ApiError(
                404,
                "model_evaluation_not_found",
                f"Model evaluation '{version}' was not found.",
            )
        with evaluation_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def list_versions(self) -> list[str]:
        return sorted(path.stem for path in self.evaluations_path.glob("*.json"))
