from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.data.seed_store import load_seed_payload
from app.db.bootstrap import clamp, deterministic_delta
from app.db.spatial_filters import (
    linestring_intersects_polygon,
    overlay_bounds_intersect_polygon,
    point_within_polygon,
)
from app.ml.additive_splines import (
    predict_additive_spline_regressor,
    train_additive_spline_regressor,
)
from app.ml.beta_regression import (
    predict_beta_regression_rows,
    train_beta_regression_regressor,
)
from app.ml.features import (
    RAIN_INTENSITY_SCORES,
    SCORING_FEATURE_ORDER,
    ZoneFeatureSnapshot,
    build_scoring_feature_vector,
)
from app.ml.tree_boosting import predict_gradient_boosted_ensemble, train_gradient_boosted_regressor
from app.ml.xgboost_models import predict_xgboost_rows, train_xgboost_regressor


TRAINING_REFERENCE_AT = datetime(2026, 3, 25, tzinfo=timezone.utc)


@dataclass(frozen=True)
class TrainingRow:
    zone_id: str
    phase: str
    target_score: float
    drivers: dict[str, Any]
    feature_snapshot: ZoneFeatureSnapshot

    def scoring_features(self) -> dict[str, float]:
        return build_scoring_feature_vector(self.drivers, self.feature_snapshot)


def _deterministic_zone_bucket(zone_id: str, buckets: int = 3) -> int:
    return sum(ord(character) for character in zone_id) % buckets


def _zone_feature_snapshot_from_seed(zone: dict[str, Any], seed_payload: dict[str, Any]) -> ZoneFeatureSnapshot:
    municipality_name = zone["municipality"]
    municipality_events = [
        item for item in seed_payload["historicalEvents"] if item["municipality"] == municipality_name
    ]
    zone_events = [
        item
        for item in municipality_events
        if point_within_polygon(item["coords"], zone["polygon"])
    ]
    lookback_date = (TRAINING_REFERENCE_AT.date().replace(year=TRAINING_REFERENCE_AT.year - 3))
    intersecting_segments = [
        item
        for item in seed_payload["roadSegments"]
        if item["municipality"] == municipality_name
        and linestring_intersects_polygon(item["coords"], zone["polygon"])
    ]
    rain_overlays = [
        item
        for item in seed_payload["rainOverlays"][municipality_name]
        if overlay_bounds_intersect_polygon(item["bounds"], zone["polygon"])
    ]

    rain_overlay_peak_label = None
    rain_overlay_peak_intensity = 0
    if rain_overlays:
        rain_overlay_peak_label = max(
            (item["intensity"] for item in rain_overlays),
            key=lambda value: RAIN_INTENSITY_SCORES.get(value.lower(), 0),
        )
        rain_overlay_peak_intensity = RAIN_INTENSITY_SCORES.get(rain_overlay_peak_label.lower(), 0)

    return ZoneFeatureSnapshot(
        municipality_event_count=len(municipality_events),
        zone_event_count=len(zone_events),
        recent_zone_event_count=sum(
            1 for item in zone_events if date.fromisoformat(item["date"]) >= lookback_date
        ),
        intersecting_road_count=len(intersecting_segments),
        intersecting_road_length_km=round(sum(float(item["length_km"]) for item in intersecting_segments), 2),
        rain_overlay_count=len(rain_overlays),
        rain_overlay_peak_intensity=rain_overlay_peak_intensity,
        rain_overlay_peak_label=rain_overlay_peak_label,
    )


def build_seed_training_rows(seed_payload: dict[str, Any] | None = None) -> list[TrainingRow]:
    seed_payload = seed_payload or load_seed_payload()
    rows: list[TrainingRow] = []

    for zone in seed_payload["zones"]:
        feature_snapshot = _zone_feature_snapshot_from_seed(zone, seed_payload)
        current_drivers = dict(zone["drivers"])
        current_target = float(zone["riskScore"])
        rows.append(
            TrainingRow(
                zone_id=zone["id"],
                phase="latest",
                target_score=current_target,
                drivers=current_drivers,
                feature_snapshot=feature_snapshot,
            )
        )

        previous_target = clamp(current_target - deterministic_delta(zone["id"]), 0.08, 0.92)
        previous_drivers = {
            **current_drivers,
            "rain_6h": max(0, int(current_drivers["rain_6h"]) - 4),
            "rain_24h": max(0, int(current_drivers["rain_24h"]) - 7),
            "rain_72h": max(0, int(current_drivers["rain_72h"]) - 11),
        }
        rows.append(
            TrainingRow(
                zone_id=zone["id"],
                phase="previous",
                target_score=float(previous_target),
                drivers=previous_drivers,
                feature_snapshot=feature_snapshot,
            )
        )

    return rows


def _feature_stats(rows: list[TrainingRow]) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    feature_vectors = [row.scoring_features() for row in rows]

    for feature_name in SCORING_FEATURE_ORDER:
        values = [vector[feature_name] for vector in feature_vectors]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
        scale = variance ** 0.5
        means[feature_name] = mean
        scales[feature_name] = scale if scale > 1e-9 else 1.0

    return means, scales


def split_training_rows(rows: list[TrainingRow]) -> tuple[list[TrainingRow], list[TrainingRow]]:
    training_rows = [row for row in rows if _deterministic_zone_bucket(row.zone_id) != 0]
    validation_rows = [row for row in rows if _deterministic_zone_bucket(row.zone_id) == 0]
    if not validation_rows:
        return rows, []
    return training_rows, validation_rows


def _resolve_training_splits(
    rows: list[TrainingRow] | None,
    *,
    train_rows: list[TrainingRow] | None = None,
    validation_rows: list[TrainingRow] | None = None,
) -> tuple[list[TrainingRow], list[TrainingRow], list[TrainingRow]]:
    explicit_train_rows = list(train_rows or [])
    explicit_validation_rows = list(validation_rows or [])
    if train_rows is None and validation_rows is None:
        resolved_rows = rows or build_seed_training_rows()
        resolved_train_rows, resolved_validation_rows = split_training_rows(
            resolved_rows
        )
        return resolved_rows, resolved_train_rows, resolved_validation_rows

    if not explicit_train_rows:
        raise ValueError("Explicit training splits require at least one training row.")
    resolved_rows = rows or [*explicit_train_rows, *explicit_validation_rows]
    return resolved_rows, explicit_train_rows, explicit_validation_rows


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]

    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda row_index: abs(augmented[row_index][pivot_index]))
        pivot_value = augmented[pivot_row][pivot_index]
        if abs(pivot_value) < 1e-10:
            raise ValueError("Training matrix is singular; increase regularization or inspect feature redundancy.")
        if pivot_row != pivot_index:
            augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]

        pivot_value = augmented[pivot_index][pivot_index]
        augmented[pivot_index] = [value / pivot_value for value in augmented[pivot_index]]

        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            augmented[row_index] = [
                current - factor * pivot
                for current, pivot in zip(augmented[row_index], augmented[pivot_index])
            ]

    return [row[-1] for row in augmented]


def _predict_with_linear_model(
    rows: list[TrainingRow],
    means: dict[str, float],
    scales: dict[str, float],
    coefficients: list[float],
    intercept: float,
) -> list[float]:
    predictions: list[float] = []
    for row in rows:
        feature_vector = row.scoring_features()
        normalized = [
            (feature_vector[feature_name] - means[feature_name]) / scales[feature_name]
            for feature_name in SCORING_FEATURE_ORDER
        ]
        predictions.append(intercept + sum(weight * value for weight, value in zip(coefficients, normalized)))
    return predictions


def _regression_metrics(predictions: list[float], targets: list[float]) -> dict[str, float]:
    if not predictions:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "max_abs_error": 0.0,
            "r2": 0.0,
        }

    absolute_errors = [abs(prediction - target) for prediction, target in zip(predictions, targets)]
    squared_errors = [(prediction - target) ** 2 for prediction, target in zip(predictions, targets)]
    target_mean = sum(targets) / len(targets)
    target_variance = sum((target - target_mean) ** 2 for target in targets)
    explained_error = sum(squared_errors)
    r2 = 1.0 if target_variance <= 1e-12 else 1 - (explained_error / target_variance)
    return {
        "mae": round(sum(absolute_errors) / len(absolute_errors), 6),
        "rmse": round((sum(squared_errors) / len(squared_errors)) ** 0.5, 6),
        "max_abs_error": round(max(absolute_errors), 6),
        "r2": round(r2, 6),
    }


def _artifact_runtime_defaults() -> dict[str, Any]:
    return {
        "freshness_penalties": {
            "Desactualizado": 0.04,
            "Retrasado": 0.015,
        },
        "confidence_scores": {
            "Fresco": 2,
            "Retrasado": 1,
            "Desactualizado": 0,
            "Estatico": 2,
        },
        "bounds": {
            "min": 0.08,
            "max": 0.92,
        },
    }


def _phase_breakdown(rows: list[TrainingRow]) -> dict[str, int]:
    return {
        "latest": sum(1 for row in rows if row.phase == "latest"),
        "previous": sum(1 for row in rows if row.phase == "previous"),
    }


def _feature_matrix(rows: list[TrainingRow]) -> tuple[list[dict[str, float]], list[list[float]], list[float]]:
    feature_rows = [row.scoring_features() for row in rows]
    matrix = [
        [float(feature_row[feature_name]) for feature_name in SCORING_FEATURE_ORDER]
        for feature_row in feature_rows
    ]
    targets = [float(row.target_score) for row in rows]
    return feature_rows, matrix, targets


def _fit_affine_calibration(predictions: list[float], targets: list[float]) -> dict[str, float | str]:
    if len(predictions) < 2:
        return {
            "method": "identity",
            "slope": 1.0,
            "intercept": 0.0,
        }

    prediction_mean = sum(predictions) / len(predictions)
    target_mean = sum(targets) / len(targets)
    covariance = sum(
        (prediction - prediction_mean) * (target - target_mean)
        for prediction, target in zip(predictions, targets)
    )
    variance = sum((prediction - prediction_mean) ** 2 for prediction in predictions)
    if abs(variance) <= 1e-12:
        return {
            "method": "identity",
            "slope": 1.0,
            "intercept": 0.0,
        }
    slope = covariance / variance
    intercept = target_mean - (slope * prediction_mean)
    return {
        "method": "affine",
        "slope": round(slope, 6),
        "intercept": round(intercept, 6),
    }


def apply_affine_calibration(score: float, calibration: dict[str, Any] | None) -> float:
    if not calibration:
        return score
    slope = float(calibration.get("slope", 1.0))
    intercept = float(calibration.get("intercept", 0.0))
    return (score * slope) + intercept


def train_seed_linear_artifact(
    *,
    version: str,
    alpha: float,
    rows: list[TrainingRow] | None = None,
    train_rows: list[TrainingRow] | None = None,
    validation_rows: list[TrainingRow] | None = None,
    dataset_name: str = "frontend_seed_bootstrap",
) -> dict[str, Any]:
    rows, training_rows, validation_rows = _resolve_training_splits(
        rows,
        train_rows=train_rows,
        validation_rows=validation_rows,
    )
    means, scales = _feature_stats(training_rows)
    feature_vectors = [row.scoring_features() for row in training_rows]
    training_targets = [row.target_score for row in training_rows]
    target_mean = sum(training_targets) / len(training_targets)

    normalized_rows = [
        [
            (feature_vector[feature_name] - means[feature_name]) / scales[feature_name]
            for feature_name in SCORING_FEATURE_ORDER
        ]
        for feature_vector in feature_vectors
    ]
    centered_targets = [target - target_mean for target in training_targets]

    feature_count = len(SCORING_FEATURE_ORDER)
    gram_matrix = [[0.0 for _ in range(feature_count)] for _ in range(feature_count)]
    response_vector = [0.0 for _ in range(feature_count)]

    for row_index, normalized_row in enumerate(normalized_rows):
        for first_index in range(feature_count):
            response_vector[first_index] += normalized_row[first_index] * centered_targets[row_index]
            for second_index in range(feature_count):
                gram_matrix[first_index][second_index] += (
                    normalized_row[first_index] * normalized_row[second_index]
                )

    for index in range(feature_count):
        gram_matrix[index][index] += alpha

    coefficients = _solve_linear_system(gram_matrix, response_vector)

    train_predictions = _predict_with_linear_model(training_rows, means, scales, coefficients, target_mean)
    validation_predictions = _predict_with_linear_model(validation_rows, means, scales, coefficients, target_mean)
    validation_targets = [row.target_score for row in validation_rows]
    calibration = _fit_affine_calibration(validation_predictions, validation_targets)
    calibrated_validation_predictions = [
        apply_affine_calibration(prediction, calibration)
        for prediction in validation_predictions
    ]
    all_predictions = _predict_with_linear_model(rows, means, scales, coefficients, target_mean)
    all_targets = [row.target_score for row in rows]
    runtime_defaults = _artifact_runtime_defaults()

    return {
        "model_id": "trained-spatial-seed-linear-regression",
        "version": version,
        "artifact_type": "trained_linear_model",
        "model_family": "linear_ridge",
        "description": "Seed-trained spatial risk model exported from the current bootstrap dataset.",
        "feature_order": SCORING_FEATURE_ORDER,
        "feature_means": {key: round(value, 6) for key, value in means.items()},
        "feature_scales": {key: round(value, 6) for key, value in scales.items()},
        "coefficients": {
            feature_name: round(coefficients[index], 6)
            for index, feature_name in enumerate(SCORING_FEATURE_ORDER)
        },
        "intercept": round(target_mean, 6),
        "calibration": {
            **calibration,
            "validation_metrics_before": _regression_metrics(validation_predictions, validation_targets),
            "validation_metrics_after": _regression_metrics(calibrated_validation_predictions, validation_targets),
            "validation_rows": len(validation_rows),
        },
        **runtime_defaults,
        "training": {
            "dataset": dataset_name,
            "rows": len(rows),
            "model_family": "linear_ridge",
            "alpha": alpha,
            "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "metrics": _regression_metrics(all_predictions, all_targets),
            "phase_breakdown": _phase_breakdown(rows),
            "splits": {
                "train_rows": len(training_rows),
                "validation_rows": len(validation_rows),
                "train_metrics": _regression_metrics(train_predictions, training_targets),
                "validation_metrics": _regression_metrics(validation_predictions, validation_targets),
            },
        },
    }


def train_beta_regression_artifact(
    *,
    version: str,
    alpha: float,
    rows: list[TrainingRow] | None = None,
    train_rows: list[TrainingRow] | None = None,
    validation_rows: list[TrainingRow] | None = None,
    dataset_name: str = "frontend_seed_bootstrap",
    max_iterations: int = 250,
    learning_rate: float = 0.2,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    rows, training_rows, validation_rows = _resolve_training_splits(
        rows,
        train_rows=train_rows,
        validation_rows=validation_rows,
    )
    train_feature_rows, train_matrix, train_targets = _feature_matrix(training_rows)
    validation_feature_rows, _, validation_targets = _feature_matrix(validation_rows)
    all_feature_rows, _, all_targets = _feature_matrix(rows)

    beta_model = train_beta_regression_regressor(
        feature_names=SCORING_FEATURE_ORDER,
        train_matrix=train_matrix,
        train_targets=train_targets,
        alpha=alpha,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        tolerance=tolerance,
    )
    raw_train_predictions = predict_beta_regression_rows(beta_model, train_feature_rows)
    raw_validation_predictions = predict_beta_regression_rows(
        beta_model, validation_feature_rows
    )
    calibration = _fit_affine_calibration(raw_validation_predictions, validation_targets)
    calibrated_validation_predictions = [
        apply_affine_calibration(prediction, calibration)
        for prediction in raw_validation_predictions
    ]
    raw_all_predictions = predict_beta_regression_rows(beta_model, all_feature_rows)
    runtime_defaults = _artifact_runtime_defaults()

    return {
        "model_id": "trained-spatial-seed-beta-regression",
        "version": version,
        "artifact_type": "beta_regression_model",
        "model_family": "beta_regression",
        "description": "Seed-trained bounded beta-regression spatial risk model exported from the current bootstrap dataset.",
        "feature_order": SCORING_FEATURE_ORDER,
        "feature_means": beta_model["feature_means"],
        "feature_scales": beta_model["feature_scales"],
        "coefficients": beta_model["coefficients"],
        "intercept": beta_model["intercept"],
        "precision": beta_model["precision"],
        "link_function": beta_model["link_function"],
        "response_distribution": beta_model["response_distribution"],
        "feature_importance_method": beta_model["feature_importance_method"],
        "feature_importance": beta_model["feature_importance"],
        "feature_direction": beta_model["feature_direction"],
        "component_score_space": beta_model["component_score_space"],
        "calibration": {
            **calibration,
            "validation_metrics_before": _regression_metrics(
                raw_validation_predictions, validation_targets
            ),
            "validation_metrics_after": _regression_metrics(
                calibrated_validation_predictions, validation_targets
            ),
            "validation_rows": len(validation_rows),
        },
        **runtime_defaults,
        "training": {
            "dataset": dataset_name,
            "rows": len(rows),
            "model_family": "beta_regression",
            "hyperparameters": {
                "alpha": round(alpha, 6),
                "max_iterations": max_iterations,
                "learning_rate": round(learning_rate, 6),
                "tolerance": tolerance,
                "precision": beta_model["precision"],
            },
            "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "metrics": _regression_metrics(raw_all_predictions, all_targets),
            "phase_breakdown": _phase_breakdown(rows),
            "splits": {
                "train_rows": len(training_rows),
                "validation_rows": len(validation_rows),
                "train_metrics": _regression_metrics(
                    raw_train_predictions, train_targets
                ),
                "validation_metrics": _regression_metrics(
                    raw_validation_predictions, validation_targets
                ),
                "converged": bool(beta_model["converged"]),
                "iterations": int(beta_model["iterations"]),
                "optimization_history": beta_model["optimization_history"],
            },
        },
    }


def train_gradient_boosted_tree_artifact(
    *,
    version: str,
    rows: list[TrainingRow] | None = None,
    train_rows: list[TrainingRow] | None = None,
    validation_rows: list[TrainingRow] | None = None,
    dataset_name: str = "frontend_seed_bootstrap",
    learning_rate: float = 0.1,
    estimator_count: int = 24,
    max_depth: int = 3,
    min_leaf_size: int = 2,
    min_split_gain: float = 0.0,
    early_stopping_rounds: int = 4,
) -> dict[str, Any]:
    rows, training_rows, validation_rows = _resolve_training_splits(
        rows,
        train_rows=train_rows,
        validation_rows=validation_rows,
    )
    _, train_matrix, train_targets = _feature_matrix(training_rows)
    _, validation_matrix, validation_targets = _feature_matrix(validation_rows)
    _, all_matrix, all_targets = _feature_matrix(rows)
    validation_matrix_input = validation_matrix if validation_rows else None
    validation_targets_input = validation_targets if validation_rows else None

    boosting = train_gradient_boosted_regressor(
        feature_names=SCORING_FEATURE_ORDER,
        train_matrix=train_matrix,
        train_targets=train_targets,
        validation_matrix=validation_matrix_input,
        validation_targets=validation_targets_input,
        learning_rate=learning_rate,
        estimator_count=estimator_count,
        max_depth=max_depth,
        min_leaf_size=min_leaf_size,
        min_split_gain=min_split_gain,
        early_stopping_rounds=early_stopping_rounds,
    )

    raw_train_predictions = [
        float(predict_gradient_boosted_ensemble(
            {
                "base_score": boosting["base_score"],
                "learning_rate": boosting["learning_rate"],
                "feature_order": SCORING_FEATURE_ORDER,
                "trees": boosting["trees"],
            },
            feature_row,
        ))
        for feature_row, _ in zip(train_matrix, train_targets, strict=True)
    ]
    raw_validation_predictions = [
        float(predict_gradient_boosted_ensemble(
            {
                "base_score": boosting["base_score"],
                "learning_rate": boosting["learning_rate"],
                "feature_order": SCORING_FEATURE_ORDER,
                "trees": boosting["trees"],
            },
            feature_row,
        ))
        for feature_row, _ in zip(validation_matrix, validation_targets, strict=True)
    ]
    calibration = _fit_affine_calibration(raw_validation_predictions, validation_targets)
    calibrated_validation_predictions = [
        apply_affine_calibration(prediction, calibration)
        for prediction in raw_validation_predictions
    ]
    raw_all_predictions = [
        float(predict_gradient_boosted_ensemble(
            {
                "base_score": boosting["base_score"],
                "learning_rate": boosting["learning_rate"],
                "feature_order": SCORING_FEATURE_ORDER,
                "trees": boosting["trees"],
            },
            feature_row,
        ))
        for feature_row, _ in zip(all_matrix, all_targets, strict=True)
    ]
    runtime_defaults = _artifact_runtime_defaults()

    return {
        "model_id": "trained-spatial-seed-gradient-boosted-tree",
        "version": version,
        "artifact_type": "gradient_boosted_tree_model",
        "model_family": "gradient_boosted_tree",
        "description": "Seed-trained gradient-boosted spatial risk model exported from the current bootstrap dataset.",
        "feature_order": SCORING_FEATURE_ORDER,
        "base_score": boosting["base_score"],
        "learning_rate": boosting["learning_rate"],
        "tree_count": boosting["effective_tree_count"],
        "requested_tree_count": boosting["requested_tree_count"],
        "max_depth": max_depth,
        "min_leaf_size": min_leaf_size,
        "min_split_gain": round(min_split_gain, 6),
        "early_stopping_rounds": early_stopping_rounds,
        "trees": boosting["trees"],
        "feature_importance_method": "split_gain",
        "feature_importance": boosting["feature_importance"],
        "calibration": {
            **calibration,
            "validation_metrics_before": _regression_metrics(
                raw_validation_predictions, validation_targets
            ),
            "validation_metrics_after": _regression_metrics(
                calibrated_validation_predictions, validation_targets
            ),
            "validation_rows": len(validation_rows),
        },
        **runtime_defaults,
        "training": {
            "dataset": dataset_name,
            "rows": len(rows),
            "model_family": "gradient_boosted_tree",
            "hyperparameters": {
                "learning_rate": round(learning_rate, 6),
                "estimator_count": estimator_count,
                "max_depth": max_depth,
                "min_leaf_size": min_leaf_size,
                "min_split_gain": round(min_split_gain, 6),
                "early_stopping_rounds": early_stopping_rounds,
            },
            "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "metrics": _regression_metrics(raw_all_predictions, all_targets),
            "phase_breakdown": _phase_breakdown(rows),
            "splits": {
                "train_rows": len(training_rows),
                "validation_rows": len(validation_rows),
                "train_metrics": _regression_metrics(
                    raw_train_predictions, train_targets
                ),
                "validation_metrics": _regression_metrics(
                    raw_validation_predictions, validation_targets
                ),
                "early_stopping_triggered": bool(
                    boosting["early_stopping_triggered"]
                ),
                "effective_tree_count": boosting["effective_tree_count"],
                "requested_tree_count": boosting["requested_tree_count"],
                "best_validation_rmse": boosting["best_validation_rmse"],
                "training_history": boosting["training_history"],
            },
        },
    }


def train_xgboost_artifact(
    *,
    version: str,
    rows: list[TrainingRow] | None = None,
    train_rows: list[TrainingRow] | None = None,
    validation_rows: list[TrainingRow] | None = None,
    dataset_name: str = "frontend_seed_bootstrap",
    learning_rate: float = 0.05,
    estimator_count: int = 64,
    max_depth: int = 4,
    min_leaf_size: int = 1,
    min_split_gain: float = 0.0,
    early_stopping_rounds: int = 8,
) -> dict[str, Any]:
    rows, training_rows, validation_rows = _resolve_training_splits(
        rows,
        train_rows=train_rows,
        validation_rows=validation_rows,
    )
    train_feature_rows, train_matrix, train_targets = _feature_matrix(training_rows)
    validation_feature_rows, validation_matrix, validation_targets = _feature_matrix(
        validation_rows
    )
    all_feature_rows, _, all_targets = _feature_matrix(rows)

    xgboost_model = train_xgboost_regressor(
        feature_names=SCORING_FEATURE_ORDER,
        train_matrix=train_matrix,
        train_targets=train_targets,
        validation_matrix=validation_matrix if validation_rows else None,
        validation_targets=validation_targets if validation_rows else None,
        learning_rate=learning_rate,
        estimator_count=estimator_count,
        max_depth=max_depth,
        min_leaf_size=min_leaf_size,
        min_split_gain=min_split_gain,
        early_stopping_rounds=early_stopping_rounds,
    )

    raw_train_predictions = predict_xgboost_rows(xgboost_model, train_feature_rows)
    raw_validation_predictions = predict_xgboost_rows(
        xgboost_model, validation_feature_rows
    )
    calibration = _fit_affine_calibration(raw_validation_predictions, validation_targets)
    calibrated_validation_predictions = [
        apply_affine_calibration(prediction, calibration)
        for prediction in raw_validation_predictions
    ]
    raw_all_predictions = predict_xgboost_rows(xgboost_model, all_feature_rows)
    runtime_defaults = _artifact_runtime_defaults()

    return {
        "model_id": "trained-spatial-seed-xgboost",
        "version": version,
        "artifact_type": "xgboost_model",
        "model_family": "xgboost",
        "description": "Seed-trained XGBoost spatial risk model exported from the current bootstrap dataset.",
        "feature_order": SCORING_FEATURE_ORDER,
        "base_score": xgboost_model["base_score"],
        "requested_tree_count": xgboost_model["requested_tree_count"],
        "effective_tree_count": xgboost_model["effective_tree_count"],
        "best_iteration": xgboost_model["best_iteration"],
        "model_b64": xgboost_model["model_b64"],
        "feature_importance_method": xgboost_model["feature_importance_method"],
        "feature_importance": xgboost_model["feature_importance"],
        "calibration": {
            **calibration,
            "validation_metrics_before": _regression_metrics(
                raw_validation_predictions, validation_targets
            ),
            "validation_metrics_after": _regression_metrics(
                calibrated_validation_predictions, validation_targets
            ),
            "validation_rows": len(validation_rows),
        },
        **runtime_defaults,
        "training": {
            "dataset": dataset_name,
            "rows": len(rows),
            "model_family": "xgboost",
            "hyperparameters": {
                "learning_rate": round(learning_rate, 6),
                "estimator_count": estimator_count,
                "max_depth": max_depth,
                "min_leaf_size": min_leaf_size,
                "min_child_weight": float(min_leaf_size),
                "min_split_gain": round(min_split_gain, 6),
                "gamma": round(min_split_gain, 6),
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 1.0,
                "early_stopping_rounds": early_stopping_rounds,
                "tree_method": "hist",
            },
            "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "metrics": _regression_metrics(raw_all_predictions, all_targets),
            "phase_breakdown": _phase_breakdown(rows),
            "splits": {
                "train_rows": len(training_rows),
                "validation_rows": len(validation_rows),
                "train_metrics": _regression_metrics(
                    raw_train_predictions, train_targets
                ),
                "validation_metrics": _regression_metrics(
                    raw_validation_predictions, validation_targets
                ),
                "early_stopping_triggered": bool(
                    xgboost_model["early_stopping_triggered"]
                ),
                "effective_tree_count": xgboost_model["effective_tree_count"],
                "requested_tree_count": xgboost_model["requested_tree_count"],
                "best_iteration": xgboost_model["best_iteration"],
                "best_validation_rmse": xgboost_model["best_validation_rmse"],
                "training_history": xgboost_model["training_history"],
                "native_params": xgboost_model["native_params"],
            },
        },
    }


def train_additive_spline_artifact(
    *,
    version: str,
    rows: list[TrainingRow] | None = None,
    train_rows: list[TrainingRow] | None = None,
    validation_rows: list[TrainingRow] | None = None,
    dataset_name: str = "frontend_seed_bootstrap",
    alpha: float = 1.5,
    knot_count: int = 3,
) -> dict[str, Any]:
    rows, training_rows, validation_rows = _resolve_training_splits(
        rows,
        train_rows=train_rows,
        validation_rows=validation_rows,
    )
    train_feature_rows, train_matrix, train_targets = _feature_matrix(training_rows)
    validation_feature_rows, _, validation_targets = _feature_matrix(validation_rows)
    all_feature_rows, _, all_targets = _feature_matrix(rows)

    additive_model = train_additive_spline_regressor(
        feature_names=SCORING_FEATURE_ORDER,
        train_matrix=train_matrix,
        train_targets=train_targets,
        alpha=alpha,
        knot_count=knot_count,
    )

    raw_train_predictions = [
        float(predict_additive_spline_regressor(additive_model, feature_row))
        for feature_row in train_feature_rows
    ]
    raw_validation_predictions = [
        float(predict_additive_spline_regressor(additive_model, feature_row))
        for feature_row in validation_feature_rows
    ]
    calibration = _fit_affine_calibration(raw_validation_predictions, validation_targets)
    calibrated_validation_predictions = [
        apply_affine_calibration(prediction, calibration)
        for prediction in raw_validation_predictions
    ]
    raw_all_predictions = [
        float(predict_additive_spline_regressor(additive_model, feature_row))
        for feature_row in all_feature_rows
    ]
    runtime_defaults = _artifact_runtime_defaults()

    return {
        "model_id": "trained-spatial-seed-additive-spline",
        "version": version,
        "artifact_type": "additive_spline_model",
        "model_family": "additive_spline",
        "description": "Seed-trained additive spline spatial risk model exported from the current bootstrap dataset.",
        "feature_order": SCORING_FEATURE_ORDER,
        "intercept": additive_model["intercept"],
        "feature_terms": additive_model["feature_terms"],
        "feature_importance_method": additive_model["feature_importance_method"],
        "feature_importance": additive_model["feature_importance"],
        "feature_direction": additive_model["feature_direction"],
        "basis_count": additive_model["basis_count"],
        "calibration": {
            **calibration,
            "validation_metrics_before": _regression_metrics(
                raw_validation_predictions, validation_targets
            ),
            "validation_metrics_after": _regression_metrics(
                calibrated_validation_predictions, validation_targets
            ),
            "validation_rows": len(validation_rows),
        },
        **runtime_defaults,
        "training": {
            "dataset": dataset_name,
            "rows": len(rows),
            "model_family": "additive_spline",
            "hyperparameters": {
                "alpha": round(alpha, 6),
                "knot_count": int(knot_count),
                "basis_count": additive_model["basis_count"],
            },
            "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "metrics": _regression_metrics(raw_all_predictions, all_targets),
            "phase_breakdown": _phase_breakdown(rows),
            "splits": {
                "train_rows": len(training_rows),
                "validation_rows": len(validation_rows),
                "train_metrics": _regression_metrics(
                    raw_train_predictions, train_targets
                ),
                "validation_metrics": _regression_metrics(
                    raw_validation_predictions, validation_targets
                ),
            },
        },
    }


def export_seed_linear_artifact(
    *,
    version: str,
    alpha: float = 0.75,
    rows: list[TrainingRow] | None = None,
    dataset_name: str = "frontend_seed_bootstrap",
    artifacts_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    settings = get_settings()
    target_directory = artifacts_path or settings.resolved_model_artifacts_path
    target_directory.mkdir(parents=True, exist_ok=True)
    artifact = train_seed_linear_artifact(
        version=version,
        alpha=alpha,
        rows=rows,
        dataset_name=dataset_name,
    )
    artifact_path = target_directory / f"{version}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact_path, artifact


def export_beta_regression_artifact(
    *,
    version: str,
    alpha: float = 0.75,
    rows: list[TrainingRow] | None = None,
    dataset_name: str = "frontend_seed_bootstrap",
    max_iterations: int = 250,
    learning_rate: float = 0.2,
    tolerance: float = 1e-6,
    artifacts_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    settings = get_settings()
    target_directory = artifacts_path or settings.resolved_model_artifacts_path
    target_directory.mkdir(parents=True, exist_ok=True)
    artifact = train_beta_regression_artifact(
        version=version,
        alpha=alpha,
        rows=rows,
        dataset_name=dataset_name,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        tolerance=tolerance,
    )
    artifact_path = target_directory / f"{version}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact_path, artifact


def export_gradient_boosted_tree_artifact(
    *,
    version: str,
    rows: list[TrainingRow] | None = None,
    dataset_name: str = "frontend_seed_bootstrap",
    learning_rate: float = 0.1,
    estimator_count: int = 24,
    max_depth: int = 3,
    min_leaf_size: int = 2,
    min_split_gain: float = 0.0,
    early_stopping_rounds: int = 4,
    artifacts_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    settings = get_settings()
    target_directory = artifacts_path or settings.resolved_model_artifacts_path
    target_directory.mkdir(parents=True, exist_ok=True)
    artifact = train_gradient_boosted_tree_artifact(
        version=version,
        rows=rows,
        dataset_name=dataset_name,
        learning_rate=learning_rate,
        estimator_count=estimator_count,
        max_depth=max_depth,
        min_leaf_size=min_leaf_size,
        min_split_gain=min_split_gain,
        early_stopping_rounds=early_stopping_rounds,
    )
    artifact_path = target_directory / f"{version}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact_path, artifact


def export_xgboost_artifact(
    *,
    version: str,
    rows: list[TrainingRow] | None = None,
    dataset_name: str = "frontend_seed_bootstrap",
    learning_rate: float = 0.05,
    estimator_count: int = 64,
    max_depth: int = 4,
    min_leaf_size: int = 1,
    min_split_gain: float = 0.0,
    early_stopping_rounds: int = 8,
    artifacts_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    settings = get_settings()
    target_directory = artifacts_path or settings.resolved_model_artifacts_path
    target_directory.mkdir(parents=True, exist_ok=True)
    artifact = train_xgboost_artifact(
        version=version,
        rows=rows,
        dataset_name=dataset_name,
        learning_rate=learning_rate,
        estimator_count=estimator_count,
        max_depth=max_depth,
        min_leaf_size=min_leaf_size,
        min_split_gain=min_split_gain,
        early_stopping_rounds=early_stopping_rounds,
    )
    artifact_path = target_directory / f"{version}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact_path, artifact


def export_additive_spline_artifact(
    *,
    version: str,
    rows: list[TrainingRow] | None = None,
    dataset_name: str = "frontend_seed_bootstrap",
    alpha: float = 1.5,
    knot_count: int = 3,
    artifacts_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    settings = get_settings()
    target_directory = artifacts_path or settings.resolved_model_artifacts_path
    target_directory.mkdir(parents=True, exist_ok=True)
    artifact = train_additive_spline_artifact(
        version=version,
        rows=rows,
        dataset_name=dataset_name,
        alpha=alpha,
        knot_count=knot_count,
    )
    artifact_path = target_directory / f"{version}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact_path, artifact
