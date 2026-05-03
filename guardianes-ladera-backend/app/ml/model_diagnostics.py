from __future__ import annotations

from typing import Any


def _ordered_feature_names(artifact: dict[str, Any]) -> list[str]:
    feature_order = list(artifact.get("feature_order") or [])
    if artifact.get("artifact_type") in {
        "trained_linear_model",
        "beta_regression_model",
    }:
        source_names = list((artifact.get("coefficients") or {}).keys())
    elif artifact.get("artifact_type") == "additive_spline_model":
        source_names = list((artifact.get("feature_terms") or {}).keys())
    elif artifact.get("artifact_type") in {"gradient_boosted_tree_model", "xgboost_model"}:
        source_names = list((artifact.get("feature_importance") or {}).keys())
    else:
        source_names = list((artifact.get("weights") or {}).keys())
    ordered = []
    for feature_name in [*feature_order, *source_names]:
        if feature_name not in ordered:
            ordered.append(feature_name)
    return ordered


def _feature_magnitude_map(
    artifact: dict[str, Any],
) -> tuple[str, dict[str, float], dict[str, str]]:
    feature_values: dict[str, float] = {}
    directions: dict[str, str] = {}
    if artifact.get("artifact_type") in {
        "trained_linear_model",
        "beta_regression_model",
    }:
        method = (
            "absolute_logit_coefficient"
            if artifact.get("artifact_type") == "beta_regression_model"
            else "absolute_linear_coefficient"
        )
        coefficients = artifact.get("coefficients") or {}
        for feature_name in _ordered_feature_names(artifact):
            value = float(coefficients.get(feature_name, 0.0))
            feature_values[feature_name] = value
            directions[feature_name] = (
                "positive" if value > 0 else "negative" if value < 0 else "neutral"
            )
        return method, feature_values, directions

    if artifact.get("artifact_type") == "additive_spline_model":
        method = str(
            artifact.get("feature_importance_method")
            or "mean_abs_additive_contribution"
        )
        importance = artifact.get("feature_importance") or {}
        direction_map = artifact.get("feature_direction") or {}
        for feature_name in _ordered_feature_names(artifact):
            value = float(importance.get(feature_name, 0.0))
            feature_values[feature_name] = value
            directions[feature_name] = str(direction_map.get(feature_name) or "neutral")
        return method, feature_values, directions

    if artifact.get("artifact_type") in {"gradient_boosted_tree_model", "xgboost_model"}:
        method = str(artifact.get("feature_importance_method") or "split_gain")
        importance = artifact.get("feature_importance") or {}
        for feature_name in _ordered_feature_names(artifact):
            value = float(importance.get(feature_name, 0.0))
            feature_values[feature_name] = value
            directions[feature_name] = "mixed" if value > 0 else "neutral"
        return method, feature_values, directions

    method = "artifact_weight"
    weights = artifact.get("weights") or {}
    for feature_name in _ordered_feature_names(artifact):
        value = float(weights.get(feature_name, 0.0))
        feature_values[feature_name] = value
        directions[feature_name] = (
            "positive" if value > 0 else "negative" if value < 0 else "neutral"
        )
    return method, feature_values, directions


def feature_importance_summary(
    artifact: dict[str, Any], *, top_feature_count: int = 5
) -> dict[str, Any]:
    method, feature_values, directions = _feature_magnitude_map(artifact)
    total_absolute_magnitude = sum(abs(value) for value in feature_values.values())
    rows: list[dict[str, Any]] = []
    for feature_name, value in feature_values.items():
        absolute_magnitude = abs(value)
        share = (
            round(absolute_magnitude / total_absolute_magnitude, 6)
            if total_absolute_magnitude > 1e-12
            else 0.0
        )
        rows.append(
            {
                "feature": feature_name,
                "magnitude": round(value, 6),
                "absolute_magnitude": round(absolute_magnitude, 6),
                "share": share,
                "direction": directions[feature_name],
            }
        )
    rows.sort(key=lambda item: (-item["absolute_magnitude"], item["feature"]))
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return {
        "method": method,
        "feature_count": len(rows),
        "total_absolute_magnitude": round(total_absolute_magnitude, 6),
        "top_features": rows[:top_feature_count],
        "features": rows,
    }


def calibration_effect_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    calibration = artifact.get("calibration") or {}
    before = calibration.get("validation_metrics_before") or {}
    after = calibration.get("validation_metrics_after") or {}
    rmse_before = before.get("rmse")
    rmse_after = after.get("rmse")
    mae_before = before.get("mae")
    mae_after = after.get("mae")

    validation_rmse_improvement = None
    if rmse_before is not None and rmse_after is not None:
        validation_rmse_improvement = round(float(rmse_before) - float(rmse_after), 6)

    validation_mae_improvement = None
    if mae_before is not None and mae_after is not None:
        validation_mae_improvement = round(float(mae_before) - float(mae_after), 6)

    return {
        "method": calibration.get("method", "none"),
        "applied": calibration.get("method", "none") not in {"none", "identity"},
        "slope": calibration.get("slope"),
        "intercept": calibration.get("intercept"),
        "validation_rows": calibration.get("validation_rows"),
        "validation_rmse_before": rmse_before,
        "validation_rmse_after": rmse_after,
        "validation_rmse_improvement": validation_rmse_improvement,
        "validation_mae_before": mae_before,
        "validation_mae_after": mae_after,
        "validation_mae_improvement": validation_mae_improvement,
    }


def coefficient_drift_summary(
    active_artifact: dict[str, Any],
    challenger_artifact: dict[str, Any],
    *,
    top_feature_count: int = 5,
) -> dict[str, Any]:
    active_method, active_values, active_directions = _feature_magnitude_map(
        active_artifact
    )
    challenger_method, challenger_values, challenger_directions = (
        _feature_magnitude_map(challenger_artifact)
    )
    feature_names = sorted(set(active_values) | set(challenger_values))
    rows: list[dict[str, Any]] = []
    for feature_name in feature_names:
        active_value = float(active_values.get(feature_name, 0.0))
        challenger_value = float(challenger_values.get(feature_name, 0.0))
        delta = challenger_value - active_value
        rows.append(
            {
                "feature": feature_name,
                "active_magnitude": round(active_value, 6),
                "challenger_magnitude": round(challenger_value, 6),
                "delta": round(delta, 6),
                "absolute_delta": round(abs(delta), 6),
                "active_direction": active_directions.get(feature_name, "neutral"),
                "challenger_direction": challenger_directions.get(
                    feature_name, "neutral"
                ),
                "direction_changed": active_directions.get(feature_name, "neutral")
                != challenger_directions.get(feature_name, "neutral"),
            }
        )
    rows.sort(key=lambda item: (-item["absolute_delta"], item["feature"]))
    total_absolute_delta = round(
        sum(float(item["absolute_delta"]) for item in rows), 6
    )
    mean_absolute_delta = round(
        total_absolute_delta / len(rows), 6
    ) if rows else 0.0
    return {
        "active_method": active_method,
        "challenger_method": challenger_method,
        "feature_count": len(rows),
        "changed_feature_count": sum(
            1 for item in rows if float(item["absolute_delta"]) > 0.0
        ),
        "total_absolute_delta": total_absolute_delta,
        "mean_absolute_delta": mean_absolute_delta,
        "largest_changes": rows[:top_feature_count],
    }


def feature_importance_change_summary(
    active_artifact: dict[str, Any],
    challenger_artifact: dict[str, Any],
    *,
    top_feature_count: int = 5,
) -> dict[str, Any]:
    active_summary = feature_importance_summary(
        active_artifact, top_feature_count=top_feature_count
    )
    challenger_summary = feature_importance_summary(
        challenger_artifact, top_feature_count=top_feature_count
    )
    active_features = {
        item["feature"]: item for item in active_summary.get("features") or []
    }
    challenger_features = {
        item["feature"]: item for item in challenger_summary.get("features") or []
    }
    feature_names = sorted(set(active_features) | set(challenger_features))
    share_changes: list[dict[str, Any]] = []
    for feature_name in feature_names:
        active_item = active_features.get(feature_name) or {}
        challenger_item = challenger_features.get(feature_name) or {}
        active_share = float(active_item.get("share", 0.0))
        challenger_share = float(challenger_item.get("share", 0.0))
        share_delta = challenger_share - active_share
        share_changes.append(
            {
                "feature": feature_name,
                "active_share": round(active_share, 6),
                "challenger_share": round(challenger_share, 6),
                "share_delta": round(share_delta, 6),
                "active_rank": active_item.get("rank"),
                "challenger_rank": challenger_item.get("rank"),
            }
        )
    share_changes.sort(key=lambda item: (-abs(float(item["share_delta"])), item["feature"]))
    active_top_features = {
        item["feature"] for item in active_summary.get("top_features") or []
    }
    challenger_top_features = {
        item["feature"] for item in challenger_summary.get("top_features") or []
    }
    return {
        "active_method": active_summary.get("method"),
        "challenger_method": challenger_summary.get("method"),
        "active_top_features": sorted(active_top_features),
        "challenger_top_features": sorted(challenger_top_features),
        "entered_top_features": sorted(challenger_top_features - active_top_features),
        "exited_top_features": sorted(active_top_features - challenger_top_features),
        "top_feature_overlap_count": len(active_top_features & challenger_top_features),
        "largest_share_shifts": share_changes[:top_feature_count],
    }


def calibration_delta_summary(
    active_artifact: dict[str, Any],
    challenger_artifact: dict[str, Any],
) -> dict[str, Any]:
    active_summary = calibration_effect_summary(active_artifact)
    challenger_summary = calibration_effect_summary(challenger_artifact)

    def delta(active_key: str, challenger_key: str) -> float | None:
        active_value = active_summary.get(active_key)
        challenger_value = challenger_summary.get(challenger_key)
        if active_value is None or challenger_value is None:
            return None
        return round(float(challenger_value) - float(active_value), 6)

    return {
        "active_method": active_summary.get("method"),
        "challenger_method": challenger_summary.get("method"),
        "active_validation_rmse_improvement": active_summary.get(
            "validation_rmse_improvement"
        ),
        "challenger_validation_rmse_improvement": challenger_summary.get(
            "validation_rmse_improvement"
        ),
        "active_validation_mae_improvement": active_summary.get(
            "validation_mae_improvement"
        ),
        "challenger_validation_mae_improvement": challenger_summary.get(
            "validation_mae_improvement"
        ),
        "validation_rmse_before_delta": delta(
            "validation_rmse_before", "validation_rmse_before"
        ),
        "validation_rmse_after_delta": delta(
            "validation_rmse_after", "validation_rmse_after"
        ),
        "validation_mae_before_delta": delta(
            "validation_mae_before", "validation_mae_before"
        ),
        "validation_mae_after_delta": delta(
            "validation_mae_after", "validation_mae_after"
        ),
        "calibration_gain_delta": delta(
            "validation_rmse_improvement", "validation_rmse_improvement"
        ),
    }
