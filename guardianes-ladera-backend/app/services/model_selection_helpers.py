"""Pure helpers for model-selection service workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import product

from app.core.exceptions import ApiError
from app.ml.model_selection import alpha_slug


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def time_window_gap_days(
    current_time_window: dict, historical_time_window: dict
) -> float | None:
    current_start = parse_iso_datetime(current_time_window.get("start_at"))
    current_end = parse_iso_datetime(current_time_window.get("end_at"))
    historical_start = parse_iso_datetime(historical_time_window.get("start_at"))
    historical_end = parse_iso_datetime(historical_time_window.get("end_at"))

    if current_start and current_end and historical_start and historical_end:
        if historical_end < current_start:
            gap_seconds = (current_start - historical_end).total_seconds()
        elif current_end < historical_start:
            gap_seconds = (historical_start - current_end).total_seconds()
        else:
            gap_seconds = 0.0
        return round(gap_seconds / 86400, 6)

    current_reference = parse_iso_datetime(current_time_window.get("reference_at"))
    historical_reference = parse_iso_datetime(
        historical_time_window.get("reference_at")
    )
    if current_reference and historical_reference:
        return round(
            abs((current_reference - historical_reference).total_seconds()) / 86400,
            6,
        )
    return None


def taxonomy_group(dataset_context: dict) -> str | None:
    taxonomy = dict(dataset_context.get("dataset_taxonomy") or {})
    group = taxonomy.get("taxonomy_group")
    return str(group) if group else None


def cohort_distance(current_cohort: dict, historical_cohort: dict) -> int | None:
    current_group = current_cohort.get("cohort_group")
    historical_group = historical_cohort.get("cohort_group")
    if not current_group or not historical_group:
        return None
    if current_group != historical_group:
        return None
    current_index = current_cohort.get("bucket_index")
    historical_index = historical_cohort.get("bucket_index")
    if current_index is None or historical_index is None:
        return None
    try:
        return abs(int(current_index) - int(historical_index))
    except (TypeError, ValueError):
        return None


def candidate_signature(candidate: dict) -> str:
    model_family = str(candidate.get("model_family") or "unknown")
    alpha = candidate.get("alpha")
    hyperparameters = dict(candidate.get("hyperparameters") or {})
    if (
        alpha is not None
        and model_family in {"linear_ridge", "unknown", "baseline"}
        and set(hyperparameters) <= {"alpha"}
    ):
        return f"trained_linear_alpha:{alpha_slug(float(alpha))}"
    hyperparameters.pop("basis_count", None)
    if hyperparameters:
        ordered_items = ",".join(
            f"{key}={hyperparameters[key]}" for key in sorted(hyperparameters)
        )
        return f"{model_family}:{ordered_items}"
    return str(candidate.get("model_version") or "unknown")


def unique_floats(values: list[float] | None, *, default: list[float]) -> list[float]:
    source = values or default
    unique: list[float] = []
    for value in source:
        numeric = float(value)
        if numeric not in unique:
            unique.append(numeric)
    return unique


def unique_ints(values: list[int] | None, *, default: list[int]) -> list[int]:
    source = values or default
    unique: list[int] = []
    for value in source:
        numeric = int(value)
        if numeric not in unique:
            unique.append(numeric)
    return unique


def resolve_model_families(
    *,
    model_family: str,
    model_families: list[str] | None,
) -> list[str]:
    supported = {
        "linear_ridge",
        "beta_regression",
        "additive_spline",
        "gradient_boosted_tree",
        "xgboost",
    }
    requested = list(model_families or [model_family])
    resolved: list[str] = []
    for family in requested:
        normalized = str(family or "").strip()
        if normalized not in supported:
            raise ApiError(
                400,
                "invalid_model_family",
                f"Unsupported model family: {family}",
            )
        if normalized not in resolved:
            resolved.append(normalized)
    return resolved


def benchmark_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def family_rollups(candidates: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for candidate in candidates:
        family = str(candidate.get("model_family") or "unknown")
        grouped.setdefault(family, []).append(candidate)

    rollups: list[dict] = []
    for family, family_candidates in grouped.items():
        ranked = sorted(family_candidates, key=lambda item: int(item.get("rank") or 0))
        best_candidate = ranked[0]
        rollups.append(
            {
                "model_family": family,
                "candidate_count": len(family_candidates),
                "best_rank": int(best_candidate.get("rank") or 0),
                "best_model_version": best_candidate.get("model_version"),
                "best_validation_rmse": best_candidate.get("validation_rmse"),
                "best_validation_risk_level_accuracy": best_candidate.get(
                    "validation_risk_level_accuracy"
                ),
                "best_validation_brier_score": best_candidate.get(
                    "validation_brier_score"
                ),
                "best_validation_auprc": best_candidate.get("validation_auprc"),
                "best_validation_recall": best_candidate.get("validation_recall"),
                "best_validation_mcc": best_candidate.get("validation_mcc"),
                "best_validation_rows": best_candidate.get("validation_rows"),
                "best_hyperparameters": best_candidate.get("hyperparameters") or {},
                "mean_validation_rmse": round(
                    sum(
                        float(candidate.get("validation_rmse") or 0.0)
                        for candidate in family_candidates
                    )
                    / len(family_candidates),
                    6,
                ),
            }
        )

    rollups.sort(
        key=lambda item: (
            int(item.get("best_rank") or 0),
            float(item.get("best_validation_rmse") or 0.0),
            str(item.get("model_family") or ""),
        )
    )
    return rollups


def resolve_boosted_tree_candidate_configs(
    *,
    learning_rates: list[float] | None,
    estimator_counts: list[int] | None,
    max_depths: list[int] | None,
    min_leaf_sizes: list[int] | None,
    min_split_gains: list[float] | None,
    early_stopping_rounds: int | None,
) -> list[dict[str, int | float]]:
    resolved_learning_rates = unique_floats(learning_rates, default=[0.05, 0.1])
    resolved_estimator_counts = unique_ints(estimator_counts, default=[12, 24])
    resolved_max_depths = unique_ints(max_depths, default=[2, 3])
    resolved_min_leaf_sizes = unique_ints(min_leaf_sizes, default=[1])
    resolved_min_split_gains = unique_floats(min_split_gains, default=[0.0])

    if any(value <= 0 or value > 1 for value in resolved_learning_rates):
        raise ApiError(
            400,
            "invalid_learning_rate",
            "All learningRates must be greater than zero and at most one.",
        )
    if any(value <= 0 for value in resolved_estimator_counts):
        raise ApiError(
            400,
            "invalid_estimator_count",
            "All estimatorCounts must be greater than zero.",
        )
    if any(value <= 0 for value in resolved_max_depths):
        raise ApiError(
            400,
            "invalid_max_depth",
            "All maxDepths must be greater than zero.",
        )
    if any(value <= 0 for value in resolved_min_leaf_sizes):
        raise ApiError(
            400,
            "invalid_min_leaf_size",
            "All minLeafSizes must be greater than zero.",
        )
    if any(value < 0 for value in resolved_min_split_gains):
        raise ApiError(
            400,
            "invalid_min_split_gain",
            "All minSplitGains must be zero or greater.",
        )

    candidate_configs = [
        {
            "learning_rate": learning_rate,
            "estimator_count": estimator_count,
            "max_depth": max_depth,
            "min_leaf_size": min_leaf_size,
            "min_split_gain": min_split_gain,
            "early_stopping_rounds": early_stopping_rounds
            if early_stopping_rounds is not None
            else 4,
        }
        for learning_rate, estimator_count, max_depth, min_leaf_size, min_split_gain in product(
            resolved_learning_rates,
            resolved_estimator_counts,
            resolved_max_depths,
            resolved_min_leaf_sizes,
            resolved_min_split_gains,
        )
    ]
    if len(candidate_configs) > 24:
        raise ApiError(
            400,
            "model_selection_grid_too_large",
            "The requested gradient-boosted-tree search grid is too large. Reduce the hyperparameter combinations to 24 or fewer.",
        )
    return candidate_configs


def resolve_additive_spline_candidate_configs(
    *,
    alphas: list[float] | None,
    knot_counts: list[int] | None,
) -> list[dict[str, int | float]]:
    resolved_alphas = unique_floats(alphas, default=[0.75, 1.5])
    resolved_knot_counts = unique_ints(knot_counts, default=[2, 3, 4])

    if any(alpha <= 0 for alpha in resolved_alphas):
        raise ApiError(
            400,
            "invalid_alpha",
            "All alpha candidates must be greater than zero.",
        )
    if any(knot_count < 0 for knot_count in resolved_knot_counts):
        raise ApiError(
            400,
            "invalid_knot_count",
            "All knotCounts must be zero or greater.",
        )

    candidate_configs = [
        {"alpha": alpha, "knot_count": knot_count}
        for alpha, knot_count in product(resolved_alphas, resolved_knot_counts)
    ]
    if len(candidate_configs) > 24:
        raise ApiError(
            400,
            "model_selection_grid_too_large",
            "The requested additive-spline search grid is too large. Reduce the hyperparameter combinations to 24 or fewer.",
        )
    return candidate_configs


def boosted_tree_candidate_suffix(config: dict[str, int | float]) -> str:
    suffix = (
        f"gbt-lr-{alpha_slug(float(config['learning_rate']))}"
        f"-trees-{int(config['estimator_count'])}"
        f"-depth-{int(config['max_depth'])}"
        f"-leaf-{int(config['min_leaf_size'])}"
    )
    min_split_gain = float(config.get("min_split_gain", 0.0))
    if min_split_gain > 0:
        suffix += f"-gain-{alpha_slug(min_split_gain)}"
    return suffix


def xgboost_candidate_suffix(config: dict[str, int | float]) -> str:
    suffix = (
        f"xgb-lr-{alpha_slug(float(config['learning_rate']))}"
        f"-trees-{int(config['estimator_count'])}"
        f"-depth-{int(config['max_depth'])}"
        f"-leaf-{int(config['min_leaf_size'])}"
    )
    min_split_gain = float(config.get("min_split_gain", 0.0))
    if min_split_gain > 0:
        suffix += f"-gamma-{alpha_slug(min_split_gain)}"
    return suffix


def additive_spline_candidate_suffix(config: dict[str, int | float]) -> str:
    return (
        f"gam-alpha-{alpha_slug(float(config['alpha']))}"
        f"-knots-{int(config['knot_count'])}"
    )


def probability_metrics(bucket: dict) -> dict:
    return dict(bucket.get("probability_metrics") or {})


def dataset_row_contexts(dataset: dict) -> list[dict]:
    return [dict(item.get("context") or {}) for item in dataset.get("rows") or []]


def temporal_bucket_reference(
    context: dict[str, object], fallback_label: str
) -> datetime | None:
    timestamp_candidates = [
        context.get("observedAt"),
        context.get("featureRunCompletedAt"),
        context.get("runCompletedAt"),
    ]
    for candidate in timestamp_candidates:
        parsed = parse_iso_datetime(str(candidate) if candidate is not None else None)
        if parsed is not None:
            return parsed

    if ":" in fallback_label:
        _, _, suffix = fallback_label.partition(":")
        parsed = parse_iso_datetime(f"{suffix}T00:00:00+00:00")
        if parsed is not None:
            return parsed
    return None


def metric_delta(
    active_value: float | None, challenger_value: float | None
) -> float | None:
    if active_value is None or challenger_value is None:
        return None
    return round(float(challenger_value) - float(active_value), 6)


def lower_is_better_improvement(
    active_value: float | None, challenger_value: float | None
) -> float | None:
    if active_value is None or challenger_value is None:
        return None
    return round(float(active_value) - float(challenger_value), 6)


def higher_is_better_gain(
    active_value: float | None, challenger_value: float | None
) -> float | None:
    if active_value is None or challenger_value is None:
        return None
    return round(float(challenger_value) - float(active_value), 6)
