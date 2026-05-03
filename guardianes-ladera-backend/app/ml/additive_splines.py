from __future__ import annotations

from typing import Any


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _standard_deviation(values: list[float], mean: float) -> float:
    if not values:
        return 1.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    deviation = variance**0.5
    return deviation if deviation > 1e-9 else 1.0


def _quantile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    if quantile <= 0:
        return sorted_values[0]
    if quantile >= 1:
        return sorted_values[-1]
    position = (len(sorted_values) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return (lower_value * (1 - weight)) + (upper_value * weight)


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]

    for pivot_index in range(size):
        pivot_row = max(
            range(pivot_index, size),
            key=lambda row_index: abs(augmented[row_index][pivot_index]),
        )
        pivot_value = augmented[pivot_row][pivot_index]
        if abs(pivot_value) < 1e-10:
            raise ValueError(
                "Additive spline training matrix is singular; increase regularization or reduce knot count."
            )
        if pivot_row != pivot_index:
            augmented[pivot_index], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[pivot_index],
            )

        pivot_value = augmented[pivot_index][pivot_index]
        augmented[pivot_index] = [
            value / pivot_value for value in augmented[pivot_index]
        ]

        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            augmented[row_index] = [
                current - factor * pivot
                for current, pivot in zip(
                    augmented[row_index], augmented[pivot_index], strict=True
                )
            ]

    return [row[-1] for row in augmented]


def _feature_value(
    feature_row: dict[str, float] | list[float], feature_name: str, feature_index: int
) -> float:
    if isinstance(feature_row, dict):
        return float(feature_row.get(feature_name, 0.0))
    return float(feature_row[feature_index])


def _standardized_feature_value(term: dict[str, Any], feature_value: float) -> float:
    return (feature_value - float(term.get("mean", 0.0))) / float(
        term.get("scale", 1.0)
    )


def _basis_value(spec: dict[str, Any], feature_row: dict[str, float] | list[float]) -> float:
    feature_value = _feature_value(
        feature_row, str(spec["feature"]), int(spec["feature_index"])
    )
    standardized = _standardized_feature_value(
        {"mean": spec["mean"], "scale": spec["scale"]},
        feature_value,
    )
    if spec["kind"] == "linear":
        return standardized
    return max(0.0, standardized - float(spec["knot"]))


def _effect_at_standardized_value(term: dict[str, Any], standardized_value: float) -> float:
    effect = float(term.get("linear_coefficient", 0.0)) * standardized_value
    knots = term.get("knots_standardized") or []
    hinge_coefficients = term.get("hinge_coefficients") or []
    for knot, coefficient in zip(knots, hinge_coefficients, strict=True):
        effect += float(coefficient) * max(0.0, standardized_value - float(knot))
    return effect


def _effect_profile(
    *,
    term: dict[str, Any],
    raw_knots: list[float],
    training_min: float,
    training_max: float,
) -> tuple[list[dict[str, float]], str]:
    sample_points = [training_min, *raw_knots, training_max]
    unique_points: list[float] = []
    for point in sample_points:
        if not unique_points or abs(point - unique_points[-1]) > 1e-9:
            unique_points.append(point)

    profile = [
        {
            "value": round(point, 6),
            "effect": round(
                _effect_at_standardized_value(
                    term,
                    _standardized_feature_value(term, point),
                ),
                6,
            ),
        }
        for point in unique_points
    ]

    directions: list[int] = []
    for previous, current in zip(profile, profile[1:]):
        delta = float(current["effect"]) - float(previous["effect"])
        if abs(delta) <= 1e-6:
            continue
        directions.append(1 if delta > 0 else -1)
    if not directions:
        direction = "neutral"
    elif len(set(directions)) > 1:
        direction = "mixed"
    else:
        direction = "positive" if directions[0] > 0 else "negative"
    return profile, direction


def train_additive_spline_regressor(
    *,
    feature_names: list[str],
    train_matrix: list[list[float]],
    train_targets: list[float],
    alpha: float = 1.5,
    knot_count: int = 3,
) -> dict[str, Any]:
    if not train_matrix or not train_targets:
        raise ValueError("Additive spline regression requires non-empty training data.")
    if alpha <= 0:
        raise ValueError("Additive spline alpha must be greater than zero.")
    if knot_count < 0:
        raise ValueError("Additive spline knot_count must be zero or greater.")

    feature_terms: dict[str, dict[str, Any]] = {}
    basis_specs: list[dict[str, Any]] = []

    for feature_index, feature_name in enumerate(feature_names):
        values = [float(row[feature_index]) for row in train_matrix]
        feature_mean = _mean(values)
        feature_scale = _standard_deviation(values, feature_mean)
        standardized_values = sorted(
            (value - feature_mean) / feature_scale for value in values
        )
        unique_standardized = sorted(set(standardized_values))
        internal_capacity = max(len(unique_standardized) - 2, 0)
        resolved_knot_count = min(int(knot_count), internal_capacity)
        knots_standardized: list[float] = []
        if resolved_knot_count > 0:
            for knot_index in range(1, resolved_knot_count + 1):
                candidate = _quantile(
                    standardized_values, knot_index / (resolved_knot_count + 1)
                )
                if not knots_standardized or abs(candidate - knots_standardized[-1]) > 1e-9:
                    knots_standardized.append(candidate)
        knots_raw = [
            (knot * feature_scale) + feature_mean for knot in knots_standardized
        ]
        feature_terms[feature_name] = {
            "basis_type": "piecewise_linear_truncated_power",
            "feature_index": feature_index,
            "mean": round(feature_mean, 6),
            "scale": round(feature_scale, 6),
            "training_min": round(min(values), 6),
            "training_max": round(max(values), 6),
            "knots_standardized": [round(knot, 6) for knot in knots_standardized],
            "knots_raw": [round(knot, 6) for knot in knots_raw],
        }
        basis_specs.append(
            {
                "feature": feature_name,
                "feature_index": feature_index,
                "kind": "linear",
                "mean": feature_mean,
                "scale": feature_scale,
            }
        )
        for knot in knots_standardized:
            basis_specs.append(
                {
                    "feature": feature_name,
                    "feature_index": feature_index,
                    "kind": "hinge",
                    "mean": feature_mean,
                    "scale": feature_scale,
                    "knot": knot,
                }
            )

    design_matrix = [
        [_basis_value(spec, feature_row) for spec in basis_specs]
        for feature_row in train_matrix
    ]
    target_mean = _mean(train_targets)
    centered_targets = [target - target_mean for target in train_targets]
    basis_count = len(basis_specs)
    gram_matrix = [[0.0 for _ in range(basis_count)] for _ in range(basis_count)]
    response_vector = [0.0 for _ in range(basis_count)]

    for row_index, design_row in enumerate(design_matrix):
        for first_index in range(basis_count):
            response_vector[first_index] += (
                design_row[first_index] * centered_targets[row_index]
            )
            for second_index in range(basis_count):
                gram_matrix[first_index][second_index] += (
                    design_row[first_index] * design_row[second_index]
                )

    for diagonal_index in range(basis_count):
        gram_matrix[diagonal_index][diagonal_index] += alpha

    coefficients = _solve_linear_system(gram_matrix, response_vector)

    coefficients_by_feature: dict[str, dict[str, Any]] = {
        feature_name: {"linear_coefficient": 0.0, "hinge_coefficients": []}
        for feature_name in feature_names
    }
    for coefficient, spec in zip(coefficients, basis_specs, strict=True):
        feature_bucket = coefficients_by_feature[str(spec["feature"])]
        if spec["kind"] == "linear":
            feature_bucket["linear_coefficient"] = round(float(coefficient), 6)
        else:
            feature_bucket["hinge_coefficients"].append(round(float(coefficient), 6))

    model = {
        "intercept": round(target_mean, 6),
        "feature_order": feature_names,
        "feature_terms": {},
        "feature_importance_method": "mean_abs_additive_contribution",
        "feature_importance": {},
        "feature_direction": {},
        "basis_count": basis_count,
    }

    for feature_name in feature_names:
        base_term = dict(feature_terms[feature_name])
        coefficients_bucket = coefficients_by_feature[feature_name]
        base_term["linear_coefficient"] = coefficients_bucket["linear_coefficient"]
        base_term["hinge_coefficients"] = coefficients_bucket["hinge_coefficients"]
        effects = []
        for feature_row in train_matrix:
            standardized_value = _standardized_feature_value(
                base_term,
                _feature_value(
                    feature_row, feature_name, int(base_term["feature_index"])
                ),
            )
            effects.append(_effect_at_standardized_value(base_term, standardized_value))
        mean_abs_effect = _mean([abs(effect) for effect in effects])
        effect_range = max(effects) - min(effects) if effects else 0.0
        effect_profile, direction = _effect_profile(
            term=base_term,
            raw_knots=list(base_term.get("knots_raw") or []),
            training_min=float(base_term["training_min"]),
            training_max=float(base_term["training_max"]),
        )
        base_term["mean_abs_effect"] = round(mean_abs_effect, 6)
        base_term["effect_range"] = round(effect_range, 6)
        base_term["direction"] = direction
        base_term["effect_profile"] = effect_profile
        model["feature_terms"][feature_name] = base_term
        model["feature_importance"][feature_name] = round(mean_abs_effect, 6)
        model["feature_direction"][feature_name] = direction

    return model


def predict_additive_spline_regressor(
    model: dict[str, Any],
    feature_row: dict[str, float] | list[float],
    *,
    with_contributions: bool = False,
) -> float | tuple[float, dict[str, float]]:
    prediction = float(model.get("intercept", 0.0))
    contributions = {
        feature_name: 0.0 for feature_name in model.get("feature_order") or []
    }
    for feature_name in model.get("feature_order") or []:
        term = dict((model.get("feature_terms") or {}).get(feature_name) or {})
        if not term:
            continue
        feature_value = _feature_value(
            feature_row, feature_name, int(term.get("feature_index", 0))
        )
        standardized_value = _standardized_feature_value(term, feature_value)
        effect = _effect_at_standardized_value(term, standardized_value)
        prediction += effect
        contributions[feature_name] = round(effect, 6)
    if with_contributions:
        return prediction, contributions
    return prediction
