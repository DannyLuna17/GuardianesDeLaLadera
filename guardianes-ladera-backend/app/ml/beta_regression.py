from __future__ import annotations

from math import exp, lgamma, log, sqrt
from typing import Any


def _clip_probability(value: float, *, epsilon: float = 1e-5) -> float:
    return min(max(float(value), epsilon), 1.0 - epsilon)


def _sigmoid(value: float) -> float:
    if value >= 0:
        scale = exp(-value)
        return 1.0 / (1.0 + scale)
    scale = exp(value)
    return scale / (1.0 + scale)


def _logit(value: float) -> float:
    bounded = _clip_probability(value)
    return log(bounded / (1.0 - bounded))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _standard_deviation(values: list[float], mean_value: float) -> float:
    if not values:
        return 1.0
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    deviation = variance**0.5
    return deviation if deviation > 1e-9 else 1.0


def _digamma(value: float) -> float:
    if value <= 0:
        raise ValueError("digamma requires a positive input.")
    result = 0.0
    x = float(value)
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    inverse = 1.0 / x
    inverse_squared = inverse * inverse
    return (
        result
        + log(x)
        - 0.5 * inverse
        - inverse_squared
        * (
            (1.0 / 12.0)
            - inverse_squared * ((1.0 / 120.0) - inverse_squared * (1.0 / 252.0))
        )
    )


def _feature_value(
    feature_row: dict[str, float] | list[float], feature_name: str, feature_index: int
) -> float:
    if isinstance(feature_row, dict):
        return float(feature_row.get(feature_name, 0.0))
    return float(feature_row[feature_index])


def _standardize_matrix(
    feature_names: list[str], matrix: list[list[float]]
) -> tuple[dict[str, float], dict[str, float], list[list[float]]]:
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    standardized_matrix: list[list[float]] = []
    for feature_index, feature_name in enumerate(feature_names):
        values = [float(row[feature_index]) for row in matrix]
        means[feature_name] = _mean(values)
        scales[feature_name] = _standard_deviation(values, means[feature_name])
    for row in matrix:
        standardized_matrix.append(
            [
                (float(row[feature_index]) - means[feature_name])
                / scales[feature_name]
                for feature_index, feature_name in enumerate(feature_names)
            ]
        )
    return means, scales, standardized_matrix


def _estimate_precision(
    targets: list[float], predicted_means: list[float] | None = None
) -> float:
    clipped_targets = [_clip_probability(value) for value in targets]
    if not clipped_targets:
        return 25.0

    if predicted_means is None:
        target_mean = _mean(clipped_targets)
        variance = _mean(
            [(target - target_mean) ** 2 for target in clipped_targets]
        )
        if variance <= 1e-6:
            return 25.0
        estimate = (target_mean * (1.0 - target_mean) / variance) - 1.0
        return round(min(max(estimate, 2.0), 200.0), 6)

    residual_ratios: list[float] = []
    for target, mean_value in zip(clipped_targets, predicted_means, strict=True):
        bounded_mean = _clip_probability(mean_value)
        denominator = bounded_mean * (1.0 - bounded_mean)
        if denominator <= 1e-9:
            continue
        residual_ratios.append(((target - bounded_mean) ** 2) / denominator)
    if not residual_ratios:
        return 25.0
    average_ratio = _mean(residual_ratios)
    if average_ratio <= 1e-9:
        return 200.0
    estimate = (1.0 / average_ratio) - 1.0
    return round(min(max(estimate, 2.0), 200.0), 6)


def _objective_and_gradient(
    *,
    intercept: float,
    coefficients: list[float],
    standardized_matrix: list[list[float]],
    targets: list[float],
    alpha: float,
    precision: float,
) -> tuple[float, float, list[float]]:
    row_count = len(targets)
    coefficient_count = len(coefficients)
    penalized_log_likelihood = 0.0
    intercept_gradient = 0.0
    coefficient_gradient = [0.0 for _ in range(coefficient_count)]

    for row, target in zip(standardized_matrix, targets, strict=True):
        eta = intercept + sum(
            weight * value for weight, value in zip(coefficients, row, strict=True)
        )
        mean_value = _clip_probability(_sigmoid(eta))
        target_value = _clip_probability(target)
        mean_precision = mean_value * precision
        complement_precision = (1.0 - mean_value) * precision
        y_star = _logit(target_value)
        mean_star = _digamma(mean_precision) - _digamma(complement_precision)
        score_core = (
            precision * mean_value * (1.0 - mean_value) * (y_star - mean_star)
        )
        intercept_gradient += score_core
        for feature_index in range(coefficient_count):
            coefficient_gradient[feature_index] += score_core * row[feature_index]
        penalized_log_likelihood += (
            lgamma(precision)
            - lgamma(mean_precision)
            - lgamma(complement_precision)
            + ((mean_precision - 1.0) * log(target_value))
            + ((complement_precision - 1.0) * log(1.0 - target_value))
        )

    penalty = 0.5 * alpha * sum(value * value for value in coefficients)
    penalized_log_likelihood -= penalty
    for feature_index in range(coefficient_count):
        coefficient_gradient[feature_index] -= alpha * coefficients[feature_index]

    scale = 1.0 / max(row_count, 1)
    return (
        penalized_log_likelihood * scale,
        intercept_gradient * scale,
        [gradient * scale for gradient in coefficient_gradient],
    )


def _predict_standardized_rows(
    *,
    intercept: float,
    coefficients: list[float],
    standardized_matrix: list[list[float]],
) -> list[float]:
    predictions: list[float] = []
    for row in standardized_matrix:
        eta = intercept + sum(
            weight * value for weight, value in zip(coefficients, row, strict=True)
        )
        predictions.append(_sigmoid(eta))
    return predictions


def train_beta_regression_regressor(
    *,
    feature_names: list[str],
    train_matrix: list[list[float]],
    train_targets: list[float],
    alpha: float = 0.75,
    max_iterations: int = 250,
    learning_rate: float = 0.2,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    if not train_matrix or not train_targets:
        raise ValueError("Beta regression requires non-empty training data.")
    if alpha <= 0:
        raise ValueError("Beta regression alpha must be greater than zero.")
    if max_iterations <= 0:
        raise ValueError("Beta regression max_iterations must be greater than zero.")
    if learning_rate <= 0:
        raise ValueError("Beta regression learning_rate must be greater than zero.")
    if tolerance <= 0:
        raise ValueError("Beta regression tolerance must be greater than zero.")

    feature_means, feature_scales, standardized_matrix = _standardize_matrix(
        feature_names, train_matrix
    )
    clipped_targets = [_clip_probability(value) for value in train_targets]
    intercept = _logit(_mean(clipped_targets))
    coefficients = [0.0 for _ in feature_names]
    precision = _estimate_precision(clipped_targets)
    history: list[dict[str, Any]] = []
    converged = False
    total_iterations = 0

    for precision_round in range(2):
        for _ in range(max_iterations):
            total_iterations += 1
            objective, intercept_gradient, coefficient_gradient = (
                _objective_and_gradient(
                    intercept=intercept,
                    coefficients=coefficients,
                    standardized_matrix=standardized_matrix,
                    targets=clipped_targets,
                    alpha=alpha,
                    precision=precision,
                )
            )
            gradient_norm = sqrt(
                (intercept_gradient**2)
                + sum(gradient * gradient for gradient in coefficient_gradient)
            )
            if gradient_norm <= tolerance:
                history.append(
                    {
                        "iteration": total_iterations,
                        "precision_round": precision_round + 1,
                        "objective": round(objective, 6),
                        "gradient_norm": round(gradient_norm, 6),
                        "step_size": 0.0,
                        "precision": round(precision, 6),
                        "accepted": False,
                    }
                )
                converged = True
                break

            step_size = learning_rate
            accepted = False
            while step_size >= 1e-6:
                candidate_intercept = intercept + (step_size * intercept_gradient)
                candidate_coefficients = [
                    coefficient + (step_size * gradient)
                    for coefficient, gradient in zip(
                        coefficients, coefficient_gradient, strict=True
                    )
                ]
                candidate_objective, _, _ = _objective_and_gradient(
                    intercept=candidate_intercept,
                    coefficients=candidate_coefficients,
                    standardized_matrix=standardized_matrix,
                    targets=clipped_targets,
                    alpha=alpha,
                    precision=precision,
                )
                if candidate_objective >= objective:
                    intercept = candidate_intercept
                    coefficients = candidate_coefficients
                    accepted = True
                    history.append(
                        {
                            "iteration": total_iterations,
                            "precision_round": precision_round + 1,
                            "objective": round(candidate_objective, 6),
                            "gradient_norm": round(gradient_norm, 6),
                            "step_size": round(step_size, 6),
                            "precision": round(precision, 6),
                            "accepted": True,
                        }
                    )
                    learning_rate = min(step_size * 1.1, 1.0)
                    break
                step_size *= 0.5

            if not accepted:
                history.append(
                    {
                        "iteration": total_iterations,
                        "precision_round": precision_round + 1,
                        "objective": round(objective, 6),
                        "gradient_norm": round(gradient_norm, 6),
                        "step_size": 0.0,
                        "precision": round(precision, 6),
                        "accepted": False,
                    }
                )
                break

        train_predictions = _predict_standardized_rows(
            intercept=intercept,
            coefficients=coefficients,
            standardized_matrix=standardized_matrix,
        )
        refined_precision = _estimate_precision(
            clipped_targets, predicted_means=train_predictions
        )
        if precision_round == 1 or abs(refined_precision - precision) <= 1e-3:
            precision = refined_precision
            break
        precision = refined_precision

    feature_importance = {
        feature_name: round(abs(coefficients[feature_index]), 6)
        for feature_index, feature_name in enumerate(feature_names)
    }
    feature_direction = {
        feature_name: (
            "positive"
            if coefficients[feature_index] > 0
            else "negative"
            if coefficients[feature_index] < 0
            else "neutral"
        )
        for feature_index, feature_name in enumerate(feature_names)
    }
    return {
        "feature_order": feature_names,
        "feature_means": {
            feature_name: round(feature_means[feature_name], 6)
            for feature_name in feature_names
        },
        "feature_scales": {
            feature_name: round(feature_scales[feature_name], 6)
            for feature_name in feature_names
        },
        "coefficients": {
            feature_name: round(coefficients[feature_index], 6)
            for feature_index, feature_name in enumerate(feature_names)
        },
        "intercept": round(intercept, 6),
        "precision": round(precision, 6),
        "link_function": "logit",
        "response_distribution": "beta",
        "feature_importance_method": "absolute_logit_coefficient",
        "feature_importance": feature_importance,
        "feature_direction": feature_direction,
        "max_iterations": max_iterations,
        "iterations": total_iterations,
        "converged": converged,
        "optimization_history": history,
        "component_score_space": "logit",
    }


def predict_beta_regression_model(
    model: dict[str, Any],
    feature_row: dict[str, float] | list[float],
    *,
    with_contributions: bool = False,
) -> float | tuple[float, dict[str, float], float]:
    feature_order = list(model.get("feature_order") or [])
    coefficients = model.get("coefficients") or {}
    feature_means = model.get("feature_means") or {}
    feature_scales = model.get("feature_scales") or {}
    logit_intercept = float(model.get("intercept", 0.0))
    contributions: dict[str, float] = {}
    linear_score = logit_intercept
    for feature_index, feature_name in enumerate(feature_order):
        raw_value = _feature_value(feature_row, feature_name, feature_index)
        standardized_value = (
            raw_value - float(feature_means.get(feature_name, 0.0))
        ) / float(feature_scales.get(feature_name, 1.0))
        contribution = standardized_value * float(coefficients.get(feature_name, 0.0))
        contributions[feature_name] = round(contribution, 6)
        linear_score += contribution
    prediction = _sigmoid(linear_score)
    if not with_contributions:
        return float(prediction)
    return float(prediction), contributions, round(logit_intercept, 6)


def predict_beta_regression_rows(
    model: dict[str, Any], feature_rows: list[dict[str, float] | list[float]]
) -> list[float]:
    return [
        float(predict_beta_regression_model(model, feature_row))
        for feature_row in feature_rows
    ]
