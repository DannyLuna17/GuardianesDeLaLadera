from __future__ import annotations

import base64
import json
from functools import lru_cache
from typing import Any

import xgboost as xgb


def _matrix_from_feature_rows(
    feature_order: list[str], feature_rows: list[dict[str, float] | list[float]]
) -> list[list[float]]:
    matrix: list[list[float]] = []
    for feature_row in feature_rows:
        if isinstance(feature_row, dict):
            matrix.append(
                [float(feature_row.get(feature_name, 0.0)) for feature_name in feature_order]
            )
        else:
            matrix.append([float(value) for value in feature_row])
    return matrix


def _dmatrix(
    *,
    feature_order: list[str],
    feature_rows: list[dict[str, float] | list[float]],
    targets: list[float] | None = None,
) -> xgb.DMatrix:
    matrix = _matrix_from_feature_rows(feature_order, feature_rows)
    if targets is None:
        return xgb.DMatrix(matrix, feature_names=feature_order)
    return xgb.DMatrix(matrix, label=targets, feature_names=feature_order)


def _base_score(booster: xgb.Booster) -> float:
    config = json.loads(booster.save_config())
    raw_base_score = config["learner"]["learner_model_param"].get("base_score", 0.5)
    if isinstance(raw_base_score, str) and raw_base_score.startswith("["):
        parsed = json.loads(raw_base_score)
        if isinstance(parsed, list) and parsed:
            raw_base_score = parsed[0]
    return round(float(raw_base_score), 6)


def _iteration_range(model: dict[str, Any]) -> tuple[int, int]:
    effective_tree_count = int(model.get("effective_tree_count") or 0)
    if effective_tree_count > 0:
        return (0, effective_tree_count)
    return (0, 0)


@lru_cache(maxsize=32)
def _load_booster(model_b64: str) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(bytearray(base64.b64decode(model_b64.encode("ascii"))))
    return booster


def train_xgboost_regressor(
    *,
    feature_names: list[str],
    train_matrix: list[list[float]],
    train_targets: list[float],
    validation_matrix: list[list[float]] | None = None,
    validation_targets: list[float] | None = None,
    learning_rate: float = 0.05,
    estimator_count: int = 64,
    max_depth: int = 4,
    min_leaf_size: int = 1,
    min_split_gain: float = 0.0,
    early_stopping_rounds: int = 8,
) -> dict[str, Any]:
    if not train_matrix or not train_targets:
        raise ValueError("XGBoost requires non-empty training data.")

    dtrain = _dmatrix(
        feature_order=feature_names,
        feature_rows=train_matrix,
        targets=train_targets,
    )
    evals = [(dtrain, "train")]
    validation_enabled = validation_matrix is not None and validation_targets is not None
    if validation_enabled:
        dvalidation = _dmatrix(
            feature_order=feature_names,
            feature_rows=validation_matrix or [],
            targets=validation_targets or [],
        )
        evals.append((dvalidation, "validation"))

    evals_result: dict[str, dict[str, list[float]]] = {}
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": learning_rate,
        "max_depth": max_depth,
        "min_child_weight": float(min_leaf_size),
        "gamma": float(min_split_gain),
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "lambda": 1.0,
        "tree_method": "hist",
        "seed": 42,
    }
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=estimator_count,
        evals=evals,
        evals_result=evals_result,
        verbose_eval=False,
        early_stopping_rounds=(
            early_stopping_rounds
            if validation_enabled and early_stopping_rounds > 0
            else None
        ),
    )

    raw_best_iteration = getattr(booster, "best_iteration", None)
    if raw_best_iteration is None:
        best_iteration = estimator_count - 1
    else:
        best_iteration = int(raw_best_iteration)
    effective_tree_count = best_iteration + 1
    model_b64 = base64.b64encode(
        bytes(booster.save_raw(raw_format="json"))
    ).decode("ascii")
    gain_importance = booster.get_score(importance_type="gain")
    feature_importance = {
        feature_name: round(float(gain_importance.get(feature_name, 0.0)), 6)
        for feature_name in feature_names
    }

    train_history = [float(value) for value in (evals_result.get("train") or {}).get("rmse", [])]
    validation_history = [
        float(value) for value in (evals_result.get("validation") or {}).get("rmse", [])
    ]
    training_history = [
        {
            "iteration": iteration,
            "train_rmse": round(train_history[iteration - 1], 6),
            "validation_rmse": (
                round(validation_history[iteration - 1], 6)
                if iteration - 1 < len(validation_history)
                else None
            ),
        }
        for iteration in range(1, len(train_history) + 1)
    ]

    best_validation_rmse = None
    if validation_history and best_iteration < len(validation_history):
        best_validation_rmse = round(validation_history[best_iteration], 6)

    return {
        "feature_order": feature_names,
        "model_b64": model_b64,
        "base_score": _base_score(booster),
        "requested_tree_count": estimator_count,
        "effective_tree_count": effective_tree_count,
        "feature_importance_method": "gain",
        "feature_importance": feature_importance,
        "best_iteration": best_iteration,
        "best_validation_rmse": best_validation_rmse,
        "early_stopping_triggered": effective_tree_count < estimator_count,
        "training_history": training_history,
        "native_params": {
            "eta": round(float(params["eta"]), 6),
            "max_depth": int(params["max_depth"]),
            "min_child_weight": round(float(params["min_child_weight"]), 6),
            "gamma": round(float(params["gamma"]), 6),
            "subsample": round(float(params["subsample"]), 6),
            "colsample_bytree": round(float(params["colsample_bytree"]), 6),
            "lambda": round(float(params["lambda"]), 6),
            "tree_method": str(params["tree_method"]),
        },
    }


def predict_xgboost_model(
    model: dict[str, Any],
    feature_row: dict[str, float] | list[float],
    *,
    with_contributions: bool = False,
) -> float | tuple[float, dict[str, float], float]:
    feature_order = list(model.get("feature_order") or [])
    dmatrix = _dmatrix(feature_order=feature_order, feature_rows=[feature_row])
    booster = _load_booster(str(model["model_b64"]))
    iteration_range = _iteration_range(model)

    prediction = float(
        booster.predict(
            dmatrix,
            validate_features=True,
            iteration_range=iteration_range,
        )[0]
    )
    if not with_contributions:
        return prediction

    contribution_row = booster.predict(
        dmatrix,
        pred_contribs=True,
        validate_features=True,
        iteration_range=iteration_range,
    )[0].tolist()
    feature_contributions = {
        feature_name: round(float(contribution_row[index]), 6)
        for index, feature_name in enumerate(feature_order)
    }
    bias = round(float(contribution_row[-1]), 6)
    return prediction, feature_contributions, bias


def predict_xgboost_rows(
    model: dict[str, Any], feature_rows: list[dict[str, float] | list[float]]
) -> list[float]:
    if not feature_rows:
        return []
    feature_order = list(model.get("feature_order") or [])
    dmatrix = _dmatrix(feature_order=feature_order, feature_rows=feature_rows)
    booster = _load_booster(str(model["model_b64"]))
    iteration_range = _iteration_range(model)
    predictions = booster.predict(
        dmatrix,
        validate_features=True,
        iteration_range=iteration_range,
    )
    return [float(value) for value in predictions.tolist()]
