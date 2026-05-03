"""Candidate artifact helpers for model-selection workflows."""

from __future__ import annotations

from pathlib import Path

from app.ml.training import export_additive_spline_artifact
from app.ml.training import export_beta_regression_artifact
from app.ml.training import export_gradient_boosted_tree_artifact
from app.ml.training import export_seed_linear_artifact
from app.ml.training import export_xgboost_artifact
from app.ml.training import train_additive_spline_artifact
from app.ml.training import train_beta_regression_artifact
from app.ml.training import train_gradient_boosted_tree_artifact
from app.ml.training import train_seed_linear_artifact
from app.ml.training import train_xgboost_artifact
from app.services import model_selection_helpers as selection_helpers


def train_candidate_artifact(
    *,
    model_family: str,
    version: str,
    rows: list,
    dataset_name: str,
    hyperparameters: dict,
    train_rows: list | None = None,
    validation_rows: list | None = None,
) -> dict:
    if model_family == "linear_ridge":
        return train_seed_linear_artifact(
            version=version,
            alpha=float(hyperparameters["alpha"]),
            rows=rows,
            train_rows=train_rows,
            validation_rows=validation_rows,
            dataset_name=dataset_name,
        )
    if model_family == "beta_regression":
        return train_beta_regression_artifact(
            version=version,
            alpha=float(hyperparameters["alpha"]),
            rows=rows,
            train_rows=train_rows,
            validation_rows=validation_rows,
            dataset_name=dataset_name,
        )
    if model_family == "additive_spline":
        return train_additive_spline_artifact(
            version=version,
            rows=rows,
            train_rows=train_rows,
            validation_rows=validation_rows,
            dataset_name=dataset_name,
            alpha=float(hyperparameters["alpha"]),
            knot_count=int(hyperparameters["knot_count"]),
        )
    if model_family == "xgboost":
        return train_xgboost_artifact(
            version=version,
            rows=rows,
            train_rows=train_rows,
            validation_rows=validation_rows,
            dataset_name=dataset_name,
            learning_rate=float(hyperparameters["learning_rate"]),
            estimator_count=int(hyperparameters["estimator_count"]),
            max_depth=int(hyperparameters["max_depth"]),
            min_leaf_size=int(hyperparameters["min_leaf_size"]),
            min_split_gain=float(hyperparameters["min_split_gain"]),
            early_stopping_rounds=int(hyperparameters["early_stopping_rounds"]),
        )
    return train_gradient_boosted_tree_artifact(
        version=version,
        rows=rows,
        train_rows=train_rows,
        validation_rows=validation_rows,
        dataset_name=dataset_name,
        learning_rate=float(hyperparameters["learning_rate"]),
        estimator_count=int(hyperparameters["estimator_count"]),
        max_depth=int(hyperparameters["max_depth"]),
        min_leaf_size=int(hyperparameters["min_leaf_size"]),
        min_split_gain=float(hyperparameters["min_split_gain"]),
        early_stopping_rounds=int(hyperparameters["early_stopping_rounds"]),
    )


def export_candidate_artifact(
    *,
    model_family: str,
    version: str,
    rows: list,
    dataset_name: str,
    hyperparameters: dict,
) -> tuple[Path, dict]:
    if model_family == "linear_ridge":
        return export_seed_linear_artifact(
            version=version,
            alpha=float(hyperparameters["alpha"]),
            rows=rows,
            dataset_name=dataset_name,
        )
    if model_family == "beta_regression":
        return export_beta_regression_artifact(
            version=version,
            alpha=float(hyperparameters["alpha"]),
            rows=rows,
            dataset_name=dataset_name,
        )
    if model_family == "additive_spline":
        return export_additive_spline_artifact(
            version=version,
            rows=rows,
            dataset_name=dataset_name,
            alpha=float(hyperparameters["alpha"]),
            knot_count=int(hyperparameters["knot_count"]),
        )
    if model_family == "xgboost":
        return export_xgboost_artifact(
            version=version,
            rows=rows,
            dataset_name=dataset_name,
            learning_rate=float(hyperparameters["learning_rate"]),
            estimator_count=int(hyperparameters["estimator_count"]),
            max_depth=int(hyperparameters["max_depth"]),
            min_leaf_size=int(hyperparameters["min_leaf_size"]),
            min_split_gain=float(hyperparameters["min_split_gain"]),
            early_stopping_rounds=int(hyperparameters["early_stopping_rounds"]),
        )
    return export_gradient_boosted_tree_artifact(
        version=version,
        rows=rows,
        dataset_name=dataset_name,
        learning_rate=float(hyperparameters["learning_rate"]),
        estimator_count=int(hyperparameters["estimator_count"]),
        max_depth=int(hyperparameters["max_depth"]),
        min_leaf_size=int(hyperparameters["min_leaf_size"]),
        min_split_gain=float(hyperparameters["min_split_gain"]),
        early_stopping_rounds=int(hyperparameters["early_stopping_rounds"]),
    )


def candidate_payload(
    *,
    candidate_version: str,
    model_family: str,
    alpha: float | None,
    artifact_path: Path,
    artifact: dict,
    evaluation: dict,
    validation_summary: dict,
) -> dict:
    overall_probability = selection_helpers.probability_metrics(
        evaluation["metrics"]["overall"]
    )
    validation_metrics = validation_summary["metrics"]
    validation_probability = selection_helpers.probability_metrics(validation_metrics)
    return {
        "rank": 0,
        "model_version": candidate_version,
        "model_family": model_family,
        "alpha": alpha,
        "hyperparameters": dict(artifact.get("training", {}).get("hyperparameters") or {}),
        "artifact_path": str(artifact_path),
        "overall_rmse": evaluation["metrics"]["overall"]["calibrated_metrics"]["rmse"],
        "validation_rmse": validation_metrics["calibrated_metrics"]["rmse"],
        "overall_risk_level_accuracy": evaluation["metrics"]["overall"][
            "risk_level_accuracy"
        ],
        "validation_risk_level_accuracy": validation_metrics["risk_level_accuracy"],
        "overall_brier_score": overall_probability.get("brier_score"),
        "validation_brier_score": validation_probability.get("brier_score"),
        "overall_auroc": overall_probability.get("auroc"),
        "validation_auroc": validation_probability.get("auroc"),
        "overall_auprc": overall_probability.get("auprc"),
        "validation_auprc": validation_probability.get("auprc"),
        "validation_recall": validation_probability.get("recall"),
        "validation_specificity": validation_probability.get("specificity"),
        "validation_mcc": validation_probability.get("mcc"),
        "validation_ece": validation_probability.get("ece"),
        "validation_rows": validation_metrics["rows"],
        "validation_summary": {
            key: value for key, value in validation_summary.items() if key != "metrics"
        },
        "comparison": {},
        "top_errors": list(validation_summary.get("top_errors") or []),
    }
