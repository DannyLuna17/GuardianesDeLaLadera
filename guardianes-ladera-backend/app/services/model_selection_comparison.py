"""Comparison helpers for model-selection workflows."""

from __future__ import annotations

from app.ml.model_diagnostics import (
    calibration_delta_summary,
    coefficient_drift_summary,
    feature_importance_change_summary,
)
from app.services import model_selection_helpers as selection_helpers


def slice_delta_summary(
    active_slices: dict, challenger_slices: dict
) -> dict[str, dict]:
    slice_keys = sorted(set(active_slices) | set(challenger_slices))
    comparison: dict[str, dict] = {}
    for slice_key in slice_keys:
        active_slice = active_slices.get(slice_key) or {}
        challenger_slice = challenger_slices.get(slice_key) or {}
        active_rmse = (
            (active_slice.get("calibrated_metrics") or {}).get("rmse")
            if active_slice
            else None
        )
        challenger_rmse = (
            (challenger_slice.get("calibrated_metrics") or {}).get("rmse")
            if challenger_slice
            else None
        )
        active_accuracy = active_slice.get("risk_level_accuracy")
        challenger_accuracy = challenger_slice.get("risk_level_accuracy")
        active_probability = selection_helpers.probability_metrics(active_slice)
        challenger_probability = selection_helpers.probability_metrics(challenger_slice)

        rmse_delta = None
        rmse_improvement = None
        if active_rmse is not None and challenger_rmse is not None:
            rmse_delta = round(float(challenger_rmse) - float(active_rmse), 6)
            rmse_improvement = round(float(active_rmse) - float(challenger_rmse), 6)

        accuracy_delta = None
        if active_accuracy is not None and challenger_accuracy is not None:
            accuracy_delta = round(
                float(challenger_accuracy) - float(active_accuracy), 6
            )

        comparison[slice_key] = {
            "active_rows": int(active_slice.get("rows", 0)),
            "challenger_rows": int(challenger_slice.get("rows", 0)),
            "active_validation_rmse": active_rmse,
            "challenger_validation_rmse": challenger_rmse,
            "validation_rmse_delta": rmse_delta,
            "validation_rmse_improvement": rmse_improvement,
            "active_risk_level_accuracy": active_accuracy,
            "challenger_risk_level_accuracy": challenger_accuracy,
            "risk_level_accuracy_delta": accuracy_delta,
            "active_validation_brier_score": active_probability.get("brier_score"),
            "challenger_validation_brier_score": challenger_probability.get(
                "brier_score"
            ),
            "validation_brier_score_delta": selection_helpers.metric_delta(
                active_probability.get("brier_score"),
                challenger_probability.get("brier_score"),
            ),
            "validation_brier_score_improvement": selection_helpers.lower_is_better_improvement(
                active_probability.get("brier_score"),
                challenger_probability.get("brier_score"),
            ),
            "active_validation_auroc": active_probability.get("auroc"),
            "challenger_validation_auroc": challenger_probability.get("auroc"),
            "validation_auroc_delta": selection_helpers.higher_is_better_gain(
                active_probability.get("auroc"),
                challenger_probability.get("auroc"),
            ),
            "active_validation_auprc": active_probability.get("auprc"),
            "challenger_validation_auprc": challenger_probability.get("auprc"),
            "validation_auprc_delta": selection_helpers.higher_is_better_gain(
                active_probability.get("auprc"),
                challenger_probability.get("auprc"),
            ),
            "active_validation_recall": active_probability.get("recall"),
            "challenger_validation_recall": challenger_probability.get("recall"),
            "validation_recall_delta": selection_helpers.higher_is_better_gain(
                active_probability.get("recall"),
                challenger_probability.get("recall"),
            ),
            "active_validation_specificity": active_probability.get("specificity"),
            "challenger_validation_specificity": challenger_probability.get(
                "specificity"
            ),
            "validation_specificity_delta": selection_helpers.higher_is_better_gain(
                active_probability.get("specificity"),
                challenger_probability.get("specificity"),
            ),
            "active_validation_mcc": active_probability.get("mcc"),
            "challenger_validation_mcc": challenger_probability.get("mcc"),
            "validation_mcc_delta": selection_helpers.higher_is_better_gain(
                active_probability.get("mcc"),
                challenger_probability.get("mcc"),
            ),
            "active_validation_ece": active_probability.get("ece"),
            "challenger_validation_ece": challenger_probability.get("ece"),
            "validation_ece_delta": selection_helpers.metric_delta(
                active_probability.get("ece"),
                challenger_probability.get("ece"),
            ),
            "validation_ece_improvement": selection_helpers.lower_is_better_improvement(
                active_probability.get("ece"),
                challenger_probability.get("ece"),
            ),
        }
    return comparison


def slice_gate_summary(
    slice_deltas: dict[str, dict],
    *,
    min_rows: int,
) -> dict:
    considered_slices: list[dict] = []
    skipped_slices: list[dict] = []
    for slice_key, metrics in sorted(slice_deltas.items()):
        active_rows = int(metrics.get("active_rows", 0) or 0)
        challenger_rows = int(metrics.get("challenger_rows", 0) or 0)
        entry = {
            "slice_key": slice_key,
            "active_rows": active_rows,
            "challenger_rows": challenger_rows,
            "validation_rmse_regression": metrics.get("validation_rmse_delta"),
            "validation_rmse_improvement": metrics.get(
                "validation_rmse_improvement"
            ),
        }
        if active_rows < min_rows or challenger_rows < min_rows:
            skipped_slices.append(entry)
            continue
        considered_slices.append(entry)

    regression_slices = [
        entry
        for entry in considered_slices
        if (entry.get("validation_rmse_regression") or 0.0) > 0
    ]
    regression_slices = sorted(
        regression_slices,
        key=lambda item: (
            -float(item.get("validation_rmse_regression") or 0.0),
            item["slice_key"],
        ),
    )
    worst_regression = (
        regression_slices[0].get("validation_rmse_regression")
        if regression_slices
        else 0.0
    )
    return {
        "available": bool(considered_slices),
        "min_rows": min_rows,
        "considered_slice_count": len(considered_slices),
        "skipped_slice_count": len(skipped_slices),
        "regression_count": len(regression_slices),
        "non_regression_count": len(considered_slices) - len(regression_slices),
        "worst_validation_rmse_regression": (
            round(float(worst_regression), 6)
            if worst_regression is not None
            else None
        ),
        "worst_regression_slice_key": (
            regression_slices[0]["slice_key"] if regression_slices else None
        ),
        "regression_slices": regression_slices,
        "skipped_slices": skipped_slices,
    }


def best_vs_active_comparison(
    *,
    active_artifact: dict,
    challenger_artifact: dict,
    active_evaluation: dict,
    challenger_evaluation: dict,
    active_validation_summary: dict | None = None,
    challenger_validation_summary: dict | None = None,
) -> dict:
    active_validation = (
        (active_validation_summary or {}).get("metrics")
        or active_evaluation["metrics"]["validation"]
    )
    challenger_validation = (
        (challenger_validation_summary or {}).get("metrics")
        or challenger_evaluation["metrics"]["validation"]
    )
    active_overall = active_evaluation["metrics"]["overall"]
    challenger_overall = challenger_evaluation["metrics"]["overall"]
    active_validation_probability = selection_helpers.probability_metrics(
        active_validation
    )
    challenger_validation_probability = selection_helpers.probability_metrics(
        challenger_validation
    )
    active_overall_probability = selection_helpers.probability_metrics(active_overall)
    challenger_overall_probability = selection_helpers.probability_metrics(
        challenger_overall
    )
    active_slices = (
        (active_validation_summary or {}).get("validation_slices")
        or (active_evaluation.get("diagnostics") or {}).get("validation_slices")
        or {}
    )
    challenger_slices = (
        (challenger_validation_summary or {}).get("validation_slices")
        or (challenger_evaluation.get("diagnostics") or {}).get("validation_slices")
        or {}
    )

    return {
        "active_model_version": active_evaluation["model_version"],
        "active_model_family": active_artifact.get("model_family"),
        "challenger_model_version": challenger_evaluation["model_version"],
        "challenger_model_family": challenger_artifact.get("model_family"),
        "validation_strategy": (
            (challenger_validation_summary or {}).get("strategy")
            or (active_validation_summary or {}).get("strategy")
            or "dataset_holdout"
        ),
        "validation_unit": (
            (challenger_validation_summary or {}).get("validation_unit")
            or (active_validation_summary or {}).get("validation_unit")
        ),
        "challenger_hyperparameters": (
            (challenger_artifact.get("training") or {}).get("hyperparameters") or {}
        ),
        "validation_rmse_delta": round(
            challenger_validation["calibrated_metrics"]["rmse"]
            - active_validation["calibrated_metrics"]["rmse"],
            6,
        ),
        "validation_rmse_improvement": round(
            active_validation["calibrated_metrics"]["rmse"]
            - challenger_validation["calibrated_metrics"]["rmse"],
            6,
        ),
        "validation_risk_level_accuracy_delta": round(
            challenger_validation["risk_level_accuracy"]
            - active_validation["risk_level_accuracy"],
            6,
        ),
        "overall_rmse_delta": round(
            challenger_overall["calibrated_metrics"]["rmse"]
            - active_overall["calibrated_metrics"]["rmse"],
            6,
        ),
        "overall_risk_level_accuracy_delta": round(
            challenger_overall["risk_level_accuracy"]
            - active_overall["risk_level_accuracy"],
            6,
        ),
        "active_validation_brier_score": active_validation_probability.get(
            "brier_score"
        ),
        "challenger_validation_brier_score": challenger_validation_probability.get(
            "brier_score"
        ),
        "validation_brier_score_delta": selection_helpers.metric_delta(
            active_validation_probability.get("brier_score"),
            challenger_validation_probability.get("brier_score"),
        ),
        "validation_brier_score_improvement": selection_helpers.lower_is_better_improvement(
            active_validation_probability.get("brier_score"),
            challenger_validation_probability.get("brier_score"),
        ),
        "active_validation_auroc": active_validation_probability.get("auroc"),
        "challenger_validation_auroc": challenger_validation_probability.get("auroc"),
        "validation_auroc_delta": selection_helpers.higher_is_better_gain(
            active_validation_probability.get("auroc"),
            challenger_validation_probability.get("auroc"),
        ),
        "active_validation_auprc": active_validation_probability.get("auprc"),
        "challenger_validation_auprc": challenger_validation_probability.get("auprc"),
        "validation_auprc_delta": selection_helpers.higher_is_better_gain(
            active_validation_probability.get("auprc"),
            challenger_validation_probability.get("auprc"),
        ),
        "active_validation_recall": active_validation_probability.get("recall"),
        "challenger_validation_recall": challenger_validation_probability.get("recall"),
        "validation_recall_delta": selection_helpers.higher_is_better_gain(
            active_validation_probability.get("recall"),
            challenger_validation_probability.get("recall"),
        ),
        "active_validation_specificity": active_validation_probability.get(
            "specificity"
        ),
        "challenger_validation_specificity": challenger_validation_probability.get(
            "specificity"
        ),
        "validation_specificity_delta": selection_helpers.higher_is_better_gain(
            active_validation_probability.get("specificity"),
            challenger_validation_probability.get("specificity"),
        ),
        "active_validation_mcc": active_validation_probability.get("mcc"),
        "challenger_validation_mcc": challenger_validation_probability.get("mcc"),
        "validation_mcc_delta": selection_helpers.higher_is_better_gain(
            active_validation_probability.get("mcc"),
            challenger_validation_probability.get("mcc"),
        ),
        "active_validation_ece": active_validation_probability.get("ece"),
        "challenger_validation_ece": challenger_validation_probability.get("ece"),
        "validation_ece_delta": selection_helpers.metric_delta(
            active_validation_probability.get("ece"),
            challenger_validation_probability.get("ece"),
        ),
        "validation_ece_improvement": selection_helpers.lower_is_better_improvement(
            active_validation_probability.get("ece"),
            challenger_validation_probability.get("ece"),
        ),
        "overall_brier_score_delta": selection_helpers.metric_delta(
            active_overall_probability.get("brier_score"),
            challenger_overall_probability.get("brier_score"),
        ),
        "overall_auroc_delta": selection_helpers.higher_is_better_gain(
            active_overall_probability.get("auroc"),
            challenger_overall_probability.get("auroc"),
        ),
        "overall_auprc_delta": selection_helpers.higher_is_better_gain(
            active_overall_probability.get("auprc"),
            challenger_overall_probability.get("auprc"),
        ),
        "coefficient_drift": coefficient_drift_summary(
            active_artifact, challenger_artifact
        ),
        "feature_importance_change": feature_importance_change_summary(
            active_artifact, challenger_artifact
        ),
        "calibration_delta": calibration_delta_summary(
            active_artifact, challenger_artifact
        ),
        "validation_slice_deltas": {
            "by_phase": slice_delta_summary(
                active_slices.get("by_phase") or {},
                challenger_slices.get("by_phase") or {},
            ),
            "by_target_risk_level": slice_delta_summary(
                active_slices.get("by_target_risk_level") or {},
                challenger_slices.get("by_target_risk_level") or {},
            ),
            "by_spatial_block": slice_delta_summary(
                active_slices.get("by_spatial_block") or {},
                challenger_slices.get("by_spatial_block") or {},
            ),
            "by_temporal_holdout_tag": slice_delta_summary(
                active_slices.get("by_temporal_holdout_tag") or {},
                challenger_slices.get("by_temporal_holdout_tag") or {},
            ),
        },
        "validation_slice_summary": {
            "spatial_block": slice_gate_summary(
                slice_delta_summary(
                    active_slices.get("by_spatial_block") or {},
                    challenger_slices.get("by_spatial_block") or {},
                ),
                min_rows=1,
            ),
            "temporal_holdout_tag": slice_gate_summary(
                slice_delta_summary(
                    active_slices.get("by_temporal_holdout_tag") or {},
                    challenger_slices.get("by_temporal_holdout_tag") or {},
                ),
                min_rows=1,
            ),
        },
    }
