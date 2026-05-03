from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import ApiError
from app.ml.datasets import normalize_dataset_context

MODEL_DRIFT_ID = "spatial-risk-model-drift-report"


def _severity_rank(severity: str) -> int:
    ranking = {
        "unavailable": 0,
        "stable": 1,
        "warning": 2,
        "critical": 3,
    }
    return ranking.get(str(severity or "unavailable"), 0)


def build_model_drift_report(
    *,
    version: str,
    evaluation: dict[str, Any],
    baseline: dict[str, Any],
    warning_validation_rmse_increase: float,
    critical_validation_rmse_increase: float,
    warning_accuracy_drop: float,
    critical_accuracy_drop: float,
) -> dict[str, Any]:
    current_validation = ((evaluation.get("metrics") or {}).get("validation") or {})
    current_rmse = ((current_validation.get("calibrated_metrics") or {}).get("rmse"))
    current_accuracy = current_validation.get("risk_level_accuracy")
    current_rows = int(current_validation.get("rows", 0))

    baseline_validation_rmse = baseline.get("validation_rmse")
    baseline_accuracy = baseline.get("validation_risk_level_accuracy")
    baseline_rows = baseline.get("validation_rows")

    rmse_delta = None
    rmse_regression = None
    if current_rmse is not None and baseline_validation_rmse is not None:
        rmse_delta = round(float(current_rmse) - float(baseline_validation_rmse), 6)
        rmse_regression = rmse_delta

    accuracy_delta = None
    accuracy_drop = None
    if current_accuracy is not None and baseline_accuracy is not None:
        accuracy_delta = round(float(current_accuracy) - float(baseline_accuracy), 6)
        accuracy_drop = round(float(baseline_accuracy) - float(current_accuracy), 6)

    signals: list[dict[str, Any]] = []
    if rmse_regression is not None:
        severity = "stable"
        if rmse_regression >= critical_validation_rmse_increase:
            severity = "critical"
        elif rmse_regression >= warning_validation_rmse_increase:
            severity = "warning"
        signals.append(
            {
                "metric": "validation_rmse",
                "baseline": baseline_validation_rmse,
                "current": current_rmse,
                "delta": rmse_delta,
                "thresholds": {
                    "warning": warning_validation_rmse_increase,
                    "critical": critical_validation_rmse_increase,
                },
                "severity": severity,
            }
        )

    if accuracy_drop is not None:
        severity = "stable"
        if accuracy_drop >= critical_accuracy_drop:
            severity = "critical"
        elif accuracy_drop >= warning_accuracy_drop:
            severity = "warning"
        signals.append(
            {
                "metric": "validation_risk_level_accuracy",
                "baseline": baseline_accuracy,
                "current": current_accuracy,
                "delta": accuracy_delta,
                "drop": accuracy_drop,
                "thresholds": {
                    "warning": warning_accuracy_drop,
                    "critical": critical_accuracy_drop,
                },
                "severity": severity,
            }
        )

    overall_severity = "unavailable"
    if signals:
        overall_severity = max(
            (signal["severity"] for signal in signals),
            key=_severity_rank,
        )

    baseline_context = normalize_dataset_context(baseline.get("dataset_context") or {})
    current_context = normalize_dataset_context(
        ((evaluation.get("dataset_summary") or {}).get("provenance") or {})
    )
    cohort_alignment = {
        "same_taxonomy_group": (
            ((baseline_context.get("dataset_taxonomy") or {}).get("taxonomy_group"))
            == ((current_context.get("dataset_taxonomy") or {}).get("taxonomy_group"))
        ),
        "same_cohort_group": (
            ((baseline_context.get("evaluation_cohort") or {}).get("cohort_group"))
            == ((current_context.get("evaluation_cohort") or {}).get("cohort_group"))
        ),
        "baseline_cohort": baseline_context.get("evaluation_cohort") or {},
        "current_cohort": current_context.get("evaluation_cohort") or {},
    }

    return {
        "drift_id": MODEL_DRIFT_ID,
        "version": version,
        "artifact_type": "model_drift_report",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model_version": evaluation["model_version"],
        "dataset_version": evaluation["dataset_version"],
        "evaluation_version": evaluation["version"],
        "baseline": baseline,
        "current": {
            "evaluation_version": evaluation["version"],
            "dataset_version": evaluation["dataset_version"],
            "dataset_context": current_context,
            "validation_rmse": current_rmse,
            "validation_risk_level_accuracy": current_accuracy,
            "validation_rows": current_rows,
        },
        "drift_summary": {
            "severity": overall_severity,
            "drift_detected": overall_severity in {"warning", "critical"},
            "validation_rmse_delta": rmse_delta,
            "validation_rmse_regression": rmse_regression,
            "validation_risk_level_accuracy_delta": accuracy_delta,
            "validation_risk_level_accuracy_drop": accuracy_drop,
            "signals": signals,
            "cohort_alignment": cohort_alignment,
            "rows": current_rows,
        },
        "diagnostics": {
            "dataset_summary": evaluation.get("dataset_summary") or {},
            "model_summary": evaluation.get("model_summary") or {},
            "evaluation_diagnostics": evaluation.get("diagnostics") or {},
        },
        "top_errors": evaluation.get("top_errors") or [],
    }


def export_model_drift_report(
    report: dict[str, Any],
    *,
    drift_reports_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    settings = get_settings()
    target_directory = drift_reports_path or settings.resolved_model_drift_reports_path
    target_directory.mkdir(parents=True, exist_ok=True)
    report_path = target_directory / f"{report['version']}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path, report


class ModelDriftReportRegistry:
    def __init__(self, drift_reports_path: Path | None = None) -> None:
        settings = get_settings()
        self.drift_reports_path = (
            drift_reports_path or settings.resolved_model_drift_reports_path
        )

    @lru_cache
    def load(self, version: str) -> dict[str, Any]:
        report_path = self.drift_reports_path / f"{version}.json"
        if not report_path.exists():
            raise ApiError(
                404,
                "model_drift_report_not_found",
                f"Model drift report '{version}' was not found.",
            )
        with report_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def list_versions(self) -> list[str]:
        return sorted(path.stem for path in self.drift_reports_path.glob("*.json"))
