from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import ApiError

MODEL_SELECTION_ID = "spatial-risk-model-selection-run"


def alpha_slug(alpha: float) -> str:
    normalized = f"{alpha:.6f}".rstrip("0").rstrip(".")
    return normalized.replace("-", "m").replace(".", "p")


def _ascending_rank_metric(value: Any) -> float:
    return float(value) if value is not None else float("inf")


def _descending_rank_metric(value: Any) -> float:
    return -float(value) if value is not None else float("inf")


def candidate_rank_key(
    candidate: dict[str, Any],
) -> tuple[float, float, float, float, float, float, float, float, str]:
    return (
        float(candidate["validation_rmse"]),
        _ascending_rank_metric(candidate.get("validation_brier_score")),
        _descending_rank_metric(candidate.get("validation_auprc")),
        _descending_rank_metric(candidate.get("validation_recall")),
        _descending_rank_metric(candidate.get("validation_mcc")),
        -float(candidate["validation_risk_level_accuracy"]),
        float(candidate["overall_rmse"]),
        _ascending_rank_metric(candidate.get("alpha")),
        str(candidate.get("model_version") or ""),
    )


def build_model_selection_run(
    *,
    version: str,
    dataset_version: str,
    candidates: list[dict[str, Any]],
    promoted: bool,
    active_model_version: str,
    promotion: dict[str, Any] | None = None,
    gate_policy: dict[str, Any] | None = None,
    dataset_context: dict[str, Any] | None = None,
    promotion_decision: dict[str, Any] | None = None,
    best_vs_active_comparison: dict[str, Any] | None = None,
    comparison_policy: dict[str, Any] | None = None,
    family_rollups: list[dict[str, Any]] | None = None,
    nested_estimation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not candidates:
        raise ApiError(
            400,
            "model_selection_empty",
            "The model-selection run did not produce any candidates.",
        )

    ranked_candidates = sorted(candidates, key=candidate_rank_key)
    for index, candidate in enumerate(ranked_candidates, start=1):
        candidate["rank"] = index
    best_candidate = ranked_candidates[0]

    return {
        "selection_id": MODEL_SELECTION_ID,
        "version": version,
        "artifact_type": "model_selection_run",
        "dataset_version": dataset_version,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "comparison_policy": comparison_policy
        or {
            "primary_metric": "validation_rmse",
            "tie_breakers": [
                "validation_brier_score",
                "validation_auprc_desc",
                "validation_recall_desc",
                "validation_mcc_desc",
                "validation_risk_level_accuracy_desc",
                "overall_rmse",
                "alpha",
                "model_version",
            ],
        },
        "gate_policy": gate_policy or {},
        "dataset_context": dataset_context or {},
        "candidate_count": len(ranked_candidates),
        "best_model_version": best_candidate["model_version"],
        "promoted": promoted,
        "promotion": promotion
        or {
            "promoted": False,
            "reason": None,
            "promoted_at": None,
            "promoted_by": None,
            "source": None,
        },
        "promotion_decision": promotion_decision
        or {
            "eligible": promoted,
            "promoted": promoted,
            "blocking_reasons": [],
        },
        "active_model_version": active_model_version,
        "best_vs_active_comparison": best_vs_active_comparison or {},
        "family_rollups": family_rollups or [],
        "nested_estimation": nested_estimation or {},
        "candidates": ranked_candidates,
    }


def export_model_selection_run(
    selection_run: dict[str, Any],
    *,
    selection_runs_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    settings = get_settings()
    target_directory = (
        selection_runs_path or settings.resolved_model_selection_runs_path
    )
    target_directory.mkdir(parents=True, exist_ok=True)
    selection_path = target_directory / f"{selection_run['version']}.json"
    selection_path.write_text(json.dumps(selection_run, indent=2), encoding="utf-8")
    return selection_path, selection_run


class ModelSelectionRunRegistry:
    def __init__(self, selection_runs_path: Path | None = None) -> None:
        settings = get_settings()
        self.selection_runs_path = (
            selection_runs_path or settings.resolved_model_selection_runs_path
        )

    @lru_cache
    def load(self, version: str) -> dict[str, Any]:
        selection_path = self.selection_runs_path / f"{version}.json"
        if not selection_path.exists():
            raise ApiError(
                404,
                "model_selection_run_not_found",
                f"Model selection run '{version}' was not found.",
            )
        with selection_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def list_versions(self) -> list[str]:
        return sorted(path.stem for path in self.selection_runs_path.glob("*.json"))
