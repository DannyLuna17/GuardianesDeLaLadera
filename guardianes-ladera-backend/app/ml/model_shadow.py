from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import ApiError

MODEL_SHADOW_ID = "spatial-risk-model-shadow-run"


def _ascending_rank_metric(value: Any) -> float:
    return float(value) if value is not None else float("inf")


def _descending_rank_metric(value: Any) -> float:
    return -float(value) if value is not None else float("inf")


def shadow_candidate_rank_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, str]:
    return (
        float(candidate["validation_rmse"]),
        _ascending_rank_metric(candidate.get("validation_brier_score")),
        _descending_rank_metric(candidate.get("validation_auprc")),
        _descending_rank_metric(candidate.get("validation_recall")),
        _descending_rank_metric(candidate.get("validation_mcc")),
        -float(candidate["validation_risk_level_accuracy"]),
        float(candidate["overall_rmse"]),
        str(candidate["model_version"]),
    )


def build_model_shadow_run(
    *,
    version: str,
    dataset_version: str,
    dataset_context: dict[str, Any],
    active_model_version: str,
    candidates: list[dict[str, Any]],
    candidate_selection: dict[str, Any] | None = None,
    recommendation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not candidates:
        raise ApiError(
            400,
            "model_shadow_empty",
            "The model shadow run did not produce any candidates.",
        )

    ranked_candidates = sorted(candidates, key=shadow_candidate_rank_key)
    for index, candidate in enumerate(ranked_candidates, start=1):
        candidate["rank"] = index
    best_candidate = ranked_candidates[0]
    active_candidate = next(
        candidate
        for candidate in ranked_candidates
        if candidate["model_version"] == active_model_version
    )
    active_still_best = best_candidate["model_version"] == active_model_version

    return {
        "shadow_id": MODEL_SHADOW_ID,
        "version": version,
        "artifact_type": "model_shadow_run",
        "dataset_version": dataset_version,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset_context": dataset_context or {},
        "comparison_policy": {
            "primary_metric": "validation_rmse",
            "tie_breakers": [
                "validation_brier_score",
                "validation_auprc_desc",
                "validation_recall_desc",
                "validation_mcc_desc",
                "validation_risk_level_accuracy_desc",
                "overall_rmse",
                "model_version",
            ],
        },
        "active_model_version": active_model_version,
        "active_still_best": active_still_best,
        "best_model_version": best_candidate["model_version"],
        "active_candidate_rank": active_candidate["rank"],
        "candidate_count": len(ranked_candidates),
        "candidate_selection": candidate_selection or {},
        "recommendation": recommendation
        or {
            "status": "active_holds" if active_still_best else "review_challenger",
            "reason": "Active model remains best on the evaluated labeled cohort."
            if active_still_best
            else "A challenger outperformed the active model on the evaluated labeled cohort.",
        },
        "candidates": ranked_candidates,
    }


def export_model_shadow_run(
    shadow_run: dict[str, Any],
    *,
    shadow_runs_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    settings = get_settings()
    target_directory = shadow_runs_path or settings.resolved_model_shadow_runs_path
    target_directory.mkdir(parents=True, exist_ok=True)
    shadow_path = target_directory / f"{shadow_run['version']}.json"
    shadow_path.write_text(json.dumps(shadow_run, indent=2), encoding="utf-8")
    return shadow_path, shadow_run


class ModelShadowRunRegistry:
    def __init__(self, shadow_runs_path: Path | None = None) -> None:
        settings = get_settings()
        self.shadow_runs_path = shadow_runs_path or settings.resolved_model_shadow_runs_path

    @lru_cache
    def load(self, version: str) -> dict[str, Any]:
        shadow_path = self.shadow_runs_path / f"{version}.json"
        if not shadow_path.exists():
            raise ApiError(
                404,
                "model_shadow_run_not_found",
                f"Model shadow run '{version}' was not found.",
            )
        with shadow_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def list_versions(self) -> list[str]:
        return sorted(path.stem for path in self.shadow_runs_path.glob("*.json"))
