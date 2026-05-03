"""Stability-window helpers for model-selection workflows."""

from __future__ import annotations

from datetime import datetime, timezone

from app.ml.datasets import normalize_dataset_context
from app.services import model_selection_helpers as selection_helpers


def recent_selection_runs_for_mode(
    *,
    selection_registry,
    dataset_mode: str,
    exclude_version: str | None = None,
) -> list[dict]:
    runs: list[dict] = []
    for version in selection_registry.list_versions():
        if exclude_version and version == exclude_version:
            continue
        run = selection_registry.load(version)
        run_dataset_mode = str(
            ((run.get("gate_policy") or {}).get("dataset_mode")) or "unknown"
        )
        if run_dataset_mode != dataset_mode:
            continue
        runs.append(run)
    runs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return runs


def stability_assessment(
    *,
    best_candidate: dict,
    historical_runs: list[dict],
    dataset_context: dict,
    selection_version: str,
    stability_window_runs: int,
    required_consistent_wins: int,
    require_same_dataset_family: bool,
    require_same_dataset_taxonomy: bool,
    require_same_evaluation_cohort: bool,
    max_time_window_gap_days: int,
    max_cohort_distance: int,
    current_dataset_version: str,
) -> dict:
    candidate_signature = selection_helpers.candidate_signature(best_candidate)
    current_taxonomy_group = selection_helpers.taxonomy_group(dataset_context)
    current_cohort = dict(dataset_context.get("evaluation_cohort") or {})
    considered_history: list[dict] = []
    excluded_runs: list[dict] = []
    for run in historical_runs:
        run_context = normalize_dataset_context(run.get("dataset_context") or {})
        run_taxonomy_group = selection_helpers.taxonomy_group(run_context)
        run_cohort = dict(run_context.get("evaluation_cohort") or {})
        exclusion_reason = None
        time_window_gap_days = selection_helpers.time_window_gap_days(
            dataset_context.get("time_window") or {},
            run_context.get("time_window") or {},
        )
        cohort_distance = selection_helpers.cohort_distance(
            current_cohort, run_cohort
        )
        if require_same_dataset_family and (
            run_context.get("dataset_family") != dataset_context.get("dataset_family")
        ):
            exclusion_reason = "dataset_family_mismatch"
        elif require_same_dataset_taxonomy and (
            run_taxonomy_group != current_taxonomy_group
        ):
            exclusion_reason = "dataset_taxonomy_mismatch"
        elif time_window_gap_days is None:
            exclusion_reason = "time_window_unavailable"
        elif time_window_gap_days > max_time_window_gap_days:
            exclusion_reason = "time_window_gap_exceeded"
        elif require_same_evaluation_cohort and (
            not current_cohort.get("cohort_group")
            or not run_cohort.get("cohort_group")
        ):
            exclusion_reason = "evaluation_cohort_unavailable"
        elif require_same_evaluation_cohort and cohort_distance is None:
            exclusion_reason = "evaluation_cohort_group_mismatch"
        elif require_same_evaluation_cohort and cohort_distance > max_cohort_distance:
            exclusion_reason = "evaluation_cohort_distance_exceeded"

        previous_best_candidate = next(
            candidate for candidate in run["candidates"] if candidate["rank"] == 1
        )
        run_entry = {
            "selection_version": run["version"],
            "dataset_version": run["dataset_version"],
            "best_model_version": run["best_model_version"],
            "candidate_signature": selection_helpers.candidate_signature(
                previous_best_candidate
            ),
            "created_at": run.get("created_at"),
            "dataset_family": run_context.get("dataset_family"),
            "dataset_taxonomy": run_context.get("dataset_taxonomy") or {},
            "taxonomy_group": run_taxonomy_group,
            "time_window": run_context.get("time_window") or {},
            "time_window_gap_days": time_window_gap_days,
            "evaluation_cohort": run_cohort,
            "evaluation_cohort_distance": cohort_distance,
            "source": "history",
        }
        if exclusion_reason:
            run_entry["excluded_reason"] = exclusion_reason
            excluded_runs.append(run_entry)
            continue
        considered_history.append(run_entry)
        if len(considered_history) >= max(stability_window_runs - 1, 0):
            break

    current_run_entry = {
        "selection_version": selection_version,
        "dataset_version": current_dataset_version,
        "best_model_version": best_candidate["model_version"],
        "candidate_signature": candidate_signature,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset_family": dataset_context.get("dataset_family"),
        "dataset_taxonomy": dataset_context.get("dataset_taxonomy") or {},
        "taxonomy_group": current_taxonomy_group,
        "time_window": dataset_context.get("time_window") or {},
        "time_window_gap_days": 0.0,
        "evaluation_cohort": current_cohort,
        "evaluation_cohort_distance": 0,
        "source": "current",
    }
    window_entries = [current_run_entry, *considered_history]

    matching_entries = [
        entry
        for entry in window_entries
        if entry["candidate_signature"] == candidate_signature
    ]
    return {
        "candidate_signature": candidate_signature,
        "dataset_family": dataset_context.get("dataset_family"),
        "dataset_taxonomy": dataset_context.get("dataset_taxonomy") or {},
        "taxonomy_group": current_taxonomy_group,
        "require_same_dataset_family": require_same_dataset_family,
        "require_same_dataset_taxonomy": require_same_dataset_taxonomy,
        "require_same_evaluation_cohort": require_same_evaluation_cohort,
        "max_time_window_gap_days": max_time_window_gap_days,
        "max_cohort_distance": max_cohort_distance,
        "evaluation_cohort": current_cohort,
        "stability_window_runs": stability_window_runs,
        "required_consistent_wins": required_consistent_wins,
        "window_runs_considered": len(window_entries),
        "matching_best_candidate_wins": len(matching_entries),
        "consistent_enough": len(matching_entries) >= required_consistent_wins,
        "considered_runs": window_entries,
        "matching_runs": matching_entries,
        "excluded_runs": excluded_runs,
        "excluded_reason_counts": {
            reason: sum(
                1
                for entry in excluded_runs
                if entry.get("excluded_reason") == reason
            )
            for reason in sorted(
                {entry.get("excluded_reason") for entry in excluded_runs}
            )
        },
        "distinct_dataset_versions": sorted(
            {entry["dataset_version"] for entry in window_entries}
        ),
        "matching_dataset_versions": sorted(
            {entry["dataset_version"] for entry in matching_entries}
        ),
    }
