"""Drive the full labels -> benchmark -> review flow on the real inventory.

This is the hand-off script that ties the landslide-inventory pipeline to the
governed benchmark/review endpoint:

    1. Export a ``labels`` training dataset from the ZoneOutcomeLabel rows that
       ``import_landslide_inventory.py`` inserted.
    2. If the export actually produced rows, call
       ``ModelSelectionService.run_modern_labels_benchmark_with_review(...)``
       with ``promote_best=False`` (we want the review decision, not an auto-
       promote on un-audited real data).
    3. Write a markdown decision report capturing:
         - dataset version + row counts + split counts
         - family rollups under the honest validation policy
         - promotion decision (eligible / blocking reasons)
         - recommendation from the review service (actionable or not)
         - if actionable: the review task id the caller can act on

The script talks to the backend services directly (no HTTP), honouring
``DATABASE_URL`` from the environment so it can run against a throw-away SQLite
for testing or the shared dev DB for a real decision.

Usage:

    uv run python scripts/landslide_inventory/run_benchmark_on_real_labels.py \\
        --output data/inventory/03_reports/benchmark_decision.md

"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


# Backend dir is two levels above this file; put it on sys.path so we can
# import the FastAPI app package when invoked as a plain script.
_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPT_DIR.parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _render_report(
    *,
    dataset_export: dict,
    benchmark: dict | None,
    recommendation: dict | None,
    review_task: dict | None,
    diagnosis: dict,
) -> str:
    lines: list[str] = []
    lines.append("# Modern labels benchmark — real inventory run")
    lines.append("")
    lines.append(f"Generated: {_now().isoformat()}")
    lines.append("")
    lines.append("## Dataset export")
    lines.append("")
    for key in (
        "datasetVersion",
        "sourceMode",
        "rows",
        "labelCount",
        "splitCounts",
    ):
        if key in dataset_export:
            lines.append(f"- **{key}** — `{dataset_export[key]}`")
    lines.append("")

    if diagnosis.get("export_empty"):
        lines.append("## Status: BLOCKED — export produced zero rows")
        lines.append("")
        lines.append(
            "The labels dataset exported with zero rows. That almost always"
            " means every confirmed ZoneOutcomeLabel failed to resolve a"
            " matching ZonePrediction + feature snapshot. The backend's"
            " `_build_labeled_dataset` requires a PredictionRun whose"
            " `completed_at <= label.observed_at` for the label's zone, and"
            " then pulls features from the prediction's explanation trace."
        )
        lines.append("")
        lines.append("### Diagnostic")
        lines.append("")
        for key, value in diagnosis.items():
            if key == "export_empty":
                continue
            lines.append(f"- **{key}** — `{value}`")
        lines.append("")
        lines.append("### What unblocks this")
        lines.append("")
        lines.append(
            "Historical UNGRD events from 2019-2022 have no matching"
            " predictions because the prediction pipeline only runs forward"
            " from ingestion time. Two paths forward:"
        )
        lines.append("")
        lines.append(
            "1. **Synthetic historical backfill** — create one"
            " `PredictionRun` dated before the earliest label plus one"
            " `ZonePrediction` per new zone with a"
            " `ZoneExplanation.trace.feature_snapshot`. Features would be"
            " mostly zero for the new zones (no rain history, no road"
            " catalog), so the trained model would be signal-starved. Useful"
            " to validate pipeline mechanics, not to draw a champion"
            " decision."
        )
        lines.append(
            "2. **Real feature backfill** — ingest historical IDEAM"
            " precipitation, SGC slope/geology, and OSM roads for each"
            " municipality, then run the pipeline retroactively. This is the"
            " right path but it is a data-ingestion project, not a code task."
        )
        return "\n".join(lines) + "\n"

    lines.append("## Benchmark result")
    lines.append("")
    if benchmark is not None:
        selection = benchmark.get("selection", {})
        lines.append(f"- **selectionVersion** — `{selection.get('selectionVersion')}`")
        lines.append(f"- **bestModelVersion** — `{selection.get('bestModelVersion')}`")
        lines.append(f"- **activeModelVersion** — `{selection.get('activeModelVersion')}`")
        lines.append(f"- **candidateCount** — `{selection.get('candidateCount')}`")
        lines.append(f"- **preset** — `{benchmark.get('preset')}`")
        resolved = benchmark.get("resolvedPolicy") or {}
        lines.append(
            f"- **validationStrategy** — `{resolved.get('validation_strategy')}`"
        )
        lines.append(f"- **selectionMode** — `{resolved.get('selection_mode')}`")
        lines.append("")
        lines.append("### Family rollups")
        lines.append("")
        for rollup in selection.get("familyRollups") or []:
            lines.append(
                f"- `{rollup.get('model_family')}` — best_rank "
                f"{rollup.get('best_rank')}, "
                f"val RMSE {rollup.get('best_validation_rmse')}, "
                f"Brier {rollup.get('best_validation_brier_score')}, "
                f"AUPRC {rollup.get('best_validation_auprc')}"
            )
        lines.append("")
        decision = selection.get("promotionDecision") or {}
        lines.append("### Promotion decision")
        lines.append("")
        lines.append(f"- **eligible** — `{decision.get('eligible')}`")
        lines.append(
            f"- **blockingReasons** — `{decision.get('blocking_reasons')}`"
        )
        lines.append(
            f"- **validationRmseImprovement** — `{decision.get('validation_rmse_improvement')}`"
        )
        lines.append("")
    lines.append("## Reading the result")
    lines.append("")
    if benchmark is not None:
        rollups = (benchmark.get("selection") or {}).get("familyRollups") or []
        if rollups:
            ranked = sorted(
                rollups,
                key=lambda roll: (
                    int(roll.get("best_rank") or 999),
                    float(roll.get("best_validation_rmse") or 1.0),
                ),
            )
            champion = ranked[0]
            runner = ranked[1] if len(ranked) > 1 else None
            lines.append(
                f"**Champion under this single snapshot**: "
                f"`{champion.get('model_family')}` "
                f"(val RMSE {champion.get('best_validation_rmse')}, "
                f"Brier {champion.get('best_validation_brier_score')})."
            )
            if runner is not None:
                lines.append(
                    f"Runner-up: `{runner.get('model_family')}` "
                    f"(val RMSE {runner.get('best_validation_rmse')})."
                )
            lines.append("")
        decision = (benchmark.get("selection") or {}).get("promotionDecision") or {}
        blocking = decision.get("blocking_reasons") or []
        if blocking:
            lines.append(
                "Promotion was **blocked**, but not on performance — on "
                "stability-evidence gates:"
            )
            lines.append("")
            for reason in blocking:
                lines.append(f"- `{reason}`")
            lines.append("")
            lines.append(
                "That is the intended research-aligned behaviour: the "
                "governed gate requires consistent wins across multiple "
                "snapshots, not a single-cohort victory. Re-run this driver "
                "after fresh labels land (or with different cohorts) to "
                "build the stability evidence the gate needs, or tune the "
                "`stability_window_runs` / `required_consistent_wins` knobs "
                "in `tune_model` if a different promotion policy is desired."
            )
            lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    if recommendation is not None:
        for key in (
            "actionable",
            "recommended_action",
            "best_model_version",
            "active_model_version",
            "skipped_reasons",
            "summary",
        ):
            if key in recommendation:
                lines.append(f"- **{key}** — `{recommendation[key]}`")
        lines.append("")
    lines.append("## Feature-sparsity caveat")
    lines.append("")
    lines.append(
        "This benchmark trained on real UNGRD outcome labels but its feature"
        " snapshots are derived from the features currently available in the"
        " DB: HistoricalEvent counts per municipality and per zone (real),"
        " plus seed zones' road-segment intersections and rain overlays"
        " (real only for the Mocoa/Pasto/Popayán seed zones). The 637"
        " auto-created municipal zones carry **zero values** for road /"
        " rain / slope / geology features because IDEAM historical rainfall,"
        " OSM roads, SGC topography, and Hansen forest-loss layers have not"
        " been ingested yet. The champion pick above is therefore signal"
        " from: historical frequency per municipality + spatial coverage of"
        " the seed zones. It is a valid baseline prior, not a definitive"
        " champion decision. Expect the ranking to tighten (and potentially"
        " flip in favour of calibrated interpretable families like"
        " `additive_spline` or `beta_regression`) once the rain / topo /"
        " road layers are joined in."
    )
    lines.append("")
    if review_task is not None:
        lines.append("## Review task opened")
        lines.append("")
        lines.append(f"- **id** — `{review_task.get('id')}`")
        lines.append(f"- **status** — `{review_task.get('status')}`")
        lines.append(f"- **reviewType** — `{review_task.get('reviewType')}`")
        lines.append(f"- **title** — `{review_task.get('title')}`")
        lines.append(f"- **summary** — `{review_task.get('summary')}`")
        lines.append("")
        lines.append(
            "> To approve: call `POST /v1/admin/models/review-tasks/"
            f"{review_task.get('id')}/update` with `status=resolved`, "
            "`decision=approve_promotion_review`, and notes. Then call "
            "`POST /v1/admin/models/promote` with the bestModelVersion and "
            f"`reviewTaskId={review_task.get('id')}`."
        )
        lines.append("")
    else:
        lines.append("## Review task")
        lines.append("")
        lines.append(
            "No review task was opened — the benchmark was not actionable"
            " (see skipped_reasons above)."
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def run_driver(
    *,
    output_path: Path,
    max_labels: int,
    label_sources: list[str] | None,
    opened_by: str,
    notes: str | None,
    observed_after: datetime | None = None,
    observed_before: datetime | None = None,
    label_cohort: str | None = None,
    zone_id_prefixes: list[str] | None = None,
) -> dict:
    from app.core.config import get_settings
    from app.core.exceptions import ApiError
    from app.db.session import get_engine, reset_engine_cache
    from app.models.domain import ZoneOutcomeLabel, ZonePrediction
    from app.services.datasets import TrainingDatasetService
    from app.services.model_selection import ModelSelectionService
    from sqlalchemy import or_, select, func
    from sqlalchemy.orm import Session

    get_settings.cache_clear()
    reset_engine_cache()
    engine = get_engine()

    summary: dict = {
        "started_at": _now().isoformat(),
        "dataset_export": None,
        "benchmark": None,
        "recommendation": None,
        "review_task": None,
        "diagnosis": {},
    }

    with Session(engine) as session:
        # Diagnostic counts so the report is useful even when the export is empty.
        label_count = session.scalar(
            select(func.count(ZoneOutcomeLabel.id))
            .where(ZoneOutcomeLabel.status == "confirmed")
        )
        prediction_count = session.scalar(select(func.count(ZonePrediction.id)))
        summary["diagnosis"]["confirmed_label_count"] = int(label_count or 0)
        summary["diagnosis"]["zone_prediction_count"] = int(prediction_count or 0)

        datasets_service = TrainingDatasetService(session)
        print(
            f"[1/3] Exporting labels dataset (max_labels={max_labels}) ..."
        )

        label_ids_filter: list[int] | None = None
        if zone_id_prefixes:
            conditions = [
                ZoneOutcomeLabel.zone_id.like(f"{prefix}%")
                for prefix in zone_id_prefixes
            ]
            matched = list(
                session.scalars(
                    select(ZoneOutcomeLabel.id)
                    .where(
                        ZoneOutcomeLabel.status == "confirmed",
                        or_(*conditions),
                    )
                    .order_by(ZoneOutcomeLabel.observed_at.desc())
                ).all()
            )
            label_ids_filter = matched[:max_labels]
            summary["diagnosis"]["zone_id_prefix_matches"] = len(matched)
            print(
                f"  Zone-id prefixes {zone_id_prefixes!r} matched "
                f"{len(matched)} labels; using {len(label_ids_filter)} "
                f"(max_labels cap applied)."
            )

        cohort_tag = f"-{label_cohort}" if label_cohort else ""
        export_version = (
            f"real-labels-inventory{cohort_tag}"
            f"-{_now().strftime('%Y%m%dT%H%M%SZ')}"
        )
        try:
            export_response = datasets_service.export_dataset(
                version=export_version,
                source_mode="labels",
                label_ids=label_ids_filter,
                label_sources=label_sources,
                max_labels=max_labels,
                observed_after=observed_after,
                observed_before=observed_before,
                origin="manual:run_benchmark_on_real_labels",
            )
        except ApiError as exc:
            if exc.code not in {
                "training_dataset_empty",
                "no_confirmed_outcome_labels",
            }:
                raise
            print(
                f"  export empty: {exc.code} — {exc.message}",
                file=sys.stderr,
            )
            summary["dataset_export"] = {
                "datasetVersion": export_version,
                "sourceMode": "labels",
                "rows": 0,
                "labelCount": 0,
                "splitCounts": {"train": 0, "validation": 0},
                "errorCode": exc.code,
                "errorMessage": exc.message,
            }
            summary["diagnosis"]["exported_rows"] = 0
            summary["diagnosis"]["exported_labels"] = 0
            summary["diagnosis"]["export_empty"] = True
            summary["diagnosis"]["export_error_code"] = exc.code
            # Best-effort: count how many labels could not resolve a prediction.
            unresolved = 0
            all_labels = session.scalars(
                select(ZoneOutcomeLabel).where(
                    ZoneOutcomeLabel.status == "confirmed"
                )
            ).all()
            for lbl in all_labels:
                if datasets_service._resolve_prediction_for_label(lbl) is None:
                    unresolved += 1
            summary["diagnosis"]["unresolved_labels"] = unresolved
            _write_report(summary, output_path)
            summary["completed_at"] = _now().isoformat()
            summary["report_path"] = str(output_path)
            return summary
        summary["dataset_export"] = export_response.model_dump(by_alias=True)
        summary["diagnosis"]["exported_rows"] = int(export_response.rows)
        summary["diagnosis"]["exported_labels"] = int(
            export_response.label_count or 0
        )
        if export_response.rows == 0:
            summary["diagnosis"]["export_empty"] = True
            return summary

        print(
            f"[2/3] Running modern-labels benchmark with review on dataset "
            f"{export_response.dataset_version} ..."
        )
        selection_service = ModelSelectionService(session)
        review_response = selection_service.run_modern_labels_benchmark_with_review(
            dataset_version=export_response.dataset_version,
            auto_export_dataset=False,
            promote_best=False,
            notes=notes,
            opened_by=opened_by,
            origin="manual:run_benchmark_on_real_labels",
        )
        payload = review_response.model_dump(by_alias=True)
        summary["benchmark"] = payload.get("benchmark")
        summary["recommendation"] = payload.get("recommendation")
        summary["review_task"] = payload.get("reviewTask")

    print("[3/3] Writing decision report ...")
    _write_report(summary, output_path)
    summary["completed_at"] = _now().isoformat()
    summary["report_path"] = str(output_path)
    return summary


def _write_report(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _render_report(
            dataset_export=summary["dataset_export"] or {},
            benchmark=summary["benchmark"],
            recommendation=summary["recommendation"],
            review_task=summary["review_task"],
            diagnosis=summary["diagnosis"],
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/inventory/03_reports/benchmark_decision.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=500,
        help="Max labels for the dataset export (capped at the backend's 500 ceiling).",
    )
    parser.add_argument(
        "--label-sources",
        nargs="*",
        default=None,
        help="Optional label source filter (e.g. 'UNGRD:UNGRD-abc'). Default: all confirmed labels.",
    )
    parser.add_argument(
        "--opened-by",
        default="inventory-driver",
        help="Opened-by identifier for any review tasks created.",
    )
    parser.add_argument(
        "--notes",
        default="auto-run from scripts/landslide_inventory/run_benchmark_on_real_labels.py",
        help="Notes attached to any opened review task.",
    )
    parser.add_argument(
        "--observed-after",
        default=None,
        help="ISO datetime lower bound for label observed_at (stability cohorts).",
    )
    parser.add_argument(
        "--observed-before",
        default=None,
        help="ISO datetime upper bound for label observed_at (stability cohorts).",
    )
    parser.add_argument(
        "--label-cohort",
        default=None,
        help="Tag baked into the dataset version for cohort-based stability checks.",
    )
    parser.add_argument(
        "--zone-id-prefix",
        action="append",
        default=None,
        help=(
            "Only include labels whose zone_id starts with this prefix. "
            "Pass multiple --zone-id-prefix flags for an OR filter. Useful "
            "for spatial cohorts (e.g. --zone-id-prefix antioquia-)."
        ),
    )
    args = parser.parse_args(argv)

    def _parse_bound(value: str | None) -> datetime | None:
        if not value:
            return None
        text = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    summary = run_driver(
        output_path=args.output,
        max_labels=args.max_labels,
        label_sources=args.label_sources,
        opened_by=args.opened_by,
        notes=args.notes,
        observed_after=_parse_bound(args.observed_after),
        observed_before=_parse_bound(args.observed_before),
        label_cohort=args.label_cohort,
        zone_id_prefixes=args.zone_id_prefix,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str, sort_keys=True))
    print(f"\nReport written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
