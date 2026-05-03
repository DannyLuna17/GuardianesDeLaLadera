"""Seed one synthetic, backdated PredictionRun so historical labels can resolve.

The backend's ``labels`` dataset export requires every ``ZoneOutcomeLabel`` to
resolve to a ``ZonePrediction`` whose ``PredictionRun.completed_at <=
label.observed_at`` and whose attached ``ZoneExplanation`` carries a feature
snapshot. That's a great design for forward-looking operations, but it
blocks any attempt to benchmark models on historical labels imported
retroactively.

This script creates one synthetic ``PredictionRun`` dated strictly before the
earliest imported label, then one ``ZonePrediction`` + ``ZoneExplanation`` per
zone that has at least one label. The ``feature_snapshot`` in the explanation
trace is left empty — the downstream dataset builder will call
``ZoneFeatureBuilder.build_for_zone`` on demand, which computes real features
from whatever the DB currently has (HistoricalEvent counts, SGC zone
polygons, seeded road segments and rain overlays for the seed zones).

**What this actually buys us**: the ``labels`` export succeeds instead of
raising ``training_dataset_empty``, and ``run_modern_labels_benchmark_with_review``
produces a decision. The resulting champion pick reflects the **historical
base rate per municipality** (via ``municipality_event_count``) plus whatever
spatial features the zone polygons carry — it is **not** informed by
precipitation, slope, or road catalog for the 637 auto-created zones because
that data is not yet ingested. Treat the first benchmark output as a
mechanical validation of the pipeline + a sanity prior, not as a definitive
champion decision. Re-run once IDEAM / OSM / Hansen ingestion lands.

Usage:

    uv run python scripts/landslide_inventory/backfill_historical_predictions.py

"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPT_DIR.parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


SYNTHETIC_MODEL_VERSION = "synthetic_historical_backfill_v1"
SYNTHETIC_RUN_NOTE = (
    "Auto-generated backdated run so historical ZoneOutcomeLabels can "
    "resolve to a PredictionRun. Features are reconstructed lazily by "
    "ZoneFeatureBuilder; see backfill_historical_predictions.py."
)
DEFAULT_DRIVERS = {
    "rain_6h": 0.0,
    "rain_24h": 0.0,
    "rain_72h": 0.0,
    "slope_deg": 15.0,
    "deforestation_proxy": 0.0,
    "geology_class": "unknown",
    "soil_class": "unknown",
}


def run_backfill(
    *,
    run_completed_at: datetime | None = None,
    dry_run: bool = False,
) -> dict:
    from app.core.config import get_settings
    from app.db.session import get_engine, reset_engine_cache
    from app.models.domain import (
        PredictionRun,
        Zone,
        ZoneExplanation,
        ZoneOutcomeLabel,
        ZonePrediction,
    )
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    get_settings.cache_clear()
    reset_engine_cache()
    engine = get_engine()

    stats = {
        "run_completed_at": None,
        "labels_total": 0,
        "zones_with_labels": 0,
        "predictions_created": 0,
        "predictions_reused": 0,
        "explanations_created": 0,
        "run_reused": False,
    }

    with Session(engine) as session, session.begin():
        labels = list(session.scalars(select(ZoneOutcomeLabel)).all())
        stats["labels_total"] = len(labels)
        if not labels:
            return stats

        # Earliest label date; backdate the run to one day before that so every
        # resolver filter (completed_at <= observed_at) hits for the oldest.
        earliest = min(
            (label.observed_at for label in labels if label.observed_at),
            default=None,
        )
        if earliest is None:
            return stats
        if earliest.tzinfo is None:
            earliest = earliest.replace(tzinfo=timezone.utc)
        resolved_completed_at = run_completed_at or (
            earliest - timedelta(days=1)
        )
        if resolved_completed_at.tzinfo is None:
            resolved_completed_at = resolved_completed_at.replace(
                tzinfo=timezone.utc
            )
        stats["run_completed_at"] = resolved_completed_at.isoformat()

        # Reuse an existing synthetic run if one was produced on a prior call.
        run = session.scalar(
            select(PredictionRun).where(
                PredictionRun.model_version == SYNTHETIC_MODEL_VERSION
            )
        )
        if run is None:
            run = PredictionRun(
                started_at=resolved_completed_at,
                completed_at=resolved_completed_at,
                status="completed",
                model_version=SYNTHETIC_MODEL_VERSION,
                partial_data=True,
                notes=SYNTHETIC_RUN_NOTE,
            )
            if not dry_run:
                session.add(run)
                session.flush()
        else:
            stats["run_reused"] = True

        zone_ids_with_labels: set[str] = {
            label.zone_id for label in labels if label.zone_id
        }
        stats["zones_with_labels"] = len(zone_ids_with_labels)

        existing_predictions = {
            prediction.zone_id: prediction
            for prediction in session.scalars(
                select(ZonePrediction).where(
                    ZonePrediction.run_id == run.id
                )
            ).all()
        }

        zones = list(
            session.scalars(
                select(Zone).where(Zone.id.in_(zone_ids_with_labels))
            ).all()
        )

        for zone in zones:
            existing = existing_predictions.get(zone.id)
            if existing is not None:
                stats["predictions_reused"] += 1
                continue
            prediction = ZonePrediction(
                run=run,
                zone=zone,
                risk_score=0.5,
                confidence="Baja",
                drivers=dict(DEFAULT_DRIVERS),
                risk_delta=0.0,
                trend="estable",
                source_snapshot={},
                created_at=resolved_completed_at,
            )
            if not dry_run:
                session.add(prediction)
                session.flush()
            stats["predictions_created"] += 1

            explanation = ZoneExplanation(
                prediction=prediction,
                mode="synthetic_backfill",
                summary=(
                    "Synthetic historical backfill. Feature snapshot is "
                    "computed lazily at dataset-export time."
                ),
                driver_chips=[],
                suggestions=[],
                data_warnings=[
                    "synthetic_historical_backfill",
                    "features_mostly_unknown",
                ],
                trace={"source": "backfill_historical_predictions_v1"},
                generated_at=resolved_completed_at,
            )
            if not dry_run:
                session.add(explanation)
            stats["explanations_created"] += 1

        if dry_run:
            session.rollback()

    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--completed-at",
        default=None,
        help=(
            "Optional ISO datetime for the synthetic run. Defaults to the "
            "earliest label's observed_at minus one day."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and count; do not commit.",
    )
    args = parser.parse_args(argv)

    run_completed_at: datetime | None = None
    if args.completed_at:
        try:
            parsed = datetime.fromisoformat(
                args.completed_at.replace("Z", "+00:00")
            )
        except ValueError as exc:
            print(
                f"Invalid --completed-at: {args.completed_at!r} ({exc})",
                file=sys.stderr,
            )
            return 1
        run_completed_at = parsed

    stats = run_backfill(
        run_completed_at=run_completed_at, dry_run=args.dry_run
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
