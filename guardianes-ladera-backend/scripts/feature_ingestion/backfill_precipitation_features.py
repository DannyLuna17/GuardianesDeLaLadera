"""Populate per-label precipitation drivers from real IDEAM history.

The existing synthetic prediction backfill creates one ``PredictionRun``
retro-dated before the earliest label and one ``ZonePrediction`` per zone
with placeholder ``rain_*`` drivers all set to zero. Every label for a given
zone resolves to that single prediction, so they all train on identical
zero-rain inputs — the model can only learn historical-frequency priors.

This backfill replaces that single-run scheme with **one PredictionRun per
label**. Each run is dated exactly at the label's ``observed_at`` and carries
a ``ZonePrediction`` whose ``drivers`` reflect the real IDEAM rolling
rainfall (1, 3, 7, 14, 30 day accumulations ending at ``observed_at``)
measured at the IDEAM station nearest to the zone centroid. The
``ZoneOutcomeLabel`` resolver picks "the most recent run with
``completed_at <= label.observed_at``" for each label, so per-label runs
naturally win over the older synthetic generic backfill.

Idempotent on ``(zone_id, completed_at)`` pairs already tagged with
``model_version = synthetic_precipitation_backfill_v1``.

Inputs:
    - ``data/feature_ingestion/00_raw/ideam_station_catalog.csv`` (from
      ``fetch_ideam_catalog.py``).
    - The IDEAM ``PrecipitacionNacionalDiaria.zip``.

Usage:

    uv run python scripts/feature_ingestion/backfill_precipitation_features.py \\
        --zip ../data-raw/PrecipitacionNacionalDiaria.zip

    # smoke test without committing:
    uv run python scripts/feature_ingestion/backfill_precipitation_features.py \\
        --zip ../data-raw/PrecipitacionNacionalDiaria.zip --dry-run

"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPT_DIR.parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


import normalize_ideam_precipitation as ideam  # noqa: E402


PRECIPITATION_MODEL_VERSION = "synthetic_precipitation_backfill_v1"
PRECIPITATION_RUN_NOTE = (
    "Auto-generated per-label PredictionRun whose drivers carry real IDEAM "
    "rolling rainfall features. See backfill_precipitation_features.py."
)
DEFAULT_MAX_DISTANCE_KM = 50.0
DEFAULT_DRIVERS_TEMPLATE = {
    "rain_6h": 0.0,  # IDEAM is daily; no sub-daily resolution
    "rain_24h": 0.0,
    "rain_72h": 0.0,
    "rain_7d": 0.0,
    "rain_14d": 0.0,
    "rain_30d": 0.0,
    "slope_deg": 15.0,
    "deforestation_proxy": 0.0,
    "geology_class": "unknown",
    "soil_class": "unknown",
}


def _ensure_utc(value):
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def run_backfill(
    *,
    zip_path: Path,
    catalog_path: Path,
    max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
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

    catalog = ideam.load_catalog(catalog_path)
    if not catalog:
        raise RuntimeError(
            f"IDEAM station catalog at {catalog_path} is empty. "
            "Run fetch_ideam_catalog.py first."
        )

    stats: dict = {
        "model_version": PRECIPITATION_MODEL_VERSION,
        "max_distance_km": max_distance_km,
        "ideam_catalog_size": len(catalog),
        "labels_total": 0,
        "labels_with_station": 0,
        "labels_without_station_too_far": 0,
        "labels_without_station_zero_centroid": 0,
        "zones_with_station": 0,
        "zones_without_station": 0,
        "stations_used": 0,
        "stations_with_zip_data": 0,
        "stations_without_zip_data": 0,
        "runs_created": 0,
        "runs_reused": 0,
        "predictions_created": 0,
        "predictions_skipped_existing": 0,
        "explanations_created": 0,
        "distance_km_examples": [],
    }

    with ideam.IdeamPrecipitationStore(zip_path, cache_max_stations=8) as store:
        zip_ids = store.available_ids()

        with Session(engine) as session, session.begin():
            zones = list(session.scalars(select(Zone)).all())
            zones_by_id = {zone.id: zone for zone in zones}

            zone_station_map: dict[str, dict] = {}
            for zone in zones:
                centroid = zone.centroid or []
                if not centroid or len(centroid) < 2:
                    continue
                lat, lon = float(centroid[0]), float(centroid[1])
                if lat == 0.0 and lon == 0.0:
                    continue
                station = ideam.find_nearest_station(
                    catalog,
                    lat,
                    lon,
                    require_in_zip=zip_ids,
                    max_km=max_distance_km,
                )
                if station is not None:
                    zone_station_map[zone.id] = station
            stats["zones_with_station"] = len(zone_station_map)
            stats["zones_without_station"] = (
                len(zones) - len(zone_station_map)
            )

            labels = list(
                session.scalars(
                    select(ZoneOutcomeLabel).where(
                        ZoneOutcomeLabel.status == "confirmed"
                    )
                ).all()
            )
            stats["labels_total"] = len(labels)

            # Pre-load existing precipitation-backfill predictions so re-runs
            # are idempotent. Key on (zone_id, completed_at iso-day).
            existing_keys: set[tuple[str, str]] = set()
            existing_runs_by_completed: dict[str, PredictionRun] = {}
            existing_predictions = session.execute(
                select(ZonePrediction, PredictionRun)
                .join(PredictionRun, ZonePrediction.run_id == PredictionRun.id)
                .where(PredictionRun.model_version == PRECIPITATION_MODEL_VERSION)
            ).all()
            for prediction, run in existing_predictions:
                completed_at = _ensure_utc(run.completed_at)
                if completed_at is None:
                    continue
                day_key = completed_at.date().isoformat()
                existing_keys.add((prediction.zone_id, day_key))
                existing_runs_by_completed[day_key] = run

            # Group labels by station so each station's series is read once.
            labels_by_station: dict[str, list[ZoneOutcomeLabel]] = defaultdict(
                list
            )
            for label in labels:
                station = zone_station_map.get(label.zone_id)
                zone = zones_by_id.get(label.zone_id)
                if station is None:
                    if zone is None or not zone.centroid:
                        stats["labels_without_station_zero_centroid"] += 1
                    else:
                        stats["labels_without_station_too_far"] += 1
                    continue
                stats["labels_with_station"] += 1
                labels_by_station[station["idestacion"]].append(label)
            stats["stations_used"] = len(labels_by_station)

            now = datetime.now(timezone.utc).replace(microsecond=0)

            for station_id, station_labels in labels_by_station.items():
                series = store.series_for(station_id)
                if not series:
                    stats["stations_without_zip_data"] += 1
                    # Even though catalog said it's available, the ZIP file
                    # may not have data for it; skip these labels.
                    stats["labels_without_station_too_far"] += len(
                        station_labels
                    )
                    stats["labels_with_station"] -= len(station_labels)
                    continue
                stats["stations_with_zip_data"] += 1

                for label in station_labels:
                    observed_at = _ensure_utc(label.observed_at)
                    if observed_at is None:
                        continue
                    day_key = observed_at.date().isoformat()
                    if (label.zone_id, day_key) in existing_keys:
                        stats["predictions_skipped_existing"] += 1
                        continue

                    rolling = ideam.rolling_window_features(
                        observed_at.date(), series
                    )

                    drivers = dict(DEFAULT_DRIVERS_TEMPLATE)
                    drivers["rain_24h"] = rolling["rain_1d"]
                    drivers["rain_72h"] = rolling["rain_3d"]
                    drivers["rain_7d"] = rolling["rain_7d"]
                    drivers["rain_14d"] = rolling["rain_14d"]
                    drivers["rain_30d"] = rolling["rain_30d"]

                    station = zone_station_map[label.zone_id]
                    run = existing_runs_by_completed.get(day_key)
                    if run is None:
                        run = PredictionRun(
                            started_at=observed_at,
                            completed_at=observed_at,
                            status="completed",
                            model_version=PRECIPITATION_MODEL_VERSION,
                            partial_data=False,
                            notes=PRECIPITATION_RUN_NOTE,
                        )
                        if not dry_run:
                            session.add(run)
                            session.flush()
                        existing_runs_by_completed[day_key] = run
                        stats["runs_created"] += 1
                    else:
                        stats["runs_reused"] += 1

                    prediction = ZonePrediction(
                        run=run,
                        zone_id=label.zone_id,
                        risk_score=0.5,
                        confidence="Baja",
                        drivers=drivers,
                        risk_delta=0.0,
                        trend="estable",
                        source_snapshot={
                            "ideam_station_id": station_id,
                            "ideam_station_name": station.get("nombre"),
                            "ideam_station_distance_km": station.get(
                                "distance_km"
                            ),
                            "ideam_station_category": station.get(
                                "idcategoria"
                            ),
                        },
                        created_at=observed_at,
                    )
                    if not dry_run:
                        session.add(prediction)
                        session.flush()
                    existing_keys.add((label.zone_id, day_key))
                    stats["predictions_created"] += 1

                    if len(stats["distance_km_examples"]) < 5:
                        stats["distance_km_examples"].append(
                            {
                                "zone_id": label.zone_id,
                                "station_id": station_id,
                                "station_name": station.get("nombre"),
                                "distance_km": station.get("distance_km"),
                                "observed_at": observed_at.isoformat(),
                                "rain_1d": rolling["rain_1d"],
                                "rain_3d": rolling["rain_3d"],
                                "rain_7d": rolling["rain_7d"],
                                "rain_30d": rolling["rain_30d"],
                                "rain_30d_observed_days": rolling[
                                    "rain_30d_observed_days"
                                ],
                            }
                        )

                    explanation = ZoneExplanation(
                        prediction=prediction,
                        mode="precipitation_backfill",
                        summary=(
                            f"Per-label IDEAM rainfall backfill from station "
                            f"{station_id} ({station.get('nombre')!r}, "
                            f"{station.get('distance_km')} km away)."
                        ),
                        driver_chips=[],
                        suggestions=[],
                        data_warnings=(
                            ["partial_window"]
                            if rolling["rain_30d_observed_days"] < 30
                            else []
                        ),
                        trace={
                            "source": PRECIPITATION_MODEL_VERSION,
                            "ideam_station_id": station_id,
                            "rolling_window_observed_days": {
                                key: rolling[key]
                                for key in (
                                    "rain_1d_observed_days",
                                    "rain_3d_observed_days",
                                    "rain_7d_observed_days",
                                    "rain_14d_observed_days",
                                    "rain_30d_observed_days",
                                )
                            },
                        },
                        generated_at=observed_at,
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
        "--zip",
        type=Path,
        required=True,
        help="Path to PrecipitacionNacionalDiaria.zip.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(
            "data/feature_ingestion/00_raw/ideam_station_catalog.csv"
        ),
    )
    parser.add_argument(
        "--max-distance-km",
        type=float,
        default=DEFAULT_MAX_DISTANCE_KM,
        help=(
            "Maximum allowed distance from a zone centroid to its nearest "
            "IDEAM station (default %(default)s km). Zones beyond this "
            "radius are skipped — their labels remain on the older "
            "synthetic backfill (zero rain) until CHIRPS or another fallback "
            "ingester ships."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute counts without committing to the DB.",
    )
    args = parser.parse_args(argv)

    if not args.zip.exists():
        print(f"ZIP not found: {args.zip}", file=sys.stderr)
        return 1
    if not args.catalog.exists():
        print(
            f"Catalog not found: {args.catalog} — run fetch_ideam_catalog.py.",
            file=sys.stderr,
        )
        return 1

    stats = run_backfill(
        zip_path=args.zip,
        catalog_path=args.catalog,
        max_distance_km=args.max_distance_km,
        dry_run=args.dry_run,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
