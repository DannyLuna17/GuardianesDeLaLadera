from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.data.seed_store import load_seed_payload
from app.db.migrations import upgrade_to_head
from app.db.spatial import (
    bounds_geometry_value,
    linestring_geometry_value,
    point_geometry_value,
    polygon_geometry_value,
    session_dialect_name,
)
from app.models import (
    HistoricalEvent,
    Municipality,
    MunicipalityRainPoint,
    PredictionRun,
    RainOverlay,
    RoadSegment,
    SourceCatalog,
    SourceSyncEvent,
    SourceSyncStatus,
    UngrdRecord,
    UserAccount,
    Zone,
    ZoneExplanation,
    ZonePrediction,
)
from app.core.security import hash_password
from app.services.explanation_builder import build_driver_chips, build_suggestions, build_summary


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def risk_level_from_score(score: float) -> str:
    if score >= 0.75:
        return "Rojo"
    if score >= 0.5:
        return "Naranja"
    if score >= 0.25:
        return "Amarillo"
    return "Verde"


def risk_text_from_level(level: str) -> str:
    return {
        "Verde": "bajo",
        "Amarillo": "moderado",
        "Naranja": "alto",
        "Rojo": "muy alto",
    }[level]


def trend_from_delta(delta: float) -> str:
    if delta > 0.01:
        return "subiendo"
    if delta < -0.01:
        return "bajando"
    return "estable"


def deterministic_delta(zone_id: str) -> float:
    base = (sum(ord(char) for char in zone_id) % 7) - 3
    return round(base * 0.01, 3)


def build_source_snapshot(source_statuses: list[SourceSyncStatus]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    now = datetime.now(timezone.utc)
    for item in source_statuses:
        if item.source.category in {"historico", "infraestructura"}:
            snapshot[item.source.id] = "Estatico"
            continue
        if item.last_synced_at is None:
            snapshot[item.source.id] = "Desactualizado"
            continue
        age = int((now - item.last_synced_at).total_seconds() // 60)
        if age <= 30:
            snapshot[item.source.id] = "Fresco"
        elif age <= 180:
            snapshot[item.source.id] = "Retrasado"
        else:
            snapshot[item.source.id] = "Desactualizado"
    return snapshot


def init_database() -> None:
    settings = get_settings()
    if not settings.run_db_migrations_on_startup:
        return
    upgrade_to_head(database_url=settings.database_url)


def ensure_seed_admin_user(session: Session) -> None:
    settings = get_settings()
    existing = session.scalar(select(UserAccount).where(UserAccount.username == settings.seed_admin_username))
    if existing is not None:
        return
    session.add(
        UserAccount(
            username=settings.seed_admin_username,
            password_hash=hash_password(settings.seed_admin_password),
            role=settings.seed_admin_role,
            is_active=True,
            created_at=datetime.now(timezone.utc).replace(microsecond=0),
        )
    )


def ensure_seed_sync_events(session: Session) -> None:
    if session.scalar(select(SourceSyncEvent.id).limit(1)) is not None:
        return

    for source in session.scalars(select(SourceCatalog)).all():
        sync_status = source.sync_status
        if sync_status is None:
            continue
        completed_at = sync_status.last_success_at or sync_status.last_synced_at
        if completed_at is None:
            continue
        message = sync_status.status_note or "Bootstrap seed synchronization snapshot."
        session.add(
            SourceSyncEvent(
                source=source,
                origin="seed",
                adapter_key="bootstrap.seed",
                transport="seed",
                status="completed",
                processed_records=0,
                started_at=completed_at,
                completed_at=completed_at,
                message=message,
                details={"backfilled": True},
            )
        )


def backfill_spatial_representations(session: Session) -> None:
    dialect_name = session_dialect_name(session)

    for zone in session.scalars(select(Zone)).all():
        if zone.centroid_geom is None:
            zone.centroid_geom = point_geometry_value(zone.centroid, dialect_name)
        if zone.polygon_geom is None:
            zone.polygon_geom = polygon_geometry_value(zone.polygon, dialect_name)

    for road_segment in session.scalars(select(RoadSegment)).all():
        if road_segment.coords_geom is None:
            road_segment.coords_geom = linestring_geometry_value(road_segment.coords, dialect_name)

    for event in session.scalars(select(HistoricalEvent)).all():
        if event.coords_geom is None:
            event.coords_geom = point_geometry_value(event.coords, dialect_name)

    for overlay in session.scalars(select(RainOverlay)).all():
        if overlay.bounds_geom is None:
            overlay.bounds_geom = bounds_geometry_value(overlay.bounds, dialect_name)


def seed_demo_data(session: Session) -> None:
    if session.scalar(select(Municipality.id).limit(1)):
        ensure_seed_admin_user(session)
        ensure_seed_sync_events(session)
        backfill_spatial_representations(session)
        return

    seed = load_seed_payload()
    settings = get_settings()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    dialect_name = session_dialect_name(session)

    municipalities_by_name: dict[str, Municipality] = {}
    for item in seed["municipalities"]:
        municipality = Municipality(
            id=item["id"],
            name=item["name"],
            center=item["center"],
            zoom=item["zoom"],
        )
        session.add(municipality)
        municipalities_by_name[item["name"]] = municipality

    road_segments_by_id: dict[str, RoadSegment] = {}
    for item in seed["roadSegments"]:
        road_segment = RoadSegment(
            id=item["id"],
            municipality=municipalities_by_name[item["municipality"]],
            name=item["name"],
            coords=item["coords"],
            coords_geom=linestring_geometry_value(item["coords"], dialect_name),
            risk_level=item["riskLevel"],
            length_km=item["length_km"],
            note=item["note"],
        )
        session.add(road_segment)
        road_segments_by_id[item["id"]] = road_segment

    zones_by_id: dict[str, Zone] = {}
    for item in seed["zones"]:
        zone = Zone(
            id=item["id"],
            municipality=municipalities_by_name[item["municipality"]],
            name=item["name"],
            type=item["type"],
            centroid=item["centroid"],
            centroid_geom=point_geometry_value(item["centroid"], dialect_name),
            polygon=item["polygon"],
            polygon_geom=polygon_geometry_value(item["polygon"], dialect_name),
            exposure=item["exposure"],
            is_active=True,
        )
        for road_segment_id in item["assets"]["road_segment_ids"]:
            zone.road_segments.append(road_segments_by_id[road_segment_id])
        session.add(zone)
        zones_by_id[item["id"]] = zone

    for item in seed["historicalEvents"]:
        session.add(
            HistoricalEvent(
                id=item["id"],
                municipality=municipalities_by_name[item["municipality"]],
                date=date.fromisoformat(item["date"]),
                severity=item["severity"],
                type=item["type"],
                coords=item["coords"],
                coords_geom=point_geometry_value(item["coords"], dialect_name),
                source=item["source"],
            )
        )

    for municipality_name, records in seed["ungrdRecords"].items():
        for record in records:
            session.add(
                UngrdRecord(
                    id=record["id"],
                    municipality=municipalities_by_name[municipality_name],
                    date=date.fromisoformat(record["date"]),
                    summary=record["summary"],
                )
            )

    for source in seed["sourceCatalog"]:
        session.add(
            SourceCatalog(
                id=source["id"],
                label=source["label"],
                category=source["category"],
            )
        )

    session.flush()

    freshness_offsets = {
        "IDEAM": 8,
        "NASA": 18,
        "UNGRD": 140,
        "SENTINEL": 220,
        "SGC": None,
        "IGAC": None,
        "DANE": None,
        "INVIAS": None,
    }
    source_statuses: list[SourceSyncStatus] = []
    for source in session.scalars(select(SourceCatalog)).all():
        offset = freshness_offsets.get(source.id)
        if offset is None:
            sync_status = SourceSyncStatus(
                source=source,
                last_synced_at=now - timedelta(days=45),
                last_success_at=now - timedelta(days=45),
                status_note="Fuente estatica o de actualizacion lenta.",
            )
        else:
            sync_status = SourceSyncStatus(
                source=source,
                last_synced_at=now - timedelta(minutes=offset),
                last_success_at=now - timedelta(minutes=offset),
                status_note="Sincronizacion semilla.",
            )
        session.add(sync_status)
        source_statuses.append(sync_status)

    ensure_seed_sync_events(session)

    for municipality_name, points in seed["rainSeries"].items():
        municipality = municipalities_by_name[municipality_name]
        for index, point in enumerate(points):
            session.add(
                MunicipalityRainPoint(
                    municipality=municipality,
                    time_label=point["time"],
                    observed=point.get("observed"),
                    forecast=point.get("forecast"),
                    forecast_low=point.get("forecastLow"),
                    forecast_high=point.get("forecastHigh"),
                    forecast_range=point.get("forecastRange"),
                    sort_order=index,
                )
            )

    for municipality_name, overlays in seed["rainOverlays"].items():
        municipality = municipalities_by_name[municipality_name]
        for overlay in overlays:
            session.add(
                RainOverlay(
                    municipality=municipality,
                    bounds=overlay["bounds"],
                    bounds_geom=bounds_geometry_value(overlay["bounds"], dialect_name),
                    intensity=overlay["intensity"],
                )
            )

    ensure_seed_admin_user(session)

    previous_run = PredictionRun(
        started_at=now - timedelta(minutes=25),
        completed_at=now - timedelta(minutes=23),
        status="completed",
        model_version=settings.model_version,
        partial_data=True,
        notes="Semilla historica previa para calcular tendencias.",
    )
    latest_run = PredictionRun(
        started_at=now - timedelta(minutes=5),
        completed_at=now - timedelta(minutes=3),
        status="completed",
        model_version=settings.model_version,
        partial_data=True,
        notes="Semilla inicial alineada con el dashboard.",
    )
    session.add_all([previous_run, latest_run])
    session.flush()

    source_snapshot = build_source_snapshot(source_statuses)
    event_counts = {
        municipality.name: len(municipality.historical_events)
        for municipality in municipalities_by_name.values()
    }

    for item in seed["zones"]:
        zone = zones_by_id[item["id"]]
        delta = deterministic_delta(item["id"])
        previous_score = clamp(item["riskScore"] - delta, 0.08, 0.92)
        previous_drivers = {
            **item["drivers"],
            "rain_6h": max(0, item["drivers"]["rain_6h"] - 4),
            "rain_24h": max(0, item["drivers"]["rain_24h"] - 7),
            "rain_72h": max(0, item["drivers"]["rain_72h"] - 11),
        }
        previous_prediction = ZonePrediction(
            run=previous_run,
            zone=zone,
            risk_score=previous_score,
            confidence=item["confidence"],
            drivers=previous_drivers,
            risk_delta=0.0,
            trend="estable",
            source_snapshot=source_snapshot,
            created_at=previous_run.completed_at,
        )
        latest_prediction = ZonePrediction(
            run=latest_run,
            zone=zone,
            risk_score=item["riskScore"],
            confidence=item["confidence"],
            drivers=item["drivers"],
            risk_delta=round(item["riskScore"] - previous_score, 3),
            trend=trend_from_delta(item["riskScore"] - previous_score),
            source_snapshot=source_snapshot,
            created_at=latest_run.completed_at,
        )
        session.add_all([previous_prediction, latest_prediction])
        session.flush()

        municipality_name = zone.municipality.name
        risk_level = risk_level_from_score(latest_prediction.risk_score)
        stale_sources = [
            source_id
            for source_id, status in source_snapshot.items()
            if status in {"Retrasado", "Desactualizado"}
        ]
        session.add(
            ZoneExplanation(
                prediction=latest_prediction,
                mode="template",
                summary=build_summary(
                    zone.name,
                    municipality_name,
                    risk_text_from_level(risk_level),
                    latest_prediction.drivers,
                    event_counts[municipality_name],
                ),
                driver_chips=build_driver_chips(latest_prediction.drivers),
                suggestions=build_suggestions(
                    zone.name,
                    municipality_name,
                    risk_level,
                    [segment.name for segment in zone.road_segments],
                ),
                data_warnings=[
                    f"Fuente {source_id} con frescura imperfecta en esta corrida."
                    for source_id in stale_sources[:3]
                ],
                trace={
                    "model_version": settings.model_version,
                    "risk_level": risk_level,
                    "event_count": event_counts[municipality_name],
                },
                generated_at=latest_run.completed_at,
            )
        )
