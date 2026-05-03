from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import ApiError
from app.db.bootstrap import ensure_seed_admin_user
from app.db.spatial import (
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
    Zone,
    ZoneExplanation,
    ZoneOutcomeLabel,
    ZonePrediction,
    zone_road_segments,
)


DEFAULT_RUNTIME_SOURCES: tuple[dict[str, str], ...] = (
    {"id": "IDEAM", "label": "IDEAM lluvia observada", "category": "tiempo-real"},
    {"id": "SGC", "label": "SGC inventario MM", "category": "historico"},
    {"id": "UNGRD", "label": "UNGRD emergencias", "category": "tiempo-real"},
    {"id": "DANE", "label": "DANE catalogo territorial", "category": "historico"},
    {"id": "INVIAS", "label": "INVIAS corredores", "category": "infraestructura"},
)
STRUCTURAL_IMPORT_ADAPTER_KEY = "official.structural_bundle"
STRUCTURAL_IMPORT_TRANSPORT = "file_import"


class StructuralCatalogError(ValueError):
    pass


class StructuralSourceSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    label: str
    category: str
    source_url: str = Field(alias="sourceUrl")
    updated_at: datetime | None = Field(default=None, alias="updatedAt")
    note: str | None = None


class StructuralMunicipalitySpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    center: list[float]
    zoom: int
    source_id: str = Field(alias="sourceId")
    source_ref: str = Field(alias="sourceRef")


class StructuralRoadSegmentSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    municipality_id: str = Field(alias="municipalityId")
    name: str
    coords: list[list[float]]
    risk_level: str = Field(alias="riskLevel")
    length_km: float
    note: str
    source_id: str = Field(alias="sourceId")
    source_ref: str = Field(alias="sourceRef")


class StructuralZoneAssetsSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    road_segment_ids: list[str] = Field(alias="roadSegmentIds")


class StructuralZoneSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    municipality_id: str = Field(alias="municipalityId")
    name: str
    type: str
    centroid: list[float]
    polygon: list[list[float]]
    exposure: dict
    assets: StructuralZoneAssetsSpec
    source_id: str = Field(alias="sourceId")
    source_ref: str = Field(alias="sourceRef")


class StructuralCatalogBundle(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    version: str
    generated_at: datetime = Field(alias="generatedAt")
    sources: list[StructuralSourceSpec]
    municipalities: list[StructuralMunicipalitySpec]
    road_segments: list[StructuralRoadSegmentSpec] = Field(alias="roadSegments")
    zones: list[StructuralZoneSpec]


def is_official_source_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    return host == "gov.co" or host.endswith(".gov.co")


def _ordered_unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def load_structural_catalog_bundle(bundle_path: Path) -> StructuralCatalogBundle:
    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StructuralCatalogError(
            f"Structural catalog bundle was not found: {bundle_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise StructuralCatalogError(
            f"Structural catalog bundle is not valid JSON: {exc}"
        ) from exc

    try:
        bundle = StructuralCatalogBundle.model_validate(payload)
    except ValidationError as exc:
        raise StructuralCatalogError(
            f"Structural catalog bundle does not match the expected schema: {exc}"
        ) from exc

    source_ids = [source.id for source in bundle.sources]
    if len(source_ids) != len(set(source_ids)):
        raise StructuralCatalogError("Structural catalog bundle contains duplicate source ids.")
    if any(not is_official_source_url(source.source_url) for source in bundle.sources):
        invalid_urls = [
            source.source_url
            for source in bundle.sources
            if not is_official_source_url(source.source_url)
        ]
        raise StructuralCatalogError(
            "Structural catalog bundle contains non-official source URLs: "
            + ", ".join(invalid_urls)
        )

    municipality_ids = [item.id for item in bundle.municipalities]
    road_segment_ids = [item.id for item in bundle.road_segments]
    zone_ids = [item.id for item in bundle.zones]
    if len(municipality_ids) != len(set(municipality_ids)):
        raise StructuralCatalogError("Structural catalog bundle contains duplicate municipalities.")
    if len(road_segment_ids) != len(set(road_segment_ids)):
        raise StructuralCatalogError("Structural catalog bundle contains duplicate road segments.")
    if len(zone_ids) != len(set(zone_ids)):
        raise StructuralCatalogError("Structural catalog bundle contains duplicate zones.")

    known_sources = set(source_ids)
    known_municipalities = set(municipality_ids)
    known_road_segments = set(road_segment_ids)
    for municipality in bundle.municipalities:
        if municipality.source_id not in known_sources:
            raise StructuralCatalogError(
                f"Municipality '{municipality.id}' references unknown source '{municipality.source_id}'."
            )
    for road_segment in bundle.road_segments:
        if road_segment.source_id not in known_sources:
            raise StructuralCatalogError(
                f"Road segment '{road_segment.id}' references unknown source '{road_segment.source_id}'."
            )
        if road_segment.municipality_id not in known_municipalities:
            raise StructuralCatalogError(
                f"Road segment '{road_segment.id}' references unknown municipality '{road_segment.municipality_id}'."
            )
    for zone in bundle.zones:
        if zone.source_id not in known_sources:
            raise StructuralCatalogError(
                f"Zone '{zone.id}' references unknown source '{zone.source_id}'."
            )
        if zone.municipality_id not in known_municipalities:
            raise StructuralCatalogError(
                f"Zone '{zone.id}' references unknown municipality '{zone.municipality_id}'."
            )
        missing_roads = [
            road_segment_id
            for road_segment_id in zone.assets.road_segment_ids
            if road_segment_id not in known_road_segments
        ]
        if missing_roads:
            raise StructuralCatalogError(
                f"Zone '{zone.id}' references unknown road segments: {', '.join(missing_roads)}."
            )

    return bundle


def _upsert_source_catalog(
    session: Session, bundle_sources: list[StructuralSourceSpec]
) -> dict[str, SourceCatalog]:
    merged_sources: dict[str, dict[str, str]] = {
        item["id"]: dict(item) for item in DEFAULT_RUNTIME_SOURCES
    }
    for source in bundle_sources:
        merged_sources[source.id] = {
            "id": source.id,
            "label": source.label,
            "category": source.category,
        }

    records: dict[str, SourceCatalog] = {}
    for source_id, item in merged_sources.items():
        record = session.get(SourceCatalog, source_id)
        if record is None:
            record = SourceCatalog(
                id=item["id"],
                label=item["label"],
                category=item["category"],
            )
            session.add(record)
        else:
            record.label = item["label"]
            record.category = item["category"]
        records[source_id] = record
    return records


def purge_runtime_for_structural_reimport(session: Session) -> None:
    for model in (
        ZoneExplanation,
        ZonePrediction,
        ZoneOutcomeLabel,
        PredictionRun,
        MunicipalityRainPoint,
        RainOverlay,
        HistoricalEvent,
        UngrdRecord,
    ):
        session.execute(delete(model))

    session.execute(delete(zone_road_segments))
    session.execute(delete(Zone))
    session.execute(delete(RoadSegment))
    session.execute(delete(Municipality))


def _record_structural_source_import(
    session: Session,
    source: SourceCatalog,
    source_spec: StructuralSourceSpec,
    *,
    processed_records: int,
    imported_at: datetime,
    bundle: StructuralCatalogBundle,
) -> None:
    status = source.sync_status
    if status is None:
        status = SourceSyncStatus(source=source)
        session.add(status)
    status.last_synced_at = imported_at
    status.last_success_at = source_spec.updated_at or imported_at
    status.last_error = None
    status.status_note = source_spec.note or "Imported from official structural catalog bundle."

    details = {
        "structural_import": True,
        "source_url": source_spec.source_url,
        "bundle_version": bundle.version,
        "bundle_generated_at": bundle.generated_at.isoformat(),
        "processed_records": processed_records,
    }
    if source_spec.updated_at is not None:
        details["provider_updated_at"] = source_spec.updated_at.isoformat()

    session.add(
        SourceSyncEvent(
            source=source,
            origin="manual",
            adapter_key=STRUCTURAL_IMPORT_ADAPTER_KEY,
            transport=STRUCTURAL_IMPORT_TRANSPORT,
            status="completed",
            processed_records=processed_records,
            started_at=imported_at,
            completed_at=imported_at,
            message=source_spec.note or "Imported from official structural catalog bundle.",
            details=details,
        )
    )


def import_structural_catalog_bundle(
    session: Session, bundle: StructuralCatalogBundle
) -> dict[str, int]:
    imported_at = datetime.now(timezone.utc).replace(microsecond=0)
    source_records = _upsert_source_catalog(session, bundle.sources)

    purge_runtime_for_structural_reimport(session)

    dialect_name = session_dialect_name(session)
    municipalities_by_id: dict[str, Municipality] = {}
    for item in bundle.municipalities:
        municipality = Municipality(
            id=item.id,
            name=item.name,
            center=item.center,
            zoom=item.zoom,
            source_id=item.source_id,
            source_ref=item.source_ref,
        )
        session.add(municipality)
        municipalities_by_id[item.id] = municipality

    road_segments_by_id: dict[str, RoadSegment] = {}
    for item in bundle.road_segments:
        road_segment = RoadSegment(
            id=item.id,
            municipality=municipalities_by_id[item.municipality_id],
            name=item.name,
            coords=item.coords,
            coords_geom=linestring_geometry_value(item.coords, dialect_name),
            risk_level=item.risk_level,
            length_km=item.length_km,
            note=item.note,
            source_id=item.source_id,
            source_ref=item.source_ref,
        )
        session.add(road_segment)
        road_segments_by_id[item.id] = road_segment

    for item in bundle.zones:
        zone = Zone(
            id=item.id,
            municipality=municipalities_by_id[item.municipality_id],
            name=item.name,
            type=item.type,
            centroid=item.centroid,
            centroid_geom=point_geometry_value(item.centroid, dialect_name),
            polygon=item.polygon,
            polygon_geom=polygon_geometry_value(item.polygon, dialect_name),
            exposure=item.exposure,
            is_active=True,
            source_id=item.source_id,
            source_ref=item.source_ref,
        )
        for road_segment_id in item.assets.road_segment_ids:
            zone.road_segments.append(road_segments_by_id[road_segment_id])
        session.add(zone)

    ensure_seed_admin_user(session)

    processed_by_source: dict[str, int] = {}
    for item in bundle.municipalities:
        processed_by_source[item.source_id] = processed_by_source.get(item.source_id, 0) + 1
    for item in bundle.road_segments:
        processed_by_source[item.source_id] = processed_by_source.get(item.source_id, 0) + 1
    for item in bundle.zones:
        processed_by_source[item.source_id] = processed_by_source.get(item.source_id, 0) + 1

    for source in bundle.sources:
        _record_structural_source_import(
            session,
            source_records[source.id],
            source,
            processed_records=processed_by_source.get(source.id, 0),
            imported_at=imported_at,
            bundle=bundle,
        )

    session.flush()

    return {
        "municipalities": len(bundle.municipalities),
        "road_segments": len(bundle.road_segments),
        "zones": len(bundle.zones),
    }


def _latest_sync_event(session: Session, source_id: str) -> SourceSyncEvent | None:
    statement = (
        select(SourceSyncEvent)
        .where(SourceSyncEvent.source_id == source_id)
        .order_by(SourceSyncEvent.completed_at.desc(), SourceSyncEvent.id.desc())
        .limit(1)
    )
    return session.scalar(statement)


def _entity_provenance_errors(
    session: Session, model: type[Municipality] | type[Zone] | type[RoadSegment], label: str
) -> list[str]:
    rows = session.execute(select(model.id, model.source_id, model.source_ref)).all()
    if not rows:
        return [f"no {label} records are loaded"]
    invalid_ids = [
        str(row[0])
        for row in rows
        if not str(row[1] or "").strip() or not str(row[2] or "").strip()
    ]
    if not invalid_ids:
        return []
    return [
        f"{label} records are missing provenance fields: {', '.join(invalid_ids[:5])}"
        + ("..." if len(invalid_ids) > 5 else "")
    ]


def collect_structural_catalog_violations(session: Session) -> list[str]:
    violations: list[str] = []
    violations.extend(_entity_provenance_errors(session, Municipality, "municipality"))
    violations.extend(_entity_provenance_errors(session, Zone, "zone"))
    violations.extend(_entity_provenance_errors(session, RoadSegment, "road segment"))

    used_source_ids = _ordered_unique(
        [
            source_id
            for source_id in (
                list(session.scalars(select(Municipality.source_id)).all())
                + list(session.scalars(select(Zone.source_id)).all())
                + list(session.scalars(select(RoadSegment.source_id)).all())
            )
            if source_id
        ]
    )
    if not used_source_ids:
        violations.append("no structural provenance sources are registered")
        return violations

    available_sources = {
        source.id: source for source in session.scalars(select(SourceCatalog)).all()
    }
    missing_sources = [source_id for source_id in used_source_ids if source_id not in available_sources]
    if missing_sources:
        violations.append(
            "structural provenance sources are missing from source_catalog: "
            + ", ".join(missing_sources)
        )

    for source_id in used_source_ids:
        latest_event = _latest_sync_event(session, source_id)
        if latest_event is None:
            violations.append(f"source '{source_id}' has no structural import event")
            continue
        details = latest_event.details or {}
        if latest_event.transport == "seed" or "seed" in latest_event.adapter_key.lower():
            violations.append(f"source '{source_id}' is still backed by seed transport")
        if not details.get("structural_import"):
            violations.append(f"source '{source_id}' has not been imported as structural catalog")
        source_url = details.get("source_url")
        if not isinstance(source_url, str) or not is_official_source_url(source_url):
            violations.append(
                f"source '{source_id}' does not have an official .gov.co provenance URL"
            )

    return violations


def ensure_real_data_structural_catalog(
    session: Session, *, for_api: bool
) -> None:
    settings = get_settings()
    if not settings.real_data_only:
        return

    violations = collect_structural_catalog_violations(session)
    if not violations:
        return

    detail = "; ".join(violations)
    if for_api:
        raise ApiError(
            409,
            "structural_catalog_not_official",
            "The structural catalog is not backed by official provenance while REAL_DATA_ONLY is enabled: "
            + detail
            + ". Import an official structural bundle before serving or scoring data.",
        )
    raise RuntimeError(
        "REAL_DATA_ONLY is enabled but the structural catalog is not backed by official provenance: "
        + detail
        + ". Import an official structural bundle before starting the runtime."
    )
