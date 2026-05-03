"""Import the canonical landslide inventory CSV into the backend database.

This is the final stage of the landslide-inventory pipeline. It reads the
``colombia_landslide_events_v1.csv`` produced by ``merge_inventory.py`` and
writes:

1. ``Municipality`` rows for every municipality that does not yet exist,
   using a small built-in department-centroid gazetteer for coarse coordinates.
2. One default ``Zone`` per new municipality (``<muni-slug>-cab``, type
   ``Cabecera municipal``), whose polygon is a small box around the centroid.
3. ``ZoneOutcomeLabel`` rows for every UNGRD row in the CSV, keyed by
   ``(zone_id, observed_at, source)`` — matching the uniqueness constraint the
   domain model enforces. The target score is derived from the severity
   classifier the normalizer already applied:

        fatal    -> 1.00
        severe   -> 0.85
        moderate -> 0.60
        minor    -> 0.35

SIMMA rows (``record_quality == "spatial_prior_only"``) are **skipped** — they
have no ``observed_at`` and the backend enforces NOT NULL on that column. They
will be imported via a separate spatial-prior flow that is not yet built.

Usage:

    uv run python scripts/landslide_inventory/import_landslide_inventory.py \\
        --csv data/inventory/02_final/colombia_landslide_events_v1.csv

    # Validate schema without writing:
    uv run python scripts/landslide_inventory/import_landslide_inventory.py \\
        --csv data/inventory/02_final/colombia_landslide_events_v1.csv --dry-run

The script uses the same SQLAlchemy session the backend uses (via
``app.db.session``). It honours ``DATABASE_URL`` from the environment, so it
works against a throw-away SQLite for testing or against the shared dev DB.

"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


# Repo root lives three levels up from this file, so we can import the backend
# package when invoked as a plain script.
_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPT_DIR.parent.parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _lib.dates import parse_iso_datetime as _shared_parse_iso  # noqa: E402
from _lib.geo import extract_first_polygon_ring as _shared_extract_first_polygon_ring  # noqa: E402
from _lib.geo import polygon_box as _make_polygon_box  # noqa: E402
from _lib.slug import normalize_admin_name as _shared_normalize_admin_name  # noqa: E402
from _lib.slug import slugify as _shared_slugify  # noqa: E402
from _lib.slug import truncate_with_hash as _shared_truncate  # noqa: E402


DEPARTMENT_CENTROIDS: dict[str, tuple[float, float]] = {
    # Best-effort department centroids (lat, lon). Used to coarsely place
    # municipalities whose exact centroid is unknown. Good enough for zone
    # bookkeeping; precise geocoding is refined later by SIMMA spatial priors.
    "AMAZONAS": (-1.4, -71.5),
    "ANTIOQUIA": (7.0, -75.5),
    "ARAUCA": (6.7, -71.0),
    "ATLANTICO": (10.7, -75.0),
    "BOLIVAR": (8.7, -74.5),
    "BOYACA": (5.4, -72.9),
    "CALDAS": (5.3, -75.3),
    "CAQUETA": (1.0, -74.0),
    "CASANARE": (5.3, -72.0),
    "CAUCA": (2.7, -76.8),
    "CESAR": (9.3, -73.5),
    "CHOCO": (6.0, -77.0),
    "CORDOBA": (8.8, -75.7),
    "CUNDINAMARCA": (5.0, -74.0),
    "DISTRITO CAPITAL": (4.66, -74.08),
    "BOGOTA D.C.": (4.66, -74.08),
    "BOGOTA D.C": (4.66, -74.08),
    "GUAINIA": (2.6, -68.5),
    "GUAVIARE": (2.0, -72.5),
    "HUILA": (2.6, -75.5),
    "LA GUAJIRA": (11.5, -72.5),
    "MAGDALENA": (10.1, -74.2),
    "META": (3.5, -73.0),
    "NARINO": (1.4, -77.9),
    "NARIÑO": (1.4, -77.9),
    "NORTE DE SANTANDER": (8.1, -72.9),
    "PUTUMAYO": (0.6, -76.0),
    "QUINDIO": (4.5, -75.7),
    "RISARALDA": (5.1, -75.9),
    "SAN ANDRES Y PROVIDENCIA": (12.6, -81.7),
    "SANTANDER": (6.8, -73.3),
    "SUCRE": (9.0, -75.0),
    "TOLIMA": (4.1, -75.2),
    "VALLE DEL CAUCA": (3.8, -76.5),
    "VAUPES": (0.5, -70.5),
    "VICHADA": (5.0, -69.5),
}

MUNICIPALITY_CENTROIDS: dict[str, tuple[float, float]] = {
    # High-recurrence municipalities from the merged inventory — a handful of
    # precise centroids so their zones are not lumped at the department centroid.
    "MOCOA": (1.147, -76.648),
    "PASTO": (1.214, -77.281),
    "POPAYAN": (2.444, -76.614),
    "MEDELLIN": (6.244, -75.574),
    "BOGOTA": (4.66, -74.08),
    "BOGOTA D.C.": (4.66, -74.08),
    "PEREIRA": (4.813, -75.694),
    "MANIZALES": (5.069, -75.521),
    "BUCARAMANGA": (7.119, -73.122),
    "VILLAVICENCIO": (4.142, -73.626),
    "IBAGUE": (4.438, -75.232),
    "CUCUTA": (7.893, -72.508),
    "CARTAGENA": (10.391, -75.479),
    "BARRANQUILLA": (10.963, -74.796),
    "CALI": (3.437, -76.522),
    "DOSQUEBRADAS": (4.836, -75.672),
    "MARSELLA": (4.937, -75.735),
    "MISTRATO": (5.295, -75.884),
    "ARMENIA": (4.536, -75.681),
    "NEIVA": (2.925, -75.283),
    "SAN ANDRES DE TUMACO": (1.800, -78.770),
    "VALLEDUPAR": (10.472, -73.248),
    "MONTERIA": (8.749, -75.883),
    "TULUA": (4.080, -76.200),
    "OCANA": (8.238, -73.356),
    "OCAÑA": (8.238, -73.356),
    "SAMANIEGO": (1.335, -77.595),
    "LA VEGA": (5.000, -74.335),
    "LA UNION": (1.601, -77.132),
    "SAN FRANCISCO": (1.177, -76.880),
    "TOLEDO": (7.313, -72.491),
    "CHAPARRAL": (3.722, -75.483),
    "INZA": (2.550, -76.065),
    "EL TAMBO": (2.452, -76.810),
    "ARGELIA": (4.730, -75.945),
    "BALBOA": (1.633, -77.213),
    "SUCRE": (8.815, -74.721),
    "VENECIA": (5.965, -75.741),
    "LA PALMA": (5.362, -74.389),
    "PACHO": (5.132, -74.159),
    "SUAREZ": (2.960, -76.692),
    "SANTA BARBARA": (5.875, -75.564),
    "BUCARASICA": (8.042, -72.862),
    "LA ESPERANZA": (7.638, -72.671),
}

SEVERITY_TO_TARGET_SCORE = {
    "fatal": 1.0,
    "severe": 0.85,
    "moderate": 0.6,
    "minor": 0.35,
}

ZONE_POLYGON_HALF_WIDTH_DEG = 0.02  # ~2.2 km half-side; fine for a v1 zone box.
SPATIAL_PRIOR_ANCHOR_DATE = "2000-01-01"
# Fixed anchor date for date-less SIMMA spatial priors. Chosen so they
# contribute to `zone_event_count` / `municipality_event_count` (no date
# filter) but fall outside any label's 3-year lookback window and therefore
# never pollute `recent_zone_event_count`.


def _slugify(text: str) -> str:
    """Turn 'San José' into 'san-jose' — ASCII-safe, hyphenated, lower-case."""
    return _shared_slugify(text)


def _truncate(value: str, *, limit: int) -> str:
    return _shared_truncate(value, limit=limit)


def _muni_id(department: str, municipality: str) -> str:
    raw = f"{_slugify(department)}-{_slugify(municipality)}"
    return _truncate(raw, limit=32)


def _zone_id(muni_id: str) -> str:
    return _truncate(f"{muni_id}-cab", limit=32)


def _centroid_for(municipality: str, department: str) -> tuple[float, float]:
    muni_key = _normalize_admin_name(municipality)
    if muni_key in MUNICIPALITY_CENTROIDS:
        return MUNICIPALITY_CENTROIDS[muni_key]
    dep_key = _normalize_admin_name(department)
    if dep_key in DEPARTMENT_CENTROIDS:
        return DEPARTMENT_CENTROIDS[dep_key]
    # Geographic centre of Colombia as last-resort fallback.
    return (4.57, -74.30)


def _polygon_box(centroid: tuple[float, float]) -> list[list[float]]:
    return _make_polygon_box(centroid, half_width_deg=ZONE_POLYGON_HALF_WIDTH_DEG)


def _normalize_admin_name(raw: str | None) -> str:
    """ASCII-upper, diacritics stripped, replacement chars dropped.

    Colombian UNGRD and SGC municipal names disagree on accents (``APARTADÓ``
    vs ``APARTADO``) and a handful of UNGRD rows also carry U+FFFD from
    upstream encoding loss (``NARI\ufffdO`` when it should be ``NARIÑO``).
    This helper collapses both to a safe ASCII key so the gazetteer join
    actually hits.
    """
    return _shared_normalize_admin_name(raw)


def _load_sgc_gazetteer(path: Path) -> dict[tuple[str, str], dict]:
    """Index SGC Capas_Generales/Municipios polygons by (name, dept).

    Returns a dict keyed by ``(ascii-upper muni name, ascii-upper dept name)``
    → {polygon: [[lat, lon], ...], centroid: (lat, lon), divipola: str,
    area_km: float | None}. Missing file → empty dict, callers fall back.
    """
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    index: dict[tuple[str, str], dict] = {}
    for feature in payload.get("features") or []:
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        name = _normalize_admin_name(props.get("NOMBRE_ENT"))
        dept = _normalize_admin_name(props.get("DEPARTAMEN"))
        if not name or not dept:
            continue
        rings = _extract_first_polygon_ring(geom)
        if not rings:
            continue
        # Convert GeoJSON [lon, lat] to [lat, lon] to match the seed convention.
        polygon_latlon = [[pt[1], pt[0]] for pt in rings]
        # Centroid: simple mean of outer-ring vertices. Good enough for a
        # municipal polygon; we do not need geodesic precision here.
        centroid_lat = sum(pt[0] for pt in polygon_latlon) / len(polygon_latlon)
        centroid_lon = sum(pt[1] for pt in polygon_latlon) / len(polygon_latlon)
        divipola: str | None = None
        cod_dept = props.get("COD_DEPART")
        cod_muni = props.get("COD_MUNICI")
        if cod_dept is not None and cod_muni is not None:
            try:
                divipola = f"{int(cod_dept):02d}{int(cod_muni):03d}"
            except (TypeError, ValueError):
                divipola = None
        area_km = props.get("AREA_KM")
        index[(name, dept)] = {
            "polygon": polygon_latlon,
            "centroid": (centroid_lat, centroid_lon),
            "divipola": divipola,
            "area_km": float(area_km) if isinstance(area_km, (int, float)) else None,
        }
    return index


def _extract_first_polygon_ring(geometry: dict) -> list[list[float]]:
    """Return the outer ring of the first polygon in a GeoJSON geometry.

    For MultiPolygon we keep only the first part; that is enough for the v1
    coarse zone polygon. Picking the largest polygon by area could be a nicer
    improvement later.
    """
    return _shared_extract_first_polygon_ring(geometry)


def _parse_iso(value: str | None) -> datetime | None:
    return _shared_parse_iso(value)


def _severity_to_target_score(severity: str | None) -> float:
    key = (severity or "").strip().lower()
    return SEVERITY_TO_TARGET_SCORE.get(key, 0.5)


def run_import(
    *,
    csv_path: Path,
    dry_run: bool,
    sgc_municipios_path: Path | None = None,
    emit_historical_events: bool = True,
) -> dict:
    from datetime import date as date_type

    from app.core.config import get_settings
    from app.db.session import get_engine, reset_engine_cache
    from app.db.spatial import point_geometry_value, session_dialect_name
    from app.models import Municipality, Zone
    from app.models.domain import HistoricalEvent, ZoneOutcomeLabel
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    get_settings.cache_clear()
    reset_engine_cache()
    engine = get_engine()

    sgc_gazetteer: dict[tuple[str, str], dict] = {}
    if sgc_municipios_path is not None:
        sgc_gazetteer = _load_sgc_gazetteer(sgc_municipios_path)
        if sgc_gazetteer:
            print(
                f"Loaded {len(sgc_gazetteer)} SGC municipal polygons "
                f"from {sgc_municipios_path}"
            )

    stats = {
        "csv_rows": 0,
        "skipped_no_observed_at": 0,
        "considered": 0,
        "municipalities_created": 0,
        "municipalities_reused": 0,
        "zones_created": 0,
        "zones_reused": 0,
        "labels_inserted": 0,
        "labels_duplicate": 0,
        "skipped_by_source": {},
        "sgc_polygon_hits": 0,
        "sgc_polygon_misses": 0,
        "historical_events_inserted": 0,
        "historical_events_duplicate": 0,
        "spatial_priors_inserted": 0,
        "spatial_priors_skipped_no_municipality": 0,
        "municipalities_unknown_centroid": [],
    }

    with Session(engine) as session, session.begin():
        municipalities_by_id: dict[str, Municipality] = {
            muni.id: muni
            for muni in session.scalars(select(Municipality)).all()
        }
        municipalities_by_name: dict[str, Municipality] = {
            muni.name.strip().upper(): muni for muni in municipalities_by_id.values()
        }
        all_zones = list(session.scalars(select(Zone)).all())
        zones_by_id: dict[str, Zone] = {zone.id: zone for zone in all_zones}
        zones_by_municipality_id: dict[str, list[Zone]] = {}
        for zone in all_zones:
            zones_by_municipality_id.setdefault(zone.municipality_id, []).append(
                zone
            )

        def _pick_zone(muni_id: str) -> Zone | None:
            """Return the preferred existing zone for a municipality, or None.

            Prefer "real" zones (anything whose id does not end in ``-cab``)
            over auto-generated ``<muni>-cab`` placeholders, so labels attach
            to feature-equipped seed zones when they exist. Within each
            bucket, the zone with the lexicographically smallest id wins —
            deterministic and reproducible.
            """
            bucket = zones_by_municipality_id.get(muni_id) or []
            if not bucket:
                return None
            real = [z for z in bucket if not z.id.endswith("-cab")]
            if real:
                return sorted(real, key=lambda z: z.id)[0]
            return sorted(bucket, key=lambda z: z.id)[0]
        def _key_for(
            zone_id: str, observed_at: datetime, source: str
        ) -> tuple[str, str, str]:
            # SQLite strips the tzinfo when the column is ``DateTime(timezone=True)``,
            # so a stored row and a freshly parsed row from CSV otherwise don't
            # match. Normalize both sides to a naive-UTC ISO string.
            if observed_at.tzinfo is not None:
                observed_at = observed_at.astimezone(timezone.utc).replace(
                    tzinfo=None
                )
            return (zone_id, observed_at.isoformat(timespec="seconds"), source)

        existing_label_keys: set[tuple[str, str, str]] = {
            _key_for(label.zone_id, label.observed_at, label.source)
            for label in session.scalars(select(ZoneOutcomeLabel)).all()
        }
        existing_historical_event_ids: set[str] = (
            {
                event.id
                for event in session.scalars(select(HistoricalEvent)).all()
            }
            if emit_historical_events
            else set()
        )

        dialect = session_dialect_name(session) if emit_historical_events else ""
        now = datetime.now(timezone.utc).replace(microsecond=0)

        spatial_prior_anchor = date_type.fromisoformat(SPATIAL_PRIOR_ANCHOR_DATE)

        def _ensure_municipality_for_row(
            municipality_name: str,
            department: str,
            row_source: str,
            row_divipola: str | None,
        ) -> Municipality | None:
            """Reuse or create a Municipality, matching the main-loop logic."""
            muni_name_key = municipality_name.upper()
            existing = municipalities_by_name.get(muni_name_key)
            if existing is not None:
                return existing
            sgc = sgc_gazetteer.get(
                (
                    _normalize_admin_name(municipality_name),
                    _normalize_admin_name(department),
                )
            )
            muni_id = _muni_id(department, municipality_name)
            if muni_id in municipalities_by_id and (
                municipalities_by_id[muni_id].name.strip().upper()
                != muni_name_key
            ):
                suffix = _slugify(municipality_name)[:6]
                muni_id = _truncate(f"{muni_id}-{suffix}", limit=32)
            centroid = (
                sgc["centroid"]
                if sgc is not None
                else _centroid_for(municipality_name, department)
            )
            if sgc is not None:
                stats["sgc_polygon_hits"] += 1
            else:
                stats["sgc_polygon_misses"] += 1
                muni_norm = _normalize_admin_name(municipality_name)
                dept_norm = _normalize_admin_name(department)
                if (
                    muni_norm not in MUNICIPALITY_CENTROIDS
                    and dept_norm not in DEPARTMENT_CENTROIDS
                ):
                    stats["municipalities_unknown_centroid"].append(
                        f"{municipality_name} ({department})"
                    )
            source_ref = (
                (sgc.get("divipola") if sgc else None)
                or row_divipola
                or None
            )
            new_muni = Municipality(
                id=muni_id,
                name=municipality_name,
                center=[centroid[0], centroid[1]],
                zoom=11,
                source_id=row_source,
                source_ref=source_ref,
            )
            if not dry_run:
                session.add(new_muni)
                session.flush()
            municipalities_by_id[muni_id] = new_muni
            municipalities_by_name[muni_name_key] = new_muni
            stats["municipalities_created"] += 1
            return new_muni

        with open(csv_path, "r", encoding="utf-8", newline="") as src:
            reader = csv.DictReader(src)
            row_index_counter = 0
            for row in reader:
                row_index_counter += 1
                stats["csv_rows"] += 1
                observed_at = _parse_iso(row.get("observed_at"))
                if observed_at is None:
                    # Spatial priors (SIMMA): no date, but still useful as a
                    # historical-density signal. When we have a municipality
                    # + coords, emit a date-anchored HistoricalEvent and stop.
                    # Anything else (genuinely missing data) is just counted.
                    row_source_inner = (row.get("source") or "UNKNOWN").strip()
                    row_muni_name = (row.get("municipality") or "").strip()
                    row_department = (row.get("department") or "").strip()
                    has_coords = bool(
                        row.get("latitude") and row.get("longitude")
                    )
                    if (
                        emit_historical_events
                        and row.get("record_quality") != "spatial_prior_only"
                        and row_muni_name
                        and has_coords
                    ):
                        muni = _ensure_municipality_for_row(
                            row_muni_name,
                            row_department or "UNKNOWN",
                            row_source_inner,
                            row.get("divipola"),
                        )
                        if muni is not None:
                            raw_event_id = (
                                row.get("event_id")
                                or f"prior-{row_index_counter}"
                            )
                            prior_id = _truncate(str(raw_event_id), limit=32)
                            if prior_id not in existing_historical_event_ids:
                                coords_latlon = [
                                    float(row["latitude"]),
                                    float(row["longitude"]),
                                ]
                                session.add(
                                    HistoricalEvent(
                                        id=prior_id,
                                        municipality_id=muni.id,
                                        date=spatial_prior_anchor,
                                        severity=(
                                            row.get("severity") or "unknown"
                                        ),
                                        type=(
                                            row.get("movement_type") or "unknown"
                                        ),
                                        coords=coords_latlon,
                                        coords_geom=point_geometry_value(
                                            coords_latlon, dialect
                                        ),
                                        source=row_source_inner,
                                    )
                                ) if not dry_run else None
                                existing_historical_event_ids.add(prior_id)
                                stats["spatial_priors_inserted"] += 1
                            else:
                                stats["historical_events_duplicate"] += 1
                        else:
                            stats["spatial_priors_skipped_no_municipality"] += 1
                    else:
                        stats["spatial_priors_skipped_no_municipality"] += (
                            1 if row.get("source") == "SGC_SIMMA" else 0
                        )
                    stats["skipped_no_observed_at"] += 1
                    stats["skipped_by_source"][row_source_inner] = (
                        stats["skipped_by_source"].get(row_source_inner, 0)
                        + 1
                    )
                    continue

                stats["considered"] += 1

                row_source = (row.get("source") or "UNKNOWN").strip()
                municipality_name = (row.get("municipality") or "UNKNOWN").strip()
                department = (row.get("department") or "UNKNOWN").strip()

                muni_name_key = municipality_name.upper()
                dept_key = department.upper()
                muni = municipalities_by_name.get(muni_name_key)
                sgc_entry = sgc_gazetteer.get(
                    (
                        _normalize_admin_name(municipality_name),
                        _normalize_admin_name(department),
                    )
                )
                if muni is None:
                    muni_id = _muni_id(department, municipality_name)
                    # collision-avoid: if the ID collides with a pre-existing
                    # municipality whose name doesn't match, suffix the ID.
                    if muni_id in municipalities_by_id and (
                        municipalities_by_id[muni_id].name.strip().upper() != muni_name_key
                    ):
                        suffix = _slugify(municipality_name)[:6]
                        muni_id = _truncate(f"{muni_id}-{suffix}", limit=32)
                    if sgc_entry is not None:
                        centroid = sgc_entry["centroid"]
                        stats["sgc_polygon_hits"] += 1
                    else:
                        centroid = _centroid_for(municipality_name, department)
                        stats["sgc_polygon_misses"] += 1
                        muni_norm = _normalize_admin_name(municipality_name)
                        dept_norm = _normalize_admin_name(department)
                        if (
                            muni_norm not in MUNICIPALITY_CENTROIDS
                            and dept_norm not in DEPARTMENT_CENTROIDS
                        ):
                            stats["municipalities_unknown_centroid"].append(
                                f"{municipality_name} ({department})"
                            )
                    source_ref = (
                        (sgc_entry.get("divipola") if sgc_entry else None)
                        or row.get("divipola")
                        or None
                    )
                    muni = Municipality(
                        id=muni_id,
                        name=municipality_name,
                        center=[centroid[0], centroid[1]],
                        zoom=11,
                        source_id=row_source,
                        source_ref=source_ref,
                    )
                    if not dry_run:
                        session.add(muni)
                        session.flush()
                    municipalities_by_id[muni_id] = muni
                    municipalities_by_name[muni_name_key] = muni
                    stats["municipalities_created"] += 1
                else:
                    stats["municipalities_reused"] += 1

                zone = _pick_zone(muni.id)
                if zone is None:
                    zone_id = _zone_id(muni.id)
                    centroid = tuple(muni.center) if muni.center else _centroid_for(
                        municipality_name, department
                    )
                    if sgc_entry is not None:
                        polygon = sgc_entry["polygon"]
                        area_km = sgc_entry.get("area_km")
                    else:
                        polygon = _polygon_box(centroid)
                        area_km = None
                    source_ref = (
                        (sgc_entry.get("divipola") if sgc_entry else None)
                        or row.get("divipola")
                        or None
                    )
                    zone = Zone(
                        id=zone_id,
                        municipality_id=muni.id,
                        name=f"Cabecera {municipality_name.title()}",
                        type="Cabecera municipal",
                        centroid=[centroid[0], centroid[1]],
                        polygon=polygon,
                        exposure={
                            "population_estimate": None,
                            "households_estimate": None,
                            "auto_generated": True,
                            "area_km": area_km,
                            "has_sgc_polygon": sgc_entry is not None,
                        },
                        is_active=True,
                        source_id=row_source,
                        source_ref=source_ref,
                    )
                    if not dry_run:
                        session.add(zone)
                        session.flush()
                    zones_by_id[zone_id] = zone
                    zones_by_municipality_id.setdefault(muni.id, []).append(zone)
                    stats["zones_created"] += 1
                else:
                    stats["zones_reused"] += 1

                source_tag = (
                    f"{row_source}:{row.get('event_id') or 'unknown'}"
                )
                key = _key_for(zone.id, observed_at, source_tag)
                if key in existing_label_keys:
                    stats["labels_duplicate"] += 1
                    continue

                evidence = {
                    "severity": row.get("severity"),
                    "movement_type": row.get("movement_type"),
                    "department": department,
                    "divipola": row.get("divipola"),
                    "deaths": row.get("deaths") or None,
                    "injured": row.get("injured") or None,
                    "homes_destroyed": row.get("homes_destroyed") or None,
                    "homes_damaged": row.get("homes_damaged") or None,
                    "description": row.get("description"),
                    "source_url": row.get("source_url"),
                    "record_quality": row.get("record_quality"),
                    "inventory_version": "v1",
                }
                label = ZoneOutcomeLabel(
                    zone_id=zone.id,
                    feature_run_id=None,
                    observed_at=observed_at,
                    target_score=_severity_to_target_score(row.get("severity")),
                    source=source_tag,
                    status="confirmed",
                    notes=None,
                    evidence=evidence,
                    created_at=now,
                    updated_at=now,
                )
                if not dry_run:
                    session.add(label)
                existing_label_keys.add(key)
                stats["labels_inserted"] += 1

                if emit_historical_events:
                    raw_event_id = row.get("event_id") or f"auto-{row_index_counter}"
                    historical_id = _truncate(str(raw_event_id), limit=32)
                    if historical_id in existing_historical_event_ids:
                        stats["historical_events_duplicate"] += 1
                    else:
                        coords_latlon = (
                            list(zone.centroid)
                            if zone.centroid
                            else [0.0, 0.0]
                        )
                        historical = HistoricalEvent(
                            id=historical_id,
                            municipality_id=muni.id,
                            date=observed_at.date(),
                            severity=(row.get("severity") or "unknown"),
                            type=(row.get("movement_type") or "unknown"),
                            coords=coords_latlon,
                            coords_geom=point_geometry_value(
                                coords_latlon, dialect
                            ),
                            source=row_source,
                        )
                        if not dry_run:
                            session.add(historical)
                        existing_historical_event_ids.add(historical_id)
                        stats["historical_events_inserted"] += 1

        if dry_run:
            session.rollback()

    # De-duplicate the unknown-centroid list and cap it for readability.
    unique_unknown = sorted(set(stats["municipalities_unknown_centroid"]))
    stats["municipalities_unknown_centroid"] = unique_unknown[:20]
    stats["municipalities_unknown_centroid_count"] = len(unique_unknown)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to colombia_landslide_events_v1.csv.",
    )
    parser.add_argument(
        "--sgc-municipios",
        type=Path,
        default=Path(
            "data/inventory/00_raw/simma/boundaries/municipios.geojson"
        ),
        help=(
            "Optional SGC Capas_Generales Municipios GeoJSON. When present, "
            "auto-created zones use the real municipal polygon and DIVIPOLA "
            "from SGC instead of the hardcoded 0.02° box + gazetteer fallback."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse, validate, and count — do not commit to the database.",
    )
    args = parser.parse_args(argv)

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1

    stats = run_import(
        csv_path=args.csv,
        dry_run=args.dry_run,
        sgc_municipios_path=args.sgc_municipios,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
