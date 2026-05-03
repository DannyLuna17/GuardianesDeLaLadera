from __future__ import annotations

import sys

from sqlalchemy import select

from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings
from app.data.seed_store import load_seed_payload
from app.db.bootstrap import ensure_seed_admin_user
from app.db.session import session_scope
from app.db.spatial import (
    linestring_geometry_value,
    point_geometry_value,
    polygon_geometry_value,
    session_dialect_name,
)
from app.models import Municipality, RoadSegment, SourceCatalog, Zone


def main() -> None:
    settings = get_settings()
    if settings.real_data_only:
        raise SystemExit(
            "bootstrap_operational_catalog.py is disabled while REAL_DATA_ONLY=true. "
            "Use scripts/import_official_structural_catalog.py with an official .gov.co-backed bundle instead."
        )

    seed = load_seed_payload()

    with session_scope() as session:
        if session.scalar(select(Municipality.id).limit(1)) is not None:
            print("Municipalities already exist. Skipping structural bootstrap.")
            return

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

        for source in seed["sourceCatalog"]:
            session.add(
                SourceCatalog(
                    id=source["id"],
                    label=source["label"],
                    category=source["category"],
                )
            )

        ensure_seed_admin_user(session)

    print("Bootstrapped structural catalog (municipalities, zones, roads, sources, admin).")


if __name__ == "__main__":
    main()
