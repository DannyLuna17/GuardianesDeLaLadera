"""Normalize SIMMA raw GeoJSON chunks into our canonical event schema.

Input:
  - every ``lote_*.geojson`` produced by ``fetch_simma.py`` (landslide points)
  - optional ``boundaries/municipios.geojson`` from ``fetch_simma.py --fetch
    -admin-boundaries``. When present, every SIMMA point is spatial-joined
    against the 1,123 Colombian municipal polygons and the record's
    ``municipality`` + ``department`` + ``divipola`` (``COD_DEPART`` +
    ``COD_MUNICI``) are populated. This bumps the record-quality tier from
    ``spatial_prior_only`` (point-only) to ``medium`` (point + muni + dept).

**Important reality check**: the SIMMA public MapServer service still exposes
only spatial geometry and movement-type taxonomy — no dates, no severity. The
admin-boundary enrichment fills municipality/department but does NOT add an
``observed_at``. SIMMA records therefore remain a **spatial prior** rather
than time-stamped labels. They are excellent for:

    - defining zones in the backend catalog (with real admin polygons)
    - cross-referencing UNGRD records whose municipality matches
    - pseudo-absence sampling (avoid placing negatives on SIMMA points)

They are still NOT usable by themselves as outcome labels because
``observed_at`` is required by the backend's label schema.

Output: JSONL — one canonical record per line — matching the schema used by
``normalize_ungrd.py`` with the spatial-only caveat above.

Usage:

    uv run python scripts/landslide_inventory/normalize_simma.py \\
        --lotes-dir data/inventory/00_raw/simma/lotes \\
        --output data/inventory/01_staging/simma_normalized.jsonl

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _lib.geo import extract_polygon_rings as _shared_extract_polygon_rings  # noqa: E402
from _lib.geo import point_in_polygon as _shared_point_in_polygon  # noqa: E402
from _lib.geo import ring_contains as _shared_ring_contains  # noqa: E402


# Real SIMMA public-service attribute layout (layer 1 / "Inventario/Punto"):
#   OBJECTID, ID, INV_MOVIMIENTO_MASA_ID, F35DOV_TIPO_MOVIMIENTO_ID,
#   TIPO, SUBTIPO_MOVIMIENTO_ID, SUBTIPO, CLAS_MAPA, ETIQUETA_MAPA
# Dates/municipalities/severities live on internal SGC tables that the public
# MapServer does not expose; we keep candidate-field lists empty for those and
# provide candidates only for the taxonomy and identifier slots.
TYPE_FIELDS = ("CLAS_MAPA", "TIPO")
SUBTYPE_FIELDS = ("SUBTIPO", "ETIQUETA_MAPA")
ID_FIELDS = ("INV_MOVIMIENTO_MASA_ID", "CAT_MOVIMIENTO_MASA_ID", "OBJECTID", "ID")


MOVEMENT_TYPE_MAP = {
    # SGC class -> our canonical taxonomy. Matches the researcher's brief:
    #   Caida / Caída de rocas / Volcamiento               -> fall
    #   Deslizamiento / rotacional / por licuación / Reptación -> slide
    #   Flujo / Avalancha                                   -> flow
    #   Propagación Lateral / Deformación gravitacional profunda -> complex
    # Missing keys fall through to "unknown" (researcher's last row).
    "CAIDA": "fall",
    "CAÍDA": "fall",
    "CAIDA DE ROCA": "fall",
    "CAÍDA DE ROCAS": "fall",
    "CAÍDA DE ROCA": "fall",
    "CAIDO": "fall",
    "VOLCAMIENTO": "fall",
    "VOLCAMIENTO FLEXURAL": "fall",
    "VOLCAMIENTO FLEXURAL DE ROCA": "fall",
    "DESLIZAMIENTO": "slide",
    "DESLIZAMIENTO ROTACIONAL": "slide",
    "DESLIZAMIENTO TRASLACIONAL": "slide",
    "DESLIZAMIENTO POR LICUACION": "slide",
    "DESLIZAMIENTO POR LICUACIÓN": "slide",
    "REPTACION": "slide",
    "REPTACIÓN": "slide",
    "FLUJO": "flow",
    "FLUJO DE DETRITOS": "flow",
    "FLUJO DE LODO": "flow",
    "AVALANCHA": "flow",
    "AVENIDA TORRENCIAL": "flow",
    "PROPAGACION LATERAL": "complex",
    "PROPAGACIÓN LATERAL": "complex",
    "DEFORMACION GRAVITACIONAL PROFUNDA": "complex",
    "DEFORMACIÓN GRAVITACIONAL PROFUNDA": "complex",
    "COMPLEJO": "complex",
}


def _lookup(attributes: dict, candidates: tuple[str, ...]) -> str | None:
    normalized = {
        (key or "").strip().upper(): value for key, value in (attributes or {}).items()
    }
    for name in candidates:
        value = normalized.get(name.upper())
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _movement_type(type_hint: str | None, subtype_hint: str | None) -> str:
    for candidate in (type_hint, subtype_hint):
        if not candidate:
            continue
        token = candidate.strip().upper()
        if token in MOVEMENT_TYPE_MAP:
            return MOVEMENT_TYPE_MAP[token]
        for key, value in MOVEMENT_TYPE_MAP.items():
            if key in token:
                return value
    return "unknown"


def _extract_coordinates(geometry: dict | None) -> tuple[float | None, float | None]:
    if not geometry:
        return None, None
    geom_type = str(geometry.get("type") or "").lower()
    coords = geometry.get("coordinates")
    if not coords:
        return None, None
    try:
        if geom_type == "point":
            lon, lat = coords[0], coords[1]
        elif geom_type in {"multipoint", "linestring"}:
            lon, lat = coords[0][0], coords[0][1]
        elif geom_type in {"polygon", "multilinestring"}:
            lon, lat = coords[0][0][0], coords[0][0][1]
        elif geom_type == "multipolygon":
            lon, lat = coords[0][0][0][0], coords[0][0][0][1]
        else:
            return None, None
        return float(lat), float(lon)
    except (TypeError, ValueError, IndexError):
        return None, None


def _resolve_record_id(properties: dict) -> str:
    for key in ID_FIELDS:
        value = properties.get(key) or properties.get(key.lower())
        if value not in (None, ""):
            return str(value)
    return "unknown"


class MunicipioIndex:
    """Spatial index over Colombian municipal polygons.

    Each entry is a list of rings (lists of [lon, lat] vertices) plus a bbox
    that the point-in-polygon test uses as a cheap pre-filter. Ray-casting
    runs only against polygons whose bbox contains the query point, which
    reduces the per-point cost from ~1,123 polygon tests to typically <5.
    """

    def __init__(self) -> None:
        # Each item: (min_lat, max_lat, min_lon, max_lon, rings, properties)
        self._entries: list[tuple] = []

    @classmethod
    def from_geojson(cls, path: Path) -> "MunicipioIndex":
        index = cls()
        if not path.exists():
            return index
        payload = json.loads(path.read_text(encoding="utf-8"))
        for feature in payload.get("features") or []:
            properties = feature.get("properties") or {}
            geometry = feature.get("geometry") or {}
            rings_per_polygon = cls._extract_rings(geometry)
            if not rings_per_polygon:
                continue
            for rings in rings_per_polygon:
                outer = rings[0]
                min_lat = min(point[1] for point in outer)
                max_lat = max(point[1] for point in outer)
                min_lon = min(point[0] for point in outer)
                max_lon = max(point[0] for point in outer)
                index._entries.append(
                    (min_lat, max_lat, min_lon, max_lon, rings, properties)
                )
        return index

    @staticmethod
    def _extract_rings(geometry: dict) -> list[list[list[list[float]]]]:
        """Flatten a GeoJSON geometry to a list-of-polygons; each polygon is a
        list of rings (outer first, then holes). We evaluate each polygon
        independently so a MultiPolygon produces multiple index entries."""
        return _shared_extract_polygon_rings(geometry)

    def __len__(self) -> int:
        return len(self._entries)

    def find(self, lat: float, lon: float) -> dict | None:
        candidate_count = 0
        for min_lat, max_lat, min_lon, max_lon, rings, properties in self._entries:
            if lat < min_lat or lat > max_lat or lon < min_lon or lon > max_lon:
                continue
            candidate_count += 1
            if self._point_in_polygon(lat, lon, rings):
                return properties
        return None

    @staticmethod
    def _point_in_polygon(
        lat: float, lon: float, rings: list[list[list[float]]]
    ) -> bool:
        """GeoJSON convention: each ring is a list of [lon, lat] vertices.

        Outer ring is rings[0]; inner rings (holes) follow. Classic ray-casting
        with even-odd rule, then XOR'd against any hole that also contains the
        point.
        """
        return _shared_point_in_polygon(lat, lon, rings)

    @staticmethod
    def _ring_contains(lat: float, lon: float, ring: list[list[float]]) -> bool:
        return _shared_ring_contains(lat, lon, ring)


def _divipola_from(properties: dict) -> str | None:
    dept = properties.get("COD_DEPART")
    muni = properties.get("COD_MUNICI")
    if dept is None or muni is None:
        return None
    try:
        return f"{int(dept):02d}{int(muni):03d}"
    except (TypeError, ValueError):
        return None


def normalize_features(
    features: list[dict],
    *,
    observer: dict,
    municipios: MunicipioIndex | None = None,
) -> list[dict]:
    events: list[dict] = []
    for feature in features:
        properties = feature.get("properties") or feature.get("attributes") or {}
        geometry = feature.get("geometry")

        latitude, longitude = _extract_coordinates(geometry)
        observer["geometry_populated"] += int(latitude is not None)

        type_hint = _lookup(properties, TYPE_FIELDS)
        subtype_hint = _lookup(properties, SUBTYPE_FIELDS)

        record_id = _resolve_record_id(properties)
        event_id = f"SIMMA-{record_id}"

        municipality: str | None = None
        department: str | None = None
        divipola: str | None = None
        if (
            municipios is not None
            and latitude is not None
            and longitude is not None
        ):
            match = municipios.find(latitude, longitude)
            if match is not None:
                name = match.get("NOMBRE_ENT")
                dept = match.get("DEPARTAMEN")
                municipality = str(name).strip().title() if name else None
                department = str(dept).strip().title() if dept else None
                divipola = _divipola_from(match)
                observer["spatial_join_matched"] += 1
            else:
                observer["spatial_join_unmatched"] += 1

        has_geom = latitude is not None and longitude is not None
        if has_geom and municipality:
            # Point + muni + dept = usable spatial prior at medium quality.
            record_quality = "medium"
        elif has_geom:
            # Point only; municipality unresolved. Keep flagged as prior-only.
            record_quality = "spatial_prior_only"
        else:
            record_quality = "low"

        description_bits = [
            bit for bit in (type_hint, subtype_hint) if bit
        ]
        description = " / ".join(description_bits) if description_bits else None

        event = {
            "event_id": event_id,
            "source": "SGC_SIMMA",
            "observed_at": None,
            "municipality": municipality,
            "department": department,
            "divipola": divipola,
            "latitude": latitude,
            "longitude": longitude,
            "severity": None,
            "movement_type": _movement_type(type_hint, subtype_hint),
            "deaths": None,
            "injured": None,
            "missing": None,
            "homes_destroyed": None,
            "homes_damaged": None,
            "description": description,
            "source_url": (
                "https://srvags.sgc.gov.co/arcgis/rest/services/SIMMA/"
                "Capas_Principales/MapServer"
            ),
            "record_quality": record_quality,
            "raw_attributes": properties,
        }
        events.append(event)
        observer["by_record_quality"][record_quality] = (
            observer["by_record_quality"].get(record_quality, 0) + 1
        )
    return events


def normalize_dir(
    *,
    lotes_dir: Path,
    output_path: Path,
    municipios_path: Path | None = None,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observer = {
        "files_read": 0,
        "features_read": 0,
        "geometry_populated": 0,
        "spatial_join_matched": 0,
        "spatial_join_unmatched": 0,
        "municipios_index_size": 0,
        "by_record_quality": {},
    }
    files = sorted(path for path in lotes_dir.glob("*.geojson"))
    if not files:
        raise RuntimeError(
            f"No *.geojson files found under {lotes_dir}. Did fetch_simma.py run?"
        )

    municipios: MunicipioIndex | None = None
    if municipios_path is not None and municipios_path.exists():
        municipios = MunicipioIndex.from_geojson(municipios_path)
        observer["municipios_index_size"] = len(municipios)
        print(
            f"Loaded {len(municipios)} municipal polygons for spatial join."
        )
    elif municipios_path is not None:
        print(
            f"Municipios file not found at {municipios_path}; "
            f"SIMMA records will stay at record_quality='spatial_prior_only'."
        )

    with open(output_path, "w", encoding="utf-8") as dst:
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            features = payload.get("features") or []
            observer["files_read"] += 1
            observer["features_read"] += len(features)
            events = normalize_features(
                features, observer=observer, municipios=municipios
            )
            for event in events:
                dst.write(json.dumps(event, ensure_ascii=False) + "\n")
    return observer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lotes-dir",
        type=Path,
        default=Path("data/inventory/00_raw/simma/lotes"),
        help="Directory containing the lote_*.geojson files.",
    )
    parser.add_argument(
        "--municipios",
        type=Path,
        default=Path(
            "data/inventory/00_raw/simma/boundaries/municipios.geojson"
        ),
        help=(
            "Optional municipios GeoJSON from fetch_simma.py's admin-boundary "
            "download. When present, every SIMMA point is spatial-joined to a "
            "municipality and record_quality is bumped to 'medium'."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/inventory/01_staging/simma_normalized.jsonl"),
        help="Where to write the normalized JSONL.",
    )
    args = parser.parse_args(argv)

    if not args.lotes_dir.exists():
        print(f"Lotes directory not found: {args.lotes_dir}", file=sys.stderr)
        return 1

    stats = normalize_dir(
        lotes_dir=args.lotes_dir,
        output_path=args.output,
        municipios_path=args.municipios,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
