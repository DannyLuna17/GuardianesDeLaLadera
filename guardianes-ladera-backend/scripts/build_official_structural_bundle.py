from __future__ import annotations

import json
import math
import shutil
import zipfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import shapefile


BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_ROOT / "app" / "data"
OUTPUT_PATH = DATA_DIR / "official-structural" / "official_structural_bundle.json"

MUNICIPALITY_ARCHIVE = DATA_DIR / "MGN2025_MPIO_GRAFICO.zip"
URBAN_SECTION_ARCHIVE = DATA_DIR / "MGN2025_URB_SECCION.zip"
VEREDA_ARCHIVE = DATA_DIR / "shp_CRVeredas_2024.zip"
ROADS_GEOJSON = DATA_DIR / "invias.geojson"
SEED_PATH = DATA_DIR / "frontend_seed.json"

MUNICIPALITY_SOURCE_URL = (
    "https://geoportal.dane.gov.co/mparcgis/rest/services/MGN2025/"
    "Serv_CapasMGN_2025/MapServer/317"
)
URBAN_SECTION_SOURCE_URL = (
    "https://geoportal.dane.gov.co/mparcgis/rest/services/MGN2025/"
    "Serv_CapasMGN_2025/MapServer/214"
)
VEREDA_SOURCE_URL = (
    "https://geoportal.dane.gov.co/mparcgis/rest/services/"
    "NIVEL_DE_REFERENCIA_DE_VEREDAS/Serv_CapasNivelReferenciaVeredas_2024/MapServer"
)
ROADS_SOURCE_URL = (
    "https://hermes.invias.gov.co/arcgis/rest/services/OpenData/"
    "ServiciosOpenData1/FeatureServer/0"
)

MUNICIPALITY_CONFIG = {
    "Mocoa": {"id": "mocoa", "code": "86001", "zoom": 12},
    "Pasto": {"id": "pasto", "code": "52001", "zoom": 12},
    "Popayan": {"id": "popayan", "code": "19001", "zoom": 12},
}


def load_seed() -> dict[str, Any]:
    return json.loads(SEED_PATH.read_text(encoding="utf-8"))


def shape_parts(shape: shapefile.Shape) -> list[list[tuple[float, float]]]:
    points = [(float(x), float(y)) for x, y in shape.points]
    part_starts = list(shape.parts) + [len(points)]
    rings: list[list[tuple[float, float]]] = []
    for start, end in zip(part_starts, part_starts[1:]):
        ring = points[start:end]
        if len(ring) < 3:
            continue
        if ring[0] != ring[-1]:
            ring = ring + [ring[0]]
        rings.append(ring)
    return rings


def ring_area(ring: list[tuple[float, float]]) -> float:
    area = 0.0
    for index in range(len(ring) - 1):
        x1, y1 = ring[index]
        x2, y2 = ring[index + 1]
        area += (x1 * y2) - (x2 * y1)
    return area / 2.0


def largest_ring(shape: shapefile.Shape) -> list[tuple[float, float]]:
    rings = shape_parts(shape)
    if not rings:
        raise ValueError("Polygon shape has no valid rings.")
    return max(rings, key=lambda item: abs(ring_area(item)))


def polygon_centroid(ring: list[tuple[float, float]]) -> tuple[float, float]:
    area = ring_area(ring)
    if abs(area) < 1e-12:
        xs = [point[0] for point in ring]
        ys = [point[1] for point in ring]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    cx = 0.0
    cy = 0.0
    for index in range(len(ring) - 1):
        x1, y1 = ring[index]
        x2, y2 = ring[index + 1]
        cross = (x1 * y2) - (x2 * y1)
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    factor = 1.0 / (6.0 * area)
    return (cx * factor, cy * factor)


def to_latlng(point: tuple[float, float]) -> list[float]:
    lon, lat = point
    return [round(lat, 6), round(lon, 6)]


def ring_to_latlng(ring: list[tuple[float, float]]) -> list[list[float]]:
    if ring and ring[0] == ring[-1]:
        ring = ring[:-1]
    return [to_latlng(point) for point in ring]


def bbox_from_points(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def bboxes_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    return not (
        first[2] < second[0]
        or first[0] > second[2]
        or first[3] < second[1]
        or first[1] > second[3]
    )


def point_in_ring(point: tuple[float, float], ring: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    for index in range(len(ring) - 1):
        x1, y1 = ring[index]
        x2, y2 = ring[index + 1]
        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1
        )
        if intersects:
            inside = not inside
    return inside


def shape_contains_point(shape: shapefile.Shape, point: tuple[float, float]) -> bool:
    return any(point_in_ring(point, ring) for ring in shape_parts(shape))


def bbox_center(bbox: Iterable[float]) -> tuple[float, float]:
    xmin, ymin, xmax, ymax = [float(value) for value in bbox]
    return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)


def point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def orient(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
    return (
        min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12
        and min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12
    )


def segments_intersect(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> bool:
    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)

    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True

    if abs(o1) < 1e-12 and on_segment(a1, a2, b1):
        return True
    if abs(o2) < 1e-12 and on_segment(a1, a2, b2):
        return True
    if abs(o3) < 1e-12 and on_segment(b1, b2, a1):
        return True
    if abs(o4) < 1e-12 and on_segment(b1, b2, a2):
        return True
    return False


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lon1, lat1 = a
    lon2, lat2 = b
    radius_km = 6371.0
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    hav = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(hav))


def line_length_km(line: list[tuple[float, float]]) -> float:
    if len(line) < 2:
        return 0.0
    return round(
        sum(haversine_km(line[index], line[index + 1]) for index in range(len(line) - 1)),
        3,
    )


def extract_archive(archive_path: Path, target_dir: Path) -> Path:
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(target_dir)
    shapefiles = list(target_dir.glob("*.shp"))
    if len(shapefiles) != 1:
        raise RuntimeError(f"Expected exactly one shapefile in {archive_path.name}.")
    return shapefiles[0]


def select_official_units(
    seed: dict[str, Any],
    urban_reader: shapefile.Reader,
    vereda_reader: shapefile.Reader,
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}

    urban_by_code: dict[str, list[tuple[dict[str, Any], shapefile.Shape]]] = {}
    for shape_record in urban_reader.iterShapeRecords():
        record = shape_record.record.as_dict()
        urban_by_code.setdefault(record["mpio_cdpmp"], []).append((record, shape_record.shape))

    vereda_by_code: dict[str, list[tuple[dict[str, Any], shapefile.Shape]]] = {}
    for shape_record in vereda_reader.iterShapeRecords():
        record = shape_record.record.as_dict()
        vereda_by_code.setdefault(record["DPTOMPIO"], []).append((record, shape_record.shape))

    for seed_zone in seed["zones"]:
        municipality_name = seed_zone["municipality"]
        municipality = MUNICIPALITY_CONFIG[municipality_name]
        seed_centroid = (float(seed_zone["centroid"][1]), float(seed_zone["centroid"][0]))
        is_rural = seed_zone["type"].lower() == "vereda"
        source_bucket = vereda_by_code if is_rural else urban_by_code
        candidates = source_bucket[municipality["code"]]

        containing: list[tuple[dict[str, Any], shapefile.Shape]] = [
            candidate
            for candidate in candidates
            if shape_contains_point(candidate[1], seed_centroid)
        ]
        if containing:
            record, shape = containing[0]
        else:
            record, shape = min(
                candidates,
                key=lambda item: point_distance(seed_centroid, bbox_center(item[1].bbox)),
            )

        polygon_ring = largest_ring(shape)
        centroid = polygon_centroid(polygon_ring)
        if is_rural:
            zone_id = f"{municipality['id']}-v-{record['CODIGO_VER']}"
            name = str(record["NOMBRE_VER"]).strip()
            source_id = "DANE_VEREDAS_2024"
            source_ref = f"vereda:{record['CODIGO_VER']}"
            zone_type = "Vereda"
        else:
            section_code = str(record["secu_ccnct"]).strip()
            zone_id = f"{municipality['id']}-u-{section_code}"
            name = f"Seccion Urbana {record['setu_ccdgo']}-{record['secu_ccdgo']}"
            source_id = "DANE_MGN2025_URB_SECCION"
            source_ref = f"seccion_urbana:{section_code}"
            zone_type = "Seccion Urbana"

        selected[source_ref] = {
            "id": zone_id[:32],
            "municipalityId": municipality["id"],
            "name": name,
            "type": zone_type,
            "centroid": to_latlng(centroid),
            "polygon": ring_to_latlng(polygon_ring),
            "polygon_lonlat": polygon_ring,
            "polygon_bbox": bbox_from_points(polygon_ring),
            "exposure": {
                "population_estimate": None,
                "households_estimate": None,
            },
            "assets": {"roadSegmentIds": []},
            "sourceId": source_id,
            "sourceRef": source_ref,
        }

    return list(selected.values())


def municipality_records(municipality_reader: shapefile.Reader) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for shape_record in municipality_reader.iterShapeRecords():
        record = shape_record.record.as_dict()
        code = str(record["mpio_cdpmp"]).strip()
        match = next(
            (item for item in MUNICIPALITY_CONFIG.values() if item["code"] == code),
            None,
        )
        if match is None:
            continue
        ring = largest_ring(shape_record.shape)
        centroid = polygon_centroid(ring)
        result[match["id"]] = {
            "id": match["id"],
            "name": str(record["mpio_cnmbr"]).title(),
            "center": to_latlng(centroid),
            "zoom": match["zoom"],
            "sourceId": "DANE_MGN2025_MPIO",
            "sourceRef": f"municipio:{code}",
            "polygon_lonlat": ring,
        }
    return result


def road_name(properties: dict[str, Any]) -> str:
    tramo = str(properties.get("tramo") or "").strip()
    sector = str(properties.get("sector") or "").strip()
    if tramo and sector:
        return f"{tramo} - {sector}"[:128]
    return (sector or tramo or f"Via {properties.get('codigo_via') or properties.get('key')}")[:128]


def road_source_ref(properties: dict[str, Any]) -> str:
    cod_tramo = str(properties.get("cod_tramo") or "").strip()
    if cod_tramo:
        return f"via:{cod_tramo}"
    codigo_via = str(properties.get("codigo_via") or "").strip()
    key = str(properties.get("key") or "").strip()
    suffix = codigo_via or key or str(properties.get("objectid"))
    return f"via:{suffix}"


def road_id(municipality_id: str, properties: dict[str, Any]) -> str:
    token = str(properties.get("cod_tramo") or properties.get("codigo_via") or properties.get("objectid"))
    return f"{municipality_id}-r-{token}"[:32]


def build_road_segments(
    zones: list[dict[str, Any]],
    municipalities: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    roads_payload = json.loads(ROADS_GEOJSON.read_text(encoding="utf-8"))
    road_segments: dict[str, dict[str, Any]] = {}

    zone_lookup = {zone["id"]: zone for zone in zones}
    municipality_scope_bboxes: dict[str, tuple[float, float, float, float]] = {}
    for municipality_id in municipalities:
        zone_bboxes = [
            zone["polygon_bbox"]
            for zone in zones
            if zone["municipalityId"] == municipality_id
        ]
        municipality_scope_bboxes[municipality_id] = (
            min(bbox[0] for bbox in zone_bboxes),
            min(bbox[1] for bbox in zone_bboxes),
            max(bbox[2] for bbox in zone_bboxes),
            max(bbox[3] for bbox in zone_bboxes),
        )

    for feature in roads_payload.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "LineString":
            continue
        coords = [(float(lon), float(lat)) for lon, lat in geometry.get("coordinates", [])]
        if len(coords) < 2:
            continue
        line_bbox = bbox_from_points(coords)

        municipality_ids = [
            municipality_id
            for municipality_id, scope_bbox in municipality_scope_bboxes.items()
            if bboxes_overlap(line_bbox, scope_bbox)
        ]
        if len(municipality_ids) != 1:
            continue
        municipality_id = municipality_ids[0]

        intersecting_zone_ids = [
            zone["id"]
            for zone in zones
            if zone["municipalityId"] == municipality_id
            and (
                bboxes_overlap(line_bbox, zone["polygon_bbox"])
                or min(
                    point_distance(point, (zone["centroid"][1], zone["centroid"][0]))
                    for point in coords
                )
                <= 0.03
            )
        ]
        if not intersecting_zone_ids:
            continue

        properties = feature.get("properties") or {}
        segment_ref = road_source_ref(properties)
        segment_id = road_id(municipality_id, properties)

        road_segments.setdefault(
            segment_ref,
            {
                "id": segment_id,
                "municipalityId": municipality_id,
                "name": road_name(properties),
                "coords": [to_latlng(point) for point in coords],
                "riskLevel": "Sin clasificar",
                "length_km": line_length_km(coords),
                "note": (
                    "Segmento oficial INVIAS. La fuente descargada no incluye una "
                    "clasificacion vial de riesgo."
                ),
                "sourceId": "INVIAS_RED_VIAL",
                "sourceRef": segment_ref,
            },
        )

        for zone_id in intersecting_zone_ids:
            road_segment_id = road_segments[segment_ref]["id"]
            assets = zone_lookup[zone_id]["assets"]["roadSegmentIds"]
            if road_segment_id not in assets:
                assets.append(road_segment_id)

    return list(road_segments.values())


def bundle_payload() -> dict[str, Any]:
    seed = load_seed()
    tmp_root = DATA_DIR / "_tmp_bundle_build"
    municipality_reader: shapefile.Reader | None = None
    urban_reader: shapefile.Reader | None = None
    vereda_reader: shapefile.Reader | None = None
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    try:
        municipality_shp = extract_archive(MUNICIPALITY_ARCHIVE, tmp_root / "municipalities")
        urban_shp = extract_archive(URBAN_SECTION_ARCHIVE, tmp_root / "urban_sections")
        vereda_shp = extract_archive(VEREDA_ARCHIVE, tmp_root / "veredas")

        municipality_reader = shapefile.Reader(str(municipality_shp), encoding="utf-8")
        urban_reader = shapefile.Reader(str(urban_shp), encoding="utf-8")
        vereda_reader = shapefile.Reader(str(vereda_shp), encoding="utf-8")

        municipalities = municipality_records(municipality_reader)
        zones = select_official_units(seed, urban_reader, vereda_reader)
        road_segments = build_road_segments(zones, municipalities)

        for zone in zones:
            zone.pop("polygon_lonlat", None)
            zone.pop("polygon_bbox", None)

        return {
            "version": "official-structural-pilot-v1",
            "generatedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "sources": [
                {
                    "id": "DANE_MGN2025_MPIO",
                    "label": "DANE MGN 2025 nivel municipio",
                    "category": "historico",
                    "sourceUrl": MUNICIPALITY_SOURCE_URL,
                },
                {
                    "id": "DANE_MGN2025_URB_SECCION",
                    "label": "DANE MGN 2025 seccion urbana",
                    "category": "historico",
                    "sourceUrl": URBAN_SECTION_SOURCE_URL,
                },
                {
                    "id": "DANE_VEREDAS_2024",
                    "label": "DANE veredas 2024",
                    "category": "historico",
                    "sourceUrl": VEREDA_SOURCE_URL,
                },
                {
                    "id": "INVIAS_RED_VIAL",
                    "label": "INVIAS red vial oficial",
                    "category": "infraestructura",
                    "sourceUrl": ROADS_SOURCE_URL,
                },
            ],
            "municipalities": [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "center": item["center"],
                    "zoom": item["zoom"],
                    "sourceId": item["sourceId"],
                    "sourceRef": item["sourceRef"],
                }
                for item in municipalities.values()
            ],
            "roadSegments": road_segments,
            "zones": zones,
        }
    finally:
        for reader in (municipality_reader, urban_reader, vereda_reader):
            if reader is not None:
                reader.close()
        shutil.rmtree(tmp_root, ignore_errors=True)


def main() -> None:
    payload = bundle_payload()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "Generated official structural bundle:",
        OUTPUT_PATH,
        f"{len(payload['municipalities'])} municipalities,",
        f"{len(payload['zones'])} zones,",
        f"{len(payload['roadSegments'])} road segments.",
    )


if __name__ == "__main__":
    main()
