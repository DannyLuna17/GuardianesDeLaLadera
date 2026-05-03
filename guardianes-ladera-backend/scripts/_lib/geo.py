"""Geospatial helpers shared by standalone pipeline scripts."""

from __future__ import annotations

import math


EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometres for WGS84 lat/lon points."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def polygon_box(
    centroid: tuple[float, float],
    *,
    half_width_deg: float,
) -> list[list[float]]:
    """Return a simple [lat, lon] box around a centroid."""
    lat, lon = centroid
    half = half_width_deg
    return [
        [lat + half, lon - half],
        [lat + half, lon + half],
        [lat - half, lon + half],
        [lat - half, lon - half],
    ]


def extract_first_polygon_ring(geometry: dict) -> list[list[float]]:
    """Return the outer [lon, lat] ring of a Polygon or first MultiPolygon part."""
    geom_type = (geometry.get("type") or "").lower()
    coords = geometry.get("coordinates") or []
    if geom_type == "polygon" and coords:
        return coords[0]
    if geom_type == "multipolygon" and coords and coords[0]:
        return coords[0][0]
    return []


def extract_polygon_rings(geometry: dict) -> list[list[list[list[float]]]]:
    """Flatten a GeoJSON Polygon/MultiPolygon into polygons with rings."""
    geom_type = (geometry.get("type") or "").lower()
    coords = geometry.get("coordinates") or []
    if geom_type == "polygon":
        return [coords] if coords else []
    if geom_type == "multipolygon":
        return [polygon for polygon in coords if polygon]
    return []


def ring_contains(lat: float, lon: float, ring: list[list[float]]) -> bool:
    """Ray-casting test for a GeoJSON ring of [lon, lat] vertices."""
    count = 0
    length = len(ring)
    for i in range(length):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % length][0], ring[(i + 1) % length][1]
        if (y1 > lat) != (y2 > lat):
            denom = y2 - y1
            if denom == 0:
                continue
            x_intersect = x1 + (lat - y1) * (x2 - x1) / denom
            if lon < x_intersect:
                count += 1
    return count % 2 == 1


def point_in_polygon(
    lat: float,
    lon: float,
    rings: list[list[list[float]]],
) -> bool:
    """Return whether a WGS84 point falls inside GeoJSON polygon rings."""
    if not rings:
        return False
    if not ring_contains(lat, lon, rings[0]):
        return False
    for hole in rings[1:]:
        if ring_contains(lat, lon, hole):
            return False
    return True

