from __future__ import annotations

from collections.abc import Iterable

from geoalchemy2 import Geometry
from geoalchemy2.elements import WKTElement
from sqlalchemy import JSON
from sqlalchemy.orm import Session


def point_geometry_type():
    return Geometry("POINT", srid=4326).with_variant(JSON(), "sqlite")


def polygon_geometry_type():
    return Geometry("POLYGON", srid=4326).with_variant(JSON(), "sqlite")


def linestring_geometry_type():
    return Geometry("LINESTRING", srid=4326).with_variant(JSON(), "sqlite")


def session_dialect_name(session: Session) -> str:
    bind = session.get_bind()
    if bind is None:
        return "sqlite"
    return bind.dialect.name


def _ensure_point(coords: list[float]) -> tuple[float, float]:
    if len(coords) != 2:
        raise ValueError("Point geometry requires exactly two coordinates.")
    lat, lon = coords
    return float(lat), float(lon)


def _ensure_sequence(points: Iterable[list[float]]) -> list[tuple[float, float]]:
    normalized: list[tuple[float, float]] = []
    for point in points:
        lat, lon = _ensure_point(point)
        normalized.append((lat, lon))
    if not normalized:
        raise ValueError("Geometry sequence cannot be empty.")
    return normalized


def _point_geojson(coords: list[float]) -> dict:
    lat, lon = _ensure_point(coords)
    return {"type": "Point", "coordinates": [lon, lat]}


def _linestring_geojson(points: list[list[float]]) -> dict:
    normalized = _ensure_sequence(points)
    return {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in normalized]}


def _polygon_ring(points: list[list[float]]) -> list[tuple[float, float]]:
    normalized = _ensure_sequence(points)
    if normalized[0] != normalized[-1]:
        normalized.append(normalized[0])
    return normalized


def _polygon_geojson(points: list[list[float]]) -> dict:
    ring = _polygon_ring(points)
    return {"type": "Polygon", "coordinates": [[[lon, lat] for lat, lon in ring]]}


def point_geometry_value(coords: list[float], dialect_name: str):
    lat, lon = _ensure_point(coords)
    if dialect_name == "postgresql":
        return WKTElement(f"POINT({lon} {lat})", srid=4326)
    return _point_geojson(coords)


def linestring_geometry_value(points: list[list[float]], dialect_name: str):
    normalized = _ensure_sequence(points)
    if dialect_name == "postgresql":
        wkt = ",".join(f"{lon} {lat}" for lat, lon in normalized)
        return WKTElement(f"LINESTRING({wkt})", srid=4326)
    return _linestring_geojson(points)


def polygon_geometry_value(points: list[list[float]], dialect_name: str):
    ring = _polygon_ring(points)
    if dialect_name == "postgresql":
        wkt = ",".join(f"{lon} {lat}" for lat, lon in ring)
        return WKTElement(f"POLYGON(({wkt}))", srid=4326)
    return _polygon_geojson(points)


def bounds_geometry_value(bounds: list[list[float]], dialect_name: str):
    if len(bounds) != 2:
        raise ValueError("Bounds geometry requires two corner coordinates.")
    first_lat, first_lon = _ensure_point(bounds[0])
    second_lat, second_lon = _ensure_point(bounds[1])
    north = max(first_lat, second_lat)
    south = min(first_lat, second_lat)
    east = max(first_lon, second_lon)
    west = min(first_lon, second_lon)
    polygon_points = [
        [north, west],
        [north, east],
        [south, east],
        [south, west],
    ]
    return polygon_geometry_value(polygon_points, dialect_name)
