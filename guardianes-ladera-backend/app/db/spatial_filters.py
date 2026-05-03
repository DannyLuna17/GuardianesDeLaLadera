from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import func


@dataclass(frozen=True)
class BoundingBox:
    north: float
    south: float
    east: float
    west: float

    @classmethod
    def from_edges(cls, north: float, south: float, east: float, west: float) -> "BoundingBox":
        if south > north:
            raise ValueError("Bounding box south edge cannot be greater than north edge.")
        if west > east:
            raise ValueError("Bounding box west edge cannot be greater than east edge.")
        if north < -90 or north > 90 or south < -90 or south > 90:
            raise ValueError("Bounding box latitude must stay within -90 and 90 degrees.")
        if east < -180 or east > 180 or west < -180 or west > 180:
            raise ValueError("Bounding box longitude must stay within -180 and 180 degrees.")
        return cls(
            north=float(north),
            south=float(south),
            east=float(east),
            west=float(west),
        )

    def to_postgis_envelope(self) -> Any:
        return func.ST_MakeEnvelope(self.west, self.south, self.east, self.north, 4326)


def _normalize_point(point: list[float]) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError("Spatial coordinates must contain exactly two values.")
    lat, lon = point
    return float(lat), float(lon)


def _coordinates_envelope(points: list[list[float]]) -> tuple[float, float, float, float]:
    if not points:
        raise ValueError("Spatial coordinate sequences cannot be empty.")
    normalized = [_normalize_point(point) for point in points]
    latitudes = [lat for lat, _ in normalized]
    longitudes = [lon for _, lon in normalized]
    return min(latitudes), max(latitudes), min(longitudes), max(longitudes)


def _envelope_intersects_bbox(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    bounds: BoundingBox,
) -> bool:
    return not (
        max_lat < bounds.south
        or min_lat > bounds.north
        or max_lon < bounds.west
        or min_lon > bounds.east
    )


def point_within_bbox(coords: list[float], bounds: BoundingBox) -> bool:
    lat, lon = _normalize_point(coords)
    return bounds.south <= lat <= bounds.north and bounds.west <= lon <= bounds.east


def linestring_intersects_bbox(points: list[list[float]], bounds: BoundingBox) -> bool:
    min_lat, max_lat, min_lon, max_lon = _coordinates_envelope(points)
    return _envelope_intersects_bbox(min_lat, max_lat, min_lon, max_lon, bounds)


def polygon_intersects_bbox(points: list[list[float]], bounds: BoundingBox) -> bool:
    min_lat, max_lat, min_lon, max_lon = _coordinates_envelope(points)
    return _envelope_intersects_bbox(min_lat, max_lat, min_lon, max_lon, bounds)


def overlay_bounds_intersect_bbox(overlay_bounds: list[list[float]], bounds: BoundingBox) -> bool:
    min_lat, max_lat, min_lon, max_lon = _coordinates_envelope(overlay_bounds)
    return _envelope_intersects_bbox(min_lat, max_lat, min_lon, max_lon, bounds)


def _normalize_ring(points: list[list[float]]) -> list[tuple[float, float]]:
    normalized = [_normalize_point(point) for point in points]
    if not normalized:
        raise ValueError("Polygon coordinates cannot be empty.")
    if normalized[0] != normalized[-1]:
        normalized.append(normalized[0])
    return normalized


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> int:
    value = ((b[1] - a[1]) * (c[0] - b[0])) - ((b[0] - a[0]) * (c[1] - b[1]))
    epsilon = 1e-12
    if abs(value) <= epsilon:
        return 0
    return 1 if value > 0 else 2


def _point_on_segment(
    point: tuple[float, float],
    segment_start: tuple[float, float],
    segment_end: tuple[float, float],
) -> bool:
    lat, lon = point
    start_lat, start_lon = segment_start
    end_lat, end_lon = segment_end
    epsilon = 1e-12

    cross = ((lon - start_lon) * (end_lat - start_lat)) - ((lat - start_lat) * (end_lon - start_lon))
    if abs(cross) > epsilon:
        return False

    return (
        min(start_lat, end_lat) - epsilon <= lat <= max(start_lat, end_lat) + epsilon
        and min(start_lon, end_lon) - epsilon <= lon <= max(start_lon, end_lon) + epsilon
    )


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    first_orientation = _orientation(first_start, first_end, second_start)
    second_orientation = _orientation(first_start, first_end, second_end)
    third_orientation = _orientation(second_start, second_end, first_start)
    fourth_orientation = _orientation(second_start, second_end, first_end)

    if first_orientation != second_orientation and third_orientation != fourth_orientation:
        return True

    if first_orientation == 0 and _point_on_segment(second_start, first_start, first_end):
        return True
    if second_orientation == 0 and _point_on_segment(second_end, first_start, first_end):
        return True
    if third_orientation == 0 and _point_on_segment(first_start, second_start, second_end):
        return True
    if fourth_orientation == 0 and _point_on_segment(first_end, second_start, second_end):
        return True

    return False


def point_within_polygon(coords: list[float], polygon_points: list[list[float]]) -> bool:
    point = _normalize_point(coords)
    polygon = _normalize_ring(polygon_points)

    for index in range(len(polygon) - 1):
        if _point_on_segment(point, polygon[index], polygon[index + 1]):
            return True

    lat, lon = point
    inside = False
    for index in range(len(polygon) - 1):
        first_lat, first_lon = polygon[index]
        second_lat, second_lon = polygon[index + 1]

        if (first_lat > lat) == (second_lat > lat):
            continue

        edge_lon = ((second_lon - first_lon) * (lat - first_lat) / (second_lat - first_lat)) + first_lon
        if lon <= edge_lon:
            inside = not inside

    return inside


def linestring_intersects_polygon(line_points: list[list[float]], polygon_points: list[list[float]]) -> bool:
    line = [_normalize_point(point) for point in line_points]
    polygon = _normalize_ring(polygon_points)

    if any(point_within_polygon([lat, lon], polygon_points) for lat, lon in line):
        return True

    if any(_point_on_segment(vertex, line_start, line_end) for vertex in polygon[:-1] for line_start, line_end in zip(line, line[1:])):
        return True

    polygon_segments = list(zip(polygon, polygon[1:]))
    for line_start, line_end in zip(line, line[1:]):
        for polygon_start, polygon_end in polygon_segments:
            if _segments_intersect(line_start, line_end, polygon_start, polygon_end):
                return True

    return False


def polygon_intersects_polygon(first_polygon: list[list[float]], second_polygon: list[list[float]]) -> bool:
    first_ring = _normalize_ring(first_polygon)
    second_ring = _normalize_ring(second_polygon)

    if any(point_within_polygon([lat, lon], second_polygon) for lat, lon in first_ring[:-1]):
        return True
    if any(point_within_polygon([lat, lon], first_polygon) for lat, lon in second_ring[:-1]):
        return True

    first_segments = list(zip(first_ring, first_ring[1:]))
    second_segments = list(zip(second_ring, second_ring[1:]))
    for first_start, first_end in first_segments:
        for second_start, second_end in second_segments:
            if _segments_intersect(first_start, first_end, second_start, second_end):
                return True

    return False


def overlay_bounds_intersect_polygon(overlay_bounds: list[list[float]], polygon_points: list[list[float]]) -> bool:
    if len(overlay_bounds) != 2:
        raise ValueError("Overlay bounds must contain exactly two corners.")
    first_lat, first_lon = _normalize_point(overlay_bounds[0])
    second_lat, second_lon = _normalize_point(overlay_bounds[1])
    north = max(first_lat, second_lat)
    south = min(first_lat, second_lat)
    east = max(first_lon, second_lon)
    west = min(first_lon, second_lon)
    overlay_polygon = [
        [north, west],
        [north, east],
        [south, east],
        [south, west],
    ]
    return polygon_intersects_polygon(overlay_polygon, polygon_points)
