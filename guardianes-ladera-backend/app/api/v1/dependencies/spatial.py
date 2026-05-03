from __future__ import annotations

from fastapi import Query

from app.core.exceptions import ApiError
from app.db.spatial_filters import BoundingBox


def get_bounding_box(
    north: float | None = Query(default=None),
    south: float | None = Query(default=None),
    east: float | None = Query(default=None),
    west: float | None = Query(default=None),
) -> BoundingBox | None:
    edges = [north, south, east, west]
    if all(edge is None for edge in edges):
        return None
    if any(edge is None for edge in edges):
        raise ApiError(
            400,
            "invalid_bbox",
            "north, south, east, and west must all be provided together.",
        )
    try:
        return BoundingBox.from_edges(north=north, south=south, east=east, west=west)
    except ValueError as exc:
        raise ApiError(400, "invalid_bbox", str(exc)) from exc
