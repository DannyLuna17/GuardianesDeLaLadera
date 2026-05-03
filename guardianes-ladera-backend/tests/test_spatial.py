from geoalchemy2.elements import WKTElement

from app.db.spatial import (
    bounds_geometry_value,
    linestring_geometry_value,
    point_geometry_value,
    polygon_geometry_value,
)


def test_point_geometry_value_returns_geojson_for_sqlite():
    value = point_geometry_value([1.155, -76.661], "sqlite")

    assert value == {
        "type": "Point",
        "coordinates": [-76.661, 1.155],
    }


def test_point_geometry_value_returns_wkt_for_postgresql():
    value = point_geometry_value([1.155, -76.661], "postgresql")

    assert isinstance(value, WKTElement)
    assert value.data == "POINT(-76.661 1.155)"
    assert value.srid == 4326


def test_linestring_and_polygon_values_normalize_coordinate_order():
    line = linestring_geometry_value([[1.15, -76.66], [1.14, -76.62]], "sqlite")
    polygon = polygon_geometry_value(
        [[1.16, -76.67], [1.16, -76.63], [1.12, -76.63], [1.12, -76.67]],
        "sqlite",
    )

    assert line == {
        "type": "LineString",
        "coordinates": [[-76.66, 1.15], [-76.62, 1.14]],
    }
    assert polygon["type"] == "Polygon"
    assert polygon["coordinates"][0][0] == [-76.67, 1.16]
    assert polygon["coordinates"][0][-1] == [-76.67, 1.16]


def test_bounds_geometry_value_builds_closed_polygon():
    bounds = bounds_geometry_value([[1.18, -76.69], [1.11, -76.61]], "sqlite")

    assert bounds == {
        "type": "Polygon",
        "coordinates": [
            [
                [-76.69, 1.18],
                [-76.61, 1.18],
                [-76.61, 1.11],
                [-76.69, 1.11],
                [-76.69, 1.18],
            ]
        ],
    }
