"""Add spatial shadow columns for PostGIS-backed persistence.

Revision ID: 20260325_0002
Revises: 20260324_0001
Create Date: 2026-03-25 00:02:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from geoalchemy2 import Geometry


revision = "20260325_0002"
down_revision = "20260324_0001"
branch_labels = None
depends_on = None


def _point_type(dialect_name: str):
    if dialect_name == "postgresql":
        return Geometry("POINT", srid=4326)
    return sa.JSON()


def _polygon_type(dialect_name: str):
    if dialect_name == "postgresql":
        return Geometry("POLYGON", srid=4326)
    return sa.JSON()


def _linestring_type(dialect_name: str):
    if dialect_name == "postgresql":
        return Geometry("LINESTRING", srid=4326)
    return sa.JSON()


def upgrade() -> None:
    bind = op.get_bind()
    dialect_name = bind.dialect.name

    if dialect_name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    op.add_column("zones", sa.Column("centroid_geom", _point_type(dialect_name), nullable=True))
    op.add_column("zones", sa.Column("polygon_geom", _polygon_type(dialect_name), nullable=True))
    op.add_column("road_segments", sa.Column("coords_geom", _linestring_type(dialect_name), nullable=True))
    op.add_column("historical_events", sa.Column("coords_geom", _point_type(dialect_name), nullable=True))
    op.add_column("rain_overlays", sa.Column("bounds_geom", _polygon_type(dialect_name), nullable=True))

    if dialect_name == "postgresql":
        op.create_index("ix_zones_centroid_geom", "zones", ["centroid_geom"], unique=False, postgresql_using="gist")
        op.create_index("ix_zones_polygon_geom", "zones", ["polygon_geom"], unique=False, postgresql_using="gist")
        op.create_index(
            "ix_road_segments_coords_geom",
            "road_segments",
            ["coords_geom"],
            unique=False,
            postgresql_using="gist",
        )
        op.create_index(
            "ix_historical_events_coords_geom",
            "historical_events",
            ["coords_geom"],
            unique=False,
            postgresql_using="gist",
        )
        op.create_index(
            "ix_rain_overlays_bounds_geom",
            "rain_overlays",
            ["bounds_geom"],
            unique=False,
            postgresql_using="gist",
        )


def downgrade() -> None:
    bind = op.get_bind()
    dialect_name = bind.dialect.name

    if dialect_name == "postgresql":
        op.drop_index("ix_rain_overlays_bounds_geom", table_name="rain_overlays")
        op.drop_index("ix_historical_events_coords_geom", table_name="historical_events")
        op.drop_index("ix_road_segments_coords_geom", table_name="road_segments")
        op.drop_index("ix_zones_polygon_geom", table_name="zones")
        op.drop_index("ix_zones_centroid_geom", table_name="zones")

    op.drop_column("rain_overlays", "bounds_geom")
    op.drop_column("historical_events", "coords_geom")
    op.drop_column("road_segments", "coords_geom")
    op.drop_column("zones", "polygon_geom")
    op.drop_column("zones", "centroid_geom")
