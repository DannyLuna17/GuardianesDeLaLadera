"""Add provenance columns for structural catalog entities.

Revision ID: 20260328_0014
Revises: 20260327_0013
Create Date: 2026-03-28 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260328_0014"
down_revision = "20260327_0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("municipalities", sa.Column("source_id", sa.String(length=32), nullable=True))
    op.add_column("municipalities", sa.Column("source_ref", sa.String(length=160), nullable=True))
    op.create_index(op.f("ix_municipalities_source_id"), "municipalities", ["source_id"], unique=False)

    op.add_column("zones", sa.Column("source_id", sa.String(length=32), nullable=True))
    op.add_column("zones", sa.Column("source_ref", sa.String(length=160), nullable=True))
    op.create_index(op.f("ix_zones_source_id"), "zones", ["source_id"], unique=False)

    op.add_column("road_segments", sa.Column("source_id", sa.String(length=32), nullable=True))
    op.add_column("road_segments", sa.Column("source_ref", sa.String(length=160), nullable=True))
    op.create_index(op.f("ix_road_segments_source_id"), "road_segments", ["source_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_road_segments_source_id"), table_name="road_segments")
    op.drop_column("road_segments", "source_ref")
    op.drop_column("road_segments", "source_id")

    op.drop_index(op.f("ix_zones_source_id"), table_name="zones")
    op.drop_column("zones", "source_ref")
    op.drop_column("zones", "source_id")

    op.drop_index(op.f("ix_municipalities_source_id"), table_name="municipalities")
    op.drop_column("municipalities", "source_ref")
    op.drop_column("municipalities", "source_id")
