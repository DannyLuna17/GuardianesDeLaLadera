"""Add assignment fields for governed label review queues.

Revision ID: 20260325_0005
Revises: 20260325_0004
Create Date: 2026-03-25 05:45:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260325_0005"
down_revision = "20260325_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("zone_outcome_labels", sa.Column("assigned_reviewer", sa.String(length=64), nullable=True))
    op.add_column("zone_outcome_labels", sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("zone_outcome_labels", sa.Column("review_due_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index(
        "ix_zone_outcome_labels_assigned_reviewer",
        "zone_outcome_labels",
        ["assigned_reviewer"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_zone_outcome_labels_assigned_reviewer", table_name="zone_outcome_labels")
    op.drop_column("zone_outcome_labels", "review_due_at")
    op.drop_column("zone_outcome_labels", "assigned_at")
    op.drop_column("zone_outcome_labels", "assigned_reviewer")
