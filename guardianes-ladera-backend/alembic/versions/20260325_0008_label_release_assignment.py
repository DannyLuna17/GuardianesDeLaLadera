"""Add assignment fields for pending training release reviews.

Revision ID: 20260325_0008
Revises: 20260325_0007
Create Date: 2026-03-25 08:05:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260325_0008"
down_revision = "20260325_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "zone_outcome_labels",
        sa.Column("training_release_assigned_reviewer", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "zone_outcome_labels",
        sa.Column("training_release_assigned_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "zone_outcome_labels",
        sa.Column("training_release_due_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_zone_outcome_labels_training_release_assigned_reviewer",
        "zone_outcome_labels",
        ["training_release_assigned_reviewer"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_zone_outcome_labels_training_release_assigned_reviewer", table_name="zone_outcome_labels")
    op.drop_column("zone_outcome_labels", "training_release_due_at")
    op.drop_column("zone_outcome_labels", "training_release_assigned_at")
    op.drop_column("zone_outcome_labels", "training_release_assigned_reviewer")
