"""Add training eligibility fields to governed labels.

Revision ID: 20260325_0006
Revises: 20260325_0005
Create Date: 2026-03-25 06:35:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260325_0006"
down_revision = "20260325_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("zone_outcome_labels", sa.Column("training_eligibility_status", sa.String(length=32), nullable=True))
    op.add_column("zone_outcome_labels", sa.Column("training_eligibility_updated_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("zone_outcome_labels", sa.Column("training_eligibility_updated_by", sa.String(length=64), nullable=True))
    op.add_column("zone_outcome_labels", sa.Column("training_eligibility_notes", sa.Text(), nullable=True))
    op.create_index(
        "ix_zone_outcome_labels_training_eligibility_status",
        "zone_outcome_labels",
        ["training_eligibility_status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_zone_outcome_labels_training_eligibility_status", table_name="zone_outcome_labels")
    op.drop_column("zone_outcome_labels", "training_eligibility_notes")
    op.drop_column("zone_outcome_labels", "training_eligibility_updated_by")
    op.drop_column("zone_outcome_labels", "training_eligibility_updated_at")
    op.drop_column("zone_outcome_labels", "training_eligibility_status")
