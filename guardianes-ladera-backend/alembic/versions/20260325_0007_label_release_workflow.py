"""Add auditable release workflow fields for held labels.

Revision ID: 20260325_0007
Revises: 20260325_0006
Create Date: 2026-03-25 07:25:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260325_0007"
down_revision = "20260325_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("zone_outcome_labels", sa.Column("training_release_status", sa.String(length=32), nullable=True))
    op.add_column("zone_outcome_labels", sa.Column("training_release_criteria", sa.JSON(), nullable=True))
    op.add_column("zone_outcome_labels", sa.Column("training_release_requested_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("zone_outcome_labels", sa.Column("training_release_requested_by", sa.String(length=64), nullable=True))
    op.add_column("zone_outcome_labels", sa.Column("training_release_requested_notes", sa.Text(), nullable=True))
    op.add_column("zone_outcome_labels", sa.Column("training_release_reviewed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("zone_outcome_labels", sa.Column("training_release_reviewed_by", sa.String(length=64), nullable=True))
    op.add_column("zone_outcome_labels", sa.Column("training_release_review_notes", sa.Text(), nullable=True))
    op.create_index(
        "ix_zone_outcome_labels_training_release_status",
        "zone_outcome_labels",
        ["training_release_status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_zone_outcome_labels_training_release_status", table_name="zone_outcome_labels")
    op.drop_column("zone_outcome_labels", "training_release_review_notes")
    op.drop_column("zone_outcome_labels", "training_release_reviewed_by")
    op.drop_column("zone_outcome_labels", "training_release_reviewed_at")
    op.drop_column("zone_outcome_labels", "training_release_requested_notes")
    op.drop_column("zone_outcome_labels", "training_release_requested_by")
    op.drop_column("zone_outcome_labels", "training_release_requested_at")
    op.drop_column("zone_outcome_labels", "training_release_criteria")
    op.drop_column("zone_outcome_labels", "training_release_status")
