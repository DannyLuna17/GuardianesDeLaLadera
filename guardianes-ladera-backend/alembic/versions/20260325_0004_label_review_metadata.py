"""Add review metadata to governed zone outcome labels.

Revision ID: 20260325_0004
Revises: 20260325_0003
Create Date: 2026-03-25 03:05:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260325_0004"
down_revision = "20260325_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("zone_outcome_labels", sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("zone_outcome_labels", sa.Column("reviewed_by", sa.String(length=64), nullable=True))
    op.add_column("zone_outcome_labels", sa.Column("review_notes", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("zone_outcome_labels", "review_notes")
    op.drop_column("zone_outcome_labels", "reviewed_by")
    op.drop_column("zone_outcome_labels", "reviewed_at")
