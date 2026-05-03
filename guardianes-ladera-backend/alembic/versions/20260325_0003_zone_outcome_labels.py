"""Add governed zone outcome labels for supervised training.

Revision ID: 20260325_0003
Revises: 20260325_0002
Create Date: 2026-03-25 02:10:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260325_0003"
down_revision = "20260325_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "zone_outcome_labels",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("zone_id", sa.String(length=32), nullable=False),
        sa.Column("feature_run_id", sa.Integer(), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("target_score", sa.Float(), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="confirmed"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("evidence", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["feature_run_id"], ["prediction_runs.id"]),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"]),
        sa.UniqueConstraint("zone_id", "observed_at", "source", name="uq_zone_outcome_label_observation"),
    )
    op.create_index("ix_zone_outcome_labels_zone_id", "zone_outcome_labels", ["zone_id"], unique=False)
    op.create_index(
        "ix_zone_outcome_labels_feature_run_id",
        "zone_outcome_labels",
        ["feature_run_id"],
        unique=False,
    )
    op.create_index(
        "ix_zone_outcome_labels_observed_at",
        "zone_outcome_labels",
        ["observed_at"],
        unique=False,
    )
    op.create_index("ix_zone_outcome_labels_source", "zone_outcome_labels", ["source"], unique=False)
    op.create_index("ix_zone_outcome_labels_status", "zone_outcome_labels", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_zone_outcome_labels_status", table_name="zone_outcome_labels")
    op.drop_index("ix_zone_outcome_labels_source", table_name="zone_outcome_labels")
    op.drop_index("ix_zone_outcome_labels_observed_at", table_name="zone_outcome_labels")
    op.drop_index("ix_zone_outcome_labels_feature_run_id", table_name="zone_outcome_labels")
    op.drop_index("ix_zone_outcome_labels_zone_id", table_name="zone_outcome_labels")
    op.drop_table("zone_outcome_labels")
