"""add training release escalation metadata

Revision ID: 20260325_0009
Revises: 20260325_0008
Create Date: 2026-03-25 17:10:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "20260325_0009"
down_revision = "20260325_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "zone_outcome_labels",
        sa.Column("training_release_escalation_status", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "zone_outcome_labels",
        sa.Column("training_release_escalation_level", sa.Integer(), nullable=True),
    )
    op.add_column(
        "zone_outcome_labels",
        sa.Column("training_release_escalated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "zone_outcome_labels",
        sa.Column("training_release_escalated_by", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "zone_outcome_labels",
        sa.Column("training_release_escalation_reason", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_zone_outcome_labels_training_release_escalation_status",
        "zone_outcome_labels",
        ["training_release_escalation_status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_zone_outcome_labels_training_release_escalation_status",
        table_name="zone_outcome_labels",
    )
    op.drop_column("zone_outcome_labels", "training_release_escalation_reason")
    op.drop_column("zone_outcome_labels", "training_release_escalated_by")
    op.drop_column("zone_outcome_labels", "training_release_escalated_at")
    op.drop_column("zone_outcome_labels", "training_release_escalation_level")
    op.drop_column("zone_outcome_labels", "training_release_escalation_status")
