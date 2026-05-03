"""add notification events table

Revision ID: 20260325_0010
Revises: 20260325_0009
Create Date: 2026-03-25 18:05:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "20260325_0010"
down_revision = "20260325_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "notification_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("severity", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("channel", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=160), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("target_username", sa.String(length=64), nullable=True),
        sa.Column("related_label_id", sa.Integer(), nullable=True),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("acknowledged_by", sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(["related_label_id"], ["zone_outcome_labels.id"]),
    )
    op.create_index("ix_notification_events_event_type", "notification_events", ["event_type"], unique=False)
    op.create_index("ix_notification_events_severity", "notification_events", ["severity"], unique=False)
    op.create_index("ix_notification_events_status", "notification_events", ["status"], unique=False)
    op.create_index("ix_notification_events_target_username", "notification_events", ["target_username"], unique=False)
    op.create_index("ix_notification_events_related_label_id", "notification_events", ["related_label_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_notification_events_related_label_id", table_name="notification_events")
    op.drop_index("ix_notification_events_target_username", table_name="notification_events")
    op.drop_index("ix_notification_events_status", table_name="notification_events")
    op.drop_index("ix_notification_events_severity", table_name="notification_events")
    op.drop_index("ix_notification_events_event_type", table_name="notification_events")
    op.drop_table("notification_events")
