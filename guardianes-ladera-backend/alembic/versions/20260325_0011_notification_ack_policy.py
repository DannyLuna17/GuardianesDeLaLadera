"""add notification delivery policy metadata

Revision ID: 20260325_0011
Revises: 20260325_0010
Create Date: 2026-03-25 21:10:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "20260325_0011"
down_revision = "20260325_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("notification_events", sa.Column("delivery_channels", sa.JSON(), nullable=True))
    op.add_column("notification_events", sa.Column("ack_due_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("notification_events", sa.Column("reminder_count", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("notification_events", sa.Column("last_reminder_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_notification_events_ack_due_at", "notification_events", ["ack_due_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_notification_events_ack_due_at", table_name="notification_events")
    op.drop_column("notification_events", "last_reminder_at")
    op.drop_column("notification_events", "reminder_count")
    op.drop_column("notification_events", "ack_due_at")
    op.drop_column("notification_events", "delivery_channels")
