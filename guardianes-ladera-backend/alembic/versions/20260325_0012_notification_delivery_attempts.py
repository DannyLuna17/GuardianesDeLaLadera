"""add notification delivery attempts

Revision ID: 20260325_0012
Revises: 20260325_0011
Create Date: 2026-03-25 22:05:00.000000
"""

from __future__ import annotations

import json

from alembic import op
import sqlalchemy as sa


revision = "20260325_0012"
down_revision = "20260325_0011"
branch_labels = None
depends_on = None


def _normalize_channels(raw_channels, fallback_channel: str | None) -> list[str]:
    if isinstance(raw_channels, list):
        values = [str(item).strip() for item in raw_channels if str(item).strip()]
        if values:
            return values
    if isinstance(raw_channels, str) and raw_channels.strip():
        try:
            parsed = json.loads(raw_channels)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            values = [str(item).strip() for item in parsed if str(item).strip()]
            if values:
                return values
    if fallback_channel and fallback_channel.strip():
        return [fallback_channel.strip()]
    return ["in_app"]


def upgrade() -> None:
    op.create_table(
        "notification_delivery_attempts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("notification_event_id", sa.Integer(), sa.ForeignKey("notification_events.id"), nullable=False),
        sa.Column("channel", sa.String(length=32), nullable=False),
        sa.Column("adapter_key", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="completed"),
        sa.Column("attempted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivery_reference", sa.String(length=160), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("details", sa.JSON(), nullable=False),
    )
    op.create_index(
        "ix_notification_delivery_attempts_notification_event_id",
        "notification_delivery_attempts",
        ["notification_event_id"],
        unique=False,
    )
    op.create_index("ix_notification_delivery_attempts_channel", "notification_delivery_attempts", ["channel"], unique=False)
    op.create_index("ix_notification_delivery_attempts_status", "notification_delivery_attempts", ["status"], unique=False)
    op.create_index(
        "ix_notification_delivery_attempts_attempted_at",
        "notification_delivery_attempts",
        ["attempted_at"],
        unique=False,
    )

    bind = op.get_bind()
    notification_events = sa.table(
        "notification_events",
        sa.column("id", sa.Integer()),
        sa.column("channel", sa.String()),
        sa.column("delivery_channels", sa.JSON()),
        sa.column("created_at", sa.DateTime(timezone=True)),
    )
    notification_delivery_attempts = sa.table(
        "notification_delivery_attempts",
        sa.column("notification_event_id", sa.Integer()),
        sa.column("channel", sa.String()),
        sa.column("adapter_key", sa.String()),
        sa.column("status", sa.String()),
        sa.column("attempted_at", sa.DateTime(timezone=True)),
        sa.column("completed_at", sa.DateTime(timezone=True)),
        sa.column("delivery_reference", sa.String()),
        sa.column("error_message", sa.Text()),
        sa.column("details", sa.JSON()),
    )

    existing_notifications = bind.execute(
        sa.select(
            notification_events.c.id,
            notification_events.c.channel,
            notification_events.c.delivery_channels,
            notification_events.c.created_at,
        )
    ).mappings()
    attempt_rows = []
    for row in existing_notifications:
        channels = _normalize_channels(row["delivery_channels"], row["channel"])
        for channel in channels:
            attempt_rows.append(
                {
                    "notification_event_id": row["id"],
                    "channel": channel,
                    "adapter_key": f"legacy.{channel}",
                    "status": "completed",
                    "attempted_at": row["created_at"],
                    "completed_at": row["created_at"],
                    "delivery_reference": f"legacy:{row['id']}:{channel}",
                    "error_message": None,
                    "details": {
                        "backfilled": True,
                        "backfill_reason": "preexisting_notification_event",
                    },
                }
            )
    if attempt_rows:
        op.bulk_insert(notification_delivery_attempts, attempt_rows)


def downgrade() -> None:
    op.drop_index("ix_notification_delivery_attempts_attempted_at", table_name="notification_delivery_attempts")
    op.drop_index("ix_notification_delivery_attempts_status", table_name="notification_delivery_attempts")
    op.drop_index("ix_notification_delivery_attempts_channel", table_name="notification_delivery_attempts")
    op.drop_index("ix_notification_delivery_attempts_notification_event_id", table_name="notification_delivery_attempts")
    op.drop_table("notification_delivery_attempts")
