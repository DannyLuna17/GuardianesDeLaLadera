"""add model review tasks

Revision ID: 20260327_0013
Revises: 20260325_0012
Create Date: 2026-03-27 09:30:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260327_0013"
down_revision = "20260325_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_review_tasks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("review_type", sa.String(length=32), nullable=False),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default="open",
        ),
        sa.Column(
            "source_notification_id",
            sa.Integer(),
            sa.ForeignKey("notification_events.id"),
            nullable=False,
        ),
        sa.Column("source_event_type", sa.String(length=64), nullable=False),
        sa.Column("source_alert_severity", sa.String(length=32), nullable=False),
        sa.Column("source_alert_status", sa.String(length=32), nullable=False),
        sa.Column("active_model_version", sa.String(length=64), nullable=False),
        sa.Column("candidate_model_version", sa.String(length=64), nullable=True),
        sa.Column("dataset_version", sa.String(length=128), nullable=True),
        sa.Column("title", sa.String(length=160), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("recommended_action", sa.Text(), nullable=True),
        sa.Column("assigned_reviewer", sa.String(length=64), nullable=True),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decision", sa.String(length=64), nullable=True),
        sa.Column("resolution_notes", sa.Text(), nullable=True),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(length=64), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_by", sa.String(length=64), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_by", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_model_review_tasks_review_type",
        "model_review_tasks",
        ["review_type"],
        unique=False,
    )
    op.create_index(
        "ix_model_review_tasks_status",
        "model_review_tasks",
        ["status"],
        unique=False,
    )
    op.create_index(
        "ix_model_review_tasks_source_notification_id",
        "model_review_tasks",
        ["source_notification_id"],
        unique=False,
    )
    op.create_index(
        "ix_model_review_tasks_source_event_type",
        "model_review_tasks",
        ["source_event_type"],
        unique=False,
    )
    op.create_index(
        "ix_model_review_tasks_source_alert_severity",
        "model_review_tasks",
        ["source_alert_severity"],
        unique=False,
    )
    op.create_index(
        "ix_model_review_tasks_source_alert_status",
        "model_review_tasks",
        ["source_alert_status"],
        unique=False,
    )
    op.create_index(
        "ix_model_review_tasks_active_model_version",
        "model_review_tasks",
        ["active_model_version"],
        unique=False,
    )
    op.create_index(
        "ix_model_review_tasks_candidate_model_version",
        "model_review_tasks",
        ["candidate_model_version"],
        unique=False,
    )
    op.create_index(
        "ix_model_review_tasks_dataset_version",
        "model_review_tasks",
        ["dataset_version"],
        unique=False,
    )
    op.create_index(
        "ix_model_review_tasks_assigned_reviewer",
        "model_review_tasks",
        ["assigned_reviewer"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_model_review_tasks_assigned_reviewer",
        table_name="model_review_tasks",
    )
    op.drop_index(
        "ix_model_review_tasks_dataset_version",
        table_name="model_review_tasks",
    )
    op.drop_index(
        "ix_model_review_tasks_candidate_model_version",
        table_name="model_review_tasks",
    )
    op.drop_index(
        "ix_model_review_tasks_active_model_version",
        table_name="model_review_tasks",
    )
    op.drop_index(
        "ix_model_review_tasks_source_alert_status",
        table_name="model_review_tasks",
    )
    op.drop_index(
        "ix_model_review_tasks_source_alert_severity",
        table_name="model_review_tasks",
    )
    op.drop_index(
        "ix_model_review_tasks_source_event_type",
        table_name="model_review_tasks",
    )
    op.drop_index(
        "ix_model_review_tasks_source_notification_id",
        table_name="model_review_tasks",
    )
    op.drop_index(
        "ix_model_review_tasks_status",
        table_name="model_review_tasks",
    )
    op.drop_index(
        "ix_model_review_tasks_review_type",
        table_name="model_review_tasks",
    )
    op.drop_table("model_review_tasks")
