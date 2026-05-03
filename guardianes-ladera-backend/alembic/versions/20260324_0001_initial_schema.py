"""Initial backend schema.

Revision ID: 20260324_0001
Revises:
Create Date: 2026-03-24 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260324_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "municipalities",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("center", sa.JSON(), nullable=False),
        sa.Column("zoom", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_municipalities_name"), "municipalities", ["name"], unique=True)

    op.create_table(
        "source_catalog",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_source_catalog_category"), "source_catalog", ["category"], unique=False)

    op.create_table(
        "user_accounts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("password_hash", sa.String(length=256), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_user_accounts_username"), "user_accounts", ["username"], unique=True)

    op.create_table(
        "prediction_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("partial_data", sa.Boolean(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "job_executions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_job_executions_job_type"), "job_executions", ["job_type"], unique=False)

    op.create_table(
        "road_segments",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("municipality_id", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("coords", sa.JSON(), nullable=False),
        sa.Column("risk_level", sa.String(length=16), nullable=False),
        sa.Column("length_km", sa.Float(), nullable=False),
        sa.Column("note", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["municipality_id"], ["municipalities.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_road_segments_municipality_id"), "road_segments", ["municipality_id"], unique=False)

    op.create_table(
        "zones",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("municipality_id", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("centroid", sa.JSON(), nullable=False),
        sa.Column("polygon", sa.JSON(), nullable=False),
        sa.Column("exposure", sa.JSON(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["municipality_id"], ["municipalities.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_zones_municipality_id"), "zones", ["municipality_id"], unique=False)
    op.create_index(op.f("ix_zones_type"), "zones", ["type"], unique=False)

    op.create_table(
        "historical_events",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("municipality_id", sa.String(length=32), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("coords", sa.JSON(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(["municipality_id"], ["municipalities.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_historical_events_municipality_id"), "historical_events", ["municipality_id"], unique=False)

    op.create_table(
        "municipality_rain_points",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("municipality_id", sa.String(length=32), nullable=False),
        sa.Column("time_label", sa.String(length=32), nullable=False),
        sa.Column("observed", sa.Float(), nullable=True),
        sa.Column("forecast", sa.Float(), nullable=True),
        sa.Column("forecast_low", sa.Float(), nullable=True),
        sa.Column("forecast_high", sa.Float(), nullable=True),
        sa.Column("forecast_range", sa.Float(), nullable=True),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["municipality_id"], ["municipalities.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_municipality_rain_points_municipality_id"),
        "municipality_rain_points",
        ["municipality_id"],
        unique=False,
    )

    op.create_table(
        "rain_overlays",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("municipality_id", sa.String(length=32), nullable=False),
        sa.Column("bounds", sa.JSON(), nullable=False),
        sa.Column("intensity", sa.String(length=16), nullable=False),
        sa.ForeignKeyConstraint(["municipality_id"], ["municipalities.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_rain_overlays_municipality_id"), "rain_overlays", ["municipality_id"], unique=False)

    op.create_table(
        "source_sync_status",
        sa.Column("source_id", sa.String(length=32), nullable=False),
        sa.Column("last_synced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_success_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("status_note", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["source_id"], ["source_catalog.id"]),
        sa.PrimaryKeyConstraint("source_id"),
    )

    op.create_table(
        "source_sync_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source_id", sa.String(length=32), nullable=False),
        sa.Column("origin", sa.String(length=32), nullable=False),
        sa.Column("adapter_key", sa.String(length=64), nullable=False),
        sa.Column("transport", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("processed_records", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["source_id"], ["source_catalog.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_source_sync_events_source_id"), "source_sync_events", ["source_id"], unique=False)

    op.create_table(
        "ungrd_records",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("municipality_id", sa.String(length=32), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["municipality_id"], ["municipalities.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_ungrd_records_municipality_id"), "ungrd_records", ["municipality_id"], unique=False)

    op.create_table(
        "zone_road_segments",
        sa.Column("zone_id", sa.String(length=32), nullable=False),
        sa.Column("road_segment_id", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(["road_segment_id"], ["road_segments.id"]),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"]),
        sa.PrimaryKeyConstraint("zone_id", "road_segment_id"),
    )

    op.create_table(
        "zone_predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("zone_id", sa.String(length=32), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("confidence", sa.String(length=16), nullable=False),
        sa.Column("drivers", sa.JSON(), nullable=False),
        sa.Column("risk_delta", sa.Float(), nullable=False),
        sa.Column("trend", sa.String(length=16), nullable=False),
        sa.Column("source_snapshot", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["prediction_runs.id"]),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "zone_id", name="uq_zone_prediction_run"),
    )
    op.create_index(op.f("ix_zone_predictions_run_id"), "zone_predictions", ["run_id"], unique=False)
    op.create_index(op.f("ix_zone_predictions_zone_id"), "zone_predictions", ["zone_id"], unique=False)

    op.create_table(
        "zone_explanations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("prediction_id", sa.Integer(), nullable=False),
        sa.Column("mode", sa.String(length=32), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("driver_chips", sa.JSON(), nullable=False),
        sa.Column("suggestions", sa.JSON(), nullable=False),
        sa.Column("data_warnings", sa.JSON(), nullable=False),
        sa.Column("trace", sa.JSON(), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["prediction_id"], ["zone_predictions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("prediction_id"),
    )


def downgrade() -> None:
    op.drop_table("zone_explanations")
    op.drop_index(op.f("ix_zone_predictions_zone_id"), table_name="zone_predictions")
    op.drop_index(op.f("ix_zone_predictions_run_id"), table_name="zone_predictions")
    op.drop_table("zone_predictions")
    op.drop_table("zone_road_segments")
    op.drop_index(op.f("ix_ungrd_records_municipality_id"), table_name="ungrd_records")
    op.drop_table("ungrd_records")
    op.drop_index(op.f("ix_source_sync_events_source_id"), table_name="source_sync_events")
    op.drop_table("source_sync_events")
    op.drop_table("source_sync_status")
    op.drop_index(op.f("ix_rain_overlays_municipality_id"), table_name="rain_overlays")
    op.drop_table("rain_overlays")
    op.drop_index(op.f("ix_municipality_rain_points_municipality_id"), table_name="municipality_rain_points")
    op.drop_table("municipality_rain_points")
    op.drop_index(op.f("ix_historical_events_municipality_id"), table_name="historical_events")
    op.drop_table("historical_events")
    op.drop_index(op.f("ix_zones_type"), table_name="zones")
    op.drop_index(op.f("ix_zones_municipality_id"), table_name="zones")
    op.drop_table("zones")
    op.drop_index(op.f("ix_road_segments_municipality_id"), table_name="road_segments")
    op.drop_table("road_segments")
    op.drop_index(op.f("ix_job_executions_job_type"), table_name="job_executions")
    op.drop_table("job_executions")
    op.drop_table("prediction_runs")
    op.drop_index(op.f("ix_user_accounts_username"), table_name="user_accounts")
    op.drop_table("user_accounts")
    op.drop_index(op.f("ix_source_catalog_category"), table_name="source_catalog")
    op.drop_table("source_catalog")
    op.drop_index(op.f("ix_municipalities_name"), table_name="municipalities")
    op.drop_table("municipalities")
