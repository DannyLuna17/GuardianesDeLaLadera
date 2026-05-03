from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import JSON, Boolean, Column, Date, DateTime, Float, ForeignKey, Integer, String, Table, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.spatial import linestring_geometry_type, point_geometry_type, polygon_geometry_type


zone_road_segments = Table(
    "zone_road_segments",
    Base.metadata,
    Column("zone_id", ForeignKey("zones.id"), primary_key=True),
    Column("road_segment_id", ForeignKey("road_segments.id"), primary_key=True),
)


class Municipality(Base):
    __tablename__ = "municipalities"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    center: Mapped[list[float]] = mapped_column(JSON)
    zoom: Mapped[int] = mapped_column(Integer)
    source_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    source_ref: Mapped[str | None] = mapped_column(String(160), nullable=True)

    zones: Mapped[list[Zone]] = relationship(back_populates="municipality")
    road_segments: Mapped[list[RoadSegment]] = relationship(back_populates="municipality")
    historical_events: Mapped[list[HistoricalEvent]] = relationship(back_populates="municipality")
    ungrd_records: Mapped[list[UngrdRecord]] = relationship(back_populates="municipality")
    rain_points: Mapped[list[MunicipalityRainPoint]] = relationship(back_populates="municipality")
    rain_overlays: Mapped[list[RainOverlay]] = relationship(back_populates="municipality")


class Zone(Base):
    __tablename__ = "zones"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    municipality_id: Mapped[str] = mapped_column(ForeignKey("municipalities.id"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    type: Mapped[str] = mapped_column(String(32), index=True)
    centroid: Mapped[list[float]] = mapped_column(JSON)
    centroid_geom: Mapped[object | dict | None] = mapped_column(point_geometry_type(), nullable=True)
    polygon: Mapped[list[list[float]]] = mapped_column(JSON)
    polygon_geom: Mapped[object | dict | None] = mapped_column(polygon_geometry_type(), nullable=True)
    exposure: Mapped[dict] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    source_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    source_ref: Mapped[str | None] = mapped_column(String(160), nullable=True)

    municipality: Mapped[Municipality] = relationship(back_populates="zones")
    road_segments: Mapped[list[RoadSegment]] = relationship(
        secondary=zone_road_segments,
        back_populates="zones",
    )
    predictions: Mapped[list[ZonePrediction]] = relationship(back_populates="zone")
    outcome_labels: Mapped[list[ZoneOutcomeLabel]] = relationship(back_populates="zone")


class RoadSegment(Base):
    __tablename__ = "road_segments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    municipality_id: Mapped[str] = mapped_column(ForeignKey("municipalities.id"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    coords: Mapped[list[list[float]]] = mapped_column(JSON)
    coords_geom: Mapped[object | dict | None] = mapped_column(linestring_geometry_type(), nullable=True)
    risk_level: Mapped[str] = mapped_column(String(16))
    length_km: Mapped[float] = mapped_column(Float)
    note: Mapped[str] = mapped_column(Text)
    source_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    source_ref: Mapped[str | None] = mapped_column(String(160), nullable=True)

    municipality: Mapped[Municipality] = relationship(back_populates="road_segments")
    zones: Mapped[list[Zone]] = relationship(
        secondary=zone_road_segments,
        back_populates="road_segments",
    )


class HistoricalEvent(Base):
    __tablename__ = "historical_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    municipality_id: Mapped[str] = mapped_column(ForeignKey("municipalities.id"), index=True)
    date: Mapped[date] = mapped_column(Date)
    severity: Mapped[str] = mapped_column(String(16))
    type: Mapped[str] = mapped_column(String(32))
    coords: Mapped[list[float]] = mapped_column(JSON)
    coords_geom: Mapped[object | dict | None] = mapped_column(point_geometry_type(), nullable=True)
    source: Mapped[str] = mapped_column(String(32))

    municipality: Mapped[Municipality] = relationship(back_populates="historical_events")


class UngrdRecord(Base):
    __tablename__ = "ungrd_records"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    municipality_id: Mapped[str] = mapped_column(ForeignKey("municipalities.id"), index=True)
    date: Mapped[date] = mapped_column(Date)
    summary: Mapped[str] = mapped_column(Text)

    municipality: Mapped[Municipality] = relationship(back_populates="ungrd_records")


class SourceCatalog(Base):
    __tablename__ = "source_catalog"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    label: Mapped[str] = mapped_column(String(128))
    category: Mapped[str] = mapped_column(String(32), index=True)

    sync_status: Mapped[SourceSyncStatus | None] = relationship(back_populates="source", uselist=False)
    sync_events: Mapped[list[SourceSyncEvent]] = relationship(back_populates="source")


class SourceSyncStatus(Base):
    __tablename__ = "source_sync_status"

    source_id: Mapped[str] = mapped_column(ForeignKey("source_catalog.id"), primary_key=True)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    status_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    source: Mapped[SourceCatalog] = relationship(back_populates="sync_status")


class SourceSyncEvent(Base):
    __tablename__ = "source_sync_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("source_catalog.id"), index=True)
    origin: Mapped[str] = mapped_column(String(32), default="manual")
    adapter_key: Mapped[str] = mapped_column(String(64))
    transport: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default="completed")
    processed_records: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    message: Mapped[str] = mapped_column(Text)
    details: Mapped[dict] = mapped_column(JSON, default=dict)

    source: Mapped[SourceCatalog] = relationship(back_populates="sync_events")


class PredictionRun(Base):
    __tablename__ = "prediction_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(32), default="completed")
    model_version: Mapped[str] = mapped_column(String(64))
    partial_data: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    predictions: Mapped[list[ZonePrediction]] = relationship(back_populates="run")
    outcome_labels: Mapped[list[ZoneOutcomeLabel]] = relationship(back_populates="feature_run")


class ZonePrediction(Base):
    __tablename__ = "zone_predictions"
    __table_args__ = (UniqueConstraint("run_id", "zone_id", name="uq_zone_prediction_run"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("prediction_runs.id"), index=True)
    zone_id: Mapped[str] = mapped_column(ForeignKey("zones.id"), index=True)
    risk_score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[str] = mapped_column(String(16))
    drivers: Mapped[dict] = mapped_column(JSON)
    risk_delta: Mapped[float] = mapped_column(Float, default=0.0)
    trend: Mapped[str] = mapped_column(String(16), default="estable")
    source_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    run: Mapped[PredictionRun] = relationship(back_populates="predictions")
    zone: Mapped[Zone] = relationship(back_populates="predictions")
    explanation: Mapped[ZoneExplanation | None] = relationship(back_populates="prediction", uselist=False)


class ZoneExplanation(Base):
    __tablename__ = "zone_explanations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("zone_predictions.id"), unique=True)
    mode: Mapped[str] = mapped_column(String(32), default="template")
    summary: Mapped[str] = mapped_column(Text)
    driver_chips: Mapped[list[str]] = mapped_column(JSON)
    suggestions: Mapped[list[str]] = mapped_column(JSON)
    data_warnings: Mapped[list[str]] = mapped_column(JSON)
    trace: Mapped[dict] = mapped_column(JSON, default=dict)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    prediction: Mapped[ZonePrediction] = relationship(back_populates="explanation")


class MunicipalityRainPoint(Base):
    __tablename__ = "municipality_rain_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    municipality_id: Mapped[str] = mapped_column(ForeignKey("municipalities.id"), index=True)
    time_label: Mapped[str] = mapped_column(String(32))
    observed: Mapped[float | None] = mapped_column(Float, nullable=True)
    forecast: Mapped[float | None] = mapped_column(Float, nullable=True)
    forecast_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    forecast_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    forecast_range: Mapped[float | None] = mapped_column(Float, nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer)

    municipality: Mapped[Municipality] = relationship(back_populates="rain_points")


class RainOverlay(Base):
    __tablename__ = "rain_overlays"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    municipality_id: Mapped[str] = mapped_column(ForeignKey("municipalities.id"), index=True)
    bounds: Mapped[list[list[float]]] = mapped_column(JSON)
    bounds_geom: Mapped[object | dict | None] = mapped_column(polygon_geometry_type(), nullable=True)
    intensity: Mapped[str] = mapped_column(String(16))

    municipality: Mapped[Municipality] = relationship(back_populates="rain_overlays")


class JobExecution(Base):
    __tablename__ = "job_executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_type: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="completed")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    details: Mapped[dict] = mapped_column(JSON, default=dict)


class UserAccount(Base):
    __tablename__ = "user_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    role: Mapped[str] = mapped_column(String(32), default="viewer")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class NotificationEvent(Base):
    __tablename__ = "notification_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[str] = mapped_column(String(32), index=True, default="info")
    status: Mapped[str] = mapped_column(String(32), index=True, default="open")
    channel: Mapped[str] = mapped_column(String(32), default="in_app")
    title: Mapped[str] = mapped_column(String(160))
    message: Mapped[str] = mapped_column(Text)
    target_username: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    related_label_id: Mapped[int | None] = mapped_column(ForeignKey("zone_outcome_labels.id"), nullable=True, index=True)
    delivery_channels: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ack_due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    reminder_count: Mapped[int] = mapped_column(Integer, default=0)
    last_reminder_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String(64), nullable=True)

    delivery_attempts: Mapped[list["NotificationDeliveryAttempt"]] = relationship(
        back_populates="notification",
        cascade="all, delete-orphan",
        order_by="NotificationDeliveryAttempt.attempted_at",
    )


class NotificationDeliveryAttempt(Base):
    __tablename__ = "notification_delivery_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    notification_event_id: Mapped[int] = mapped_column(ForeignKey("notification_events.id"), index=True)
    channel: Mapped[str] = mapped_column(String(32), index=True)
    adapter_key: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), index=True, default="completed")
    attempted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    delivery_reference: Mapped[str | None] = mapped_column(String(160), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    details: Mapped[dict] = mapped_column(JSON, default=dict)

    notification: Mapped[NotificationEvent] = relationship(back_populates="delivery_attempts")


class ModelReviewTask(Base):
    __tablename__ = "model_review_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    review_type: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True, default="open")
    source_notification_id: Mapped[int] = mapped_column(
        ForeignKey("notification_events.id"), index=True
    )
    source_event_type: Mapped[str] = mapped_column(String(64), index=True)
    source_alert_severity: Mapped[str] = mapped_column(String(32), index=True)
    source_alert_status: Mapped[str] = mapped_column(String(32), index=True)
    active_model_version: Mapped[str] = mapped_column(String(64), index=True)
    candidate_model_version: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    dataset_version: Mapped[str | None] = mapped_column(
        String(128), nullable=True, index=True
    )
    title: Mapped[str] = mapped_column(String(160))
    summary: Mapped[str] = mapped_column(Text)
    recommended_action: Mapped[str | None] = mapped_column(Text, nullable=True)
    assigned_reviewer: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    due_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    decision: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resolution_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_by: Mapped[str] = mapped_column(String(64))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    resolved_by: Mapped[str | None] = mapped_column(String(64), nullable=True)

    source_notification: Mapped[NotificationEvent] = relationship()


class ZoneOutcomeLabel(Base):
    __tablename__ = "zone_outcome_labels"
    __table_args__ = (
        UniqueConstraint("zone_id", "observed_at", "source", name="uq_zone_outcome_label_observation"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    zone_id: Mapped[str] = mapped_column(ForeignKey("zones.id"), index=True)
    feature_run_id: Mapped[int | None] = mapped_column(ForeignKey("prediction_runs.id"), index=True, nullable=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    target_score: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="confirmed", index=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)
    assigned_reviewer: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    assigned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    review_due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    training_eligibility_status: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    training_eligibility_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    training_eligibility_updated_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    training_eligibility_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    training_release_status: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    training_release_criteria: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    training_release_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    training_release_requested_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    training_release_requested_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    training_release_reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    training_release_reviewed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    training_release_review_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    training_release_assigned_reviewer: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    training_release_assigned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    training_release_due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    training_release_escalation_status: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    training_release_escalation_level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    training_release_escalated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    training_release_escalated_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    training_release_escalation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    review_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    zone: Mapped[Zone] = relationship(back_populates="outcome_labels")
    feature_run: Mapped[PredictionRun | None] = relationship(back_populates="outcome_labels")
