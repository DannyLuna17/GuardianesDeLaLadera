from sqlalchemy import inspect, select


def test_init_database_runs_alembic_and_creates_version_table(tmp_path, monkeypatch):
    database_path = tmp_path / "guardianes_migrations.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("RUN_DB_MIGRATIONS_ON_STARTUP", "true")

    from app.core.config import get_settings
    from app.db.bootstrap import init_database
    from app.db.session import get_engine, reset_engine_cache

    get_settings.cache_clear()
    reset_engine_cache()

    init_database()

    inspector = inspect(get_engine())
    tables = set(inspector.get_table_names())
    assert "alembic_version" in tables
    assert "municipalities" in tables
    assert "notification_delivery_attempts" in tables
    assert "notification_events" in tables
    assert "model_review_tasks" in tables
    assert "zone_predictions" in tables
    assert "zone_outcome_labels" in tables

    zone_columns = {column["name"] for column in inspector.get_columns("zones")}
    road_columns = {column["name"] for column in inspector.get_columns("road_segments")}
    historical_event_columns = {column["name"] for column in inspector.get_columns("historical_events")}
    overlay_columns = {column["name"] for column in inspector.get_columns("rain_overlays")}
    outcome_label_columns = {column["name"] for column in inspector.get_columns("zone_outcome_labels")}
    notification_columns = {column["name"] for column in inspector.get_columns("notification_events")}
    notification_attempt_columns = {column["name"] for column in inspector.get_columns("notification_delivery_attempts")}
    model_review_task_columns = {column["name"] for column in inspector.get_columns("model_review_tasks")}

    assert {"centroid_geom", "polygon_geom"} <= zone_columns
    assert "coords_geom" in road_columns
    assert "coords_geom" in historical_event_columns
    assert "bounds_geom" in overlay_columns
    assert {
        "delivery_channels",
        "ack_due_at",
        "reminder_count",
        "last_reminder_at",
    } <= notification_columns
    assert {
        "notification_event_id",
        "channel",
        "adapter_key",
        "status",
        "attempted_at",
        "completed_at",
        "delivery_reference",
        "error_message",
        "details",
    } <= notification_attempt_columns
    assert {
        "review_type",
        "status",
        "source_notification_id",
        "source_event_type",
        "source_alert_severity",
        "source_alert_status",
        "active_model_version",
        "candidate_model_version",
        "dataset_version",
        "title",
        "summary",
        "recommended_action",
        "assigned_reviewer",
        "due_at",
        "decision",
        "resolution_notes",
        "details",
        "created_at",
        "created_by",
        "updated_at",
        "updated_by",
        "resolved_at",
        "resolved_by",
    } <= model_review_task_columns
    assert {
        "assigned_reviewer",
        "assigned_at",
        "review_due_at",
        "training_eligibility_status",
        "training_eligibility_updated_at",
        "training_eligibility_updated_by",
        "training_eligibility_notes",
        "training_release_status",
        "training_release_criteria",
        "training_release_requested_at",
        "training_release_requested_by",
        "training_release_requested_notes",
        "training_release_reviewed_at",
        "training_release_reviewed_by",
        "training_release_review_notes",
        "training_release_assigned_reviewer",
        "training_release_assigned_at",
        "training_release_due_at",
        "training_release_escalation_status",
        "training_release_escalation_level",
        "training_release_escalated_at",
        "training_release_escalated_by",
        "training_release_escalation_reason",
        "reviewed_at",
        "reviewed_by",
        "review_notes",
    } <= outcome_label_columns


def test_seed_demo_data_populates_spatial_shadow_columns(tmp_path, monkeypatch):
    database_path = tmp_path / "guardianes_seed_spatial.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("RUN_DB_MIGRATIONS_ON_STARTUP", "true")

    from app.core.config import get_settings
    from app.db.bootstrap import init_database, seed_demo_data
    from app.db.session import reset_engine_cache, session_scope
    from app.models import HistoricalEvent, RainOverlay, RoadSegment, Zone

    get_settings.cache_clear()
    reset_engine_cache()

    init_database()

    with session_scope() as session:
        seed_demo_data(session)

    with session_scope() as session:
        zone = session.scalars(select(Zone)).first()
        road_segment = session.scalars(select(RoadSegment)).first()
        historical_event = session.scalars(select(HistoricalEvent)).first()
        overlay = session.scalars(select(RainOverlay)).first()

        assert zone is not None
        assert zone.centroid_geom == {
            "type": "Point",
            "coordinates": [zone.centroid[1], zone.centroid[0]],
        }
        assert zone.polygon_geom["type"] == "Polygon"

        assert road_segment is not None
        assert road_segment.coords_geom["type"] == "LineString"

        assert historical_event is not None
        assert historical_event.coords_geom == {
            "type": "Point",
            "coordinates": [historical_event.coords[1], historical_event.coords[0]],
        }

        assert overlay is not None
        assert overlay.bounds_geom["type"] == "Polygon"
