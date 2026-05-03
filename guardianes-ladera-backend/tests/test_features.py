from datetime import datetime, timezone


def test_zone_feature_builder_derives_spatial_features_from_zone_geometry(tmp_path, monkeypatch):
    database_path = tmp_path / "guardianes_feature_builder.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("RUN_DB_MIGRATIONS_ON_STARTUP", "true")
    monkeypatch.setenv("SEED_DEMO_DATA", "true")

    from app.core.config import get_settings
    from app.db.bootstrap import init_database, seed_demo_data
    from app.db.session import reset_engine_cache, session_scope
    from app.ml.features import ZoneFeatureBuilder
    from app.models import Zone

    get_settings.cache_clear()
    reset_engine_cache()

    init_database()
    with session_scope() as session:
        seed_demo_data(session)

    with session_scope() as session:
        zone = session.get(Zone, "moc-01")
        assert zone is not None
        builder = ZoneFeatureBuilder(session)
        snapshot = builder.build_for_zone(
            zone,
            as_of=datetime(2026, 3, 25, tzinfo=timezone.utc),
        )

        assert snapshot.municipality_event_count == 3
        assert snapshot.zone_event_count == 2
        assert snapshot.recent_zone_event_count == 1
        assert snapshot.intersecting_road_count == 2
        assert snapshot.intersecting_road_length_km == 70.0
        assert snapshot.rain_overlay_count == 2
        assert snapshot.rain_overlay_peak_intensity == 3
        assert snapshot.rain_overlay_peak_label == "alta"
