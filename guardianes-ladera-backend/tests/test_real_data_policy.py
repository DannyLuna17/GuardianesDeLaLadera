import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, select

from app.core.exceptions import ApiError


def promote_seed_structure_to_official_catalog(session) -> None:
    from app.models import Municipality, RoadSegment, SourceCatalog, SourceSyncEvent, SourceSyncStatus, Zone

    imported_at = datetime(2026, 3, 28, 12, 0, tzinfo=timezone.utc)
    source_specs = {
        "DANE": {
            "label": "DANE catalogo territorial",
            "category": "historico",
            "source_url": "https://geoportal.dane.gov.co/mparcgis/rest/services/MGN2024/Serv_CapasMGN_2024/MapServer/317",
        },
        "INVIAS": {
            "label": "INVIAS corredores",
            "category": "infraestructura",
            "source_url": "https://hermes.invias.gov.co/arcgis/rest/services/OpenData/ServiciosOpenData/FeatureServer/0",
        },
    }
    for source_id, spec in source_specs.items():
        source = session.get(SourceCatalog, source_id)
        if source is None:
            source = SourceCatalog(
                id=source_id,
                label=spec["label"],
                category=spec["category"],
            )
            session.add(source)
        else:
            source.label = spec["label"]
            source.category = spec["category"]

        status = source.sync_status
        if status is None:
            status = SourceSyncStatus(source=source)
            session.add(status)
        status.last_synced_at = imported_at
        status.last_success_at = imported_at
        status.last_error = None
        status.status_note = "Imported from official structural catalog bundle."

        session.add(
            SourceSyncEvent(
                source=source,
                origin="manual",
                adapter_key="official.structural_bundle",
                transport="file_import",
                status="completed",
                processed_records=0,
                started_at=imported_at,
                completed_at=imported_at,
                message="Imported from official structural catalog bundle.",
                details={
                    "structural_import": True,
                    "source_url": spec["source_url"],
                    "bundle_version": "test-fixture",
                },
            )
        )

    for municipality in session.scalars(select(Municipality)).all():
        municipality.source_id = "DANE"
        municipality.source_ref = f"municipio:{municipality.id}"
    for zone in session.scalars(select(Zone)).all():
        zone.source_id = "DANE"
        zone.source_ref = f"zona:{zone.id}"
    for road_segment in session.scalars(select(RoadSegment)).all():
        road_segment.source_id = "INVIAS"
        road_segment.source_ref = f"via:{road_segment.id}"


def test_auto_transport_requires_real_base_url_when_real_data_only(monkeypatch):
    monkeypatch.setenv("REAL_DATA_ONLY", "true")
    monkeypatch.setenv("IDEAM_TRANSPORT", "auto")
    monkeypatch.setenv("IDEAM_BASE_URL", "")

    from app.core.config import get_settings

    get_settings.cache_clear()

    from app.integrations.registry import resolve_transport

    with pytest.raises(ApiError) as exc_info:
        resolve_transport("IDEAM")

    assert exc_info.value.code == "real_data_source_not_configured"


def test_seed_transport_is_disabled_when_real_data_only(monkeypatch):
    monkeypatch.setenv("REAL_DATA_ONLY", "true")
    monkeypatch.setenv("IDEAM_TRANSPORT", "seed")

    from app.core.config import get_settings

    get_settings.cache_clear()

    from app.integrations.registry import build_adapter

    with pytest.raises(ApiError) as exc_info:
        build_adapter("IDEAM")

    assert exc_info.value.code == "seed_transport_disabled"


def test_default_structural_bundle_path_points_to_official_folder(monkeypatch):
    monkeypatch.delenv("STRUCTURAL_CATALOG_BUNDLE_PATH", raising=False)

    from app.core.config import get_settings

    get_settings.cache_clear()

    settings = get_settings()
    resolved = settings.resolved_structural_catalog_bundle_path

    assert resolved.name == "official_structural_bundle.json"
    assert resolved.parent.name == "official-structural"


def test_docker_env_skips_production_secret_validation(monkeypatch):
    monkeypatch.setenv("APP_ENV", "docker")
    monkeypatch.setenv(
        "JWT_SECRET_KEY", "guardianes-ladera-dev-secret-change-me-2026"
    )
    monkeypatch.setenv("SEED_ADMIN_PASSWORD", "guardianes-admin")

    from app.core.config import get_settings

    get_settings.cache_clear()

    settings = get_settings()
    settings.validate_production_secrets()


def test_production_env_still_requires_non_default_secrets(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv(
        "JWT_SECRET_KEY", "guardianes-ladera-dev-secret-change-me-2026"
    )
    monkeypatch.setenv("SEED_ADMIN_PASSWORD", "guardianes-admin")

    from app.core.config import get_settings

    get_settings.cache_clear()

    settings = get_settings()
    with pytest.raises(RuntimeError, match="Production secrets are not configured"):
        settings.validate_production_secrets()


def test_trigger_run_requires_official_structural_catalog_when_real_data_only(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "guardianes_real_policy.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("REAL_DATA_ONLY", "true")
    monkeypatch.setenv("SEED_DEMO_DATA", "false")

    from app.core.config import get_settings
    from app.db.bootstrap import init_database, seed_demo_data
    from app.db.session import reset_engine_cache, session_scope

    get_settings.cache_clear()
    reset_engine_cache()

    init_database()
    with session_scope() as session:
        seed_demo_data(session)

    with session_scope() as session:
        from app.services.runs import RunService

        with pytest.raises(ApiError) as exc_info:
            RunService(session).trigger_run(note="blocked synthetic run")

    assert exc_info.value.code == "structural_catalog_not_official"


def test_dashboard_reads_block_legacy_seed_runs_when_real_data_only(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "guardianes_real_dashboard_policy.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("REAL_DATA_ONLY", "true")
    monkeypatch.setenv("SEED_DEMO_DATA", "false")

    from app.core.config import get_settings
    from app.db.bootstrap import init_database, seed_demo_data
    from app.db.session import reset_engine_cache, session_scope

    get_settings.cache_clear()
    reset_engine_cache()

    init_database()
    with session_scope() as session:
        seed_demo_data(session)
        promote_seed_structure_to_official_catalog(session)

    with session_scope() as session:
        from app.services.dashboard import DashboardService

        with pytest.raises(ApiError) as exc_info:
            DashboardService(session).get_dashboard_bootstrap()

    assert exc_info.value.code == "legacy_prediction_run_blocked"


def test_trigger_run_uses_operational_real_data_when_real_data_only(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "guardianes_real_operational_run.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("REAL_DATA_ONLY", "true")
    monkeypatch.setenv("SEED_DEMO_DATA", "false")

    from app.core.config import get_settings
    from app.db.bootstrap import init_database, seed_demo_data
    from app.db.session import reset_engine_cache, session_scope
    from app.models import (
        HistoricalEvent,
        MunicipalityRainPoint,
        PredictionRun,
        SourceCatalog,
        SourceSyncEvent,
        SourceSyncStatus,
        UngrdRecord,
        Zone,
        ZoneExplanation,
        ZonePrediction,
    )

    get_settings.cache_clear()
    reset_engine_cache()

    init_database()
    with session_scope() as session:
        seed_demo_data(session)
        promote_seed_structure_to_official_catalog(session)

    with session_scope() as session:
        for model in (
            ZoneExplanation,
            ZonePrediction,
            PredictionRun,
            MunicipalityRainPoint,
            HistoricalEvent,
            UngrdRecord,
            SourceSyncEvent,
            SourceSyncStatus,
        ):
            session.execute(delete(model))
        promote_seed_structure_to_official_catalog(session)

        now = datetime(2026, 3, 28, 12, 0, tzinfo=timezone.utc)
        zone = session.scalar(select(Zone).where(Zone.id == "moc-01"))
        assert zone is not None

        for source_id, provider_updated_at in (
            ("IDEAM", now - timedelta(days=1)),
            ("SGC", now - timedelta(days=30)),
            ("UNGRD", now - timedelta(days=10)),
        ):
            source = session.get(SourceCatalog, source_id)
            assert source is not None
            session.add(
                SourceSyncStatus(
                    source=source,
                    last_synced_at=now,
                    last_success_at=provider_updated_at,
                    status_note="Official HTTP sync.",
                )
            )
            session.add(
                SourceSyncEvent(
                    source=source,
                    origin="manual",
                    adapter_key=f"http.{source_id.lower()}",
                    transport="http",
                    status="completed",
                    processed_records=1,
                    started_at=now,
                    completed_at=now,
                    message="Official sync.",
                    details={"provider_updated_at": provider_updated_at.isoformat()},
                )
            )

        municipality = zone.municipality
        rain_points = [
            ("2026-03-27", 44.0),
            ("2026-03-26", 38.0),
            ("2026-03-25", 26.0),
        ]
        for sort_order, (time_label, observed) in enumerate(rain_points):
            session.add(
                MunicipalityRainPoint(
                    municipality=municipality,
                    time_label=time_label,
                    observed=observed,
                    forecast=None,
                    forecast_low=None,
                    forecast_high=None,
                    forecast_range=None,
                    sort_order=sort_order,
                )
            )

        session.add(
            HistoricalEvent(
                id="sgc-real-01",
                municipality=municipality,
                date=(now - timedelta(days=12)).date(),
                severity="Alta",
                type="Deslizamiento",
                coords=zone.centroid,
                coords_geom={
                    "type": "Point",
                    "coordinates": [zone.centroid[1], zone.centroid[0]],
                },
                source="SGC",
            )
        )
        session.add(
            UngrdRecord(
                id="ungrd-real-01",
                municipality=municipality,
                date=(now - timedelta(days=20)).date(),
                summary="Movimiento en masa con afectacion vial reportado por la UNGRD.",
            )
        )

    with session_scope() as session:
        from app.services.runs import RunService

        response = RunService(session).trigger_run(note="official strict run")
        assert response.run.model_version == "operational-real-data-v1"
        assert response.run.id > 0

        latest_prediction = session.scalar(
            select(ZonePrediction)
            .where(ZonePrediction.run_id == response.run.id, ZonePrediction.zone_id == "moc-01")
        )
        assert latest_prediction is not None
        assert latest_prediction.drivers["rain_24h"] == 44
        assert latest_prediction.drivers["rain_72h"] == 108
        assert latest_prediction.drivers["slope_deg"] is None


def test_official_structural_bundle_import_is_accepted_when_real_data_only(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "guardianes_structural_bundle.db"
    bundle_path = tmp_path / "official_structural_bundle.json"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("REAL_DATA_ONLY", "true")
    monkeypatch.setenv("SEED_DEMO_DATA", "false")

    bundle_path.write_text(
        json.dumps(
            {
                "version": "official-structural-test-v1",
                "generatedAt": "2026-03-28T12:00:00+00:00",
                "sources": [
                    {
                        "id": "DANE",
                        "label": "DANE catalogo territorial",
                        "category": "historico",
                        "sourceUrl": "https://geoportal.dane.gov.co/mparcgis/rest/services/MGN2024/Serv_CapasMGN_2024/MapServer/317",
                        "updatedAt": "2026-03-01T00:00:00+00:00",
                    },
                    {
                        "id": "INVIAS",
                        "label": "INVIAS corredores",
                        "category": "infraestructura",
                        "sourceUrl": "https://hermes.invias.gov.co/arcgis/rest/services/OpenData/ServiciosOpenData/FeatureServer/0",
                        "updatedAt": "2026-03-01T00:00:00+00:00",
                    },
                ],
                "municipalities": [
                    {
                        "id": "mocoa",
                        "name": "Mocoa",
                        "center": [1.147, -76.648],
                        "zoom": 12,
                        "sourceId": "DANE",
                        "sourceRef": "municipio:86001",
                    }
                ],
                "roadSegments": [
                    {
                        "id": "inv-moc-45",
                        "municipalityId": "mocoa",
                        "name": "Invias Ruta 45 Mocoa - Pitalito",
                        "coords": [[1.11, -76.7], [1.17, -76.63]],
                        "riskLevel": "Naranja",
                        "length_km": 12.4,
                        "note": "Segmento oficial de prueba.",
                        "sourceId": "INVIAS",
                        "sourceRef": "via:45-001",
                    }
                ],
                "zones": [
                    {
                        "id": "moc-01",
                        "municipalityId": "mocoa",
                        "name": "Vereda Alto Afan",
                        "type": "Vereda",
                        "centroid": [1.15, -76.66],
                        "polygon": [
                            [1.16, -76.67],
                            [1.16, -76.65],
                            [1.14, -76.65],
                            [1.14, -76.67],
                        ],
                        "exposure": {
                            "population_estimate": 1200,
                            "households_estimate": 320,
                        },
                        "assets": {"roadSegmentIds": ["inv-moc-45"]},
                        "sourceId": "DANE",
                        "sourceRef": "vereda:alto-afan",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    from app.core.config import get_settings
    from app.db.bootstrap import init_database
    from app.db.session import reset_engine_cache, session_scope
    from app.models import Municipality, RoadSegment, Zone
    from app.services.structural_catalog import (
        ensure_real_data_structural_catalog,
        import_structural_catalog_bundle,
        load_structural_catalog_bundle,
    )

    get_settings.cache_clear()
    reset_engine_cache()

    init_database()
    bundle = load_structural_catalog_bundle(bundle_path)

    with session_scope() as session:
        counts = import_structural_catalog_bundle(session, bundle)
        assert counts == {"municipalities": 1, "road_segments": 1, "zones": 1}
        ensure_real_data_structural_catalog(session, for_api=True)

        municipality = session.get(Municipality, "mocoa")
        zone = session.get(Zone, "moc-01")
        road_segment = session.get(RoadSegment, "inv-moc-45")
        assert municipality is not None and municipality.source_id == "DANE"
        assert zone is not None and zone.source_ref == "vereda:alto-afan"
        assert road_segment is not None and road_segment.source_id == "INVIAS"
