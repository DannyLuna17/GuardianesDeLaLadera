from datetime import datetime, timezone

from sqlalchemy import select

from app.tasks.jobs import (
    run_ingestion_cycle,
    run_model_monitoring_cycle,
    run_operational_cycle,
    run_prediction_cycle,
)


def _prepare_label_dataset(session, *, version: str) -> None:
    from app.models import PredictionRun
    from app.schemas.admin import OutcomeLabelWrite
    from app.services.datasets import TrainingDatasetService
    from app.services.labels import OutcomeLabelService

    latest_run = session.scalar(
        select(PredictionRun).order_by(
            PredictionRun.completed_at.desc(),
            PredictionRun.id.desc(),
        )
    )
    assert latest_run is not None

    label_service = OutcomeLabelService(session)
    upserted = label_service.upsert_labels(
            [
                OutcomeLabelWrite(
                    zoneId="moc-01",
                    observedAt=datetime(2026, 3, 20, tzinfo=timezone.utc),
                    targetScore=0.12,
                    source="field_validation",
                    featureRunId=latest_run.id,
                ),
                OutcomeLabelWrite(
                    zoneId="moc-02",
                    observedAt=datetime(2026, 3, 21, tzinfo=timezone.utc),
                    targetScore=0.91,
                    source="field_validation",
                    featureRunId=latest_run.id,
                ),
            ]
        )
    dataset_service = TrainingDatasetService(session)
    dataset_service.export_dataset(
        version=version,
        source_mode="labels",
        label_ids=[label.id for label in upserted.labels],
        origin="test",
    )


def test_run_prediction_cycle_creates_new_run(tmp_path, monkeypatch):
    database_path = tmp_path / "guardianes_tasks.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("SEED_DEMO_DATA", "true")
    monkeypatch.setenv("REAL_DATA_ONLY", "false")

    from app.core.config import get_settings
    from app.db.bootstrap import init_database, seed_demo_data
    from app.db.session import reset_engine_cache, session_scope

    get_settings.cache_clear()
    reset_engine_cache()

    init_database()
    with session_scope() as session:
        seed_demo_data(session)

    first_result = run_prediction_cycle(note="task test run")
    second_result = run_prediction_cycle(note="task test run 2")

    assert first_result["run"]["id"] >= 3
    assert second_result["run"]["id"] > first_result["run"]["id"]
    assert second_result["job"]["jobType"] == "prediction_run"


def test_run_ingestion_cycle_returns_synced_sources(tmp_path, monkeypatch):
    database_path = tmp_path / "guardianes_ingestion.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("SEED_DEMO_DATA", "true")
    monkeypatch.setenv("REAL_DATA_ONLY", "false")

    from app.core.config import get_settings
    from app.db.bootstrap import init_database, seed_demo_data
    from app.db.session import reset_engine_cache, session_scope

    get_settings.cache_clear()
    reset_engine_cache()

    init_database()
    with session_scope() as session:
        seed_demo_data(session)

    result = run_ingestion_cycle(sources=["IDEAM", "SGC"])

    assert result["job"]["jobType"] == "ingestion_sync"
    assert len(result["syncedSources"]) == 2


def test_run_operational_cycle_returns_combined_pipeline_response(tmp_path, monkeypatch):
    database_path = tmp_path / "guardianes_pipeline.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("SEED_DEMO_DATA", "true")
    monkeypatch.setenv("REAL_DATA_ONLY", "false")

    from app.core.config import get_settings
    from app.db.bootstrap import init_database, seed_demo_data
    from app.db.session import reset_engine_cache, session_scope

    get_settings.cache_clear()
    reset_engine_cache()

    init_database()
    with session_scope() as session:
        seed_demo_data(session)

    result = run_operational_cycle(sources=["IDEAM", "UNGRD"], note="task pipeline run")

    assert result["job"]["jobType"] == "pipeline_run"
    assert result["ingestion"]["job"]["jobType"] == "ingestion_sync"
    assert result["run"]["job"]["jobType"] == "prediction_run"
    assert result["explanations"]["job"]["jobType"] == "explanation_refresh"


def test_run_model_monitoring_cycle_skips_when_no_labels_dataset_exists(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "guardianes_monitoring_skip.db"
    # Isolate every output path the monitoring cycle might inspect.
    # Without this the test picks up labels datasets left in the default
    # source-tracked directory by previous integration runs and incorrectly
    # follows the non-skip branch.
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("SEED_DEMO_DATA", "true")
    monkeypatch.setenv("REAL_DATA_ONLY", "false")
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(tmp_path / "datasets"))
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(tmp_path / "artifacts"))
    monkeypatch.setenv(
        "MODEL_SELECTION_RUNS_PATH", str(tmp_path / "selection-runs")
    )
    monkeypatch.setenv(
        "MODEL_SHADOW_RUNS_PATH", str(tmp_path / "shadow-runs")
    )
    monkeypatch.setenv(
        "MODEL_DRIFT_REPORTS_PATH", str(tmp_path / "drift-reports")
    )

    from app.core.config import get_settings
    from app.db.bootstrap import init_database, seed_demo_data
    from app.db.session import reset_engine_cache, session_scope

    get_settings.cache_clear()
    reset_engine_cache()

    init_database()
    with session_scope() as session:
        seed_demo_data(session)

    result = run_model_monitoring_cycle(note="task monitoring skip")

    assert result["job"]["jobType"] == "model_monitoring_cycle"
    assert result["job"]["status"] == "skipped"
    assert result["skipped"] is True
    assert result["datasetVersion"] is None
    assert result["drift"] is None
    assert result["shadow"] is None


def test_run_model_monitoring_cycle_executes_drift_and_shadow_scans(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "guardianes_monitoring_cycle.db"
    datasets_path = tmp_path / "training-datasets"
    drift_reports_path = tmp_path / "drift-reports"
    shadow_runs_path = tmp_path / "shadow-runs"
    selection_runs_path = tmp_path / "selection-runs"
    artifacts_path = tmp_path / "artifacts"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("SEED_DEMO_DATA", "true")
    monkeypatch.setenv("REAL_DATA_ONLY", "false")
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_DRIFT_REPORTS_PATH", str(drift_reports_path))
    monkeypatch.setenv("MODEL_SHADOW_RUNS_PATH", str(shadow_runs_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))

    from app.core.config import get_settings
    from app.db.bootstrap import init_database, seed_demo_data
    from app.db.session import reset_engine_cache, session_scope

    get_settings.cache_clear()
    reset_engine_cache()

    init_database()
    with session_scope() as session:
        seed_demo_data(session)
        _prepare_label_dataset(session, version="task-monitoring-labels-v1")

    result = run_model_monitoring_cycle(note="task monitoring run")

    assert result["job"]["jobType"] == "model_monitoring_cycle"
    assert result["job"]["status"] == "completed"
    assert result["skipped"] is False
    assert result["datasetVersion"] == "task-monitoring-labels-v1"
    assert result["drift"]["job"]["jobType"] == "model_drift_scan"
    assert result["drift"]["datasetVersion"] == "task-monitoring-labels-v1"
    assert result["shadow"]["job"]["jobType"] == "model_shadow_scan"
    assert result["shadow"]["datasetVersion"] == "task-monitoring-labels-v1"
    assert result["shadow"]["candidateCount"] >= 1
