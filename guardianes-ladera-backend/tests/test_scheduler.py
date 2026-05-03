from app.tasks.scheduler import BackendScheduler


def test_backend_scheduler_pipeline_mode_configures_single_operational_job(monkeypatch):
    monkeypatch.setenv("SCHEDULER_EXECUTION_MODE", "pipeline")
    monkeypatch.setenv("SCHEDULER_SOURCES", "IDEAM,UNGRD")
    monkeypatch.setenv("OPERATIONAL_PIPELINE_INTERVAL_MINUTES", "11")
    monkeypatch.setenv("ENABLE_TRAINING_RELEASE_SLA_MONITOR", "true")
    monkeypatch.setenv("TRAINING_RELEASE_SLA_INTERVAL_MINUTES", "6")
    monkeypatch.setenv("ENABLE_TRAINING_RELEASE_REASSIGNMENT_MONITOR", "true")
    monkeypatch.setenv("TRAINING_RELEASE_REASSIGNMENT_INTERVAL_MINUTES", "8")
    monkeypatch.setenv("TRAINING_RELEASE_AUTO_REASSIGN_REVIEWER", "admin")
    monkeypatch.setenv("ENABLE_NOTIFICATION_ACK_MONITOR", "true")
    monkeypatch.setenv("NOTIFICATION_ACK_MONITOR_INTERVAL_MINUTES", "13")
    monkeypatch.setenv("ENABLE_NOTIFICATION_DELIVERY_RETRY_MONITOR", "true")
    monkeypatch.setenv("NOTIFICATION_DELIVERY_RETRY_INTERVAL_MINUTES", "15")
    monkeypatch.setenv("ENABLE_NOTIFICATION_DELIVERY_FAILURE_MONITOR", "true")
    monkeypatch.setenv("NOTIFICATION_DELIVERY_FAILURE_INTERVAL_MINUTES", "19")
    monkeypatch.setenv("ENABLE_MODEL_MONITORING_CYCLE", "true")
    monkeypatch.setenv("MODEL_MONITORING_INTERVAL_MINUTES", "21")
    monkeypatch.setenv("MODEL_MONITORING_DRIFT_TOP_ERROR_COUNT", "7")
    monkeypatch.setenv("MODEL_MONITORING_SHADOW_TOP_ERROR_COUNT", "6")
    monkeypatch.setenv("MODEL_MONITORING_SHADOW_MAX_CANDIDATES", "5")
    monkeypatch.setenv("ENABLE_MODEL_MONITORING_ALERTS", "true")
    monkeypatch.setenv("NOTIFICATION_MODEL_MONITORING_USERNAMES", "ops-1,ops-2")

    from app.core.config import get_settings

    get_settings.cache_clear()

    scheduler = BackendScheduler()
    status = scheduler.status()

    assert status["execution_mode"] == "pipeline"
    assert status["scheduler_sources"] == ["IDEAM", "UNGRD"]
    assert status["operational_pipeline_interval_minutes"] == 11
    assert status["training_release_sla_monitor_enabled"] is True
    assert status["training_release_sla_interval_minutes"] == 6
    assert status["training_release_reassignment_monitor_enabled"] is True
    assert status["training_release_reassignment_interval_minutes"] == 8
    assert status["training_release_auto_reassign_reviewer"] == "admin"
    assert status["notification_ack_monitor_enabled"] is True
    assert status["notification_ack_monitor_interval_minutes"] == 13
    assert status["notification_delivery_retry_monitor_enabled"] is True
    assert status["notification_delivery_retry_interval_minutes"] == 15
    assert status["notification_delivery_failure_monitor_enabled"] is True
    assert status["notification_delivery_failure_interval_minutes"] == 19
    assert status["model_monitoring_cycle_enabled"] is True
    assert status["model_monitoring_interval_minutes"] == 21
    assert status["model_monitoring_drift_top_error_count"] == 7
    assert status["model_monitoring_shadow_top_error_count"] == 6
    assert status["model_monitoring_shadow_max_candidates"] == 5
    assert status["model_monitoring_alerts_enabled"] is True
    assert status["notification_model_monitoring_usernames"] == ["ops-1", "ops-2"]
    assert {job["id"] for job in status["jobs"]} == {
        "operational_pipeline_cycle",
        "training_release_sla_cycle",
        "training_release_reassignment_cycle",
        "notification_ack_deadline_cycle",
        "notification_delivery_retry_cycle",
        "notification_delivery_failure_cycle",
        "model_monitoring_cycle",
    }


def test_backend_scheduler_split_mode_configures_ingestion_and_prediction_jobs(
    monkeypatch,
):
    monkeypatch.setenv("SCHEDULER_EXECUTION_MODE", "split")
    monkeypatch.setenv("SCHEDULER_SOURCES", "IDEAM,SGC,UNGRD")
    monkeypatch.setenv("INGESTION_JOB_INTERVAL_MINUTES", "17")
    monkeypatch.setenv("PREDICTION_JOB_INTERVAL_MINUTES", "9")
    monkeypatch.setenv("ENABLE_TRAINING_RELEASE_SLA_MONITOR", "false")
    monkeypatch.setenv("ENABLE_TRAINING_RELEASE_REASSIGNMENT_MONITOR", "false")
    monkeypatch.setenv("ENABLE_NOTIFICATION_ACK_MONITOR", "false")
    monkeypatch.setenv("ENABLE_NOTIFICATION_DELIVERY_RETRY_MONITOR", "false")
    monkeypatch.setenv("ENABLE_NOTIFICATION_DELIVERY_FAILURE_MONITOR", "false")
    monkeypatch.setenv("ENABLE_MODEL_MONITORING_CYCLE", "false")
    monkeypatch.setenv("ENABLE_MODEL_MONITORING_ALERTS", "false")

    from app.core.config import get_settings

    get_settings.cache_clear()

    scheduler = BackendScheduler()
    status = scheduler.status()

    assert status["execution_mode"] == "split"
    assert status["scheduler_sources"] == ["IDEAM", "SGC", "UNGRD"]
    assert status["ingestion_interval_minutes"] == 17
    assert status["prediction_interval_minutes"] == 9
    assert status["training_release_sla_monitor_enabled"] is False
    assert status["training_release_reassignment_monitor_enabled"] is False
    assert status["notification_ack_monitor_enabled"] is False
    assert status["notification_delivery_retry_monitor_enabled"] is False
    assert status["notification_delivery_failure_monitor_enabled"] is False
    assert status["model_monitoring_cycle_enabled"] is False
    assert status["model_monitoring_alerts_enabled"] is False
    assert {job["id"] for job in status["jobs"]} == {
        "ingestion_cycle",
        "prediction_cycle",
    }
