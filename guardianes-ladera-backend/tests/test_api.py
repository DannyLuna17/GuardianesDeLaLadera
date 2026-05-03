import json
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient


def create_test_client(
    tmp_path,
    monkeypatch,
    *,
    seed_demo_data: bool = True,
    real_data_only: bool = False,
):
    database_path = tmp_path / "guardianes_test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("SEED_DEMO_DATA", "true" if seed_demo_data else "false")
    monkeypatch.setenv("REAL_DATA_ONLY", "true" if real_data_only else "false")

    from app.core.config import get_settings
    from app.db.session import reset_engine_cache

    get_settings.cache_clear()
    reset_engine_cache()

    from app.main import create_app

    app = create_app()
    return TestClient(app)


def get_admin_headers(client: TestClient) -> dict[str, str]:
    login_response = client.post(
        "/v1/auth/login",
        json={"username": "admin", "password": "guardianes-admin"},
    )
    assert login_response.status_code == 200
    token = login_response.json()["accessToken"]
    return {"Authorization": f"Bearer {token}"}


def create_failed_warning_notification(*, target_username: str = "admin") -> int:
    from app.db.session import session_scope
    from app.services.notifications import NotificationService

    with session_scope() as session:
        service = NotificationService(session)
        notification = service.create_event(
            event_type="notification_test_warning",
            severity="warning",
            title="Notification delivery warning test",
            message="Synthetic warning notification for delivery monitoring tests.",
            target_username=target_username,
            details={"test_case": "notification_delivery_monitoring"},
        )
        return notification.id


def write_historical_selection_run(
    selection_runs_path,
    *,
    version: str,
    dataset_version: str,
    created_at: str,
    dataset_mode: str,
    dataset_family: str,
    time_window: dict,
    alpha: float = 0.75,
    dataset_taxonomy: dict | None = None,
    evaluation_cohort: dict | None = None,
):
    selection_runs_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection_id": "spatial-risk-model-selection-run",
        "version": version,
        "artifact_type": "model_selection_run",
        "dataset_version": dataset_version,
        "created_at": created_at,
        "comparison_policy": {
            "primary_metric": "validation_rmse",
            "tie_breakers": [
                "validation_risk_level_accuracy_desc",
                "overall_rmse",
                "alpha",
            ],
        },
        "gate_policy": {"dataset_mode": dataset_mode},
        "dataset_context": {
            "dataset_mode": dataset_mode,
            "dataset_family": dataset_family,
            "time_window": time_window,
            "source": dataset_mode,
            "dataset_taxonomy": dataset_taxonomy or {},
            "evaluation_cohort": evaluation_cohort or {},
        },
        "candidate_count": 1,
        "best_model_version": f"historical-candidate-alpha-{str(alpha).replace('.', 'p')}",
        "promoted": False,
        "promotion": {
            "promoted": False,
            "reason": None,
            "promoted_at": None,
            "promoted_by": None,
            "source": "history",
        },
        "promotion_decision": {
            "eligible": False,
            "promoted": False,
            "blocking_reasons": [],
        },
        "active_model_version": "trained-spatial-seed-v1",
        "best_vs_active_comparison": {},
        "candidates": [
            {
                "rank": 1,
                "model_version": f"historical-candidate-alpha-{str(alpha).replace('.', 'p')}",
                "alpha": alpha,
                "artifact_path": "",
                "overall_rmse": 0.03,
                "validation_rmse": 0.03,
                "overall_risk_level_accuracy": 1.0,
                "validation_risk_level_accuracy": 1.0,
                "validation_rows": 8,
                "comparison": {},
                "top_errors": [],
            }
        ],
    }
    (selection_runs_path / f"{version}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def probe_best_alpha(
    client: TestClient,
    headers: dict[str, str],
    *,
    dataset_version: str,
    selection_runs_path,
    version: str,
    version_prefix: str,
) -> float:
    response = client.post(
        "/v1/admin/models/tune",
        json={
            "version": version,
            "datasetVersion": dataset_version,
            "alphas": [0.25, 0.75, 1.5],
            "versionPrefix": version_prefix,
            "promoteBest": False,
        },
        headers=headers,
    )
    assert response.status_code == 200
    payload = response.json()
    selection_path = selection_runs_path / f"{version}.json"
    if selection_path.exists():
        selection_path.unlink()
    return float(payload["candidates"][0]["alpha"])


def latest_run_zone_scores(
    client: TestClient, *, zone_ids: list[str]
) -> dict[str, float]:
    latest_run = client.get("/v1/runs/latest").json()
    detail = client.get(f"/v1/runs/{latest_run['id']}").json()
    scores = {
        zone["id"]: float(zone["riskScore"])
        for zone in detail["zones"]
        if zone["id"] in set(zone_ids)
    }
    assert len(scores) == len(zone_ids)
    return scores


def prepare_predictive_monitoring_alert_scenario(
    client: TestClient,
    headers: dict[str, str],
    *,
    prefix: str,
) -> dict[str, object]:
    latest_run = client.get("/v1/runs/latest").json()

    source_label_response = client.post(
        "/v1/admin/labels/upsert",
        json={
            "labels": [
                {
                    "zoneId": "moc-01",
                    "observedAt": "2026-03-20T00:00:00Z",
                    "targetScore": 0.05,
                    "source": "field_validation",
                    "featureRunId": latest_run["id"],
                },
                {
                    "zoneId": "moc-02",
                    "observedAt": "2026-03-21T00:00:00Z",
                    "targetScore": 0.95,
                    "source": "field_validation",
                    "featureRunId": latest_run["id"],
                },
            ]
        },
        headers=headers,
    )
    assert source_label_response.status_code == 200
    source_label_ids = [item["id"] for item in source_label_response.json()["labels"]]

    source_dataset_version = f"{prefix}-source-labels-v1"
    selection_version = f"selection-{prefix}-source-labels-v1"
    review_dataset_version = f"{prefix}-review-labels-v1"
    active_evaluation_version = f"{prefix}-active-baseline-v1"
    version_prefix = f"{prefix}-candidate"

    export_source_dataset = client.post(
        "/v1/admin/training-datasets/export",
        json={
            "version": source_dataset_version,
            "sourceMode": "labels",
            "labelIds": source_label_ids,
        },
        headers=headers,
    )
    assert export_source_dataset.status_code == 200

    tune_response = client.post(
        "/v1/admin/models/tune",
        json={
            "version": selection_version,
            "datasetVersion": source_dataset_version,
            "alphas": [0.25, 0.75, 1.5],
            "versionPrefix": version_prefix,
            "promoteBest": False,
            "topErrorCount": 5,
        },
        headers=headers,
    )
    assert tune_response.status_code == 200
    challenger_versions = [
        candidate["modelVersion"]
        for candidate in tune_response.json()["candidates"]
        if candidate["modelVersion"].startswith(f"{version_prefix}-alpha-")
    ]
    assert challenger_versions

    active_eval_response = client.post(
        "/v1/admin/models/evaluate",
        json={
            "version": active_evaluation_version,
            "modelVersion": "trained-spatial-seed-v1",
            "datasetVersion": source_dataset_version,
            "topErrorCount": 10,
        },
        headers=headers,
    )
    assert active_eval_response.status_code == 200
    active_predictions = {
        row["zoneId"]: row["predictedScore"]
        for row in active_eval_response.json()["topErrors"]
    }

    selected_candidate_version = ""
    selected_candidate_predictions: dict[str, float] = {}
    max_prediction_distance = -1.0
    for candidate_version in challenger_versions:
        candidate_eval_response = client.post(
            "/v1/admin/models/evaluate",
            json={
                "version": f"{candidate_version}-{prefix}-eval-v1",
                "modelVersion": candidate_version,
                "datasetVersion": source_dataset_version,
                "topErrorCount": 10,
            },
            headers=headers,
        )
        assert candidate_eval_response.status_code == 200
        candidate_predictions = {
            row["zoneId"]: row["predictedScore"]
            for row in candidate_eval_response.json()["topErrors"]
        }
        prediction_distance = round(
            sum(
                abs(candidate_predictions[zone_id] - active_predictions[zone_id])
                for zone_id in active_predictions
            ),
            6,
        )
        if prediction_distance > max_prediction_distance:
            max_prediction_distance = prediction_distance
            selected_candidate_version = candidate_version
            selected_candidate_predictions = candidate_predictions

    assert selected_candidate_version.startswith(f"{version_prefix}-alpha-")
    assert max_prediction_distance > 0

    review_label_response = client.post(
        "/v1/admin/labels/upsert",
        json={
            "labels": [
                {
                    "zoneId": "moc-01",
                    "observedAt": "2026-03-22T00:00:00Z",
                    "targetScore": selected_candidate_predictions["moc-01"],
                    "source": "field_validation",
                    "featureRunId": latest_run["id"],
                },
                {
                    "zoneId": "moc-02",
                    "observedAt": "2026-03-23T00:00:00Z",
                    "targetScore": selected_candidate_predictions["moc-02"],
                    "source": "field_validation",
                    "featureRunId": latest_run["id"],
                },
            ]
        },
        headers=headers,
    )
    assert review_label_response.status_code == 200
    review_label_ids = [item["id"] for item in review_label_response.json()["labels"]]

    export_review_dataset = client.post(
        "/v1/admin/training-datasets/export",
        json={
            "version": review_dataset_version,
            "sourceMode": "labels",
            "labelIds": review_label_ids,
        },
        headers=headers,
    )
    assert export_review_dataset.status_code == 200

    return {
        "latest_run": latest_run,
        "source_dataset_version": source_dataset_version,
        "review_dataset_version": review_dataset_version,
        "selected_candidate_version": selected_candidate_version,
        "selected_candidate_predictions": selected_candidate_predictions,
    }


def test_health_endpoint(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_dashboard_bootstrap_contains_frontend_contract(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        response = client.get("/v1/dashboard/bootstrap")
        assert response.status_code == 200
        payload = response.json()
        assert "municipalities" in payload
        assert "zones" in payload
        assert "roadSegments" in payload
        assert "rainSeries" in payload
        assert "ungrdRecords" in payload
        assert "sourceStatus" in payload
        assert "latestRun" in payload
        assert "dataProvenance" in payload
        assert payload["dataProvenance"]["mockDataPresent"] is True
        assert any(
            item["key"] == "structural_base" and item["state"] == "mock"
            for item in payload["dataProvenance"]["items"]
        )
        assert len(payload["zones"]) == 12


def test_zone_filters_and_explanation(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        zones_response = client.get(
            "/v1/zones", params={"municipality": "Mocoa", "minRiskLevel": "Naranja"}
        )
        assert zones_response.status_code == 200
        zones = zones_response.json()
        assert zones
        assert all(zone["municipality"] == "Mocoa" for zone in zones)

        zone_id = zones[0]["id"]
        explanation_response = client.get(f"/v1/zones/{zone_id}/explanation")
        assert explanation_response.status_code == 200
        explanation = explanation_response.json()
        assert explanation["zoneId"] == zone_id
        assert explanation["summary"]


def test_zone_events_are_filtered_by_zone_geometry(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        response = client.get("/v1/zones/moc-01/events")
        assert response.status_code == 200
        payload = response.json()
        event_ids = {event["id"] for event in payload}
        assert event_ids == {"sgc-01", "sgc-03"}


def test_zone_spatial_summary_returns_spatial_analytics(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        response = client.get("/v1/zones/moc-01/spatial-summary")
        assert response.status_code == 200
        payload = response.json()
        assert payload["zoneId"] == "moc-01"
        assert payload["historicalEventCount"] == 2
        assert payload["historicalEventIds"] == ["sgc-01", "sgc-03"]
        assert payload["severityBreakdown"] == {"Alta": 1, "Baja": 1}
        assert payload["intersectingRoadSegmentCount"] == 2
        assert payload["intersectingRoadSegmentIds"] == ["inv-moc-11", "inv-moc-45"]
        assert payload["intersectingRoadLengthKm"] == 70.0
        assert payload["rainOverlayCount"] == 2
        assert payload["rainOverlayIntensities"] == ["alta", "media"]


def test_zone_bbox_filter_uses_map_window_constraints(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        baseline = client.get("/v1/zones", params={"municipality": "Mocoa"}).json()
        assert baseline
        target_zone = baseline[0]
        lat, lon = target_zone["centroid"]

        response = client.get(
            "/v1/zones",
            params={
                "municipality": "Mocoa",
                "north": lat + 0.01,
                "south": lat - 0.01,
                "east": lon + 0.01,
                "west": lon - 0.01,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload
        assert any(item["id"] == target_zone["id"] for item in payload)
        assert len(payload) <= len(baseline)


def test_historical_events_support_bbox_filter(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        baseline = client.get(
            "/v1/historical-events", params={"municipality": "Mocoa"}
        ).json()
        assert baseline
        target_event = baseline[0]
        lat, lon = target_event["coords"]
        north = lat + 0.01
        south = lat - 0.01
        east = lon + 0.01
        west = lon - 0.01

        response = client.get(
            "/v1/historical-events",
            params={
                "municipality": "Mocoa",
                "north": north,
                "south": south,
                "east": east,
                "west": west,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload
        assert any(item["id"] == target_event["id"] for item in payload)
        assert all(south <= item["coords"][0] <= north for item in payload)
        assert all(west <= item["coords"][1] <= east for item in payload)


def test_partial_bbox_is_rejected(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        response = client.get(
            "/v1/zones",
            params={"north": 1.17, "south": 1.14},
        )
        assert response.status_code == 400
        payload = response.json()
        assert payload["error"]["code"] == "invalid_bbox"


def test_run_detail_contains_zone_predictions(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        latest_response = client.get("/v1/runs/latest")
        latest_run_id = latest_response.json()["id"]

        detail_response = client.get(f"/v1/runs/{latest_run_id}")
        assert detail_response.status_code == 200
        payload = detail_response.json()
        assert payload["zones"]
        assert payload["zonesMonitored"] == len(payload["zones"])


def test_trigger_run_creates_new_latest_run_and_job(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        before = client.get("/v1/runs/latest").json()

        trigger_response = client.post(
            "/v1/admin/runs/trigger",
            json={"note": "manual test trigger"},
            headers=headers,
        )
        assert trigger_response.status_code == 200
        trigger_payload = trigger_response.json()
        assert trigger_payload["job"]["jobType"] == "prediction_run"
        assert trigger_payload["run"]["id"] > before["id"]

        after = client.get("/v1/runs/latest").json()
        assert after["id"] == trigger_payload["run"]["id"]

        jobs_response = client.get("/v1/admin/jobs", headers=headers)
        assert jobs_response.status_code == 200
        jobs = jobs_response.json()
        assert jobs
        assert jobs[0]["jobType"] in {"prediction_run", "explanation_refresh"}


def test_refresh_explanations_creates_job_for_latest_run(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()
        response = client.post(
            "/v1/admin/explanations/trigger",
            json={"runId": latest_run["id"]},
            headers=headers,
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["runId"] == latest_run["id"]
        assert payload["refreshedCount"] > 0
        assert payload["job"]["jobType"] == "explanation_refresh"


def test_admin_endpoints_require_authentication(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        response = client.get("/v1/admin/jobs")
        assert response.status_code == 401


def test_login_and_me_endpoint(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        me_response = client.get("/v1/auth/me", headers=headers)
        assert me_response.status_code == 200
        payload = me_response.json()
        assert payload["username"] == "admin"
        assert payload["role"] == "admin"


def test_admin_user_management_and_password_rotation(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        admin_headers = get_admin_headers(client)

        create_response = client.post(
            "/v1/admin/users",
            json={
                "username": "analyst-1",
                "password": "TempSecret123",
                "role": "analyst",
                "isActive": True,
            },
            headers=admin_headers,
        )
        assert create_response.status_code == 200
        created_user = create_response.json()
        assert created_user["username"] == "analyst-1"
        assert created_user["role"] == "analyst"
        assert created_user["isActive"] is True

        users_response = client.get("/v1/admin/users", headers=admin_headers)
        assert users_response.status_code == 200
        assert any(
            item["username"] == "analyst-1" for item in users_response.json()
        )

        first_login = client.post(
            "/v1/auth/login",
            json={"username": "analyst-1", "password": "TempSecret123"},
            headers={"X-Forwarded-For": "10.0.0.2"},
        )
        assert first_login.status_code == 200
        analyst_headers = {
            "Authorization": f"Bearer {first_login.json()['accessToken']}"
        }

        change_password = client.post(
            "/v1/auth/change-password",
            json={
                "currentPassword": "TempSecret123",
                "newPassword": "RotatedSecret123",
            },
            headers=analyst_headers,
        )
        assert change_password.status_code == 200
        assert change_password.json()["username"] == "analyst-1"

        old_password_login = client.post(
            "/v1/auth/login",
            json={"username": "analyst-1", "password": "TempSecret123"},
            headers={"X-Forwarded-For": "10.0.0.3"},
        )
        assert old_password_login.status_code == 401

        rotated_password_login = client.post(
            "/v1/auth/login",
            json={"username": "analyst-1", "password": "RotatedSecret123"},
            headers={"X-Forwarded-For": "10.0.0.4"},
        )
        assert rotated_password_login.status_code == 200

        reset_password = client.post(
            "/v1/admin/users/analyst-1/password-reset",
            json={"newPassword": "ResetSecret123"},
            headers=admin_headers,
        )
        assert reset_password.status_code == 200

        rotated_password_after_reset = client.post(
            "/v1/auth/login",
            json={"username": "analyst-1", "password": "RotatedSecret123"},
            headers={"X-Forwarded-For": "10.0.0.5"},
        )
        assert rotated_password_after_reset.status_code == 401

        reset_password_login = client.post(
            "/v1/auth/login",
            json={"username": "analyst-1", "password": "ResetSecret123"},
            headers={"X-Forwarded-For": "10.0.0.6"},
        )
        assert reset_password_login.status_code == 200

        deactivate_user = client.patch(
            "/v1/admin/users/analyst-1",
            json={"isActive": False},
            headers=admin_headers,
        )
        assert deactivate_user.status_code == 200
        assert deactivate_user.json()["isActive"] is False

        inactive_login = client.post(
            "/v1/auth/login",
            json={"username": "analyst-1", "password": "ResetSecret123"},
            headers={"X-Forwarded-For": "10.0.0.7"},
        )
        assert inactive_login.status_code == 403
        assert inactive_login.json()["error"]["code"] == "user_inactive"

        jobs = client.get("/v1/admin/jobs", headers=admin_headers).json()
        job_types = {job["jobType"] for job in jobs}
        assert {
            "auth_user_create",
            "auth_password_change",
            "auth_password_reset",
            "auth_user_update",
        }.issubset(job_types)


def test_last_active_admin_cannot_be_deactivated(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        admin_headers = get_admin_headers(client)
        response = client.patch(
            "/v1/admin/users/admin",
            json={"isActive": False},
            headers=admin_headers,
        )
        assert response.status_code == 400
        assert (
            response.json()["error"]["code"] == "last_admin_deactivation_not_allowed"
        )


def test_scheduler_status_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_SCHEDULER", "false")
    monkeypatch.setenv("SCHEDULER_EXECUTION_MODE", "pipeline")
    monkeypatch.setenv("SCHEDULER_SOURCES", "IDEAM,SGC")
    monkeypatch.setenv("INGESTION_JOB_INTERVAL_MINUTES", "15")
    monkeypatch.setenv("PREDICTION_JOB_INTERVAL_MINUTES", "10")
    monkeypatch.setenv("OPERATIONAL_PIPELINE_INTERVAL_MINUTES", "12")
    monkeypatch.setenv("ENABLE_TRAINING_RELEASE_SLA_MONITOR", "true")
    monkeypatch.setenv("TRAINING_RELEASE_SLA_INTERVAL_MINUTES", "7")
    monkeypatch.setenv("ENABLE_TRAINING_RELEASE_REASSIGNMENT_MONITOR", "true")
    monkeypatch.setenv("TRAINING_RELEASE_REASSIGNMENT_INTERVAL_MINUTES", "9")
    monkeypatch.setenv("TRAINING_RELEASE_AUTO_REASSIGN_REVIEWER", "admin")
    monkeypatch.setenv("ENABLE_NOTIFICATION_ACK_MONITOR", "true")
    monkeypatch.setenv("NOTIFICATION_ACK_MONITOR_INTERVAL_MINUTES", "14")
    monkeypatch.setenv("ENABLE_NOTIFICATION_DELIVERY_RETRY_MONITOR", "true")
    monkeypatch.setenv("NOTIFICATION_DELIVERY_RETRY_INTERVAL_MINUTES", "16")
    monkeypatch.setenv("ENABLE_NOTIFICATION_DELIVERY_FAILURE_MONITOR", "true")
    monkeypatch.setenv("NOTIFICATION_DELIVERY_FAILURE_INTERVAL_MINUTES", "18")
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        response = client.get("/v1/admin/scheduler/status", headers=headers)
        assert response.status_code == 200
        payload = response.json()
        assert payload["enabled"] is False
        assert payload["executionMode"] == "pipeline"
        assert payload["schedulerSources"] == ["IDEAM", "SGC"]
        assert payload["ingestionIntervalMinutes"] == 15
        assert payload["predictionIntervalMinutes"] == 10
        assert payload["operationalPipelineIntervalMinutes"] == 12
        assert payload["trainingReleaseSlaMonitorEnabled"] is True
        assert payload["trainingReleaseSlaIntervalMinutes"] == 7
        assert payload["trainingReleaseReassignmentMonitorEnabled"] is True
        assert payload["trainingReleaseReassignmentIntervalMinutes"] == 9
        assert payload["trainingReleaseAutoReassignReviewer"] == "admin"
        assert payload["notificationAckMonitorEnabled"] is True
        assert payload["notificationAckMonitorIntervalMinutes"] == 14
        assert payload["notificationDeliveryRetryMonitorEnabled"] is True
        assert payload["notificationDeliveryRetryIntervalMinutes"] == 16
        assert payload["notificationDeliveryFailureMonitorEnabled"] is True
        assert payload["notificationDeliveryFailureIntervalMinutes"] == 18
        assert payload["jobs"]
        assert {job["id"] for job in payload["jobs"]} == {
            "operational_pipeline_cycle",
            "training_release_sla_cycle",
            "training_release_reassignment_cycle",
            "notification_ack_deadline_cycle",
            "notification_delivery_retry_cycle",
            "notification_delivery_failure_cycle",
        }


def test_trigger_ingestion_updates_sources_and_returns_job(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        response = client.post(
            "/v1/admin/ingestion/trigger",
            json={"sources": ["IDEAM", "UNGRD"]},
            headers=headers,
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["job"]["jobType"] == "ingestion_sync"
        assert len(payload["syncedSources"]) == 2

        source_status = client.get("/v1/source-status").json()
        ideam = next(item for item in source_status if item["id"] == "IDEAM")
        ungrd = next(item for item in source_status if item["id"] == "UNGRD")
        assert ideam["status"] in {"Fresco", "Retrasado"}
        assert ungrd["status"] in {"Fresco", "Retrasado"}


def test_trigger_ingestion_marks_partial_failures_without_aborting(
    tmp_path, monkeypatch
):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        from app.services import ingestion as ingestion_module
        from app.integrations.registry import build_adapter as original_build_adapter

        class FailingIdeamAdapter:
            transport = "seed"
            adapter_key = "seed.ideam"

            def sync(self, session):
                raise RuntimeError("IDEAM adapter unavailable in test")

        def patched_build_adapter(source_id: str):
            if source_id == "IDEAM":
                return FailingIdeamAdapter()
            return original_build_adapter(source_id)

        monkeypatch.setattr(ingestion_module, "build_adapter", patched_build_adapter)
        response = client.post(
            "/v1/admin/ingestion/trigger",
            json={"sources": ["IDEAM", "UNGRD"]},
            headers=headers,
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["job"]["status"] == "completed_with_errors"
        statuses = {
            item["sourceId"]: item["status"] for item in payload["syncedSources"]
        }
        assert statuses["IDEAM"] == "failed"
        assert statuses["UNGRD"] == "completed"


def test_trigger_ingestion_can_use_http_transport(tmp_path, monkeypatch):
    monkeypatch.setenv("IDEAM_TRANSPORT", "http")
    monkeypatch.setenv("IDEAM_BASE_URL", "https://providers.test/ideam")

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        from app.data.seed_store import load_seed_payload
        from app.integrations import provider_adapters

        seed = load_seed_payload()
        monkeypatch.setattr(
            provider_adapters.IdeamHttpAdapter,
            "fetch_payload",
            lambda self: {"rainSeries": seed["rainSeries"]},
        )

        response = client.post(
            "/v1/admin/ingestion/trigger",
            json={"sources": ["IDEAM"]},
            headers=headers,
        )
        assert response.status_code == 200
        payload = response.json()
        ideam = payload["syncedSources"][0]
        assert ideam["sourceId"] == "IDEAM"
        assert ideam["transport"] == "http"
        assert ideam["adapterKey"] == "http.ideam"
        assert ideam["details"]["payload_mode"] == "normalized_http"


def test_trigger_ingestion_can_parse_provider_style_http_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("SGC_TRANSPORT", "http")
    monkeypatch.setenv("SGC_BASE_URL", "https://providers.test/sgc")

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        from app.integrations import provider_adapters

        monkeypatch.setattr(
            provider_adapters.SgcHttpAdapter,
            "fetch_payload",
            lambda self: {
                "events": [
                    {
                        "eventId": "sgc-http-99",
                        "municipality_name": "Mocoa",
                        "occurred_at": "2026-03-01T05:00:00Z",
                        "severity": "media",
                        "event_type": "Deslizamiento",
                        "location": {"lat": 1.155, "lon": -76.661},
                    }
                ]
            },
        )

        response = client.post(
            "/v1/admin/ingestion/trigger",
            json={"sources": ["SGC"]},
            headers=headers,
        )
        assert response.status_code == 200
        payload = response.json()
        sgc = payload["syncedSources"][0]
        assert sgc["sourceId"] == "SGC"
        assert sgc["transport"] == "http"
        assert sgc["adapterKey"] == "http.sgc"
        assert sgc["details"]["payload_mode"] == "provider_http_events"

        history = client.get(
            "/v1/admin/ingestion/history?sourceId=SGC", headers=headers
        ).json()
        assert history
        assert history[0]["transport"] == "http"


def test_trigger_ingestion_auto_transport_falls_back_to_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("IDEAM_TRANSPORT", "auto")

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        response = client.post(
            "/v1/admin/ingestion/trigger",
            json={"sources": ["IDEAM"]},
            headers=headers,
        )
        assert response.status_code == 200
        payload = response.json()
        ideam = payload["syncedSources"][0]
        assert ideam["transport"] == "seed"
        assert ideam["adapterKey"] == "seed.ideam"


def test_ingestion_history_returns_sync_events(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        response = client.get("/v1/admin/ingestion/history", headers=headers)
        assert response.status_code == 200
        payload = response.json()
        assert payload
        assert payload[0]["sourceId"]
        assert payload[0]["adapterKey"]
        assert payload[0]["transport"]


def test_trigger_pipeline_runs_ingestion_prediction_and_explanations(
    tmp_path, monkeypatch
):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        before = client.get("/v1/runs/latest").json()

        response = client.post(
            "/v1/admin/pipeline/trigger",
            json={"sources": ["IDEAM", "SGC"], "note": "pipeline test"},
            headers=headers,
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["job"]["jobType"] == "pipeline_run"
        assert payload["ingestion"]["job"]["jobType"] == "ingestion_sync"
        assert payload["run"]["job"]["jobType"] == "prediction_run"
        assert payload["explanations"]["job"]["jobType"] == "explanation_refresh"
        assert payload["run"]["run"]["id"] > before["id"]

        after = client.get("/v1/runs/latest").json()
        assert after["id"] == payload["run"]["run"]["id"]

        jobs = client.get("/v1/admin/jobs", headers=headers).json()
        assert any(job["jobType"] == "pipeline_run" for job in jobs)


def test_trigger_run_persists_spatial_feature_trace(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        trigger_response = client.post(
            "/v1/admin/runs/trigger",
            json={"note": "spatial feature trace test"},
            headers=headers,
        )
        assert trigger_response.status_code == 200

        zones = client.get("/v1/zones", params={"municipality": "Mocoa"}).json()
        assert zones
        explanation = client.get(f"/v1/zones/{zones[0]['id']}/explanation").json()
        trace = explanation["trace"]

        assert trace["model_version"] == "trained-spatial-seed-v1"
        assert trace["uses_spatial_features"] is True
        assert "feature_snapshot" in trace
        assert "zone_event_count" in trace["feature_snapshot"]
        assert "intersecting_road_length_km" in trace["feature_snapshot"]


def test_retrain_endpoint_exports_trained_artifact(tmp_path, monkeypatch):
    artifacts_path = tmp_path / "artifacts"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        response = client.post(
            "/v1/admin/retrain",
            json={"version": "test-trained-via-api", "alpha": 0.5},
            headers=headers,
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["job"]["jobType"] == "model_retrain"
        assert payload["modelVersion"] == "test-trained-via-api"
        assert payload["rows"] == 24
        assert payload["overwroteActiveVersion"] is False
        assert (artifacts_path / "test-trained-via-api.json").exists()


def test_training_dataset_endpoints_export_list_and_detail(tmp_path, monkeypatch):
    datasets_path = tmp_path / "training-datasets"
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={"version": "test-dataset-v1"},
            headers=headers,
        )
        assert export_response.status_code == 200
        export_payload = export_response.json()
        assert export_payload["job"]["jobType"] == "training_dataset_export"
        assert export_payload["datasetVersion"] == "test-dataset-v1"
        assert export_payload["rows"] == 24
        assert export_payload["splitCounts"]["train"] > 0
        assert export_payload["splitCounts"]["validation"] > 0
        assert (datasets_path / "test-dataset-v1.json").exists()

        list_response = client.get("/v1/admin/training-datasets", headers=headers)
        assert list_response.status_code == 200
        datasets = list_response.json()
        assert datasets
        dataset_summary = next(
            item for item in datasets if item["version"] == "test-dataset-v1"
        )
        assert dataset_summary["datasetId"] == "spatial-risk-training-dataset"
        assert dataset_summary["artifactType"] == "training_dataset"
        assert dataset_summary["rows"] == 24
        assert dataset_summary["provenanceSource"] == "frontend_seed_bootstrap"

        detail_response = client.get(
            "/v1/admin/training-datasets/test-dataset-v1",
            params={"sampleSize": 3},
            headers=headers,
        )
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["version"] == "test-dataset-v1"
        assert detail["artifactType"] == "training_dataset"
        assert detail["labelName"] == "target_score"
        assert len(detail["sampleRows"]) == 3
        assert "zone_event_count" in detail["featureOrder"]
        assert detail["provenance"]["source"] == "frontend_seed_bootstrap"


def test_operational_training_dataset_export_uses_persisted_run_history(
    tmp_path, monkeypatch
):
    datasets_path = tmp_path / "training-datasets"
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={
                "version": "test-operational-dataset-v1",
                "sourceMode": "operational",
                "runIds": [latest_run["id"]],
            },
            headers=headers,
        )
        assert export_response.status_code == 200
        export_payload = export_response.json()
        assert export_payload["job"]["jobType"] == "training_dataset_export"
        assert export_payload["sourceMode"] == "operational"
        assert export_payload["runCount"] == 1
        assert export_payload["rows"] == latest_run["zonesMonitored"]
        assert (datasets_path / "test-operational-dataset-v1.json").exists()

        list_response = client.get("/v1/admin/training-datasets", headers=headers)
        assert list_response.status_code == 200
        datasets = list_response.json()
        dataset_summary = next(
            item
            for item in datasets
            if item["version"] == "test-operational-dataset-v1"
        )
        assert dataset_summary["provenanceSource"] == "operational_prediction_history"

        detail_response = client.get(
            "/v1/admin/training-datasets/test-operational-dataset-v1",
            params={"sampleSize": 2},
            headers=headers,
        )
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["summary"]["runs"] == 1
        assert detail["summary"]["run_ids"] == [latest_run["id"]]
        assert detail["provenance"]["source"] == "operational_prediction_history"
        assert detail["provenance"]["label_source"] == "zone_predictions.risk_score"
        assert len(detail["sampleRows"]) == 2


def test_retrain_endpoint_can_use_exported_training_dataset_version(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={"version": "retrain-dataset-v1"},
            headers=headers,
        )
        assert export_response.status_code == 200

        retrain_response = client.post(
            "/v1/admin/retrain",
            json={
                "version": "test-trained-from-dataset",
                "alpha": 0.5,
                "datasetVersion": "retrain-dataset-v1",
            },
            headers=headers,
        )
        assert retrain_response.status_code == 200
        payload = retrain_response.json()
        assert payload["job"]["jobType"] == "model_retrain"
        assert payload["modelVersion"] == "test-trained-from-dataset"
        assert payload["datasetVersion"] == "retrain-dataset-v1"
        assert payload["rows"] == 24
        assert (artifacts_path / "test-trained-from-dataset.json").exists()


def test_retrain_endpoint_can_use_operational_training_dataset_version(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()
        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={
                "version": "operational-retrain-dataset-v1",
                "sourceMode": "operational",
                "runIds": [latest_run["id"]],
            },
            headers=headers,
        )
        assert export_response.status_code == 200

        retrain_response = client.post(
            "/v1/admin/retrain",
            json={
                "version": "test-trained-from-operational-dataset",
                "alpha": 0.5,
                "datasetVersion": "operational-retrain-dataset-v1",
            },
            headers=headers,
        )
        assert retrain_response.status_code == 200
        payload = retrain_response.json()
        assert payload["datasetVersion"] == "operational-retrain-dataset-v1"
        assert payload["rows"] == latest_run["zonesMonitored"]
        assert (artifacts_path / "test-trained-from-operational-dataset.json").exists()


def test_outcome_label_endpoints_create_and_list_governed_labels(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()
        observed_at = latest_run["completedAt"]

        upsert_response = client.post(
            "/v1/admin/labels/upsert",
            json={
                "labels": [
                    {
                        "zoneId": "moc-01",
                        "observedAt": observed_at,
                        "targetScore": 0.88,
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                        "notes": "Observed instability after heavy rainfall.",
                        "evidence": {"ticket": "OBS-1001"},
                    }
                ]
            },
            headers=headers,
        )
        assert upsert_response.status_code == 200
        upsert_payload = upsert_response.json()
        assert upsert_payload["createdCount"] == 1
        assert upsert_payload["updatedCount"] == 0
        assert upsert_payload["labels"][0]["zoneId"] == "moc-01"
        assert upsert_payload["labels"][0]["targetRiskLevel"] == "Rojo"
        assert upsert_payload["labels"][0]["featureRunId"] == latest_run["id"]

        list_response = client.get(
            "/v1/admin/labels",
            params={"zoneId": "moc-01", "source": "field_validation"},
            headers=headers,
        )
        assert list_response.status_code == 200
        labels = list_response.json()
        assert len(labels) == 1
        assert labels[0]["zoneId"] == "moc-01"
        assert labels[0]["source"] == "field_validation"
        assert labels[0]["targetScore"] == 0.88


def test_labels_dataset_export_can_filter_by_observed_window(tmp_path, monkeypatch):
    datasets_path = tmp_path / "training_datasets"
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()

        upsert_response = client.post(
            "/v1/admin/labels/upsert",
            json={
                "labels": [
                    {
                        "zoneId": "moc-01",
                        "observedAt": "2026-03-20T00:00:00Z",
                        "targetScore": 0.2,
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                    },
                    {
                        "zoneId": "moc-02",
                        "observedAt": "2026-03-24T00:00:00Z",
                        "targetScore": 0.8,
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                    },
                ]
            },
            headers=headers,
        )
        assert upsert_response.status_code == 200

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={
                "version": "observed-window-labels-v1",
                "sourceMode": "labels",
                "observedAfter": "2026-03-22T00:00:00Z",
            },
            headers=headers,
        )
        assert export_response.status_code == 200
        export_payload = export_response.json()
        assert export_payload["sourceMode"] == "labels"
        assert export_payload["labelCount"] == 1
        assert export_payload["rows"] == 1

        detail_response = client.get(
            "/v1/admin/training-datasets/observed-window-labels-v1",
            headers=headers,
        )
        assert detail_response.status_code == 200
        detail_payload = detail_response.json()
        assert detail_payload["summary"]["labels"] == 1
        assert detail_payload["sampleRows"][0]["zoneId"] == "moc-02"


def test_historical_event_import_creates_governed_labels_from_zone_events(
    tmp_path, monkeypatch
):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()

        response = client.post(
            "/v1/admin/labels/import/historical-events",
            json={
                "zoneId": "moc-01",
                "eventSource": "SGC",
                "status": "confirmed",
            },
            headers=headers,
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["createdCount"] == 2
        assert payload["updatedCount"] == 0
        assert payload["skippedCount"] == 0
        assert set(payload["importedEventIds"]) == {"sgc-01", "sgc-03"}
        assert all(
            item["featureRunId"] == latest_run["id"] for item in payload["labels"]
        )
        assert all(
            item["source"].startswith("historical_event:") for item in payload["labels"]
        )
        assert all(
            item["evidence"]["import_mode"] == "historical_event"
            for item in payload["labels"]
        )
        assert all(
            item["evidence"]["feature_run_resolution"] == "latest_available_backfill"
            for item in payload["labels"]
        )


def test_historical_event_imported_labels_can_feed_label_dataset_export(
    tmp_path, monkeypatch
):
    datasets_path = tmp_path / "training-datasets"
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        import_response = client.post(
            "/v1/admin/labels/import/historical-events",
            json={
                "zoneId": "moc-01",
                "eventSource": "SGC",
                "status": "confirmed",
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        imported_labels = import_response.json()["labels"]
        imported_label_ids = [item["id"] for item in imported_labels]

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={
                "version": "historical-import-label-dataset-v1",
                "sourceMode": "labels",
                "labelIds": imported_label_ids,
            },
            headers=headers,
        )
        assert export_response.status_code == 200
        export_payload = export_response.json()
        assert export_payload["sourceMode"] == "labels"
        assert export_payload["labelCount"] == 2
        assert export_payload["rows"] == 2
        assert (datasets_path / "historical-import-label-dataset-v1.json").exists()

        detail_response = client.get(
            "/v1/admin/training-datasets/historical-import-label-dataset-v1",
            headers=headers,
        )
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["provenance"]["source"] == "governed_zone_outcome_labels"
        assert detail["summary"]["labels"] == 2
        assert detail["summary"]["matched_predictions"] == 2


def test_ungrd_record_import_creates_draft_labels_and_review_confirms_them(
    tmp_path, monkeypatch
):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()

        import_response = client.post(
            "/v1/admin/labels/import/ungrd-records",
            json={
                "municipality": "Mocoa",
                "maxRecords": 1,
                "maxZonesPerRecord": 2,
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        import_payload = import_response.json()
        assert import_payload["createdCount"] == 2
        assert import_payload["updatedCount"] == 0
        assert import_payload["skippedCount"] == 0
        assert import_payload["importedRecordIds"] == ["ungrd-01"]
        assert all(item["status"] == "draft" for item in import_payload["labels"])
        assert all(
            item["source"].startswith("ungrd_record:")
            for item in import_payload["labels"]
        )
        assert all(
            item["featureRunId"] == latest_run["id"]
            for item in import_payload["labels"]
        )
        assert all(
            item["evidence"]["import_mode"] == "ungrd_record"
            for item in import_payload["labels"]
        )

        label_ids = [item["id"] for item in import_payload["labels"]]
        review_response = client.post(
            "/v1/admin/labels/review",
            json={
                "labelIds": label_ids,
                "decision": "confirmed",
                "reviewNotes": "Reviewed municipal emergency records and approved for supervised training.",
            },
            headers=headers,
        )
        assert review_response.status_code == 200
        review_payload = review_response.json()
        assert review_payload["reviewedCount"] == 2
        assert all(item["status"] == "confirmed" for item in review_payload["labels"])
        assert all(item["reviewedBy"] == "admin" for item in review_payload["labels"])
        assert all(item["reviewNotes"] for item in review_payload["labels"])

        list_response = client.get(
            "/v1/admin/labels",
            params={"source": "ungrd_record:ungrd-01", "status": "confirmed"},
            headers=headers,
        )
        assert list_response.status_code == 200
        labels = list_response.json()
        assert len(labels) == 2
        assert all(item["reviewedBy"] == "admin" for item in labels)


def test_reviewed_ungrd_labels_can_feed_label_dataset_export(tmp_path, monkeypatch):
    datasets_path = tmp_path / "training-datasets"
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        import_response = client.post(
            "/v1/admin/labels/import/ungrd-records",
            json={
                "municipality": "Mocoa",
                "maxRecords": 1,
                "maxZonesPerRecord": 2,
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_ids = [item["id"] for item in import_response.json()["labels"]]

        review_response = client.post(
            "/v1/admin/labels/review",
            json={
                "labelIds": label_ids,
                "decision": "confirmed",
                "reviewNotes": "Approved imported UNGRD municipal draft labels.",
            },
            headers=headers,
        )
        assert review_response.status_code == 200

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={
                "version": "reviewed-ungrd-label-dataset-v1",
                "sourceMode": "labels",
                "labelIds": label_ids,
            },
            headers=headers,
        )
        assert export_response.status_code == 200
        export_payload = export_response.json()
        assert export_payload["sourceMode"] == "labels"
        assert export_payload["labelCount"] == 2
        assert export_payload["rows"] == 2
        assert (datasets_path / "reviewed-ungrd-label-dataset-v1.json").exists()

        detail_response = client.get(
            "/v1/admin/training-datasets/reviewed-ungrd-label-dataset-v1",
            headers=headers,
        )
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["provenance"]["source"] == "governed_zone_outcome_labels"
        assert detail["summary"]["labels"] == 2
        assert detail["summary"]["matched_predictions"] == 2


def test_field_validation_import_creates_governed_labels_with_observer_provenance(
    tmp_path, monkeypatch
):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()

        response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-001",
                        "zoneId": "moc-01",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Alta",
                        "observer": "brigada-terreno-1",
                        "siteVisitId": "visit-001",
                        "teamId": "team-alpha",
                        "mediaRefs": ["photo://moc01/1", "photo://moc01/2"],
                        "attachmentRefs": ["report://moc01/summary"],
                        "gpsAccuracyMeters": 4.5,
                        "locationNotes": "Upper shoulder close to road cut.",
                        "status": "confirmed",
                        "notes": "Fresh cracks and active soil movement observed.",
                        "evidence": {"photoSet": "moc01-photos"},
                    },
                    {
                        "observationId": "fv-002",
                        "zoneId": "moc-02",
                        "observedAt": latest_run["completedAt"],
                        "targetScore": 0.47,
                        "observer": "brigada-terreno-2",
                        "status": "draft",
                        "notes": "Localized saturation without clear failure plane.",
                        "evidence": {"inspectionForm": "IF-2026-002"},
                    },
                ]
            },
            headers=headers,
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["createdCount"] == 2
        assert payload["updatedCount"] == 0
        assert payload["skippedCount"] == 0
        assert payload["importedObservationIds"] == ["fv-001", "fv-002"]
        assert payload["labels"][0]["source"].startswith("field_validation:")
        assert payload["labels"][0]["evidence"]["import_mode"] == "field_validation"
        assert payload["labels"][0]["evidence"]["observer"]

        list_response = client.get(
            "/v1/admin/labels",
            params={"source": "field_validation:fv-001"},
            headers=headers,
        )
        assert list_response.status_code == 200
        labels = list_response.json()
        assert len(labels) == 1
        assert labels[0]["status"] == "confirmed"
        assert labels[0]["evidence"]["observer"] == "brigada-terreno-1"
        assert labels[0]["evidence"]["site_visit_id"] == "visit-001"
        assert labels[0]["evidence"]["team_id"] == "team-alpha"
        assert labels[0]["evidence"]["media_refs"] == [
            "photo://moc01/1",
            "photo://moc01/2",
        ]


def test_field_validation_reimport_updates_existing_label_by_observation_id(
    tmp_path, monkeypatch
):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()

        first_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-dedup-001",
                        "zoneId": "moc-01",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Media",
                        "observer": "brigada-terreno-1",
                        "status": "draft",
                    }
                ]
            },
            headers=headers,
        )
        assert first_response.status_code == 200
        first_payload = first_response.json()
        first_label = first_payload["labels"][0]

        second_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-dedup-001",
                        "zoneId": "moc-01",
                        "observedAt": latest_run["startedAt"],
                        "targetScore": 0.81,
                        "observer": "brigada-terreno-1",
                        "status": "confirmed",
                        "notes": "Updated after secondary inspection.",
                    }
                ]
            },
            headers=headers,
        )
        assert second_response.status_code == 200
        second_payload = second_response.json()
        assert second_payload["createdCount"] == 0
        assert second_payload["updatedCount"] == 1
        second_label = second_payload["labels"][0]
        assert second_label["id"] == first_label["id"]
        assert second_label["targetScore"] == 0.81
        assert second_label["status"] == "confirmed"

        list_response = client.get(
            "/v1/admin/labels",
            params={"source": "field_validation:fv-dedup-001"},
            headers=headers,
        )
        assert list_response.status_code == 200
        labels = list_response.json()
        assert len(labels) == 1
        assert labels[0]["id"] == first_label["id"]
        assert labels[0]["targetScore"] == 0.81


def test_review_needs_revision_persists_review_history(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-review-001",
                        "zoneId": "moc-01",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Alta",
                        "observer": "brigada-terreno-1",
                        "status": "draft",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        review_response = client.post(
            "/v1/admin/labels/review",
            json={
                "labelIds": [label_id],
                "decision": "needs_revision",
                "reviewNotes": "Need clearer field evidence before approval.",
            },
            headers=headers,
        )
        assert review_response.status_code == 200
        review_payload = review_response.json()
        assert review_payload["reviewedCount"] == 1
        reviewed_label = review_payload["labels"][0]
        assert reviewed_label["status"] == "needs_revision"
        assert reviewed_label["reviewedBy"] == "admin"
        assert (
            reviewed_label["reviewNotes"]
            == "Need clearer field evidence before approval."
        )
        history = reviewed_label["evidence"]["review_history"]
        assert len(history) == 1
        assert history[0]["decision"] == "needs_revision"
        assert history[0]["reviewed_by"] == "admin"


def test_assign_outcome_labels_and_list_review_queue(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()
        due_at = (
            (datetime.now(timezone.utc) + timedelta(days=2))
            .replace(microsecond=0)
            .isoformat()
        )

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-queue-001",
                        "zoneId": "moc-01",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Alta",
                        "observer": "brigada-terreno-queue",
                        "siteVisitId": "visit-queue-001",
                        "teamId": "team-queue-alpha",
                        "mediaRefs": ["photo://queue/moc01/1"],
                        "gpsAccuracyMeters": 3.8,
                        "locationNotes": "Slope crack documented on the upper cut.",
                        "status": "draft",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        assign_response = client.post(
            "/v1/admin/labels/assign",
            json={
                "labelIds": [label_id],
                "reviewerUsername": "admin",
                "reviewDueAt": due_at,
                "assignmentNotes": "Prioritize this label for the next evidence review pass.",
            },
            headers=headers,
        )
        assert assign_response.status_code == 200
        assign_payload = assign_response.json()
        assert assign_payload["assignedCount"] == 1
        assigned_label = assign_payload["labels"][0]
        assert assigned_label["assignedReviewer"] == "admin"
        assert assigned_label["readyForReview"] is True
        assert assigned_label["evidenceCompletenessScore"] == 1.0
        assert assigned_label["missingEvidenceFields"] == []

        queue_response = client.get(
            "/v1/admin/labels/review-queue",
            params={"assignedReviewer": "admin", "readyForReview": "true"},
            headers=headers,
        )
        assert queue_response.status_code == 200
        queue_payload = queue_response.json()
        assert queue_payload["total"] == 1
        assert queue_payload["readyCount"] == 1
        assert queue_payload["assignedCount"] == 1
        queued_label = queue_payload["labels"][0]
        assert queued_label["id"] == label_id
        assert queued_label["assignedReviewer"] == "admin"
        assert queued_label["reviewDueAt"].startswith(due_at[:19])
        assert queued_label["isOverdue"] is False


def test_confirm_review_rejects_incomplete_label_evidence_and_queue_surfaces_missing_fields(
    tmp_path, monkeypatch
):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-incomplete-001",
                        "zoneId": "moc-01",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Media",
                        "observer": "brigada-terreno-queue",
                        "status": "draft",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        review_response = client.post(
            "/v1/admin/labels/review",
            json={
                "labelIds": [label_id],
                "decision": "confirmed",
                "reviewNotes": "Attempted confirmation should fail until the evidence package is complete.",
            },
            headers=headers,
        )
        assert review_response.status_code == 400
        payload = review_response.json()
        assert payload["error"]["code"] == "label_evidence_incomplete"
        assert "site_visit_id" in payload["error"]["message"]
        assert "media_or_attachment" in payload["error"]["message"]

        queue_response = client.get(
            "/v1/admin/labels/review-queue",
            params={"readyForReview": "false"},
            headers=headers,
        )
        assert queue_response.status_code == 200
        queue_payload = queue_response.json()
        queued_label = next(
            item for item in queue_payload["labels"] if item["id"] == label_id
        )
        assert queued_label["readyForReview"] is False
        assert "site_visit_id" in queued_label["missingEvidenceFields"]
        assert "media_or_attachment" in queued_label["missingEvidenceFields"]


def test_confirmed_label_training_eligibility_can_be_put_on_hold(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-hold-001",
                        "zoneId": "moc-01",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Alta",
                        "observer": "brigada-hold-1",
                        "siteVisitId": "visit-hold-001",
                        "teamId": "team-hold-alpha",
                        "mediaRefs": ["photo://hold/moc01/1"],
                        "gpsAccuracyMeters": 2.5,
                        "locationNotes": "Evidence package complete before governance hold.",
                        "status": "confirmed",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        hold_response = client.post(
            "/v1/admin/labels/training-eligibility",
            json={
                "labelIds": [label_id],
                "trainingEligibilityStatus": "hold",
                "notes": "Hold until secondary reviewer validates the field packet.",
            },
            headers=headers,
        )
        assert hold_response.status_code == 200
        hold_payload = hold_response.json()
        assert hold_payload["updatedCount"] == 1
        held_label = hold_payload["labels"][0]
        assert held_label["trainingEligibilityStatus"] == "hold"
        assert held_label["trainingEligibilityUpdatedBy"] == "admin"
        assert (
            held_label["trainingEligibilityNotes"]
            == "Hold until secondary reviewer validates the field packet."
        )
        history = held_label["evidence"]["training_eligibility_history"]
        assert history[-1]["training_eligibility_status"] == "hold"

        list_response = client.get(
            "/v1/admin/labels",
            params={"trainingEligibilityStatus": "hold"},
            headers=headers,
        )
        assert list_response.status_code == 200
        labels = list_response.json()
        assert any(item["id"] == label_id for item in labels)


def test_manual_training_hold_survives_source_reimport(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-hold-reimport-001",
                        "zoneId": "moc-01",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Alta",
                        "observer": "brigada-hold-1",
                        "siteVisitId": "visit-hold-reimport-001",
                        "teamId": "team-hold-alpha",
                        "mediaRefs": ["photo://hold-reimport/moc01/1"],
                        "gpsAccuracyMeters": 2.1,
                        "locationNotes": "Initial complete evidence package.",
                        "status": "confirmed",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        hold_response = client.post(
            "/v1/admin/labels/training-eligibility",
            json={
                "labelIds": [label_id],
                "trainingEligibilityStatus": "hold",
                "notes": "Hold should survive source refresh.",
            },
            headers=headers,
        )
        assert hold_response.status_code == 200

        reimport_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-hold-reimport-001",
                        "zoneId": "moc-01",
                        "observedAt": latest_run["startedAt"],
                        "targetScore": 0.82,
                        "observer": "brigada-hold-1",
                        "siteVisitId": "visit-hold-reimport-001",
                        "teamId": "team-hold-alpha",
                        "mediaRefs": [
                            "photo://hold-reimport/moc01/1",
                            "photo://hold-reimport/moc01/2",
                        ],
                        "gpsAccuracyMeters": 2.1,
                        "locationNotes": "Follow-up update from the same observation source.",
                        "status": "confirmed",
                    }
                ]
            },
            headers=headers,
        )
        assert reimport_response.status_code == 200
        reimport_payload = reimport_response.json()
        assert reimport_payload["createdCount"] == 0
        assert reimport_payload["updatedCount"] == 1
        updated_label = reimport_payload["labels"][0]
        assert updated_label["id"] == label_id
        assert updated_label["trainingEligibilityStatus"] == "hold"
        assert (
            updated_label["trainingEligibilityNotes"]
            == "Hold should survive source refresh."
        )
        history = updated_label["evidence"]["training_eligibility_history"]
        assert history[-1]["training_eligibility_status"] == "hold"


def test_held_label_requires_release_request_before_returning_to_eligible(
    tmp_path, monkeypatch
):
    datasets_path = tmp_path / "training-datasets"
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-release-001",
                        "zoneId": "moc-01",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Alta",
                        "observer": "brigada-release-1",
                        "siteVisitId": "visit-release-001",
                        "teamId": "team-release-alpha",
                        "mediaRefs": ["photo://release/moc01/1"],
                        "gpsAccuracyMeters": 2.2,
                        "locationNotes": "Complete evidence packet before hold release workflow.",
                        "status": "confirmed",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        hold_response = client.post(
            "/v1/admin/labels/training-eligibility",
            json={
                "labelIds": [label_id],
                "trainingEligibilityStatus": "hold",
                "notes": "Hold pending second-pass release approval.",
            },
            headers=headers,
        )
        assert hold_response.status_code == 200

        direct_release_response = client.post(
            "/v1/admin/labels/training-eligibility",
            json={
                "labelIds": [label_id],
                "trainingEligibilityStatus": "eligible",
                "notes": "Direct release should be blocked.",
            },
            headers=headers,
        )
        assert direct_release_response.status_code == 400
        direct_release_payload = direct_release_response.json()
        assert (
            direct_release_payload["error"]["code"]
            == "training_release_request_required"
        )

        request_response = client.post(
            "/v1/admin/labels/training-eligibility/release-request",
            json={
                "labelIds": [label_id],
                "releaseCriteria": ["secondary_photo_review", "geotech_signoff"],
                "notes": "Requesting release after secondary review packet was assembled.",
            },
            headers=headers,
        )
        assert request_response.status_code == 200
        request_payload = request_response.json()
        assert request_payload["requestedCount"] == 1
        requested_label = request_payload["labels"][0]
        assert requested_label["trainingReleaseStatus"] == "pending"
        assert requested_label["trainingReleaseRequestedBy"] == "admin"
        assert requested_label["trainingReleaseCriteria"] == [
            "secondary_photo_review",
            "geotech_signoff",
        ]

        pending_list = client.get(
            "/v1/admin/labels",
            params={"trainingReleaseStatus": "pending"},
            headers=headers,
        )
        assert pending_list.status_code == 200
        pending_labels = pending_list.json()
        assert any(item["id"] == label_id for item in pending_labels)

        release_review_response = client.post(
            "/v1/admin/labels/training-eligibility/release-review",
            json={
                "labelIds": [label_id],
                "decision": "approved",
                "notes": "Approved after second-pass evidence review.",
            },
            headers=headers,
        )
        assert release_review_response.status_code == 200
        review_payload = release_review_response.json()
        assert review_payload["reviewedCount"] == 1
        released_label = review_payload["labels"][0]
        assert released_label["trainingEligibilityStatus"] == "eligible"
        assert released_label["trainingReleaseStatus"] == "approved"
        assert released_label["trainingReleaseReviewedBy"] == "admin"
        release_history = released_label["evidence"]["training_release_history"]
        assert release_history[-1]["decision"] == "approved"

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={
                "version": "released-label-dataset-v1",
                "sourceMode": "labels",
                "labelIds": [label_id],
            },
            headers=headers,
        )
        assert export_response.status_code == 200
        assert export_response.json()["rows"] == 1
        assert (datasets_path / "released-label-dataset-v1.json").exists()


def test_rejected_release_review_keeps_label_on_hold(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-release-reject-001",
                        "zoneId": "moc-01",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Alta",
                        "observer": "brigada-release-1",
                        "siteVisitId": "visit-release-reject-001",
                        "teamId": "team-release-alpha",
                        "mediaRefs": ["photo://release-reject/moc01/1"],
                        "gpsAccuracyMeters": 2.2,
                        "locationNotes": "Complete evidence packet before rejected release review.",
                        "status": "confirmed",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        hold_response = client.post(
            "/v1/admin/labels/training-eligibility",
            json={
                "labelIds": [label_id],
                "trainingEligibilityStatus": "hold",
                "notes": "Hold before rejection path test.",
            },
            headers=headers,
        )
        assert hold_response.status_code == 200

        request_response = client.post(
            "/v1/admin/labels/training-eligibility/release-request",
            json={
                "labelIds": [label_id],
                "releaseCriteria": ["secondary_photo_review"],
                "notes": "Submitting request that will be rejected.",
            },
            headers=headers,
        )
        assert request_response.status_code == 200

        reject_response = client.post(
            "/v1/admin/labels/training-eligibility/release-review",
            json={
                "labelIds": [label_id],
                "decision": "rejected",
                "notes": "Release denied pending additional site evidence.",
            },
            headers=headers,
        )
        assert reject_response.status_code == 200
        reject_payload = reject_response.json()
        assert reject_payload["reviewedCount"] == 1
        rejected_label = reject_payload["labels"][0]
        assert rejected_label["trainingEligibilityStatus"] == "hold"
        assert rejected_label["trainingReleaseStatus"] == "rejected"
        assert rejected_label["trainingReleaseReviewedBy"] == "admin"
        release_history = rejected_label["evidence"]["training_release_history"]
        assert release_history[-1]["decision"] == "rejected"


def test_assign_pending_training_release_and_list_release_queue(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()
        due_at = (
            (datetime.now(timezone.utc) + timedelta(days=1))
            .replace(microsecond=0)
            .isoformat()
        )

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-release-queue-001",
                        "zoneId": "moc-01",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Alta",
                        "observer": "brigada-release-queue",
                        "siteVisitId": "visit-release-queue-001",
                        "teamId": "team-release-queue",
                        "mediaRefs": ["photo://release-queue/moc01/1"],
                        "gpsAccuracyMeters": 2.4,
                        "locationNotes": "Pending release assignment path test.",
                        "status": "confirmed",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        hold_response = client.post(
            "/v1/admin/labels/training-eligibility",
            json={
                "labelIds": [label_id],
                "trainingEligibilityStatus": "hold",
                "notes": "Hold before assigning pending release review.",
            },
            headers=headers,
        )
        assert hold_response.status_code == 200

        request_response = client.post(
            "/v1/admin/labels/training-eligibility/release-request",
            json={
                "labelIds": [label_id],
                "releaseCriteria": ["secondary_photo_review"],
                "notes": "Release request waiting for reviewer assignment.",
            },
            headers=headers,
        )
        assert request_response.status_code == 200

        assign_response = client.post(
            "/v1/admin/labels/training-eligibility/release-assign",
            json={
                "labelIds": [label_id],
                "reviewerUsername": "admin",
                "reviewDueAt": due_at,
                "assignmentNotes": "Assigning pending release review to admin.",
            },
            headers=headers,
        )
        assert assign_response.status_code == 200
        assign_payload = assign_response.json()
        assert assign_payload["assignedCount"] == 1
        assigned_label = assign_payload["labels"][0]
        assert assigned_label["trainingReleaseAssignedReviewer"] == "admin"
        assert assigned_label["trainingReleaseStatus"] == "pending"
        assert assigned_label["trainingReleaseIsOverdue"] is False

        queue_response = client.get(
            "/v1/admin/labels/training-release-queue",
            params={"assignedReviewer": "admin"},
            headers=headers,
        )
        assert queue_response.status_code == 200
        queue_payload = queue_response.json()
        assert queue_payload["total"] == 1
        assert queue_payload["assignedCount"] == 1
        assert queue_payload["overdueCount"] == 0
        queued_label = queue_payload["labels"][0]
        assert queued_label["id"] == label_id
        assert queued_label["trainingReleaseAssignedReviewer"] == "admin"
        assert queued_label["trainingReleaseDueAt"].startswith(due_at[:19])


def test_training_release_queue_overdue_filter_surfaces_stale_requests(
    tmp_path, monkeypatch
):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()
        past_due = (
            (datetime.now(timezone.utc) - timedelta(days=1))
            .replace(microsecond=0)
            .isoformat()
        )

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-release-overdue-001",
                        "zoneId": "moc-02",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Media",
                        "observer": "brigada-release-overdue",
                        "siteVisitId": "visit-release-overdue-001",
                        "teamId": "team-release-overdue",
                        "mediaRefs": ["photo://release-overdue/moc02/1"],
                        "gpsAccuracyMeters": 2.8,
                        "locationNotes": "Past-due release queue path test.",
                        "status": "confirmed",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        hold_response = client.post(
            "/v1/admin/labels/training-eligibility",
            json={
                "labelIds": [label_id],
                "trainingEligibilityStatus": "hold",
                "notes": "Hold before overdue release queue test.",
            },
            headers=headers,
        )
        assert hold_response.status_code == 200

        request_response = client.post(
            "/v1/admin/labels/training-eligibility/release-request",
            json={
                "labelIds": [label_id],
                "releaseCriteria": ["secondary_photo_review"],
                "notes": "Pending release request for overdue queue test.",
            },
            headers=headers,
        )
        assert request_response.status_code == 200

        assign_response = client.post(
            "/v1/admin/labels/training-eligibility/release-assign",
            json={
                "labelIds": [label_id],
                "reviewerUsername": "admin",
                "reviewDueAt": past_due,
                "assignmentNotes": "Force overdue status for queue filtering.",
            },
            headers=headers,
        )
        assert assign_response.status_code == 200

        queue_response = client.get(
            "/v1/admin/labels/training-release-queue",
            params={"overdueOnly": "true"},
            headers=headers,
        )
        assert queue_response.status_code == 200
        queue_payload = queue_response.json()
        assert queue_payload["total"] == 1
        assert queue_payload["overdueCount"] == 1
        queued_label = queue_payload["labels"][0]
        assert queued_label["id"] == label_id
        assert queued_label["trainingReleaseIsOverdue"] is True


def test_escalate_pending_training_release_and_filter_escalated_queue(
    tmp_path, monkeypatch
):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-release-escalated-001",
                        "zoneId": "moc-03",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Alta",
                        "observer": "brigada-release-escalation",
                        "siteVisitId": "visit-release-escalation-001",
                        "teamId": "team-release-escalation",
                        "mediaRefs": ["photo://release-escalation/moc03/1"],
                        "gpsAccuracyMeters": 2.1,
                        "locationNotes": "Pending release ready for escalation coverage.",
                        "status": "confirmed",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        hold_response = client.post(
            "/v1/admin/labels/training-eligibility",
            json={
                "labelIds": [label_id],
                "trainingEligibilityStatus": "hold",
                "notes": "Hold before escalation queue test.",
            },
            headers=headers,
        )
        assert hold_response.status_code == 200

        request_response = client.post(
            "/v1/admin/labels/training-eligibility/release-request",
            json={
                "labelIds": [label_id],
                "releaseCriteria": ["secondary_photo_review"],
                "notes": "Pending release that needs escalation.",
            },
            headers=headers,
        )
        assert request_response.status_code == 200

        escalate_response = client.post(
            "/v1/admin/labels/training-eligibility/release-escalate",
            json={
                "labelIds": [label_id],
                "escalationReason": "SLA breached while waiting for release review.",
            },
            headers=headers,
        )
        assert escalate_response.status_code == 200
        escalate_payload = escalate_response.json()
        assert escalate_payload["escalatedCount"] == 1
        escalated_label = escalate_payload["labels"][0]
        assert escalated_label["trainingReleaseEscalationStatus"] == "escalated"
        assert escalated_label["trainingReleaseEscalationLevel"] == 1
        assert escalated_label["trainingReleaseIsEscalated"] is True

        escalated_queue = client.get(
            "/v1/admin/labels/training-release-queue",
            params={"escalatedOnly": "true"},
            headers=headers,
        )
        assert escalated_queue.status_code == 200
        queue_payload = escalated_queue.json()
        assert queue_payload["total"] == 1
        assert queue_payload["escalatedCount"] == 1
        assert queue_payload["unassignedCount"] == 1
        queued_label = queue_payload["labels"][0]
        assert queued_label["id"] == label_id
        assert (
            queued_label["trainingReleaseEscalationReason"]
            == "SLA breached while waiting for release review."
        )

        label_inventory = client.get(
            "/v1/admin/labels",
            params={"trainingReleaseEscalationStatus": "escalated"},
            headers=headers,
        )
        assert label_inventory.status_code == 200
        inventory_payload = label_inventory.json()
        assert len(inventory_payload) == 1
        assert inventory_payload[0]["id"] == label_id


def test_release_review_clears_active_escalation_fields_after_resolution(
    tmp_path, monkeypatch
):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()
        due_at = (
            (datetime.now(timezone.utc) + timedelta(days=1))
            .replace(microsecond=0)
            .isoformat()
        )

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-release-escalated-002",
                        "zoneId": "moc-04",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Media",
                        "observer": "brigada-release-escalation-2",
                        "siteVisitId": "visit-release-escalation-002",
                        "teamId": "team-release-escalation-2",
                        "mediaRefs": ["photo://release-escalation/moc04/1"],
                        "gpsAccuracyMeters": 2.7,
                        "locationNotes": "Escalation resolution path test.",
                        "status": "confirmed",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility",
                json={
                    "labelIds": [label_id],
                    "trainingEligibilityStatus": "hold",
                    "notes": "Hold before escalation resolution test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-request",
                json={
                    "labelIds": [label_id],
                    "releaseCriteria": ["secondary_photo_review"],
                    "notes": "Pending release review for escalation resolution test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-assign",
                json={
                    "labelIds": [label_id],
                    "reviewerUsername": "admin",
                    "reviewDueAt": due_at,
                    "assignmentNotes": "Assign before escalating.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-escalate",
                json={
                    "labelIds": [label_id],
                    "escalationReason": "Escalating before final release decision.",
                    "escalationLevel": 2,
                },
                headers=headers,
            ).status_code
            == 200
        )

        review_response = client.post(
            "/v1/admin/labels/training-eligibility/release-review",
            json={
                "labelIds": [label_id],
                "decision": "approved",
                "notes": "Approved after escalated release review.",
            },
            headers=headers,
        )
        assert review_response.status_code == 200
        review_payload = review_response.json()
        assert review_payload["reviewedCount"] == 1
        reviewed_label = review_payload["labels"][0]
        assert reviewed_label["trainingEligibilityStatus"] == "eligible"
        assert reviewed_label["trainingReleaseStatus"] == "approved"
        assert reviewed_label["trainingReleaseEscalationStatus"] is None
        assert reviewed_label["trainingReleaseEscalationLevel"] is None
        assert reviewed_label["trainingReleaseAssignedReviewer"] is None
        assert reviewed_label["trainingReleaseIsEscalated"] is False
        escalation_history = reviewed_label["evidence"][
            "training_release_escalation_history"
        ]
        assert escalation_history[-1]["escalation_level"] == 2
        assert escalation_history[-1]["escalated_by"] == "admin"


def test_release_assignment_creates_notification_and_acknowledgement_flow(
    tmp_path, monkeypatch
):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()
        due_at = (
            (datetime.now(timezone.utc) + timedelta(days=1))
            .replace(microsecond=0)
            .isoformat()
        )

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-release-notification-001",
                        "zoneId": "moc-01",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Alta",
                        "observer": "brigada-release-notification",
                        "siteVisitId": "visit-release-notification-001",
                        "teamId": "team-release-notification",
                        "mediaRefs": ["photo://release-notification/moc05/1"],
                        "gpsAccuracyMeters": 2.0,
                        "locationNotes": "Notification flow for release assignment.",
                        "status": "confirmed",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility",
                json={
                    "labelIds": [label_id],
                    "trainingEligibilityStatus": "hold",
                    "notes": "Hold before notification assignment test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-request",
                json={
                    "labelIds": [label_id],
                    "releaseCriteria": ["secondary_photo_review"],
                    "notes": "Pending release assignment notification test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assign_response = client.post(
            "/v1/admin/labels/training-eligibility/release-assign",
            json={
                "labelIds": [label_id],
                "reviewerUsername": "admin",
                "reviewDueAt": due_at,
                "assignmentNotes": "Create assignment notification.",
            },
            headers=headers,
        )
        assert assign_response.status_code == 200

        notifications_response = client.get(
            "/v1/admin/notifications",
            params={
                "eventType": "training_release_assignment",
                "targetUsername": "admin",
                "status": "open",
            },
            headers=headers,
        )
        assert notifications_response.status_code == 200
        notifications = notifications_response.json()
        assert notifications
        notification = next(
            item for item in notifications if item["relatedLabelId"] == label_id
        )
        assert notification["eventType"] == "training_release_assignment"
        assert notification["targetUsername"] == "admin"
        assert notification["status"] == "open"
        assert notification["deliveryChannels"] == ["in_app"]
        assert notification["deliveryStatus"] == "delivered"
        assert notification["deliveryAttemptCount"] == 1
        assert notification["failedDeliveryCount"] == 0
        assert notification["ackDueAt"] is not None
        assert notification["isAckOverdue"] is False
        assert notification["details"]["template_key"] == "release_assignment_review"
        assert notification["details"]["template_version"] == "v2"
        assert notification["details"]["summary"] == "Release review assigned"
        assert (
            notification["details"]["routing"]["routing_audience"]
            == "assigned_reviewer"
        )
        assert notification["details"]["routing"]["is_primary"] is True

        acknowledge_response = client.post(
            "/v1/admin/notifications/acknowledge",
            json={"notificationIds": [notification["id"]]},
            headers=headers,
        )
        assert acknowledge_response.status_code == 200
        acknowledge_payload = acknowledge_response.json()
        assert acknowledge_payload["acknowledgedCount"] == 1
        assert acknowledge_payload["notifications"][0]["status"] == "acknowledged"
        assert acknowledge_payload["notifications"][0]["acknowledgedBy"] == "admin"


def test_release_escalation_routes_notifications_to_requester_reviewer_and_ops_watchers(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NOTIFICATION_RELEASE_OPS_USERNAMES", "ops-watch")
    monkeypatch.setenv("NOTIFICATION_ESCALATION_INCLUDE_REQUESTER", "true")

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()
        due_at = (
            (datetime.now(timezone.utc) + timedelta(days=1))
            .replace(microsecond=0)
            .isoformat()
        )

        from app.db.session import session_scope
        from app.models import UserAccount

        with session_scope() as session:
            session.add_all(
                [
                    UserAccount(
                        username="release-reviewer-fanout",
                        password_hash="not-used-in-tests",
                        role="admin",
                        is_active=True,
                        created_at=datetime.now(timezone.utc).replace(microsecond=0),
                    ),
                    UserAccount(
                        username="ops-watch",
                        password_hash="not-used-in-tests",
                        role="admin",
                        is_active=True,
                        created_at=datetime.now(timezone.utc).replace(microsecond=0),
                    ),
                ]
            )

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-release-escalation-routing-001",
                        "zoneId": "moc-02",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Alta",
                        "observer": "brigada-release-escalation-routing",
                        "siteVisitId": "visit-release-escalation-routing-001",
                        "teamId": "team-release-escalation-routing",
                        "mediaRefs": ["photo://release-escalation-routing/moc02/1"],
                        "gpsAccuracyMeters": 2.0,
                        "locationNotes": "Escalation routing fan-out test.",
                        "status": "confirmed",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility",
                json={
                    "labelIds": [label_id],
                    "trainingEligibilityStatus": "hold",
                    "notes": "Hold before escalation fan-out test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-request",
                json={
                    "labelIds": [label_id],
                    "releaseCriteria": ["secondary_photo_review"],
                    "notes": "Pending release before escalation fan-out test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-assign",
                json={
                    "labelIds": [label_id],
                    "reviewerUsername": "release-reviewer-fanout",
                    "reviewDueAt": due_at,
                    "assignmentNotes": "Assign before escalation routing test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        escalate_response = client.post(
            "/v1/admin/labels/training-eligibility/release-escalate",
            json={
                "labelIds": [label_id],
                "escalationReason": "Escalation routing validation.",
                "escalationLevel": 2,
            },
            headers=headers,
        )
        assert escalate_response.status_code == 200

        requester_notifications = client.get(
            "/v1/admin/notifications",
            params={
                "eventType": "training_release_escalation",
                "targetUsername": "admin",
            },
            headers=headers,
        ).json()
        reviewer_notifications = client.get(
            "/v1/admin/notifications",
            params={
                "eventType": "training_release_escalation",
                "targetUsername": "release-reviewer-fanout",
            },
            headers=headers,
        ).json()
        ops_notifications = client.get(
            "/v1/admin/notifications",
            params={
                "eventType": "training_release_escalation",
                "targetUsername": "ops-watch",
            },
            headers=headers,
        ).json()

        requester_notification = next(
            item
            for item in requester_notifications
            if item["relatedLabelId"] == label_id
        )
        reviewer_notification = next(
            item
            for item in reviewer_notifications
            if item["relatedLabelId"] == label_id
        )
        ops_notification = next(
            item for item in ops_notifications if item["relatedLabelId"] == label_id
        )

        assert (
            requester_notification["details"]["routing"]["routing_audience"]
            == "requester_copy"
        )
        assert (
            reviewer_notification["details"]["routing"]["routing_audience"]
            == "assigned_reviewer"
        )
        assert reviewer_notification["details"]["routing"]["is_primary"] is True
        assert ops_notification["details"]["routing"]["routing_audience"] == "ops_watch"
        assert (
            ops_notification["details"]["template_key"] == "release_escalation_notice"
        )
        assert ops_notification["details"]["template_version"] == "v2"
        assert ops_notification["details"]["summary"] == "Release review escalated"

        ops_attempts = client.get(
            "/v1/admin/notifications/delivery-attempts",
            params={
                "notificationId": ops_notification["id"],
                "providerName": "email_stub",
            },
            headers=headers,
        ).json()
        ops_attempt = next(
            item for item in ops_attempts if item["channel"] == "email_stub"
        )
        assert ops_attempt["payloadPreview"]["channel"] == "email_stub"
        assert (
            ops_attempt["payloadPreview"]["template_key"] == "release_escalation_notice"
        )
        assert ops_attempt["payloadPreview"]["routing_audience"] == "ops_watch"
        assert ops_attempt["providerReceipt"]["provider_code"] == "EMAIL_STUB_ACCEPTED"


def test_release_resolution_copies_assigned_reviewer_with_template_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NOTIFICATION_RESOLUTION_COPY_ASSIGNED_REVIEWER", "true")

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()
        due_at = (
            (datetime.now(timezone.utc) + timedelta(days=1))
            .replace(microsecond=0)
            .isoformat()
        )

        from app.db.session import session_scope
        from app.models import UserAccount

        with session_scope() as session:
            session.add(
                UserAccount(
                    username="resolution-reviewer-copy",
                    password_hash="not-used-in-tests",
                    role="admin",
                    is_active=True,
                    created_at=datetime.now(timezone.utc).replace(microsecond=0),
                )
            )

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-release-resolution-routing-001",
                        "zoneId": "moc-03",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Media",
                        "observer": "brigada-release-resolution-routing",
                        "siteVisitId": "visit-release-resolution-routing-001",
                        "teamId": "team-release-resolution-routing",
                        "mediaRefs": ["photo://release-resolution-routing/moc03/1"],
                        "gpsAccuracyMeters": 2.2,
                        "locationNotes": "Resolution notification routing test.",
                        "status": "confirmed",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility",
                json={
                    "labelIds": [label_id],
                    "trainingEligibilityStatus": "hold",
                    "notes": "Hold before resolution routing test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-request",
                json={
                    "labelIds": [label_id],
                    "releaseCriteria": ["secondary_photo_review"],
                    "notes": "Pending release before resolution routing test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-assign",
                json={
                    "labelIds": [label_id],
                    "reviewerUsername": "resolution-reviewer-copy",
                    "reviewDueAt": due_at,
                    "assignmentNotes": "Assign before resolution routing test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        review_response = client.post(
            "/v1/admin/labels/training-eligibility/release-review",
            json={
                "labelIds": [label_id],
                "decision": "approved",
                "notes": "Approve to test requester and reviewer copy routing.",
            },
            headers=headers,
        )
        assert review_response.status_code == 200

        requester_notifications = client.get(
            "/v1/admin/notifications",
            params={
                "eventType": "training_release_resolution",
                "targetUsername": "admin",
            },
            headers=headers,
        ).json()
        reviewer_notifications = client.get(
            "/v1/admin/notifications",
            params={
                "eventType": "training_release_resolution",
                "targetUsername": "resolution-reviewer-copy",
            },
            headers=headers,
        ).json()

        requester_notification = next(
            item
            for item in requester_notifications
            if item["relatedLabelId"] == label_id
        )
        reviewer_notification = next(
            item
            for item in reviewer_notifications
            if item["relatedLabelId"] == label_id
        )

        assert (
            requester_notification["details"]["routing"]["routing_audience"]
            == "requester"
        )
        assert (
            requester_notification["details"]["template_key"]
            == "release_resolution_notice"
        )
        assert requester_notification["details"]["template_version"] == "v2"
        assert (
            reviewer_notification["details"]["routing"]["routing_audience"]
            == "assigned_reviewer_copy"
        )
        assert reviewer_notification["details"]["summary"] == "Release review resolved"


def test_notification_ack_deadline_scan_creates_reminder_for_overdue_open_notification(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NOTIFICATION_ACK_DEADLINE_HOURS_INFO", "0")
    monkeypatch.setenv("NOTIFICATION_ACK_REMINDER_MAX_COUNT", "1")

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()
        due_at = (
            (datetime.now(timezone.utc) + timedelta(days=1))
            .replace(microsecond=0)
            .isoformat()
        )

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-notification-ack-scan-001",
                        "zoneId": "moc-01",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Alta",
                        "observer": "brigada-notification-ack-scan",
                        "siteVisitId": "visit-notification-ack-scan-001",
                        "teamId": "team-notification-ack-scan",
                        "mediaRefs": ["photo://notification-ack-scan/moc01/1"],
                        "gpsAccuracyMeters": 2.1,
                        "locationNotes": "Notification ack deadline scan test.",
                        "status": "confirmed",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility",
                json={
                    "labelIds": [label_id],
                    "trainingEligibilityStatus": "hold",
                    "notes": "Hold before notification ack scan test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-request",
                json={
                    "labelIds": [label_id],
                    "releaseCriteria": ["secondary_photo_review"],
                    "notes": "Pending release before notification ack scan test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-assign",
                json={
                    "labelIds": [label_id],
                    "reviewerUsername": "admin",
                    "reviewDueAt": due_at,
                    "assignmentNotes": "Creates assignment notification with immediate ack deadline.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        source_notifications = client.get(
            "/v1/admin/notifications",
            params={
                "eventType": "training_release_assignment",
                "targetUsername": "admin",
            },
            headers=headers,
        )
        assert source_notifications.status_code == 200
        source_notification = next(
            item
            for item in source_notifications.json()
            if item["relatedLabelId"] == label_id
        )

        from app.db.session import session_scope
        from app.models import NotificationEvent

        with session_scope() as session:
            notification = session.get(NotificationEvent, source_notification["id"])
            notification.ack_due_at = datetime.now(timezone.utc).replace(
                microsecond=0
            ) - timedelta(minutes=5)

        overdue_response = client.get(
            "/v1/admin/notifications",
            params={"eventType": "training_release_assignment", "overdueOnly": "true"},
            headers=headers,
        )
        assert overdue_response.status_code == 200
        overdue_notifications = overdue_response.json()
        assert overdue_notifications
        source_notification = next(
            item for item in overdue_notifications if item["relatedLabelId"] == label_id
        )
        assert source_notification["isAckOverdue"] is True

        scan_response = client.post(
            "/v1/admin/notifications/ack-deadline-scan",
            json={"maxNotifications": 10, "note": "Reminder scan test."},
            headers=headers,
        )
        assert scan_response.status_code == 200
        scan_payload = scan_response.json()
        assert scan_payload["job"]["jobType"] == "notification_ack_deadline_scan"
        assert scan_payload["sourceCount"] >= 1
        assert scan_payload["remindedCount"] >= 1
        reminder = next(
            item
            for item in scan_payload["notifications"]
            if item["relatedLabelId"] == label_id
        )
        assert reminder["eventType"] == "notification_ack_deadline_reminder"
        assert reminder["severity"] == "warning"
        assert reminder["deliveryChannels"] == ["in_app", "email_stub"]

        reminder_notifications = client.get(
            "/v1/admin/notifications",
            params={
                "eventType": "notification_ack_deadline_reminder",
                "targetUsername": "admin",
            },
            headers=headers,
        )
        assert reminder_notifications.status_code == 200
        reminders = reminder_notifications.json()
        reminder_notification = next(
            item for item in reminders if item["relatedLabelId"] == label_id
        )
        assert (
            reminder_notification["details"]["source_notification_id"]
            == source_notification["id"]
        )


def test_notification_delivery_attempts_are_visible_and_retryable(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NOTIFICATION_STUB_FAIL_CHANNELS", "email_stub")

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()
        first_due_at = (
            (datetime.now(timezone.utc) + timedelta(days=1))
            .replace(microsecond=0)
            .isoformat()
        )
        second_due_at = (
            (datetime.now(timezone.utc) + timedelta(days=2))
            .replace(microsecond=0)
            .isoformat()
        )

        from app.db.session import session_scope
        from app.models import UserAccount

        with session_scope() as session:
            session.add(
                UserAccount(
                    username="delivery-reviewer",
                    password_hash="not-used-in-tests",
                    role="admin",
                    is_active=True,
                    created_at=datetime.now(timezone.utc).replace(microsecond=0),
                )
            )

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-notification-delivery-001",
                        "zoneId": "moc-03",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Alta",
                        "observer": "brigada-notification-delivery",
                        "siteVisitId": "visit-notification-delivery-001",
                        "teamId": "team-notification-delivery",
                        "mediaRefs": ["photo://notification-delivery/moc03/1"],
                        "gpsAccuracyMeters": 1.8,
                        "locationNotes": "Delivery-attempt and retry workflow test.",
                        "status": "confirmed",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility",
                json={
                    "labelIds": [label_id],
                    "trainingEligibilityStatus": "hold",
                    "notes": "Hold before notification delivery retry test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-request",
                json={
                    "labelIds": [label_id],
                    "releaseCriteria": ["secondary_photo_review"],
                    "notes": "Pending release before notification delivery retry test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-assign",
                json={
                    "labelIds": [label_id],
                    "reviewerUsername": "admin",
                    "reviewDueAt": first_due_at,
                    "assignmentNotes": "Initial assignment before reassignment delivery test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        reassign_response = client.post(
            "/v1/admin/labels/training-eligibility/release-reassign",
            json={
                "labelIds": [label_id],
                "reviewerUsername": "delivery-reviewer",
                "reassignmentReason": "Move review to channel retry test reviewer.",
                "reviewDueAt": second_due_at,
            },
            headers=headers,
        )
        assert reassign_response.status_code == 200

        notifications_response = client.get(
            "/v1/admin/notifications",
            params={
                "eventType": "training_release_reassignment",
                "targetUsername": "delivery-reviewer",
                "deliveryStatus": "partial_failure",
            },
            headers=headers,
        )
        assert notifications_response.status_code == 200
        notifications = notifications_response.json()
        assert notifications
        notification = next(
            item for item in notifications if item["relatedLabelId"] == label_id
        )
        assert notification["deliveryChannels"] == ["in_app", "email_stub"]
        assert notification["deliveryStatus"] == "partial_failure"
        assert notification["deliveryAttemptCount"] == 2
        assert notification["failedDeliveryCount"] == 1

        attempts_response = client.get(
            "/v1/admin/notifications/delivery-attempts",
            params={"notificationId": notification["id"], "status": "failed"},
            headers=headers,
        )
        assert attempts_response.status_code == 200
        attempts = attempts_response.json()
        assert attempts
        failed_attempt = next(
            item for item in attempts if item["channel"] == "email_stub"
        )
        assert failed_attempt["adapterKey"] == "notifications.email_stub"
        assert failed_attempt["providerName"] == "email_stub"
        assert failed_attempt["providerStatus"] == "temporary_failure"
        assert failed_attempt["failureClassification"] == "stubbed_transient_failure"
        assert failed_attempt["retryable"] is True
        assert failed_attempt["deliveryOrigin"] == "initial"
        assert failed_attempt["payloadPreview"]["channel"] == "email_stub"
        assert (
            failed_attempt["payloadPreview"]["template_key"]
            == "release_reassignment_notice"
        )
        assert (
            failed_attempt["providerReceipt"]["provider_code"] == "EMAIL_STUB_TEMPFAIL"
        )
        assert "Stubbed email delivery failure" in failed_attempt["errorMessage"]

        from app.core.config import get_settings

        monkeypatch.setenv("NOTIFICATION_STUB_FAIL_CHANNELS", "")
        get_settings.cache_clear()

        retry_response = client.post(
            "/v1/admin/notifications/retry-delivery",
            json={
                "notificationIds": [notification["id"]],
                "note": "Retry failed delivery attempt.",
            },
            headers=headers,
        )
        assert retry_response.status_code == 200
        retry_payload = retry_response.json()
        assert retry_payload["job"]["jobType"] == "notification_delivery_retry"
        assert retry_payload["retriedCount"] == 1
        assert retry_payload["skippedCount"] == 0
        retry_attempt = retry_payload["attempts"][0]
        assert retry_attempt["notificationId"] == notification["id"]
        assert retry_attempt["channel"] == "email_stub"
        assert retry_attempt["status"] == "completed"
        assert retry_attempt["adapterKey"] == "notifications.email_stub"
        assert retry_attempt["providerStatus"] == "accepted"
        assert retry_attempt["deliveryOrigin"] == "retry"
        assert (
            retry_attempt["providerReceipt"]["provider_code"] == "EMAIL_STUB_ACCEPTED"
        )

        refreshed_notifications = client.get(
            "/v1/admin/notifications",
            params={
                "eventType": "training_release_reassignment",
                "targetUsername": "delivery-reviewer",
            },
            headers=headers,
        )
        assert refreshed_notifications.status_code == 200
        refreshed = refreshed_notifications.json()
        updated_notification = next(
            item for item in refreshed if item["relatedLabelId"] == label_id
        )
        assert updated_notification["deliveryStatus"] == "delivered"
        assert updated_notification["failedDeliveryCount"] == 0
        assert updated_notification["deliveryAttemptCount"] == 3


def test_notification_retry_scan_retries_failed_channels_automatically(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NOTIFICATION_STUB_FAIL_CHANNELS", "email_stub")
    monkeypatch.setenv("NOTIFICATION_DELIVERY_RETRY_BACKOFF_MINUTES", "0")
    monkeypatch.setenv("NOTIFICATION_DELIVERY_RETRY_MAX_ATTEMPTS_PER_CHANNEL", "3")

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-notification-retry-scan-001",
                        "zoneId": "moc-04",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Media",
                        "observer": "brigada-notification-retry-scan",
                        "siteVisitId": "visit-notification-retry-scan-001",
                        "teamId": "team-notification-retry-scan",
                        "mediaRefs": ["photo://notification-retry-scan/moc04/1"],
                        "gpsAccuracyMeters": 2.6,
                        "locationNotes": "Automatic notification retry scan test.",
                        "status": "confirmed",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility",
                json={
                    "labelIds": [label_id],
                    "trainingEligibilityStatus": "hold",
                    "notes": "Hold before retry scan test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-request",
                json={
                    "labelIds": [label_id],
                    "releaseCriteria": ["secondary_photo_review"],
                    "notes": "Pending release before retry scan test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        review_response = client.post(
            "/v1/admin/labels/training-eligibility/release-review",
            json={
                "labelIds": [label_id],
                "decision": "rejected",
                "notes": "Create warning notification with failed email channel.",
            },
            headers=headers,
        )
        assert review_response.status_code == 200

        notifications_response = client.get(
            "/v1/admin/notifications",
            params={
                "eventType": "training_release_resolution",
                "targetUsername": "admin",
                "deliveryStatus": "partial_failure",
            },
            headers=headers,
        )
        assert notifications_response.status_code == 200
        notifications = notifications_response.json()
        notification = next(
            item for item in notifications if item["relatedLabelId"] == label_id
        )
        assert notification["deliveryStatus"] == "partial_failure"
        assert notification["deliveryAttemptCount"] == 2

        first_retry_scan = client.post(
            "/v1/admin/notifications/retry-scan",
            json={
                "maxNotifications": 10,
                "note": "Retry failed channels while failure is still active.",
            },
            headers=headers,
        )
        assert first_retry_scan.status_code == 200
        first_scan_payload = first_retry_scan.json()
        assert (
            first_scan_payload["job"]["jobType"] == "notification_delivery_retry_scan"
        )
        assert first_scan_payload["candidateCount"] >= 1
        assert first_scan_payload["retriedCount"] >= 1
        failed_retry_attempt = next(
            item
            for item in first_scan_payload["attempts"]
            if item["notificationId"] == notification["id"]
        )
        assert failed_retry_attempt["channel"] == "email_stub"
        assert failed_retry_attempt["status"] == "failed"

        from app.core.config import get_settings

        monkeypatch.setenv("NOTIFICATION_STUB_FAIL_CHANNELS", "")
        get_settings.cache_clear()

        second_retry_scan = client.post(
            "/v1/admin/notifications/retry-scan",
            json={
                "maxNotifications": 10,
                "note": "Retry failed channels after clearing failure.",
            },
            headers=headers,
        )
        assert second_retry_scan.status_code == 200
        second_scan_payload = second_retry_scan.json()
        assert second_scan_payload["candidateCount"] >= 1
        assert second_scan_payload["retriedCount"] >= 1
        successful_retry_attempt = next(
            item
            for item in second_scan_payload["attempts"]
            if item["notificationId"] == notification["id"]
        )
        assert successful_retry_attempt["channel"] == "email_stub"
        assert successful_retry_attempt["status"] == "completed"

        refreshed_notifications = client.get(
            "/v1/admin/notifications",
            params={
                "eventType": "training_release_resolution",
                "targetUsername": "admin",
            },
            headers=headers,
        )
        assert refreshed_notifications.status_code == 200
        refreshed = refreshed_notifications.json()
        updated_notification = next(
            item for item in refreshed if item["id"] == notification["id"]
        )
        assert updated_notification["deliveryStatus"] == "delivered"
        assert updated_notification["failedDeliveryCount"] == 0
        assert updated_notification["deliveryAttemptCount"] == 4


def test_notification_retry_scan_skips_non_retryable_failure_classification(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NOTIFICATION_STUB_FAIL_CHANNELS", "email_stub")
    monkeypatch.setenv("NOTIFICATION_DELIVERY_RETRY_BACKOFF_MINUTES", "0")
    monkeypatch.setenv("NOTIFICATION_DELIVERY_RETRY_MAX_ATTEMPTS_PER_CHANNEL", "3")

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-notification-retry-nonretryable-001",
                        "zoneId": "moc-04",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Media",
                        "observer": "brigada-notification-retry-nonretryable",
                        "siteVisitId": "visit-notification-retry-nonretryable-001",
                        "teamId": "team-notification-retry-nonretryable",
                        "mediaRefs": [
                            "photo://notification-retry-nonretryable/moc04/1"
                        ],
                        "gpsAccuracyMeters": 2.1,
                        "locationNotes": "Non-retryable notification failure classification test.",
                        "status": "confirmed",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility",
                json={
                    "labelIds": [label_id],
                    "trainingEligibilityStatus": "hold",
                    "notes": "Hold before non-retryable retry scan test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-request",
                json={
                    "labelIds": [label_id],
                    "releaseCriteria": ["secondary_photo_review"],
                    "notes": "Pending release before non-retryable retry scan test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        review_response = client.post(
            "/v1/admin/labels/training-eligibility/release-review",
            json={
                "labelIds": [label_id],
                "decision": "rejected",
                "notes": "Create warning notification with failed email channel.",
            },
            headers=headers,
        )
        assert review_response.status_code == 200

        notifications_response = client.get(
            "/v1/admin/notifications",
            params={
                "eventType": "training_release_resolution",
                "targetUsername": "admin",
                "deliveryStatus": "partial_failure",
            },
            headers=headers,
        )
        notification = next(
            item
            for item in notifications_response.json()
            if item["relatedLabelId"] == label_id
        )

        from app.db.session import session_scope
        from app.models import NotificationDeliveryAttempt

        with session_scope() as session:
            latest_failed_attempt = (
                session.query(NotificationDeliveryAttempt)
                .filter(
                    NotificationDeliveryAttempt.notification_event_id
                    == notification["id"],
                    NotificationDeliveryAttempt.channel == "email_stub",
                )
                .order_by(NotificationDeliveryAttempt.id.desc())
                .first()
            )
            latest_failed_attempt.details = {
                **(latest_failed_attempt.details or {}),
                "failure_classification": "configuration_error",
                "retryable": False,
            }

        retry_scan = client.post(
            "/v1/admin/notifications/retry-scan",
            json={
                "maxNotifications": 20,
                "note": "Skip non-retryable configuration failures.",
            },
            headers=headers,
        )
        assert retry_scan.status_code == 200
        retry_scan_payload = retry_scan.json()
        assert retry_scan_payload["candidateCount"] == 0
        assert retry_scan_payload["retriedCount"] == 0
        assert all(
            item["notificationId"] != notification["id"]
            for item in retry_scan_payload["attempts"]
        )

        attempt_inventory = client.get(
            "/v1/admin/notifications/delivery-attempts",
            params={
                "notificationId": notification["id"],
                "failureClassification": "configuration_error",
                "providerName": "email_stub",
            },
            headers=headers,
        )
        assert attempt_inventory.status_code == 200
        attempts = attempt_inventory.json()
        assert attempts
        assert attempts[0]["retryable"] is False
        assert attempts[0]["failureClassification"] == "configuration_error"
        assert attempts[0]["providerName"] == "email_stub"


def test_notification_delivery_summary_reports_current_failure_inventory(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NOTIFICATION_STUB_FAIL_CHANNELS", "email_stub")

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        notification_id = create_failed_warning_notification()

        summary_response = client.get(
            "/v1/admin/notifications/delivery-summary", headers=headers
        )
        assert summary_response.status_code == 200
        summary = summary_response.json()

        assert summary["totalNotifications"] >= 1
        assert summary["openNotifications"] >= 1
        assert summary["deliveryStatusCounts"]["partial_failure"] >= 1
        assert summary["channelFailureCounts"]["email_stub"] >= 1
        assert summary["providerFailureCounts"]["email_stub"] >= 1
        assert summary["failureClassificationCounts"]["stubbed_transient_failure"] >= 1
        assert summary["notificationsWithFailures"] >= 1
        assert summary["retryableFailureNotificationCount"] >= 1
        assert summary["oldestOutstandingFailureAt"] is not None

        attempts_response = client.get(
            "/v1/admin/notifications/delivery-attempts",
            params={"notificationId": notification_id, "status": "failed"},
            headers=headers,
        )
        assert attempts_response.status_code == 200
        attempts = attempts_response.json()
        assert attempts
        assert attempts[0]["providerName"] == "email_stub"


def test_notification_delivery_failure_scan_alerts_and_resolves_recovered_failures(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NOTIFICATION_STUB_FAIL_CHANNELS", "email_stub")
    monkeypatch.setenv("NOTIFICATION_DELIVERY_RETRY_BACKOFF_MINUTES", "0")
    monkeypatch.setenv("NOTIFICATION_DELIVERY_FAILURE_ALERT_AFTER_ATTEMPTS", "2")
    monkeypatch.setenv("NOTIFICATION_DELIVERY_FAILURE_WATCH_USERNAMES", "ops-watch")

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        notification_id = create_failed_warning_notification(target_username="admin")

        from app.core.config import get_settings

        retry_scan_response = client.post(
            "/v1/admin/notifications/retry-scan",
            json={
                "maxNotifications": 20,
                "note": "Create a repeated retryable failure before alerting.",
            },
            headers=headers,
        )
        assert retry_scan_response.status_code == 200
        retry_payload = retry_scan_response.json()
        assert retry_payload["candidateCount"] >= 1
        assert any(
            item["notificationId"] == notification_id
            for item in retry_payload["attempts"]
        )

        failure_scan_response = client.post(
            "/v1/admin/notifications/delivery-failure-scan",
            json={
                "maxNotifications": 20,
                "note": "Create delivery alert for repeated retryable failure.",
            },
            headers=headers,
        )
        assert failure_scan_response.status_code == 200
        failure_payload = failure_scan_response.json()
        assert failure_payload["candidateCount"] >= 1
        assert failure_payload["alertedCount"] >= 1
        alert = next(
            item
            for item in failure_payload["alerts"]
            if item["details"]["source_notification_id"] == notification_id
        )
        assert alert["eventType"] == "notification_delivery_failure_alert"
        assert alert["targetUsername"] == "ops-watch"
        assert alert["details"]["alert_reason_codes"] == ["repeated_retryable_failure"]

        summary_response = client.get(
            "/v1/admin/notifications/delivery-summary", headers=headers
        )
        assert summary_response.status_code == 200
        assert summary_response.json()["activeAlertCount"] >= 1

        monkeypatch.setenv("NOTIFICATION_STUB_FAIL_CHANNELS", "")
        get_settings.cache_clear()

        recovery_retry_response = client.post(
            "/v1/admin/notifications/retry-scan",
            json={
                "maxNotifications": 20,
                "note": "Recover the failed delivery after clearing stub failures.",
            },
            headers=headers,
        )
        assert recovery_retry_response.status_code == 200
        recovery_payload = recovery_retry_response.json()
        assert any(
            item["notificationId"] == notification_id and item["status"] == "completed"
            for item in recovery_payload["attempts"]
        )

        resolve_scan_response = client.post(
            "/v1/admin/notifications/delivery-failure-scan",
            json={"maxNotifications": 20, "note": "Resolve recovered delivery alerts."},
            headers=headers,
        )
        assert resolve_scan_response.status_code == 200
        resolve_payload = resolve_scan_response.json()
        assert resolve_payload["candidateCount"] == 0
        assert resolve_payload["resolvedAlertCount"] >= 1

        alert_inventory_response = client.get(
            "/v1/admin/notifications",
            params={
                "eventType": "notification_delivery_failure_alert",
                "targetUsername": "ops-watch",
            },
            headers=headers,
        )
        assert alert_inventory_response.status_code == 200
        alerts = alert_inventory_response.json()
        resolved_alert = next(
            item
            for item in alerts
            if item["details"]["source_notification_id"] == notification_id
        )
        assert resolved_alert["status"] == "resolved"


def test_notification_ack_deadline_scan_escalates_second_reminder_delivery_policy(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NOTIFICATION_ACK_DEADLINE_HOURS_INFO", "0")
    monkeypatch.setenv("NOTIFICATION_ACK_REMINDER_MAX_COUNT", "2")
    monkeypatch.setenv("NOTIFICATION_ACK_REMINDER_ESCALATE_AFTER_COUNT", "2")

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()
        due_at = (
            (datetime.now(timezone.utc) + timedelta(days=1))
            .replace(microsecond=0)
            .isoformat()
        )

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-notification-reminder-policy-001",
                        "zoneId": "moc-01",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Alta",
                        "observer": "brigada-reminder-policy",
                        "siteVisitId": "visit-reminder-policy-001",
                        "teamId": "team-reminder-policy",
                        "mediaRefs": ["photo://notification-reminder-policy/moc01/1"],
                        "gpsAccuracyMeters": 2.0,
                        "locationNotes": "Reminder escalation policy test.",
                        "status": "confirmed",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility",
                json={
                    "labelIds": [label_id],
                    "trainingEligibilityStatus": "hold",
                    "notes": "Hold before reminder escalation policy test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-request",
                json={
                    "labelIds": [label_id],
                    "releaseCriteria": ["secondary_photo_review"],
                    "notes": "Pending release before reminder escalation policy test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-assign",
                json={
                    "labelIds": [label_id],
                    "reviewerUsername": "admin",
                    "reviewDueAt": due_at,
                    "assignmentNotes": "Create info notification with immediate ack deadline.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        source_notifications = client.get(
            "/v1/admin/notifications",
            params={
                "eventType": "training_release_assignment",
                "targetUsername": "admin",
            },
            headers=headers,
        )
        assert source_notifications.status_code == 200
        source_notification = next(
            item
            for item in source_notifications.json()
            if item["relatedLabelId"] == label_id
        )

        from app.db.session import session_scope
        from app.models import NotificationEvent

        with session_scope() as session:
            notification = session.get(NotificationEvent, source_notification["id"])
            notification.ack_due_at = datetime.now(timezone.utc).replace(
                microsecond=0
            ) - timedelta(minutes=5)

        first_scan = client.post(
            "/v1/admin/notifications/ack-deadline-scan",
            json={"maxNotifications": 10, "note": "First reminder policy scan."},
            headers=headers,
        )
        assert first_scan.status_code == 200
        first_scan_payload = first_scan.json()
        first_reminder = next(
            item
            for item in first_scan_payload["notifications"]
            if item["relatedLabelId"] == label_id
        )
        assert first_reminder["severity"] == "warning"
        assert first_reminder["deliveryChannels"] == ["in_app", "email_stub"]
        assert first_reminder["details"]["reminder_sequence"] == 1
        assert first_reminder["details"]["reminder_severity"] == "warning"

        second_scan = client.post(
            "/v1/admin/notifications/ack-deadline-scan",
            json={"maxNotifications": 10, "note": "Second reminder policy scan."},
            headers=headers,
        )
        assert second_scan.status_code == 200
        second_scan_payload = second_scan.json()
        second_reminder = next(
            item
            for item in second_scan_payload["notifications"]
            if item["relatedLabelId"] == label_id
            and item["details"]["reminder_sequence"] == 2
        )
        assert second_reminder["severity"] == "critical"
        assert second_reminder["deliveryChannels"] == [
            "in_app",
            "email_stub",
            "ops_webhook_stub",
        ]
        assert second_reminder["details"]["reminder_severity"] == "critical"


def test_manual_training_release_sla_scan_auto_escalates_overdue_label_and_creates_notification(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TRAINING_RELEASE_AUTO_ESCALATION_LEVEL", "3")

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()
        past_due = (
            (datetime.now(timezone.utc) - timedelta(days=1))
            .replace(microsecond=0)
            .isoformat()
        )

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-release-sla-001",
                        "zoneId": "moc-02",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Media",
                        "observer": "brigada-release-sla",
                        "siteVisitId": "visit-release-sla-001",
                        "teamId": "team-release-sla",
                        "mediaRefs": ["photo://release-sla/moc06/1"],
                        "gpsAccuracyMeters": 2.4,
                        "locationNotes": "SLA auto-escalation test.",
                        "status": "confirmed",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility",
                json={
                    "labelIds": [label_id],
                    "trainingEligibilityStatus": "hold",
                    "notes": "Hold before SLA scan test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-request",
                json={
                    "labelIds": [label_id],
                    "releaseCriteria": ["secondary_photo_review"],
                    "notes": "Pending release for SLA scan.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-assign",
                json={
                    "labelIds": [label_id],
                    "reviewerUsername": "admin",
                    "reviewDueAt": past_due,
                    "assignmentNotes": "Past due before SLA scan.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        sla_response = client.post(
            "/v1/admin/labels/training-eligibility/release-sla-scan",
            json={"maxLabels": 10, "note": "Automatic SLA escalation test run."},
            headers=headers,
        )
        assert sla_response.status_code == 200
        sla_payload = sla_response.json()
        assert sla_payload["job"]["jobType"] == "training_release_sla_scan"
        assert sla_payload["escalatedCount"] == 1
        assert sla_payload["notificationCount"] == 1
        escalated_label = sla_payload["labels"][0]
        assert escalated_label["id"] == label_id
        assert escalated_label["trainingReleaseEscalationStatus"] == "escalated"
        assert escalated_label["trainingReleaseEscalationLevel"] == 3

        notifications_response = client.get(
            "/v1/admin/notifications",
            params={"eventType": "training_release_auto_escalation", "status": "open"},
            headers=headers,
        )
        assert notifications_response.status_code == 200
        notifications = notifications_response.json()
        assert notifications
        notification = next(
            item for item in notifications if item["relatedLabelId"] == label_id
        )
        assert notification["severity"] == "critical"
        assert notification["targetUsername"] == "admin"


def test_manual_release_reassignment_updates_reviewer_and_creates_notification(
    tmp_path, monkeypatch
):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()
        first_due_at = (
            (datetime.now(timezone.utc) + timedelta(days=1))
            .replace(microsecond=0)
            .isoformat()
        )
        second_due_at = (
            (datetime.now(timezone.utc) + timedelta(days=2))
            .replace(microsecond=0)
            .isoformat()
        )

        from app.db.session import session_scope
        from app.models import UserAccount

        with session_scope() as session:
            session.add(
                UserAccount(
                    username="release-reviewer-ops",
                    password_hash="not-used-in-tests",
                    role="admin",
                    is_active=True,
                    created_at=datetime.now(timezone.utc).replace(microsecond=0),
                )
            )

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-release-reassign-001",
                        "zoneId": "moc-03",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Alta",
                        "observer": "brigada-release-reassign",
                        "siteVisitId": "visit-release-reassign-001",
                        "teamId": "team-release-reassign",
                        "mediaRefs": ["photo://release-reassign/moc03/1"],
                        "gpsAccuracyMeters": 2.2,
                        "locationNotes": "Manual reassignment workflow test.",
                        "status": "confirmed",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility",
                json={
                    "labelIds": [label_id],
                    "trainingEligibilityStatus": "hold",
                    "notes": "Hold before manual reassignment test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-request",
                json={
                    "labelIds": [label_id],
                    "releaseCriteria": ["secondary_photo_review"],
                    "notes": "Pending release before manual reassignment test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-assign",
                json={
                    "labelIds": [label_id],
                    "reviewerUsername": "admin",
                    "reviewDueAt": first_due_at,
                    "assignmentNotes": "Initial reviewer before reassignment.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        reassign_response = client.post(
            "/v1/admin/labels/training-eligibility/release-reassign",
            json={
                "labelIds": [label_id],
                "reviewerUsername": "release-reviewer-ops",
                "reviewDueAt": second_due_at,
                "reassignmentReason": "Escalated to alternate operator.",
            },
            headers=headers,
        )
        assert reassign_response.status_code == 200
        payload = reassign_response.json()
        assert payload["reassignedCount"] == 1
        reassigned_label = payload["labels"][0]
        assert (
            reassigned_label["trainingReleaseAssignedReviewer"]
            == "release-reviewer-ops"
        )
        assert reassigned_label["trainingReleaseDueAt"].startswith(second_due_at[:19])
        reassignment_history = reassigned_label["evidence"][
            "training_release_reassignment_history"
        ]
        assert reassignment_history[-1]["previous_reviewer_username"] == "admin"
        assert reassignment_history[-1]["reviewer_username"] == "release-reviewer-ops"

        notifications_response = client.get(
            "/v1/admin/notifications",
            params={
                "eventType": "training_release_reassignment",
                "targetUsername": "release-reviewer-ops",
                "status": "open",
            },
            headers=headers,
        )
        assert notifications_response.status_code == 200
        notifications = notifications_response.json()
        notification = next(
            item for item in notifications if item["relatedLabelId"] == label_id
        )
        assert notification["details"]["previous_reviewer_username"] == "admin"


def test_release_reassignment_scan_reassigns_overdue_escalated_label_to_fallback_reviewer(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TRAINING_RELEASE_AUTO_REASSIGN_REVIEWER", "admin")
    monkeypatch.setenv("TRAINING_RELEASE_AUTO_REASSIGN_DUE_IN_HOURS", "48")

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()
        past_due = (
            (datetime.now(timezone.utc) - timedelta(days=1))
            .replace(microsecond=0)
            .isoformat()
        )

        from app.db.session import session_scope
        from app.models import UserAccount

        with session_scope() as session:
            session.add(
                UserAccount(
                    username="release-reviewer-field",
                    password_hash="not-used-in-tests",
                    role="admin",
                    is_active=True,
                    created_at=datetime.now(timezone.utc).replace(microsecond=0),
                )
            )

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-release-reassign-scan-001",
                        "zoneId": "moc-04",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Media",
                        "observer": "brigada-release-reassign-scan",
                        "siteVisitId": "visit-release-reassign-scan-001",
                        "teamId": "team-release-reassign-scan",
                        "mediaRefs": ["photo://release-reassign-scan/moc04/1"],
                        "gpsAccuracyMeters": 2.5,
                        "locationNotes": "Automatic reassignment scan workflow test.",
                        "status": "confirmed",
                    }
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_id = import_response.json()["labels"][0]["id"]

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility",
                json={
                    "labelIds": [label_id],
                    "trainingEligibilityStatus": "hold",
                    "notes": "Hold before reassignment scan test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-request",
                json={
                    "labelIds": [label_id],
                    "releaseCriteria": ["secondary_photo_review"],
                    "notes": "Pending release before reassignment scan test.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-assign",
                json={
                    "labelIds": [label_id],
                    "reviewerUsername": "release-reviewer-field",
                    "reviewDueAt": past_due,
                    "assignmentNotes": "Initial overdue reviewer before reassignment scan.",
                },
                headers=headers,
            ).status_code
            == 200
        )

        assert (
            client.post(
                "/v1/admin/labels/training-eligibility/release-escalate",
                json={
                    "labelIds": [label_id],
                    "escalationReason": "Escalate before automatic reassignment.",
                    "escalationLevel": 2,
                },
                headers=headers,
            ).status_code
            == 200
        )

        scan_response = client.post(
            "/v1/admin/labels/training-eligibility/release-reassignment-scan",
            json={"maxLabels": 10, "note": "Automatic reassignment test run."},
            headers=headers,
        )
        assert scan_response.status_code == 200
        scan_payload = scan_response.json()
        assert scan_payload["job"]["jobType"] == "training_release_reassignment_scan"
        assert scan_payload["reassignedCount"] == 1
        assert scan_payload["notificationCount"] == 1
        reassigned_label = scan_payload["labels"][0]
        assert reassigned_label["id"] == label_id
        assert reassigned_label["trainingReleaseAssignedReviewer"] == "admin"
        assert reassigned_label["trainingReleaseIsEscalated"] is True
        reassignment_history = reassigned_label["evidence"][
            "training_release_reassignment_history"
        ]
        assert (
            reassignment_history[-1]["previous_reviewer_username"]
            == "release-reviewer-field"
        )

        notifications_response = client.get(
            "/v1/admin/notifications",
            params={
                "eventType": "training_release_reassignment",
                "targetUsername": "admin",
                "status": "open",
            },
            headers=headers,
        )
        assert notifications_response.status_code == 200
        notifications = notifications_response.json()
        notification = next(
            item for item in notifications if item["relatedLabelId"] == label_id
        )
        assert (
            notification["details"]["previous_reviewer_username"]
            == "release-reviewer-field"
        )


def test_hold_training_eligibility_blocks_label_dataset_export(tmp_path, monkeypatch):
    datasets_path = tmp_path / "training-datasets"
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-hold-export-001",
                        "zoneId": "moc-01",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Alta",
                        "observer": "brigada-hold-1",
                        "siteVisitId": "visit-hold-export-001",
                        "teamId": "team-hold-alpha",
                        "mediaRefs": ["photo://hold-export/moc01/1"],
                        "gpsAccuracyMeters": 2.5,
                        "locationNotes": "Hold candidate with complete evidence.",
                        "status": "confirmed",
                    },
                    {
                        "observationId": "fv-hold-export-002",
                        "zoneId": "moc-02",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Media",
                        "observer": "brigada-hold-2",
                        "siteVisitId": "visit-hold-export-002",
                        "teamId": "team-hold-beta",
                        "mediaRefs": ["photo://hold-export/moc02/1"],
                        "gpsAccuracyMeters": 3.1,
                        "locationNotes": "Eligible comparison label.",
                        "status": "confirmed",
                    },
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        labels_by_source = {
            item["source"]: item for item in import_response.json()["labels"]
        }
        held_label_id = labels_by_source["field_validation:fv-hold-export-001"]["id"]
        eligible_label_id = labels_by_source["field_validation:fv-hold-export-002"][
            "id"
        ]

        hold_response = client.post(
            "/v1/admin/labels/training-eligibility",
            json={
                "labelIds": [held_label_id],
                "trainingEligibilityStatus": "hold",
                "notes": "Excluded from export until secondary validation clears.",
            },
            headers=headers,
        )
        assert hold_response.status_code == 200

        blocked_export = client.post(
            "/v1/admin/training-datasets/export",
            json={
                "version": "held-label-dataset-v1",
                "sourceMode": "labels",
                "labelIds": [held_label_id],
            },
            headers=headers,
        )
        assert blocked_export.status_code == 404
        blocked_payload = blocked_export.json()
        assert blocked_payload["error"]["code"] == "training_dataset_label_not_found"

        eligible_export = client.post(
            "/v1/admin/training-datasets/export",
            json={
                "version": "eligible-label-dataset-v1",
                "sourceMode": "labels",
                "labelIds": [eligible_label_id],
            },
            headers=headers,
        )
        assert eligible_export.status_code == 200
        eligible_payload = eligible_export.json()
        assert eligible_payload["labelCount"] == 1
        assert eligible_payload["rows"] == 1
        assert (datasets_path / "eligible-label-dataset-v1.json").exists()


def test_confirmed_field_validation_labels_can_feed_label_dataset_export(
    tmp_path, monkeypatch
):
    datasets_path = tmp_path / "training-datasets"
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()

        import_response = client.post(
            "/v1/admin/labels/import/field-validations",
            json={
                "observations": [
                    {
                        "observationId": "fv-dataset-001",
                        "zoneId": "moc-01",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Alta",
                        "observer": "brigada-terreno-1",
                        "status": "confirmed",
                        "notes": "Confirmed active displacement during field validation.",
                        "evidence": {"fieldSheet": "FS-1001"},
                    },
                    {
                        "observationId": "fv-dataset-002",
                        "zoneId": "moc-02",
                        "observedAt": latest_run["completedAt"],
                        "severity": "Media",
                        "observer": "brigada-terreno-2",
                        "status": "confirmed",
                        "notes": "Moderate signs confirmed after site walk.",
                        "evidence": {"fieldSheet": "FS-1002"},
                    },
                ]
            },
            headers=headers,
        )
        assert import_response.status_code == 200
        label_ids = [item["id"] for item in import_response.json()["labels"]]

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={
                "version": "field-validation-label-dataset-v1",
                "sourceMode": "labels",
                "labelIds": label_ids,
            },
            headers=headers,
        )
        assert export_response.status_code == 200
        export_payload = export_response.json()
        assert export_payload["sourceMode"] == "labels"
        assert export_payload["labelCount"] == 2
        assert export_payload["rows"] == 2
        assert (datasets_path / "field-validation-label-dataset-v1.json").exists()

        detail_response = client.get(
            "/v1/admin/training-datasets/field-validation-label-dataset-v1",
            headers=headers,
        )
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["provenance"]["source"] == "governed_zone_outcome_labels"
        assert detail["summary"]["labels"] == 2
        assert detail["summary"]["matched_predictions"] == 2


def test_label_backed_training_dataset_export_and_retrain(tmp_path, monkeypatch):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()
        observed_at = latest_run["completedAt"]

        upsert_response = client.post(
            "/v1/admin/labels/upsert",
            json={
                "labels": [
                    {
                        "zoneId": "moc-01",
                        "observedAt": observed_at,
                        "targetScore": 0.88,
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                        "notes": "Observed instability after heavy rainfall.",
                        "evidence": {"ticket": "OBS-1001"},
                    },
                    {
                        "zoneId": "moc-02",
                        "observedAt": observed_at,
                        "targetScore": 0.41,
                        "source": "field_validation",
                        "status": "confirmed",
                        "notes": "Moderate signs confirmed by operator review.",
                        "evidence": {"ticket": "OBS-1002"},
                    },
                ]
            },
            headers=headers,
        )
        assert upsert_response.status_code == 200
        label_ids = [item["id"] for item in upsert_response.json()["labels"]]

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={
                "version": "supervised-label-dataset-v1",
                "sourceMode": "labels",
                "labelIds": label_ids,
            },
            headers=headers,
        )
        assert export_response.status_code == 200
        export_payload = export_response.json()
        assert export_payload["sourceMode"] == "labels"
        assert export_payload["labelCount"] == 2
        assert export_payload["rows"] == 2
        assert (datasets_path / "supervised-label-dataset-v1.json").exists()

        detail_response = client.get(
            "/v1/admin/training-datasets/supervised-label-dataset-v1",
            headers=headers,
        )
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["provenance"]["source"] == "governed_zone_outcome_labels"
        assert detail["summary"]["labels"] == 2
        assert detail["summary"]["matched_predictions"] == 2

        retrain_response = client.post(
            "/v1/admin/retrain",
            json={
                "version": "trained-from-supervised-labels-v1",
                "alpha": 0.5,
                "datasetVersion": "supervised-label-dataset-v1",
            },
            headers=headers,
        )
        assert retrain_response.status_code == 200
        payload = retrain_response.json()
        assert payload["datasetVersion"] == "supervised-label-dataset-v1"
        assert payload["rows"] == 2
        assert (artifacts_path / "trained-from-supervised-labels-v1.json").exists()


def test_model_endpoints_list_active_artifact_metadata(tmp_path, monkeypatch):
    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        list_response = client.get("/v1/admin/models", headers=headers)
        assert list_response.status_code == 200
        models = list_response.json()
        assert models
        active_model = next(model for model in models if model["active"] is True)
        assert active_model["version"] == "trained-spatial-seed-v1"
        assert active_model["artifactType"] == "trained_linear_model"

        detail_response = client.get(
            f"/v1/admin/models/{active_model['version']}", headers=headers
        )
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["version"] == "trained-spatial-seed-v1"
        assert detail["artifactType"] == "trained_linear_model"
        assert "calibration" in detail
        assert "training" in detail
        assert detail["training"]["splits"]["validation_rows"] > 0


def test_model_evaluation_endpoints_export_list_and_detail(tmp_path, monkeypatch):
    datasets_path = tmp_path / "training-datasets"
    evaluations_path = tmp_path / "model-evaluations"
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_EVALUATIONS_PATH", str(evaluations_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={"version": "evaluation-dataset-v1"},
            headers=headers,
        )
        assert export_response.status_code == 200

        evaluate_response = client.post(
            "/v1/admin/models/evaluate",
            json={
                "version": "trained-spatial-seed-v1-on-evaluation-dataset-v1",
                "modelVersion": "trained-spatial-seed-v1",
                "datasetVersion": "evaluation-dataset-v1",
                "topErrorCount": 3,
            },
            headers=headers,
        )
        assert evaluate_response.status_code == 200
        payload = evaluate_response.json()
        assert payload["job"]["jobType"] == "model_evaluation"
        assert (
            payload["evaluationVersion"]
            == "trained-spatial-seed-v1-on-evaluation-dataset-v1"
        )
        assert payload["modelVersion"] == "trained-spatial-seed-v1"
        assert payload["datasetVersion"] == "evaluation-dataset-v1"
        assert payload["rows"] == 24
        assert payload["metrics"]["calibrated_metrics"]["rmse"] >= 0.0
        assert payload["validationMetrics"]["rows"] > 0
        assert payload["diagnostics"]["feature_importance"]["top_features"]
        assert (
            payload["diagnostics"]["calibration_effect"]["validation_rmse_improvement"]
            > 0
        )
        assert (
            payload["diagnostics"]["validation_slices"]["by_phase"]["latest"]["rows"]
            > 0
        )
        assert len(payload["topErrors"]) == 3
        assert (
            evaluations_path / "trained-spatial-seed-v1-on-evaluation-dataset-v1.json"
        ).exists()

        list_response = client.get("/v1/admin/model-evaluations", headers=headers)
        assert list_response.status_code == 200
        evaluations = list_response.json()
        assert evaluations
        summary = next(
            item
            for item in evaluations
            if item["version"] == "trained-spatial-seed-v1-on-evaluation-dataset-v1"
        )
        assert summary["artifactType"] == "model_evaluation"
        assert summary["modelVersion"] == "trained-spatial-seed-v1"
        assert summary["datasetVersion"] == "evaluation-dataset-v1"

        detail_response = client.get(
            "/v1/admin/model-evaluations/trained-spatial-seed-v1-on-evaluation-dataset-v1",
            headers=headers,
        )
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["artifactType"] == "model_evaluation"
        assert detail["modelVersion"] == "trained-spatial-seed-v1"
        assert detail["datasetVersion"] == "evaluation-dataset-v1"
        assert detail["metrics"]["overall"]["rows"] == 24
        assert detail["metrics"]["validation"]["rows"] > 0
        assert detail["diagnostics"]["feature_importance"]["top_features"]
        assert detail["diagnostics"]["validation_slices"]["by_target_risk_level"]
        assert len(detail["topErrors"]) == 3


def test_model_tuning_can_promote_best_candidate_and_new_runs_use_it(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    manifest_path = tmp_path / "active-model.json"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))
    monkeypatch.setenv("ACTIVE_MODEL_MANIFEST_PATH", str(manifest_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={"version": "tuning-dataset-v1"},
            headers=headers,
        )
        assert export_response.status_code == 200

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-tuning-dataset-v1",
                "datasetVersion": "tuning-dataset-v1",
                "alphas": [0.25, 0.75, 1.5],
                "versionPrefix": "tuned-candidate",
                "promoteBest": True,
                "promotionReason": "Use the best validation candidate for predictive runs.",
                "topErrorCount": 2,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["job"]["jobType"] == "model_tuning"
        assert payload["selectionVersion"] == "selection-tuning-dataset-v1"
        assert payload["datasetVersion"] == "tuning-dataset-v1"
        assert payload["candidateCount"] == 3
        assert payload["promoted"] is True
        assert payload["activeModelVersion"] == payload["bestModelVersion"]
        assert payload["candidates"][0]["rank"] == 1
        assert (
            payload["candidates"][0]["comparison"][
                "vs_active_validation_rmse_improvement"
            ]
            >= 0.0
        )
        assert payload["bestCandidateComparison"]["coefficient_drift"][
            "largest_changes"
        ]
        assert payload["bestCandidateComparison"]["feature_importance_change"][
            "largest_share_shifts"
        ]
        assert len(payload["candidates"][0]["topErrors"]) == 2
        assert (selection_runs_path / "selection-tuning-dataset-v1.json").exists()
        assert manifest_path.exists()

        selection_list_response = client.get(
            "/v1/admin/model-selection-runs", headers=headers
        )
        assert selection_list_response.status_code == 200
        selection_runs = selection_list_response.json()
        assert selection_runs
        selection_summary = next(
            item
            for item in selection_runs
            if item["version"] == "selection-tuning-dataset-v1"
        )
        assert selection_summary["bestModelVersion"] == payload["bestModelVersion"]
        assert selection_summary["promoted"] is True
        assert selection_summary["activeModelVersion"] == payload["bestModelVersion"]

        selection_detail_response = client.get(
            "/v1/admin/model-selection-runs/selection-tuning-dataset-v1",
            headers=headers,
        )
        assert selection_detail_response.status_code == 200
        selection_detail = selection_detail_response.json()
        assert selection_detail["bestModelVersion"] == payload["bestModelVersion"]
        assert selection_detail["promoted"] is True
        assert selection_detail["activeModelVersion"] == payload["bestModelVersion"]
        assert selection_detail["candidates"][0]["rank"] == 1
        assert selection_detail["bestVsActiveComparison"]["calibration_delta"][
            "challenger_validation_rmse_improvement"
        ] is not None
        assert selection_detail["bestVsActiveComparison"]["validation_slice_deltas"][
            "by_phase"
        ]

        models_response = client.get("/v1/admin/models", headers=headers)
        assert models_response.status_code == 200
        models = models_response.json()
        active_model = next(model for model in models if model["active"] is True)
        assert active_model["version"] == payload["bestModelVersion"]

        trigger_response = client.post(
            "/v1/admin/runs/trigger",
            json={"note": "use promoted best model"},
            headers=headers,
        )
        assert trigger_response.status_code == 200
        run_payload = trigger_response.json()
        assert run_payload["run"]["modelVersion"] == payload["bestModelVersion"]

        zones = client.get("/v1/zones", params={"municipality": "Mocoa"}).json()
        explanation = client.get(f"/v1/zones/{zones[0]['id']}/explanation").json()
        assert explanation["trace"]["model_version"] == payload["bestModelVersion"]


def test_model_tuning_gate_can_block_promotion_for_non_label_dataset(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    manifest_path = tmp_path / "active-model.json"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))
    monkeypatch.setenv("ACTIVE_MODEL_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("MODEL_PROMOTION_REQUIRE_LABELS_DATASET", "true")

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        assert (
            client.post(
                "/v1/admin/training-datasets/export",
                json={"version": "gated-seed-dataset-v1"},
                headers=headers,
            ).status_code
            == 200
        )

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-gated-seed-dataset-v1",
                "datasetVersion": "gated-seed-dataset-v1",
                "alphas": [0.25, 0.75],
                "versionPrefix": "gated-candidate",
                "promoteBest": True,
                "promotionReason": "This should be blocked because the dataset is not label-backed.",
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["promoted"] is False
        assert payload["activeModelVersion"] == "trained-spatial-seed-v1"
        assert payload["promotionDecision"]["eligible"] is False
        assert (
            "labels_dataset_required_for_promotion"
            in payload["promotionDecision"]["blocking_reasons"]
        )
        assert (selection_runs_path / "selection-gated-seed-dataset-v1.json").exists()
        assert not manifest_path.exists()

        selection_detail = client.get(
            "/v1/admin/model-selection-runs/selection-gated-seed-dataset-v1",
            headers=headers,
        ).json()
        assert selection_detail["promoted"] is False
        assert selection_detail["promotionDecision"]["eligible"] is False
        assert (
            "labels_dataset_required_for_promotion"
            in selection_detail["promotionDecision"]["blocking_reasons"]
        )
        assert selection_detail["bestVsActiveComparison"]["coefficient_drift"][
            "largest_changes"
        ]


def test_model_tuning_gate_can_block_promotion_for_minimum_validation_rows(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    manifest_path = tmp_path / "active-model.json"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))
    monkeypatch.setenv("ACTIVE_MODEL_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("MODEL_PROMOTION_MIN_VALIDATION_ROWS", "9")

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        assert (
            client.post(
                "/v1/admin/training-datasets/export",
                json={"version": "row-gated-seed-dataset-v1"},
                headers=headers,
            ).status_code
            == 200
        )

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-row-gated-seed-dataset-v1",
                "datasetVersion": "row-gated-seed-dataset-v1",
                "alphas": [0.25, 0.75, 1.5],
                "versionPrefix": "row-gated-candidate",
                "promoteBest": True,
                "promotionReason": "This should be blocked because validation rows are below the configured minimum.",
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["promoted"] is False
        assert payload["promotionDecision"]["eligible"] is False
        assert (
            "minimum_validation_rows_not_met"
            in payload["promotionDecision"]["blocking_reasons"]
        )
        assert payload["promotionDecision"]["challenger_validation_rows"] == 8
        assert payload["promotionDecision"]["minimum_validation_rows_required"] == 9
        assert not manifest_path.exists()

        selection_detail = client.get(
            "/v1/admin/model-selection-runs/selection-row-gated-seed-dataset-v1",
            headers=headers,
        ).json()
        assert selection_detail["gatePolicy"]["min_validation_rows"] == 9
        assert selection_detail["promotionDecision"]["eligible"] is False


def test_model_tuning_gate_can_block_promotion_for_calibration_regression(
    tmp_path, monkeypatch
):
    from app.ml.model_evaluations import build_model_evaluation as base_evaluation_builder
    from app.services import model_selection as model_selection_module

    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    manifest_path = tmp_path / "active-model.json"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))
    monkeypatch.setenv("ACTIVE_MODEL_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("MODEL_PROMOTION_MIN_CALIBRATION_GAIN", "0.0")

    def fake_build_model_evaluation(*, version, artifact, dataset, top_error_count=10):
        evaluation = base_evaluation_builder(
            version=version,
            artifact=artifact,
            dataset=dataset,
            top_error_count=top_error_count,
        )
        if artifact["version"].startswith("calibration-gated-candidate"):
            evaluation["diagnostics"]["calibration_effect"][
                "validation_rmse_improvement"
            ] = -0.01
        return evaluation

    monkeypatch.setattr(
        model_selection_module, "build_model_evaluation", fake_build_model_evaluation
    )

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        assert (
            client.post(
                "/v1/admin/training-datasets/export",
                json={"version": "calibration-gated-seed-dataset-v1"},
                headers=headers,
            ).status_code
            == 200
        )

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-calibration-gated-seed-dataset-v1",
                "datasetVersion": "calibration-gated-seed-dataset-v1",
                "alphas": [0.25, 0.75, 1.5],
                "versionPrefix": "calibration-gated-candidate",
                "promoteBest": True,
                "promotionReason": "This should be blocked because the challenger calibration gain regressed.",
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["promoted"] is False
        assert payload["promotionDecision"]["eligible"] is False
        assert (
            "minimum_calibration_gain_not_met"
            in payload["promotionDecision"]["blocking_reasons"]
        )
        assert payload["promotionDecision"]["challenger_calibration_gain"] == -0.01
        assert payload["promotionDecision"]["minimum_calibration_gain_required"] == 0.0
        assert not manifest_path.exists()

        selection_detail = client.get(
            "/v1/admin/model-selection-runs/selection-calibration-gated-seed-dataset-v1",
            headers=headers,
        ).json()
        assert selection_detail["gatePolicy"]["min_calibration_gain"] == 0.0
        assert selection_detail["promotionDecision"]["eligible"] is False


def test_model_tuning_gate_can_block_promotion_for_stability_window(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    manifest_path = tmp_path / "active-model.json"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))
    monkeypatch.setenv("ACTIVE_MODEL_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("MODEL_PROMOTION_STABILITY_WINDOW_RUNS", "2")
    monkeypatch.setenv("MODEL_PROMOTION_REQUIRED_CONSISTENT_WINS", "2")

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        assert (
            client.post(
                "/v1/admin/training-datasets/export",
                json={"version": "stability-window-dataset-v1"},
                headers=headers,
            ).status_code
            == 200
        )

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-stability-window-dataset-v1",
                "datasetVersion": "stability-window-dataset-v1",
                "alphas": [0.25, 0.75, 1.5],
                "versionPrefix": "stability-window-candidate",
                "promoteBest": True,
                "promotionReason": "This should be blocked until the challenger wins consistently across the stability window.",
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["promoted"] is False
        assert payload["promotionDecision"]["eligible"] is False
        assert (
            "insufficient_stability_window_runs"
            in payload["promotionDecision"]["blocking_reasons"]
        )
        assert (
            "minimum_consistent_wins_not_met"
            in payload["promotionDecision"]["blocking_reasons"]
        )
        stability = payload["promotionDecision"]["stability_assessment"]
        assert stability["window_runs_considered"] == 1
        assert stability["matching_best_candidate_wins"] == 1
        assert stability["required_consistent_wins"] == 2
        assert not manifest_path.exists()


def test_model_tuning_can_promote_after_consistent_wins_in_stability_window(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    manifest_path = tmp_path / "active-model.json"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))
    monkeypatch.setenv("ACTIVE_MODEL_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("MODEL_PROMOTION_STABILITY_WINDOW_RUNS", "2")
    monkeypatch.setenv("MODEL_PROMOTION_REQUIRED_CONSISTENT_WINS", "2")

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        assert (
            client.post(
                "/v1/admin/training-datasets/export",
                json={"version": "stability-pass-dataset-v1"},
                headers=headers,
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/v1/admin/training-datasets/export",
                json={"version": "stability-pass-dataset-v2"},
                headers=headers,
            ).status_code
            == 200
        )

        first_tune = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-stability-pass-dataset-v1",
                "datasetVersion": "stability-pass-dataset-v1",
                "alphas": [0.25, 0.75, 1.5],
                "versionPrefix": "stability-pass-candidate",
                "promoteBest": False,
                "promotionReason": "Establish the first consistent challenger win.",
            },
            headers=headers,
        )
        assert first_tune.status_code == 200
        first_payload = first_tune.json()
        first_best_model = first_payload["bestModelVersion"]

        second_tune = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-stability-pass-dataset-v2",
                "datasetVersion": "stability-pass-dataset-v2",
                "alphas": [0.25, 0.75, 1.5],
                "versionPrefix": "stability-pass-candidate",
                "promoteBest": True,
                "promotionReason": "Promote after repeated consistent challenger wins.",
            },
            headers=headers,
        )
        assert second_tune.status_code == 200
        second_payload = second_tune.json()
        assert second_payload["promoted"] is True
        assert second_payload["bestModelVersion"] == first_best_model
        stability = second_payload["promotionDecision"]["stability_assessment"]
        assert stability["window_runs_considered"] == 2
        assert stability["matching_best_candidate_wins"] == 2
        assert stability["consistent_enough"] is True
        assert second_payload["promotionDecision"]["blocking_reasons"] == []
        assert manifest_path.exists()


def test_model_tuning_stability_window_ignores_mismatched_dataset_family(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    manifest_path = tmp_path / "active-model.json"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))
    monkeypatch.setenv("ACTIVE_MODEL_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("MODEL_PROMOTION_STABILITY_WINDOW_RUNS", "2")
    monkeypatch.setenv("MODEL_PROMOTION_REQUIRED_CONSISTENT_WINS", "2")
    monkeypatch.setenv("MODEL_PROMOTION_STABILITY_REQUIRE_SAME_DATASET_FAMILY", "true")

    write_historical_selection_run(
        selection_runs_path,
        version="historical-family-mismatch-v1",
        dataset_version="historical-family-dataset-v1",
        created_at="2026-03-25T12:00:00+00:00",
        dataset_mode="seed",
        dataset_family="seed:other_family",
        time_window={
            "kind": "bootstrap_reference",
            "reference_at": "2026-03-25T00:00:00+00:00",
            "start_at": "2026-03-25T00:00:00+00:00",
            "end_at": "2026-03-25T00:00:00+00:00",
            "span_days": 0,
        },
    )

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        assert (
            client.post(
                "/v1/admin/training-datasets/export",
                json={"version": "family-aware-dataset-v1"},
                headers=headers,
            ).status_code
            == 200
        )

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-family-aware-dataset-v1",
                "datasetVersion": "family-aware-dataset-v1",
                "alphas": [0.25, 0.75, 1.5],
                "versionPrefix": "family-aware-candidate",
                "promoteBest": True,
                "promotionReason": "This should ignore mismatched-family wins from prior selection runs.",
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["promoted"] is False
        assert (
            "stability_dataset_family_requirement_not_met"
            in payload["promotionDecision"]["blocking_reasons"]
        )
        stability = payload["promotionDecision"]["stability_assessment"]
        assert stability["dataset_family"] == "seed:frontend_seed_bootstrap"
        assert stability["excluded_reason_counts"]["dataset_family_mismatch"] == 1
        assert stability["window_runs_considered"] == 1
        assert stability["matching_best_candidate_wins"] == 1

        selection_detail = client.get(
            "/v1/admin/model-selection-runs/selection-family-aware-dataset-v1",
            headers=headers,
        ).json()
        assert selection_detail["datasetContext"]["dataset_family"] == "seed:frontend_seed_bootstrap"
        assert (
            selection_detail["promotionDecision"]["stability_assessment"][
                "excluded_runs"
            ][0]["excluded_reason"]
            == "dataset_family_mismatch"
        )
        assert not manifest_path.exists()


def test_model_tuning_stability_window_ignores_excessive_time_window_gap(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    manifest_path = tmp_path / "active-model.json"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))
    monkeypatch.setenv("ACTIVE_MODEL_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("MODEL_PROMOTION_STABILITY_WINDOW_RUNS", "2")
    monkeypatch.setenv("MODEL_PROMOTION_REQUIRED_CONSISTENT_WINS", "2")
    monkeypatch.setenv("MODEL_PROMOTION_STABILITY_MAX_TIME_WINDOW_GAP_DAYS", "30")

    write_historical_selection_run(
        selection_runs_path,
        version="historical-time-gap-v1",
        dataset_version="historical-time-gap-dataset-v1",
        created_at="2026-03-25T12:00:00+00:00",
        dataset_mode="seed",
        dataset_family="seed:frontend_seed_bootstrap",
        time_window={
            "kind": "bootstrap_reference",
            "reference_at": "2025-01-01T00:00:00+00:00",
            "start_at": "2025-01-01T00:00:00+00:00",
            "end_at": "2025-01-01T00:00:00+00:00",
            "span_days": 0,
        },
    )

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        assert (
            client.post(
                "/v1/admin/training-datasets/export",
                json={"version": "time-aware-dataset-v1"},
                headers=headers,
            ).status_code
            == 200
        )

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-time-aware-dataset-v1",
                "datasetVersion": "time-aware-dataset-v1",
                "alphas": [0.25, 0.75, 1.5],
                "versionPrefix": "time-aware-candidate",
                "promoteBest": True,
                "promotionReason": "This should ignore stale time-window wins from prior selection runs.",
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["promoted"] is False
        assert (
            "stability_time_window_requirement_not_met"
            in payload["promotionDecision"]["blocking_reasons"]
        )
        stability = payload["promotionDecision"]["stability_assessment"]
        assert stability["excluded_reason_counts"]["time_window_gap_exceeded"] == 1
        assert stability["window_runs_considered"] == 1
        assert stability["matching_best_candidate_wins"] == 1
        assert stability["excluded_runs"][0]["time_window_gap_days"] > 30
        assert not manifest_path.exists()


def test_model_tuning_stability_window_can_require_dataset_taxonomy_match(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    manifest_path = tmp_path / "active-model.json"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))
    monkeypatch.setenv("ACTIVE_MODEL_MANIFEST_PATH", str(manifest_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        assert (
            client.post(
                "/v1/admin/training-datasets/export",
                json={"version": "taxonomy-aware-dataset-v1"},
                headers=headers,
            ).status_code
            == 200
        )
        dataset_detail = client.get(
            "/v1/admin/training-datasets/taxonomy-aware-dataset-v1",
            headers=headers,
        ).json()
        assert (
            dataset_detail["provenance"]["dataset_taxonomy"]["taxonomy_group"]
            == "seed:bootstrap:zone:synthetic"
        )
        assert (
            dataset_detail["provenance"]["evaluation_cohort"]["bucket_type"] == "static"
        )

        best_alpha = probe_best_alpha(
            client,
            headers,
            dataset_version="taxonomy-aware-dataset-v1",
            selection_runs_path=selection_runs_path,
            version="selection-taxonomy-aware-probe",
            version_prefix="taxonomy-aware-probe",
        )

        write_historical_selection_run(
            selection_runs_path,
            version="historical-taxonomy-mismatch-v1",
            dataset_version="historical-taxonomy-dataset-v1",
            created_at="2026-03-25T12:00:00+00:00",
            dataset_mode="seed",
            dataset_family="seed:frontend_seed_bootstrap",
            time_window={
                "kind": "bootstrap_reference",
                "reference_at": "2026-03-25T00:00:00+00:00",
                "start_at": "2026-03-25T00:00:00+00:00",
                "end_at": "2026-03-25T00:00:00+00:00",
                "span_days": 0,
            },
            alpha=best_alpha,
            dataset_taxonomy={
                "family_root": "seed",
                "family_variant": "frontend_seed_bootstrap",
                "source": "frontend_seed_bootstrap",
                "source_family": "frontend_seed_bootstrap",
                "source_families": ["frontend_seed_bootstrap"],
                "supervision_tier": "proxy",
                "signal_type": "predicted",
                "geographic_granularity": "zone",
                "taxonomy_group": "seed:proxy:zone:predicted",
                "stability_group": "seed:proxy:zone:predicted",
            },
            evaluation_cohort={
                "cohort_key": "seed:proxy:zone:predicted:static:2026-03-25",
                "cohort_group": "seed:proxy:zone:predicted:static",
                "bucket_type": "static",
                "bucket_label": "2026-03-25",
                "bucket_index": 739335,
                "bucket_start_at": "2026-03-25T00:00:00+00:00",
                "bucket_end_at": "2026-03-25T23:59:59+00:00",
                "reference_at": "2026-03-25T00:00:00+00:00",
            },
        )

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-taxonomy-aware-dataset-v1",
                "datasetVersion": "taxonomy-aware-dataset-v1",
                "alphas": [0.25, 0.75, 1.5],
                "versionPrefix": "taxonomy-aware-candidate",
                "promoteBest": True,
                "promotionReason": "Require taxonomy-aware consistency across the stability window.",
                "stabilityWindowRuns": 2,
                "requiredConsistentWins": 2,
                "stabilityRequireSameDatasetFamily": True,
                "stabilityRequireSameDatasetTaxonomy": True,
                "stabilityMaxTimeWindowGapDays": 365,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["promoted"] is False
        assert (
            "stability_dataset_taxonomy_requirement_not_met"
            in payload["promotionDecision"]["blocking_reasons"]
        )
        stability = payload["promotionDecision"]["stability_assessment"]
        assert stability["require_same_dataset_taxonomy"] is True
        assert stability["taxonomy_group"] == "seed:bootstrap:zone:synthetic"
        assert stability["excluded_reason_counts"]["dataset_taxonomy_mismatch"] == 1

        selection_detail = client.get(
            "/v1/admin/model-selection-runs/selection-taxonomy-aware-dataset-v1",
            headers=headers,
        ).json()
        assert (
            selection_detail["datasetContext"]["dataset_taxonomy"]["taxonomy_group"]
            == "seed:bootstrap:zone:synthetic"
        )
        assert (
            selection_detail["promotionDecision"]["stability_assessment"][
                "excluded_runs"
            ][0]["excluded_reason"]
            == "dataset_taxonomy_mismatch"
        )
        assert not manifest_path.exists()


def test_model_tuning_stability_window_can_require_evaluation_cohort_distance(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    manifest_path = tmp_path / "active-model.json"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))
    monkeypatch.setenv("ACTIVE_MODEL_MANIFEST_PATH", str(manifest_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()

        upsert_response = client.post(
            "/v1/admin/labels/upsert",
            json={
                "labels": [
                    {
                        "zoneId": "moc-01",
                        "observedAt": "2026-03-20T00:00:00Z",
                        "targetScore": 0.89,
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                    },
                    {
                        "zoneId": "moc-02",
                        "observedAt": "2026-03-21T00:00:00Z",
                        "targetScore": 0.37,
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                    },
                ]
            },
            headers=headers,
        )
        assert upsert_response.status_code == 200
        label_ids = [item["id"] for item in upsert_response.json()["labels"]]

        assert (
            client.post(
                "/v1/admin/training-datasets/export",
                json={
                    "version": "cohort-aware-label-dataset-v1",
                    "sourceMode": "labels",
                    "labelIds": label_ids,
                },
                headers=headers,
            ).status_code
            == 200
        )
        dataset_detail = client.get(
            "/v1/admin/training-datasets/cohort-aware-label-dataset-v1",
            headers=headers,
        ).json()
        assert (
            dataset_detail["provenance"]["dataset_taxonomy"]["taxonomy_group"]
            == "labels:observed:zone:direct"
        )
        assert (
            dataset_detail["provenance"]["evaluation_cohort"]["bucket_type"] == "month"
        )
        assert (
            dataset_detail["provenance"]["evaluation_cohort"]["bucket_label"]
            == "2026-03"
        )

        best_alpha = probe_best_alpha(
            client,
            headers,
            dataset_version="cohort-aware-label-dataset-v1",
            selection_runs_path=selection_runs_path,
            version="selection-cohort-aware-probe",
            version_prefix="cohort-aware-probe",
        )

        write_historical_selection_run(
            selection_runs_path,
            version="historical-cohort-distance-v1",
            dataset_version="historical-cohort-distance-dataset-v1",
            created_at="2026-02-01T12:00:00+00:00",
            dataset_mode="labels",
            dataset_family="labels:field_validation",
            time_window={
                "kind": "observed_outcomes",
                "reference_at": "2026-01-25T00:00:00+00:00",
                "start_at": "2026-01-10T00:00:00+00:00",
                "end_at": "2026-01-25T00:00:00+00:00",
                "span_days": 15,
            },
            alpha=best_alpha,
            dataset_taxonomy={
                "family_root": "labels",
                "family_variant": "field_validation",
                "source": "governed_zone_outcome_labels",
                "source_family": "governed_zone_outcome_labels",
                "source_families": ["field_validation"],
                "supervision_tier": "observed",
                "signal_type": "direct",
                "geographic_granularity": "zone",
                "taxonomy_group": "labels:observed:zone:direct",
                "stability_group": "labels:observed:zone:direct",
            },
            evaluation_cohort={
                "cohort_key": "labels:observed:zone:direct:month:2026-01",
                "cohort_group": "labels:observed:zone:direct:month",
                "bucket_type": "month",
                "bucket_label": "2026-01",
                "bucket_index": 24313,
                "bucket_start_at": "2026-01-01T00:00:00+00:00",
                "bucket_end_at": "2026-01-31T23:59:59+00:00",
                "reference_at": "2026-01-25T00:00:00+00:00",
            },
        )

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-cohort-aware-label-dataset-v1",
                "datasetVersion": "cohort-aware-label-dataset-v1",
                "alphas": [0.25, 0.75, 1.5],
                "versionPrefix": "cohort-aware-candidate",
                "promoteBest": True,
                "promotionReason": "Require nearby monthly evaluation cohorts for repeated wins.",
                "stabilityWindowRuns": 2,
                "requiredConsistentWins": 2,
                "stabilityRequireSameDatasetFamily": True,
                "stabilityRequireSameDatasetTaxonomy": True,
                "stabilityRequireSameEvaluationCohort": True,
                "stabilityMaxTimeWindowGapDays": 365,
                "stabilityMaxCohortDistance": 1,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["promoted"] is False
        assert (
            "stability_evaluation_cohort_requirement_not_met"
            in payload["promotionDecision"]["blocking_reasons"]
        )
        stability = payload["promotionDecision"]["stability_assessment"]
        assert stability["require_same_evaluation_cohort"] is True
        assert stability["evaluation_cohort"]["bucket_label"] == "2026-03"
        assert (
            stability["excluded_reason_counts"]["evaluation_cohort_distance_exceeded"]
            == 1
        )
        assert (
            stability["excluded_runs"][0]["evaluation_cohort_distance"] == 2
        )
        assert not manifest_path.exists()


def test_model_tuning_stability_window_can_accept_adjacent_evaluation_cohort(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    manifest_path = tmp_path / "active-model.json"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))
    monkeypatch.setenv("ACTIVE_MODEL_MANIFEST_PATH", str(manifest_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()

        upsert_response = client.post(
            "/v1/admin/labels/upsert",
            json={
                "labels": [
                    {
                        "zoneId": "moc-01",
                        "observedAt": "2026-03-20T00:00:00Z",
                        "targetScore": 0.89,
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                    },
                    {
                        "zoneId": "moc-02",
                        "observedAt": "2026-03-21T00:00:00Z",
                        "targetScore": 0.37,
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                    },
                ]
            },
            headers=headers,
        )
        assert upsert_response.status_code == 200
        label_ids = [item["id"] for item in upsert_response.json()["labels"]]

        assert (
            client.post(
                "/v1/admin/training-datasets/export",
                json={
                    "version": "cohort-adjacent-label-dataset-v1",
                    "sourceMode": "labels",
                    "labelIds": label_ids,
                },
                headers=headers,
            ).status_code
            == 200
        )

        best_alpha = probe_best_alpha(
            client,
            headers,
            dataset_version="cohort-adjacent-label-dataset-v1",
            selection_runs_path=selection_runs_path,
            version="selection-cohort-adjacent-probe",
            version_prefix="cohort-adjacent-probe",
        )

        write_historical_selection_run(
            selection_runs_path,
            version="historical-cohort-adjacent-v1",
            dataset_version="historical-cohort-adjacent-dataset-v1",
            created_at="2026-03-01T12:00:00+00:00",
            dataset_mode="labels",
            dataset_family="labels:field_validation",
            time_window={
                "kind": "observed_outcomes",
                "reference_at": "2026-02-25T00:00:00+00:00",
                "start_at": "2026-02-10T00:00:00+00:00",
                "end_at": "2026-02-25T00:00:00+00:00",
                "span_days": 15,
            },
            alpha=best_alpha,
            dataset_taxonomy={
                "family_root": "labels",
                "family_variant": "field_validation",
                "source": "governed_zone_outcome_labels",
                "source_family": "governed_zone_outcome_labels",
                "source_families": ["field_validation"],
                "supervision_tier": "observed",
                "signal_type": "direct",
                "geographic_granularity": "zone",
                "taxonomy_group": "labels:observed:zone:direct",
                "stability_group": "labels:observed:zone:direct",
            },
            evaluation_cohort={
                "cohort_key": "labels:observed:zone:direct:month:2026-02",
                "cohort_group": "labels:observed:zone:direct:month",
                "bucket_type": "month",
                "bucket_label": "2026-02",
                "bucket_index": 24314,
                "bucket_start_at": "2026-02-01T00:00:00+00:00",
                "bucket_end_at": "2026-02-28T23:59:59+00:00",
                "reference_at": "2026-02-25T00:00:00+00:00",
            },
        )

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-cohort-adjacent-label-dataset-v1",
                "datasetVersion": "cohort-adjacent-label-dataset-v1",
                "alphas": [0.25, 0.75, 1.5],
                "versionPrefix": "cohort-adjacent-candidate",
                "promoteBest": True,
                "promotionReason": "Allow repeated wins from adjacent monthly cohorts.",
                "minValidationRmseImprovement": -1.0,
                "stabilityWindowRuns": 2,
                "requiredConsistentWins": 2,
                "stabilityRequireSameDatasetFamily": True,
                "stabilityRequireSameDatasetTaxonomy": True,
                "stabilityRequireSameEvaluationCohort": True,
                "stabilityMaxTimeWindowGapDays": 365,
                "stabilityMaxCohortDistance": 1,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["promoted"] is True
        stability = payload["promotionDecision"]["stability_assessment"]
        assert stability["window_runs_considered"] == 2
        assert stability["matching_best_candidate_wins"] == 2
        assert stability["consistent_enough"] is True
        assert payload["promotionDecision"]["blocking_reasons"] == []
        assert manifest_path.exists()


def test_model_rollback_restores_previous_active_model(tmp_path, monkeypatch):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    manifest_path = tmp_path / "active-model.json"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))
    monkeypatch.setenv("ACTIVE_MODEL_MANIFEST_PATH", str(manifest_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        assert (
            client.post(
                "/v1/admin/training-datasets/export",
                json={"version": "rollback-tuning-dataset-v1"},
                headers=headers,
            ).status_code
            == 200
        )

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-rollback-dataset-v1",
                "datasetVersion": "rollback-tuning-dataset-v1",
                "alphas": [0.25, 0.75, 1.5],
                "versionPrefix": "rollback-candidate",
                "promoteBest": True,
                "promotionReason": "Promote a challenger before rollback validation.",
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        promoted_model_version = tune_response.json()["bestModelVersion"]
        assert promoted_model_version != "trained-spatial-seed-v1"

        rollback_response = client.post(
            "/v1/admin/models/rollback",
            json={"reason": "Restore the prior champion model."},
            headers=headers,
        )
        assert rollback_response.status_code == 200
        rollback_payload = rollback_response.json()
        assert rollback_payload["activeModelVersion"] == "trained-spatial-seed-v1"
        assert rollback_payload["rolledBackFromModelVersion"] == promoted_model_version
        assert rollback_payload["previousActiveModelVersion"] == promoted_model_version
        assert manifest_path.exists()

        models = client.get("/v1/admin/models", headers=headers).json()
        active_model = next(model for model in models if model["active"] is True)
        assert active_model["version"] == "trained-spatial-seed-v1"

        trigger_response = client.post(
            "/v1/admin/runs/trigger",
            json={"note": "validate rollback restored previous champion"},
            headers=headers,
        )
        assert trigger_response.status_code == 200
        assert (
            trigger_response.json()["run"]["modelVersion"] == "trained-spatial-seed-v1"
        )

        zones = client.get("/v1/zones", params={"municipality": "Mocoa"}).json()
        explanation = client.get(f"/v1/zones/{zones[0]['id']}/explanation").json()
        assert explanation["trace"]["model_version"] == "trained-spatial-seed-v1"


def test_model_monitoring_endpoints_expose_promotion_history_and_rollups(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    manifest_path = tmp_path / "active-model.json"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))
    monkeypatch.setenv("ACTIVE_MODEL_MANIFEST_PATH", str(manifest_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        assert (
            client.post(
                "/v1/admin/training-datasets/export",
                json={"version": "monitoring-dataset-v1"},
                headers=headers,
            ).status_code
            == 200
        )
        first_tune = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-monitoring-dataset-v1",
                "datasetVersion": "monitoring-dataset-v1",
                "alphas": [0.25, 0.75, 1.5],
                "versionPrefix": "monitoring-candidate",
                "promoteBest": True,
                "promotionReason": "Promote a candidate for monitoring history tests.",
            },
            headers=headers,
        )
        assert first_tune.status_code == 200
        promoted_model_version = first_tune.json()["bestModelVersion"]

        assert (
            client.post(
                "/v1/admin/training-datasets/export",
                json={"version": "monitoring-dataset-v2"},
                headers=headers,
            ).status_code
            == 200
        )
        second_tune = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-monitoring-dataset-v2",
                "datasetVersion": "monitoring-dataset-v2",
                "alphas": [0.25, 0.75, 1.5],
                "versionPrefix": "monitoring-candidate",
                "promoteBest": False,
                "promotionReason": "Add a second best-candidate appearance for monitoring rollups.",
            },
            headers=headers,
        )
        assert second_tune.status_code == 200

        promotion_history_response = client.get(
            "/v1/admin/models/promotion-history",
            headers=headers,
        )
        assert promotion_history_response.status_code == 200
        promotion_history = promotion_history_response.json()
        assert promotion_history
        filtered_history_response = client.get(
            "/v1/admin/models/promotion-history",
            params={"modelVersion": promoted_model_version},
            headers=headers,
        )
        assert filtered_history_response.status_code == 200
        filtered_history = filtered_history_response.json()
        assert len(filtered_history) == 1
        assert filtered_history[0]["modelVersion"] == promoted_model_version
        assert filtered_history[0]["currentActive"] is True

        monitoring_list_response = client.get(
            "/v1/admin/models/monitoring",
            headers=headers,
        )
        assert monitoring_list_response.status_code == 200
        monitoring_list = monitoring_list_response.json()
        summary = next(
            item for item in monitoring_list if item["version"] == promoted_model_version
        )
        assert summary["active"] is True
        assert summary["selectionRunCount"] >= 2
        assert summary["bestCandidateCount"] >= 2
        assert summary["promotionCount"] == 1
        assert "seed:frontend_seed_bootstrap" in summary["datasetFamiliesSeen"]

        monitoring_detail_response = client.get(
            f"/v1/admin/models/monitoring/{promoted_model_version}",
            headers=headers,
        )
        assert monitoring_detail_response.status_code == 200
        detail = monitoring_detail_response.json()
        assert detail["version"] == promoted_model_version
        assert detail["active"] is True
        assert len(detail["promotionHistory"]) == 1
        assert len(detail["selectionHistory"]) >= 2
        assert detail["familyRollups"]
        assert detail["familyRollups"][0]["datasetFamily"] == "seed:frontend_seed_bootstrap"


def test_model_drift_scan_and_monitoring_use_fresh_label_datasets(
    tmp_path, monkeypatch
):
    drift_reports_path = tmp_path / "drift-reports"
    datasets_path = tmp_path / "training-datasets"
    evaluations_path = tmp_path / "evaluations"
    monkeypatch.setenv("MODEL_DRIFT_REPORTS_PATH", str(drift_reports_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_EVALUATIONS_PATH", str(evaluations_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        baseline_scores = latest_run_zone_scores(client, zone_ids=["moc-01", "moc-02"])
        latest_run = client.get("/v1/runs/latest").json()

        baseline_upsert = client.post(
            "/v1/admin/labels/upsert",
            json={
                "labels": [
                    {
                        "zoneId": "moc-01",
                        "observedAt": "2026-01-20T00:00:00Z",
                        "targetScore": baseline_scores["moc-01"],
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                    },
                    {
                        "zoneId": "moc-02",
                        "observedAt": "2026-01-21T00:00:00Z",
                        "targetScore": baseline_scores["moc-02"],
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                    },
                ]
            },
            headers=headers,
        )
        assert baseline_upsert.status_code == 200
        baseline_label_ids = [item["id"] for item in baseline_upsert.json()["labels"]]

        baseline_export = client.post(
            "/v1/admin/training-datasets/export",
            json={
                "version": "drift-baseline-labels-v1",
                "sourceMode": "labels",
                "labelIds": baseline_label_ids,
            },
            headers=headers,
        )
        assert baseline_export.status_code == 200

        baseline_evaluation = client.post(
            "/v1/admin/models/evaluate",
            json={
                "version": "eval-drift-baseline-v1",
                "modelVersion": "trained-spatial-seed-v1",
                "datasetVersion": "drift-baseline-labels-v1",
                "topErrorCount": 5,
            },
            headers=headers,
        )
        assert baseline_evaluation.status_code == 200

        drift_upsert = client.post(
            "/v1/admin/labels/upsert",
            json={
                "labels": [
                    {
                        "zoneId": "moc-01",
                        "observedAt": "2026-03-20T00:00:00Z",
                        "targetScore": 0.05,
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                    },
                    {
                        "zoneId": "moc-02",
                        "observedAt": "2026-03-21T00:00:00Z",
                        "targetScore": 0.95,
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                    },
                ]
            },
            headers=headers,
        )
        assert drift_upsert.status_code == 200
        drift_label_ids = [item["id"] for item in drift_upsert.json()["labels"]]

        drift_export = client.post(
            "/v1/admin/training-datasets/export",
            json={
                "version": "drift-fresh-labels-v1",
                "sourceMode": "labels",
                "labelIds": drift_label_ids,
            },
            headers=headers,
        )
        assert drift_export.status_code == 200

        drift_scan = client.post(
            "/v1/admin/models/drift-scan",
            json={
                "version": "drift-report-fresh-labels-v1",
                "evaluationVersion": "eval-drift-fresh-labels-v1",
                "modelVersion": "trained-spatial-seed-v1",
                "datasetVersion": "drift-fresh-labels-v1",
                "topErrorCount": 5,
                "warningValidationRmseIncrease": 0.05,
                "criticalValidationRmseIncrease": 0.1,
                "warningAccuracyDrop": 0.05,
                "criticalAccuracyDrop": 0.1,
            },
            headers=headers,
        )
        assert drift_scan.status_code == 200
        drift_payload = drift_scan.json()
        assert drift_payload["modelVersion"] == "trained-spatial-seed-v1"
        assert drift_payload["datasetVersion"] == "drift-fresh-labels-v1"
        assert drift_payload["baseline"]["source"] == "model_evaluation"
        assert drift_payload["baseline"]["reference_version"] == "eval-drift-baseline-v1"
        assert drift_payload["driftDetected"] is True
        assert drift_payload["severity"] == "critical"
        assert drift_payload["driftSummary"]["validation_rmse_delta"] > 0
        assert (
            drift_payload["driftSummary"]["validation_risk_level_accuracy_delta"] < 0
        )
        assert (drift_reports_path / "drift-report-fresh-labels-v1.json").exists()
        assert (evaluations_path / "eval-drift-fresh-labels-v1.json").exists()

        drift_reports = client.get("/v1/admin/model-drift-reports", headers=headers)
        assert drift_reports.status_code == 200
        drift_summaries = drift_reports.json()
        summary = next(
            item
            for item in drift_summaries
            if item["version"] == "drift-report-fresh-labels-v1"
        )
        assert summary["severity"] == "critical"
        assert summary["datasetFamily"] == "labels:field_validation"
        assert summary["taxonomyGroup"] == "labels:observed:zone:direct"
        assert summary["evaluationCohortLabel"] == "2026-03"

        drift_detail = client.get(
            "/v1/admin/model-drift-reports/drift-report-fresh-labels-v1",
            headers=headers,
        )
        assert drift_detail.status_code == 200
        detail = drift_detail.json()
        assert detail["driftSummary"]["signals"]
        assert detail["diagnostics"]["dataset_summary"]["provenance"]["dataset_family"] == "labels:field_validation"

        monitoring_detail = client.get(
            "/v1/admin/models/monitoring/trained-spatial-seed-v1",
            headers=headers,
        )
        assert monitoring_detail.status_code == 200
        monitoring_payload = monitoring_detail.json()
        assert monitoring_payload["latestDriftStatus"] == "critical"
        assert monitoring_payload["latestDriftDatasetVersion"] == "drift-fresh-labels-v1"
        assert monitoring_payload["driftHistory"]
        assert monitoring_payload["driftHistory"][0]["version"] == "drift-report-fresh-labels-v1"


def test_model_shadow_scan_and_monitoring_surface_recent_challengers(
    tmp_path, monkeypatch
):
    shadow_runs_path = tmp_path / "shadow-runs"
    datasets_path = tmp_path / "training-datasets"
    artifacts_path = tmp_path / "artifacts"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_SHADOW_RUNS_PATH", str(shadow_runs_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()

        upsert_response = client.post(
            "/v1/admin/labels/upsert",
            json={
                "labels": [
                    {
                        "zoneId": "moc-01",
                        "observedAt": "2026-03-20T00:00:00Z",
                        "targetScore": 0.05,
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                    },
                    {
                        "zoneId": "moc-02",
                        "observedAt": "2026-03-21T00:00:00Z",
                        "targetScore": 0.95,
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                    },
                ]
            },
            headers=headers,
        )
        assert upsert_response.status_code == 200
        label_ids = [item["id"] for item in upsert_response.json()["labels"]]

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={
                "version": "shadow-fresh-labels-v1",
                "sourceMode": "labels",
                "labelIds": label_ids,
            },
            headers=headers,
        )
        assert export_response.status_code == 200

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-shadow-fresh-labels-v1",
                "datasetVersion": "shadow-fresh-labels-v1",
                "alphas": [0.25, 0.75, 1.5],
                "versionPrefix": "shadow-candidate",
                "promoteBest": False,
                "topErrorCount": 5,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        tune_payload = tune_response.json()
        challenger_versions = [
            candidate["modelVersion"]
            for candidate in tune_payload["candidates"]
            if candidate["modelVersion"].startswith("shadow-candidate-alpha-")
        ]
        assert challenger_versions

        active_eval_response = client.post(
            "/v1/admin/models/evaluate",
            json={
                "version": "shadow-active-on-fresh-labels-v1",
                "modelVersion": "trained-spatial-seed-v1",
                "datasetVersion": "shadow-fresh-labels-v1",
                "topErrorCount": 10,
            },
            headers=headers,
        )
        assert active_eval_response.status_code == 200
        active_predictions = {
            row["zoneId"]: row["predictedScore"]
            for row in active_eval_response.json()["topErrors"]
        }

        selected_candidate_version = ""
        selected_candidate_predictions: dict[str, float] = {}
        max_prediction_distance = -1.0
        for candidate_version in challenger_versions:
            candidate_eval_response = client.post(
                "/v1/admin/models/evaluate",
                json={
                    "version": f"{candidate_version}-on-shadow-fresh-labels-v1",
                    "modelVersion": candidate_version,
                    "datasetVersion": "shadow-fresh-labels-v1",
                    "topErrorCount": 10,
                },
                headers=headers,
            )
            assert candidate_eval_response.status_code == 200
            candidate_predictions = {
                row["zoneId"]: row["predictedScore"]
                for row in candidate_eval_response.json()["topErrors"]
            }
            prediction_distance = round(
                sum(
                    abs(candidate_predictions[zone_id] - active_predictions[zone_id])
                    for zone_id in active_predictions
                ),
                6,
            )
            if prediction_distance > max_prediction_distance:
                max_prediction_distance = prediction_distance
                selected_candidate_version = candidate_version
                selected_candidate_predictions = candidate_predictions

        assert selected_candidate_version.startswith("shadow-candidate-alpha-")
        assert max_prediction_distance > 0

        challenger_label_response = client.post(
            "/v1/admin/labels/upsert",
            json={
                "labels": [
                    {
                        "zoneId": "moc-01",
                        "observedAt": "2026-03-22T00:00:00Z",
                        "targetScore": selected_candidate_predictions["moc-01"],
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                    },
                    {
                        "zoneId": "moc-02",
                        "observedAt": "2026-03-23T00:00:00Z",
                        "targetScore": selected_candidate_predictions["moc-02"],
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                    },
                ]
            },
            headers=headers,
        )
        assert challenger_label_response.status_code == 200
        challenger_label_ids = [
            item["id"] for item in challenger_label_response.json()["labels"]
        ]

        challenger_export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={
                "version": "shadow-review-labels-v1",
                "sourceMode": "labels",
                "labelIds": challenger_label_ids,
            },
            headers=headers,
        )
        assert challenger_export_response.status_code == 200

        shadow_scan = client.post(
            "/v1/admin/models/shadow-scan",
            json={
                "version": "shadow-run-review-labels-v1",
                "datasetVersion": "shadow-review-labels-v1",
                "modelVersions": [selected_candidate_version],
                "topErrorCount": 5,
            },
            headers=headers,
        )
        assert shadow_scan.status_code == 200
        shadow_payload = shadow_scan.json()
        assert shadow_payload["shadowVersion"] == "shadow-run-review-labels-v1"
        assert shadow_payload["datasetVersion"] == "shadow-review-labels-v1"
        assert shadow_payload["candidateCount"] >= 2
        assert shadow_payload["bestModelVersion"] == selected_candidate_version
        assert shadow_payload["activeStillBest"] is False
        assert shadow_payload["recommendation"]["status"] == "review_challenger"
        assert (shadow_runs_path / "shadow-run-review-labels-v1.json").exists()

        shadow_runs = client.get("/v1/admin/model-shadow-runs", headers=headers)
        assert shadow_runs.status_code == 200
        shadow_summary = next(
            item
            for item in shadow_runs.json()
            if item["version"] == "shadow-run-review-labels-v1"
        )
        assert shadow_summary["bestModelVersion"] == selected_candidate_version
        assert shadow_summary["activeStillBest"] is False

        shadow_detail = client.get(
            "/v1/admin/model-shadow-runs/shadow-run-review-labels-v1",
            headers=headers,
        )
        assert shadow_detail.status_code == 200
        detail_payload = shadow_detail.json()
        assert detail_payload["datasetContext"]["dataset_mode"] == "labels"
        assert (
            detail_payload["datasetContext"]["dataset_taxonomy"]["taxonomy_group"]
            == "labels:observed:zone:direct"
        )
        assert detail_payload["candidateSelection"]["mode"] == "explicit"
        assert any(
            candidate["modelVersion"] == selected_candidate_version
            for candidate in detail_payload["candidates"]
        )

        monitoring_detail = client.get(
            "/v1/admin/models/monitoring/trained-spatial-seed-v1",
            headers=headers,
        )
        assert monitoring_detail.status_code == 200
        monitoring_payload = monitoring_detail.json()
        assert monitoring_payload["latestShadowStatus"] == "review_challenger"
        assert monitoring_payload["latestShadowDatasetVersion"] == "shadow-review-labels-v1"
        assert (
            monitoring_payload["latestShadowBestModelVersion"]
            == selected_candidate_version
        )
        assert monitoring_payload["latestShadowActiveStillBest"] is False
        assert monitoring_payload["shadowHistory"]
        assert monitoring_payload["shadowHistory"][0]["version"] == "shadow-run-review-labels-v1"


def test_model_monitoring_scan_runs_drift_and_shadow_on_latest_labels_dataset(
    tmp_path, monkeypatch
):
    datasets_path = tmp_path / "training-datasets"
    drift_reports_path = tmp_path / "drift-reports"
    shadow_runs_path = tmp_path / "shadow-runs"
    selection_runs_path = tmp_path / "selection-runs"
    evaluations_path = tmp_path / "evaluations"
    artifacts_path = tmp_path / "artifacts"
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_DRIFT_REPORTS_PATH", str(drift_reports_path))
    monkeypatch.setenv("MODEL_SHADOW_RUNS_PATH", str(shadow_runs_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))
    monkeypatch.setenv("MODEL_EVALUATIONS_PATH", str(evaluations_path))
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest").json()

        upsert_response = client.post(
            "/v1/admin/labels/upsert",
            json={
                "labels": [
                    {
                        "zoneId": "moc-01",
                        "observedAt": "2026-03-24T00:00:00Z",
                        "targetScore": 0.15,
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                    },
                    {
                        "zoneId": "moc-02",
                        "observedAt": "2026-03-25T00:00:00Z",
                        "targetScore": 0.9,
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                    },
                ]
            },
            headers=headers,
        )
        assert upsert_response.status_code == 200
        label_ids = [item["id"] for item in upsert_response.json()["labels"]]

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={
                "version": "monitoring-cycle-labels-v1",
                "sourceMode": "labels",
                "labelIds": label_ids,
            },
            headers=headers,
        )
        assert export_response.status_code == 200

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-monitoring-cycle-labels-v1",
                "datasetVersion": "monitoring-cycle-labels-v1",
                "alphas": [0.25, 0.75, 1.5],
                "versionPrefix": "monitoring-cycle-candidate",
                "promoteBest": False,
                "topErrorCount": 5,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200

        monitoring_scan = client.post(
            "/v1/admin/models/monitoring-scan",
            headers=headers,
        )
        assert monitoring_scan.status_code == 200
        payload = monitoring_scan.json()
        assert payload["job"]["jobType"] == "model_monitoring_cycle"
        assert payload["job"]["status"] == "completed"
        assert payload["skipped"] is False
        assert payload["datasetVersion"] == "monitoring-cycle-labels-v1"
        assert payload["activeModelVersion"] == "trained-spatial-seed-v1"
        assert payload["drift"]["job"]["jobType"] == "model_drift_scan"
        assert payload["drift"]["datasetVersion"] == "monitoring-cycle-labels-v1"
        assert payload["shadow"]["job"]["jobType"] == "model_shadow_scan"
        assert payload["shadow"]["datasetVersion"] == "monitoring-cycle-labels-v1"
        assert payload["shadow"]["candidateCount"] >= 1

        monitoring_detail = client.get(
            "/v1/admin/models/monitoring/trained-spatial-seed-v1",
            headers=headers,
        )
        assert monitoring_detail.status_code == 200
        monitoring_payload = monitoring_detail.json()
        assert monitoring_payload["latestDriftDatasetVersion"] == "monitoring-cycle-labels-v1"
        assert monitoring_payload["latestShadowDatasetVersion"] == "monitoring-cycle-labels-v1"


def test_model_monitoring_scan_creates_and_deduplicates_predictive_alerts(
    tmp_path, monkeypatch
):
    datasets_path = tmp_path / "training-datasets"
    drift_reports_path = tmp_path / "drift-reports"
    shadow_runs_path = tmp_path / "shadow-runs"
    selection_runs_path = tmp_path / "selection-runs"
    evaluations_path = tmp_path / "evaluations"
    artifacts_path = tmp_path / "artifacts"
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_DRIFT_REPORTS_PATH", str(drift_reports_path))
    monkeypatch.setenv("MODEL_SHADOW_RUNS_PATH", str(shadow_runs_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))
    monkeypatch.setenv("MODEL_EVALUATIONS_PATH", str(evaluations_path))
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("NOTIFICATION_MODEL_MONITORING_USERNAMES", "ops-model")

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        scenario = prepare_predictive_monitoring_alert_scenario(
            client,
            headers,
            prefix="monitoring-alert",
        )

        first_scan = client.post(
            "/v1/admin/models/monitoring-scan",
            json={"datasetVersion": scenario["review_dataset_version"]},
            headers=headers,
        )
        assert first_scan.status_code == 200
        first_payload = first_scan.json()
        assert first_payload["createdAlertCount"] == 2
        assert first_payload["updatedAlertCount"] == 0
        assert first_payload["resolvedAlertCount"] == 0
        assert {alert["eventType"] for alert in first_payload["alerts"]} == {
            "model_monitoring_drift_alert",
            "model_monitoring_shadow_alert",
        }

        notifications_response = client.get(
            "/v1/admin/notifications",
            params={"status": "open"},
            headers=headers,
        )
        assert notifications_response.status_code == 200
        alert_notifications = [
            item
            for item in notifications_response.json()
            if item["eventType"] in {
                "model_monitoring_drift_alert",
                "model_monitoring_shadow_alert",
            }
        ]
        assert len(alert_notifications) == 2
        assert {item["targetUsername"] for item in alert_notifications} == {"ops-model"}
        assert {
            item["details"]["routing"]["routing_audience"]
            for item in alert_notifications
        } == {"model_monitoring_watch"}

        second_scan = client.post(
            "/v1/admin/models/monitoring-scan",
            json={"datasetVersion": scenario["review_dataset_version"]},
            headers=headers,
        )
        assert second_scan.status_code == 200
        second_payload = second_scan.json()
        assert second_payload["createdAlertCount"] == 0
        assert (
            second_payload["updatedAlertCount"] + second_payload["resolvedAlertCount"]
            == 2
        )
        assert {alert["eventType"] for alert in second_payload["alerts"]}.issubset(
            {
                "model_monitoring_drift_alert",
                "model_monitoring_shadow_alert",
            }
        )

        notifications_after_second_scan = client.get(
            "/v1/admin/notifications",
            headers=headers,
        )
        assert notifications_after_second_scan.status_code == 200
        deduped_alerts = [
            item
            for item in notifications_after_second_scan.json()
            if item["eventType"] in {
                "model_monitoring_drift_alert",
                "model_monitoring_shadow_alert",
            }
        ]
        assert len(deduped_alerts) == 2


def test_predictive_alerts_can_open_and_track_model_review_tasks(
    tmp_path, monkeypatch
):
    datasets_path = tmp_path / "training-datasets"
    drift_reports_path = tmp_path / "drift-reports"
    shadow_runs_path = tmp_path / "shadow-runs"
    selection_runs_path = tmp_path / "selection-runs"
    evaluations_path = tmp_path / "evaluations"
    artifacts_path = tmp_path / "artifacts"
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_DRIFT_REPORTS_PATH", str(drift_reports_path))
    monkeypatch.setenv("MODEL_SHADOW_RUNS_PATH", str(shadow_runs_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))
    monkeypatch.setenv("MODEL_EVALUATIONS_PATH", str(evaluations_path))
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("NOTIFICATION_MODEL_MONITORING_USERNAMES", "ops-model")

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        scenario = prepare_predictive_monitoring_alert_scenario(
            client,
            headers,
            prefix="model-review-task",
        )

        first_scan = client.post(
            "/v1/admin/models/monitoring-scan",
            json={"datasetVersion": scenario["review_dataset_version"]},
            headers=headers,
        )
        assert first_scan.status_code == 200
        first_payload = first_scan.json()
        alert_ids_by_type = {
            alert["eventType"]: alert["id"] for alert in first_payload["alerts"]
        }
        assert set(alert_ids_by_type) == {
            "model_monitoring_drift_alert",
            "model_monitoring_shadow_alert",
        }

        blocked_open = client.post(
            "/v1/admin/models/review-tasks/open-from-alerts",
            json={
                "notificationIds": [alert_ids_by_type["model_monitoring_shadow_alert"]],
                "reviewType": "promotion_review",
            },
            headers=headers,
        )
        assert blocked_open.status_code == 400
        assert (
            blocked_open.json()["error"]["code"]
            == "model_review_alert_not_acknowledged"
        )

        acknowledge_alerts = client.post(
            "/v1/admin/notifications/acknowledge",
            json={"notificationIds": list(alert_ids_by_type.values())},
            headers=headers,
        )
        assert acknowledge_alerts.status_code == 200
        assert acknowledge_alerts.json()["acknowledgedCount"] == 2

        invalid_review_type = client.post(
            "/v1/admin/models/review-tasks/open-from-alerts",
            json={
                "notificationIds": [alert_ids_by_type["model_monitoring_drift_alert"]],
                "reviewType": "promotion_review",
            },
            headers=headers,
        )
        assert invalid_review_type.status_code == 400
        assert (
            invalid_review_type.json()["error"]["code"]
            == "model_review_type_not_allowed_for_alert"
        )

        promotion_open = client.post(
            "/v1/admin/models/review-tasks/open-from-alerts",
            json={
                "notificationIds": [alert_ids_by_type["model_monitoring_shadow_alert"]],
                "reviewType": "promotion_review",
                "assignedReviewer": "ops-model",
                "dueAt": "2026-03-29T12:00:00Z",
                "notes": "Shadow winner should be reviewed before promotion.",
            },
            headers=headers,
        )
        assert promotion_open.status_code == 200
        promotion_payload = promotion_open.json()
        assert promotion_payload["createdCount"] == 1
        assert promotion_payload["skippedCount"] == 0
        promotion_task = promotion_payload["tasks"][0]
        assert promotion_task["reviewType"] == "promotion_review"
        assert promotion_task["assignedReviewer"] == "ops-model"
        assert promotion_task["candidateModelVersion"] == scenario["selected_candidate_version"]
        assert (
            promotion_task["details"]["source_alert"]["event_type"]
            == "model_monitoring_shadow_alert"
        )

        promotion_open_again = client.post(
            "/v1/admin/models/review-tasks/open-from-alerts",
            json={
                "notificationIds": [alert_ids_by_type["model_monitoring_shadow_alert"]],
                "reviewType": "promotion_review",
            },
            headers=headers,
        )
        assert promotion_open_again.status_code == 200
        assert promotion_open_again.json()["createdCount"] == 0
        assert promotion_open_again.json()["skippedCount"] == 1
        assert promotion_open_again.json()["tasks"][0]["id"] == promotion_task["id"]

        rollback_open = client.post(
            "/v1/admin/models/review-tasks/open-from-alerts",
            json={
                "notificationIds": [alert_ids_by_type["model_monitoring_drift_alert"]],
                "reviewType": "rollback_review",
                "notes": "Check whether current champion should be rolled back.",
            },
            headers=headers,
        )
        assert rollback_open.status_code == 200
        rollback_task = rollback_open.json()["tasks"][0]
        assert rollback_task["reviewType"] == "rollback_review"
        assert rollback_task["sourceEventType"] == "model_monitoring_drift_alert"

        retraining_open = client.post(
            "/v1/admin/models/review-tasks/open-from-alerts",
            json={
                "notificationIds": [alert_ids_by_type["model_monitoring_drift_alert"]],
                "reviewType": "retraining_review",
            },
            headers=headers,
        )
        assert retraining_open.status_code == 200
        assert retraining_open.json()["createdCount"] == 1
        retraining_task = retraining_open.json()["tasks"][0]
        assert retraining_task["reviewType"] == "retraining_review"

        promotion_update = client.post(
            f"/v1/admin/models/review-tasks/{promotion_task['id']}/update",
            json={
                "status": "in_progress",
                "assignedReviewer": "ops-model",
                "notes": "Diagnostics review started.",
            },
            headers=headers,
        )
        assert promotion_update.status_code == 200
        assert promotion_update.json()["status"] == "in_progress"
        assert promotion_update.json()["updatedBy"] == "admin"

        promotion_resolve = client.post(
            f"/v1/admin/models/review-tasks/{promotion_task['id']}/update",
            json={
                "status": "resolved",
                "decision": "approve_promotion_review",
                "notes": "Shadow diagnostics validated the challenger.",
            },
            headers=headers,
        )
        assert promotion_resolve.status_code == 200
        resolved_promotion = promotion_resolve.json()
        assert resolved_promotion["status"] == "resolved"
        assert resolved_promotion["decision"] == "approve_promotion_review"
        assert resolved_promotion["resolvedBy"] == "admin"

        list_tasks = client.get(
            "/v1/admin/models/review-tasks",
            params={"sourceNotificationId": alert_ids_by_type["model_monitoring_drift_alert"]},
            headers=headers,
        )
        assert list_tasks.status_code == 200
        listed_tasks = list_tasks.json()
        assert {item["reviewType"] for item in listed_tasks} == {
            "rollback_review",
            "retraining_review",
        }

        get_task = client.get(
            f"/v1/admin/models/review-tasks/{promotion_task['id']}",
            headers=headers,
        )
        assert get_task.status_code == 200
        detail_payload = get_task.json()
        assert detail_payload["id"] == promotion_task["id"]
        assert detail_payload["status"] == "resolved"
        assert len(detail_payload["details"]["history"]) >= 3

        notifications_response = client.get(
            "/v1/admin/notifications",
            params={"eventType": "model_monitoring_shadow_alert"},
            headers=headers,
        )
        assert notifications_response.status_code == 200
        shadow_alerts = notifications_response.json()
        assert len(shadow_alerts) == 1
        assert shadow_alerts[0]["details"]["model_review_task_ids"] == [
            promotion_task["id"]
        ]


def test_model_actions_record_governed_review_task_outcomes(tmp_path, monkeypatch):
    artifacts_path = tmp_path / "artifacts"
    evaluations_path = tmp_path / "evaluations"
    training_datasets_path = tmp_path / "training_datasets"
    selection_runs_path = tmp_path / "selection_runs"
    shadow_runs_path = tmp_path / "shadow_runs"
    drift_reports_path = tmp_path / "drift_reports"
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(training_datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))
    monkeypatch.setenv("MODEL_SHADOW_RUNS_PATH", str(shadow_runs_path))
    monkeypatch.setenv("MODEL_DRIFT_REPORTS_PATH", str(drift_reports_path))
    monkeypatch.setenv("MODEL_EVALUATIONS_PATH", str(evaluations_path))
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("NOTIFICATION_MODEL_MONITORING_USERNAMES", "ops-model")

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        scenario = prepare_predictive_monitoring_alert_scenario(
            client,
            headers,
            prefix="model-governed-actions",
        )

        monitoring_response = client.post(
            "/v1/admin/models/monitoring-scan",
            json={"datasetVersion": scenario["review_dataset_version"]},
            headers=headers,
        )
        assert monitoring_response.status_code == 200
        alert_ids_by_type = {
            alert["eventType"]: alert["id"]
            for alert in monitoring_response.json()["alerts"]
        }

        acknowledge_alerts = client.post(
            "/v1/admin/notifications/acknowledge",
            json={"notificationIds": list(alert_ids_by_type.values())},
            headers=headers,
        )
        assert acknowledge_alerts.status_code == 200

        promotion_open = client.post(
            "/v1/admin/models/review-tasks/open-from-alerts",
            json={
                "notificationIds": [alert_ids_by_type["model_monitoring_shadow_alert"]],
                "reviewType": "promotion_review",
                "assignedReviewer": "ops-model",
                "notes": "Promotion requires governed approval.",
            },
            headers=headers,
        )
        assert promotion_open.status_code == 200
        promotion_task = promotion_open.json()["tasks"][0]

        retraining_open = client.post(
            "/v1/admin/models/review-tasks/open-from-alerts",
            json={
                "notificationIds": [alert_ids_by_type["model_monitoring_drift_alert"]],
                "reviewType": "retraining_review",
                "assignedReviewer": "ops-model",
                "notes": "Retraining requires governed approval.",
            },
            headers=headers,
        )
        assert retraining_open.status_code == 200
        retraining_task = retraining_open.json()["tasks"][0]

        blocked_promotion = client.post(
            "/v1/admin/models/promote",
            json={
                "modelVersion": scenario["selected_candidate_version"],
                "reviewTaskId": promotion_task["id"],
                "reason": "Blocked until task is resolved.",
            },
            headers=headers,
        )
        assert blocked_promotion.status_code == 409
        assert (
            blocked_promotion.json()["error"]["code"]
            == "model_review_task_not_resolved"
        )

        resolve_promotion = client.post(
            f"/v1/admin/models/review-tasks/{promotion_task['id']}/update",
            json={
                "status": "resolved",
                "decision": "approve_promotion_review",
                "notes": "Promotion approved after reviewing the challenger.",
            },
            headers=headers,
        )
        assert resolve_promotion.status_code == 200

        resolve_retraining = client.post(
            f"/v1/admin/models/review-tasks/{retraining_task['id']}/update",
            json={
                "status": "resolved",
                "decision": "approve_retraining_review",
                "notes": "Retraining approved for the monitored labels cohort.",
            },
            headers=headers,
        )
        assert resolve_retraining.status_code == 200

        wrong_review_type = client.post(
            "/v1/admin/models/promote",
            json={
                "modelVersion": scenario["selected_candidate_version"],
                "reviewTaskId": retraining_task["id"],
                "reason": "Wrong task type should fail.",
            },
            headers=headers,
        )
        assert wrong_review_type.status_code == 409
        assert (
            wrong_review_type.json()["error"]["code"]
            == "model_review_task_wrong_type"
        )

        retrain_response = client.post(
            "/v1/admin/retrain",
            json={
                "version": "governed-retrain-v1",
                "datasetVersion": scenario["source_dataset_version"],
                "reviewTaskId": retraining_task["id"],
            },
            headers=headers,
        )
        assert retrain_response.status_code == 200

        promote_response = client.post(
            "/v1/admin/models/promote",
            json={
                "modelVersion": scenario["selected_candidate_version"],
                "reviewTaskId": promotion_task["id"],
                "reason": "Promotion approved from a governed review task.",
            },
            headers=headers,
        )
        assert promote_response.status_code == 200

        promotion_task_detail = client.get(
            f"/v1/admin/models/review-tasks/{promotion_task['id']}",
            headers=headers,
        )
        assert promotion_task_detail.status_code == 200
        promotion_detail_payload = promotion_task_detail.json()
        assert promotion_detail_payload["details"]["last_governed_action"] == "promotion"
        assert (
            promotion_detail_payload["details"]["governed_actions"][-1]["outcome"][
                "model_version"
            ]
            == scenario["selected_candidate_version"]
        )

        retraining_task_detail = client.get(
            f"/v1/admin/models/review-tasks/{retraining_task['id']}",
            headers=headers,
        )
        assert retraining_task_detail.status_code == 200
        retraining_detail_payload = retraining_task_detail.json()
        assert retraining_detail_payload["details"]["last_governed_action"] == "retraining"
        assert (
            retraining_detail_payload["details"]["governed_actions"][-1]["outcome"][
                "dataset_version"
            ]
            == scenario["source_dataset_version"]
        )
