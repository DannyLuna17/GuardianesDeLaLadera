import json

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


def write_temporal_validation_dataset(datasets_path) -> None:
    from app.ml.datasets import build_dataset_context, build_training_dataset
    from app.ml.training import build_seed_training_rows

    rows = build_seed_training_rows()[:8]
    time_window = {
        "kind": "observed_window",
        "reference_at": "2026-01-04T00:00:00+00:00",
        "start_at": "2026-01-01T00:00:00+00:00",
        "end_at": "2026-01-04T00:00:00+00:00",
        "span_days": 3,
    }
    context = build_dataset_context(
        dataset_mode="labels",
        dataset_family="labels:field_validation",
        time_window=time_window,
        source="field_validation",
        label_source_families=["field_validation"],
    )
    row_contexts = []
    for index, _ in enumerate(rows):
        bucket_day = 1 + (index // 2)
        row_contexts.append(
            {
                "spatialBlockId": f"block-{index % 3}",
                "eventGroupId": f"temporal-test-{index // 2}",
                "temporalHoldoutTag": f"bucket:2026-01-0{bucket_day}",
                "observedAt": f"2026-01-0{bucket_day}T00:00:00+00:00",
            }
        )

    dataset = build_training_dataset(
        version="temporal-validation-dataset-v1",
        rows=rows,
        description="Synthetic labels dataset with ordered temporal buckets for backtesting.",
        provenance={
            "exported_at": "2026-04-19T00:00:00+00:00",
            **context,
        },
        row_contexts=row_contexts,
        summary_extra={
            "dataset_family": context["dataset_family"],
            "time_window": context["time_window"],
            "dataset_taxonomy": context["dataset_taxonomy"],
            "evaluation_cohort": context["evaluation_cohort"],
        },
    )
    datasets_path.mkdir(parents=True, exist_ok=True)
    (datasets_path / "temporal-validation-dataset-v1.json").write_text(
        json.dumps(dataset),
        encoding="utf-8",
    )


def write_spatial_only_labels_dataset(datasets_path) -> None:
    from app.ml.datasets import build_dataset_context, build_training_dataset
    from app.ml.training import build_seed_training_rows

    rows = build_seed_training_rows()[:6]
    time_window = {
        "kind": "observed_window",
        "reference_at": "2026-02-10T00:00:00+00:00",
        "start_at": "2026-02-01T00:00:00+00:00",
        "end_at": "2026-02-10T00:00:00+00:00",
        "span_days": 9,
    }
    context = build_dataset_context(
        dataset_mode="labels",
        dataset_family="labels:field_validation",
        time_window=time_window,
        source="governed_zone_outcome_labels",
        label_source_families=["field_validation"],
    )
    row_contexts = []
    for index, _ in enumerate(rows):
        row_contexts.append(
            {
                "spatialBlockId": f"block-{index % 3}",
                "eventGroupId": f"spatial-only-{index // 2}",
            }
        )

    dataset = build_training_dataset(
        version="spatial-only-labels-dataset-v1",
        rows=rows,
        description="Synthetic labels dataset with spatial blocks but without temporal backtest buckets.",
        provenance={
            "exported_at": "2026-04-19T00:00:00+00:00",
            **context,
        },
        row_contexts=row_contexts,
        summary_extra={
            "dataset_family": context["dataset_family"],
            "time_window": context["time_window"],
            "dataset_taxonomy": context["dataset_taxonomy"],
            "evaluation_cohort": context["evaluation_cohort"],
        },
    )
    datasets_path.mkdir(parents=True, exist_ok=True)
    (datasets_path / "spatial-only-labels-dataset-v1.json").write_text(
        json.dumps(dataset),
        encoding="utf-8",
    )


def test_retrain_endpoint_supports_gradient_boosted_tree(tmp_path, monkeypatch):
    artifacts_path = tmp_path / "artifacts"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        response = client.post(
            "/v1/admin/retrain",
            json={
                "version": "gbt-retrained-v1",
                "modelFamily": "gradient_boosted_tree",
                "learningRate": 0.1,
                "estimatorCount": 12,
                "maxDepth": 2,
                "minLeafSize": 1,
                "earlyStoppingRounds": 2,
            },
            headers=headers,
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["modelVersion"] == "gbt-retrained-v1"
        assert payload["modelFamily"] == "gradient_boosted_tree"
        assert payload["alpha"] is None
        assert payload["hyperparameters"]["learning_rate"] == 0.1
        assert payload["hyperparameters"]["estimator_count"] == 12
        assert (artifacts_path / "gbt-retrained-v1.json").exists()

        detail = client.get("/v1/admin/models/gbt-retrained-v1", headers=headers).json()
        assert detail["artifactType"] == "gradient_boosted_tree_model"
        assert detail["modelFamily"] == "gradient_boosted_tree"
        assert detail["training"]["hyperparameters"]["max_depth"] == 2
        assert detail["training"]["splits"]["effective_tree_count"] > 0


def test_retrain_endpoint_supports_additive_spline(tmp_path, monkeypatch):
    artifacts_path = tmp_path / "artifacts"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        response = client.post(
            "/v1/admin/retrain",
            json={
                "version": "gam-retrained-v1",
                "modelFamily": "additive_spline",
                "alpha": 1.25,
                "knotCount": 3,
            },
            headers=headers,
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["modelVersion"] == "gam-retrained-v1"
        assert payload["modelFamily"] == "additive_spline"
        assert payload["alpha"] == 1.25
        assert payload["hyperparameters"]["alpha"] == 1.25
        assert payload["hyperparameters"]["knot_count"] == 3
        assert (artifacts_path / "gam-retrained-v1.json").exists()

        detail = client.get("/v1/admin/models/gam-retrained-v1", headers=headers).json()
        assert detail["artifactType"] == "additive_spline_model"
        assert detail["modelFamily"] == "additive_spline"
        assert detail["training"]["hyperparameters"]["knot_count"] == 3
        assert detail["training"]["hyperparameters"]["basis_count"] >= 9


def test_retrain_endpoint_supports_beta_regression(tmp_path, monkeypatch):
    artifacts_path = tmp_path / "artifacts"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        response = client.post(
            "/v1/admin/retrain",
            json={
                "version": "beta-retrained-v1",
                "modelFamily": "beta_regression",
                "alpha": 0.75,
            },
            headers=headers,
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["modelVersion"] == "beta-retrained-v1"
        assert payload["modelFamily"] == "beta_regression"
        assert payload["alpha"] == 0.75
        assert payload["hyperparameters"]["alpha"] == 0.75
        assert (artifacts_path / "beta-retrained-v1.json").exists()

        detail = client.get("/v1/admin/models/beta-retrained-v1", headers=headers).json()
        assert detail["artifactType"] == "beta_regression_model"
        assert detail["modelFamily"] == "beta_regression"
        assert detail["training"]["hyperparameters"]["precision"] >= 2.0


def test_retrain_endpoint_supports_xgboost(tmp_path, monkeypatch):
    artifacts_path = tmp_path / "artifacts"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        response = client.post(
            "/v1/admin/retrain",
            json={
                "version": "xgb-retrained-v1",
                "modelFamily": "xgboost",
                "learningRate": 0.1,
                "estimatorCount": 16,
                "maxDepth": 3,
                "minLeafSize": 1,
                "earlyStoppingRounds": 4,
            },
            headers=headers,
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["modelVersion"] == "xgb-retrained-v1"
        assert payload["modelFamily"] == "xgboost"
        assert payload["alpha"] is None
        assert payload["hyperparameters"]["learning_rate"] == 0.1
        assert payload["hyperparameters"]["estimator_count"] == 16
        assert (artifacts_path / "xgb-retrained-v1.json").exists()

        detail = client.get("/v1/admin/models/xgb-retrained-v1", headers=headers).json()
        assert detail["artifactType"] == "xgboost_model"
        assert detail["modelFamily"] == "xgboost"
        assert detail["training"]["hyperparameters"]["subsample"] == 1.0
        assert detail["training"]["splits"]["effective_tree_count"] > 0


def test_tune_endpoint_supports_gradient_boosted_tree_candidates(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={"version": "gbt-tuning-dataset-v1"},
            headers=headers,
        )
        assert export_response.status_code == 200

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-gbt-tuning-dataset-v1",
                "datasetVersion": "gbt-tuning-dataset-v1",
                "modelFamily": "gradient_boosted_tree",
                "learningRates": [0.05, 0.1],
                "estimatorCounts": [8],
                "maxDepths": [2],
                "minLeafSizes": [1],
                "earlyStoppingRounds": 2,
                "versionPrefix": "gbt-candidate",
                "promoteBest": False,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["candidateCount"] == 2
        assert payload["bestCandidateComparison"]["challenger_model_family"] == (
            "gradient_boosted_tree"
        )
        assert (
            "by_spatial_block"
            in payload["bestCandidateComparison"]["validation_slice_deltas"]
        )
        assert (
            "by_temporal_holdout_tag"
            in payload["bestCandidateComparison"]["validation_slice_deltas"]
        )
        assert all(
            candidate["modelFamily"] == "gradient_boosted_tree"
            for candidate in payload["candidates"]
        )
        assert all(candidate["alpha"] is None for candidate in payload["candidates"])
        assert all(
            candidate["hyperparameters"]["estimator_count"] == 8
            for candidate in payload["candidates"]
        )
        assert (
            selection_runs_path / "selection-gbt-tuning-dataset-v1.json"
        ).exists()

        detail_response = client.get(
            "/v1/admin/model-selection-runs/selection-gbt-tuning-dataset-v1",
            headers=headers,
        )
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["comparisonPolicy"]["primary_metric"] == "validation_rmse"
        assert detail["candidates"][0]["modelFamily"] == "gradient_boosted_tree"
        assert detail["candidates"][0]["hyperparameters"]["max_depth"] == 2


def test_tune_endpoint_supports_additive_spline_candidates(tmp_path, monkeypatch):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={"version": "gam-tuning-dataset-v1"},
            headers=headers,
        )
        assert export_response.status_code == 200

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-gam-tuning-dataset-v1",
                "datasetVersion": "gam-tuning-dataset-v1",
                "modelFamily": "additive_spline",
                "alphas": [0.75],
                "knotCounts": [2, 4],
                "versionPrefix": "gam-candidate",
                "promoteBest": False,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["candidateCount"] == 2
        assert payload["bestCandidateComparison"]["challenger_model_family"] == (
            "additive_spline"
        )
        assert (
            "by_spatial_block"
            in payload["bestCandidateComparison"]["validation_slice_deltas"]
        )
        assert (
            "by_temporal_holdout_tag"
            in payload["bestCandidateComparison"]["validation_slice_deltas"]
        )
        assert all(
            candidate["modelFamily"] == "additive_spline"
            for candidate in payload["candidates"]
        )
        assert all(candidate["alpha"] == 0.75 for candidate in payload["candidates"])
        assert {
            candidate["hyperparameters"]["knot_count"]
            for candidate in payload["candidates"]
        } == {2, 4}
        assert (
            selection_runs_path / "selection-gam-tuning-dataset-v1.json"
        ).exists()

        detail_response = client.get(
            "/v1/admin/model-selection-runs/selection-gam-tuning-dataset-v1",
            headers=headers,
        )
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["comparisonPolicy"]["primary_metric"] == "validation_rmse"
        assert detail["candidates"][0]["modelFamily"] == "additive_spline"
        assert detail["candidates"][0]["hyperparameters"]["alpha"] == 0.75


def test_tune_endpoint_supports_beta_regression_candidates(tmp_path, monkeypatch):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={"version": "beta-tuning-dataset-v1"},
            headers=headers,
        )
        assert export_response.status_code == 200

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-beta-tuning-dataset-v1",
                "datasetVersion": "beta-tuning-dataset-v1",
                "modelFamily": "beta_regression",
                "alphas": [0.5, 0.75],
                "versionPrefix": "beta-candidate",
                "promoteBest": False,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["candidateCount"] == 2
        assert payload["bestCandidateComparison"]["challenger_model_family"] == (
            "beta_regression"
        )
        assert all(
            candidate["modelFamily"] == "beta_regression"
            for candidate in payload["candidates"]
        )
        assert {candidate["alpha"] for candidate in payload["candidates"]} == {
            0.5,
            0.75,
        }
        assert (
            selection_runs_path / "selection-beta-tuning-dataset-v1.json"
        ).exists()

        detail_response = client.get(
            "/v1/admin/model-selection-runs/selection-beta-tuning-dataset-v1",
            headers=headers,
        )
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["candidates"][0]["modelFamily"] == "beta_regression"
        assert detail["candidates"][0]["hyperparameters"]["precision"] >= 2.0


def test_tune_endpoint_supports_xgboost_candidates(tmp_path, monkeypatch):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={"version": "xgb-tuning-dataset-v1"},
            headers=headers,
        )
        assert export_response.status_code == 200

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-xgb-tuning-dataset-v1",
                "datasetVersion": "xgb-tuning-dataset-v1",
                "modelFamily": "xgboost",
                "learningRates": [0.05, 0.1],
                "estimatorCounts": [12],
                "maxDepths": [2],
                "minLeafSizes": [1],
                "earlyStoppingRounds": 4,
                "versionPrefix": "xgb-candidate",
                "promoteBest": False,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["candidateCount"] == 2
        assert payload["bestCandidateComparison"]["challenger_model_family"] == (
            "xgboost"
        )
        assert (
            "by_spatial_block"
            in payload["bestCandidateComparison"]["validation_slice_deltas"]
        )
        assert (
            "by_temporal_holdout_tag"
            in payload["bestCandidateComparison"]["validation_slice_deltas"]
        )
        assert all(candidate["modelFamily"] == "xgboost" for candidate in payload["candidates"])
        assert all(candidate["alpha"] is None for candidate in payload["candidates"])
        assert all(
            candidate["hyperparameters"]["estimator_count"] == 12
            for candidate in payload["candidates"]
        )
        assert (
            selection_runs_path / "selection-xgb-tuning-dataset-v1.json"
        ).exists()

        detail_response = client.get(
            "/v1/admin/model-selection-runs/selection-xgb-tuning-dataset-v1",
            headers=headers,
        )
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["comparisonPolicy"]["primary_metric"] == "validation_rmse"
        assert detail["candidates"][0]["modelFamily"] == "xgboost"
        assert detail["candidates"][0]["hyperparameters"]["subsample"] == 1.0


def test_tune_endpoint_supports_spatial_block_kfold_validation(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={"version": "spatial-kfold-dataset-v1"},
            headers=headers,
        )
        assert export_response.status_code == 200

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-spatial-kfold-v1",
                "datasetVersion": "spatial-kfold-dataset-v1",
                "modelFamily": "xgboost",
                "validationStrategy": "spatial_block_kfold",
                "validationFoldCount": 3,
                "learningRates": [0.05],
                "estimatorCounts": [12],
                "maxDepths": [2],
                "minLeafSizes": [1],
                "earlyStoppingRounds": 4,
                "versionPrefix": "spatial-kfold-candidate",
                "promoteBest": False,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["candidateCount"] == 1
        assert payload["bestCandidateComparison"]["validation_strategy"] == (
            "spatial_block_kfold"
        )
        assert payload["candidates"][0]["validationSummary"]["strategy"] == (
            "spatial_block_kfold"
        )
        assert payload["candidates"][0]["validationSummary"]["fold_count"] == 3

        detail_response = client.get(
            "/v1/admin/model-selection-runs/selection-spatial-kfold-v1",
            headers=headers,
        )
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["comparisonPolicy"]["validation_strategy"] == (
            "spatial_block_kfold"
        )
        assert detail["comparisonPolicy"]["folds_evaluated"] == 3
        assert detail["candidates"][0]["validationSummary"]["validation_unit"] == (
            "spatialBlockId"
        )


def test_tune_endpoint_supports_temporal_holdout_backtest_validation(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    write_temporal_validation_dataset(datasets_path)

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-temporal-backtest-v1",
                "datasetVersion": "temporal-validation-dataset-v1",
                "modelFamily": "linear_ridge",
                "validationStrategy": "temporal_holdout_backtest",
                "validationFoldCount": 2,
                "alphas": [0.75],
                "versionPrefix": "temporal-candidate",
                "promoteBest": False,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["candidateCount"] == 1
        assert payload["bestCandidateComparison"]["validation_strategy"] == (
            "temporal_holdout_backtest"
        )
        assert payload["candidates"][0]["validationSummary"]["strategy"] == (
            "temporal_holdout_backtest"
        )
        assert payload["candidates"][0]["validationSummary"]["fold_count"] == 2
        assert (
            payload["candidates"][0]["validationSummary"]["folds"][0]["metadata"][
                "validation_bucket"
            ]
            != ""
        )

        detail_response = client.get(
            "/v1/admin/model-selection-runs/selection-temporal-backtest-v1",
            headers=headers,
        )
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["comparisonPolicy"]["validation_strategy"] == (
            "temporal_holdout_backtest"
        )
        assert detail["comparisonPolicy"]["folds_evaluated"] == 2


def test_tune_endpoint_supports_mixed_family_selection_under_shared_validation(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={"version": "mixed-family-dataset-v1"},
            headers=headers,
        )
        assert export_response.status_code == 200

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-mixed-family-v1",
                "datasetVersion": "mixed-family-dataset-v1",
                "modelFamilies": ["additive_spline", "xgboost"],
                "validationStrategy": "spatial_block_kfold",
                "validationFoldCount": 3,
                "alphas": [0.75],
                "knotCounts": [3],
                "learningRates": [0.05],
                "estimatorCounts": [12],
                "maxDepths": [2],
                "minLeafSizes": [1],
                "earlyStoppingRounds": 4,
                "versionPrefix": "mixed-family-candidate",
                "promoteBest": False,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["candidateCount"] == 2
        assert {
            candidate["modelFamily"] for candidate in payload["candidates"]
        } == {"additive_spline", "xgboost"}
        assert payload["bestCandidateComparison"]["validation_strategy"] == (
            "spatial_block_kfold"
        )
        assert len(payload["familyRollups"]) == 2
        assert {item["model_family"] for item in payload["familyRollups"]} == {
            "additive_spline",
            "xgboost",
        }

        detail_response = client.get(
            "/v1/admin/model-selection-runs/selection-mixed-family-v1",
            headers=headers,
        )
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["gatePolicy"]["requested_model_families"] == [
            "additive_spline",
            "xgboost",
        ]
        assert len(detail["familyRollups"]) == 2
        assert detail["familyRollups"][0]["best_rank"] == 1


def test_tune_endpoint_supports_nested_outer_estimation_for_search_process(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={"version": "nested-estimation-dataset-v1"},
            headers=headers,
        )
        assert export_response.status_code == 200

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-nested-estimation-v1",
                "datasetVersion": "nested-estimation-dataset-v1",
                "modelFamilies": ["additive_spline", "xgboost"],
                "selectionMode": "nested_outer_estimate",
                "validationStrategy": "spatial_block_kfold",
                "validationFoldCount": 2,
                "nestedOuterFoldCount": 2,
                "alphas": [0.75],
                "knotCounts": [3],
                "learningRates": [0.05],
                "estimatorCounts": [12],
                "maxDepths": [2],
                "minLeafSizes": [1],
                "earlyStoppingRounds": 4,
                "versionPrefix": "nested-estimation-candidate",
                "promoteBest": False,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["candidateCount"] == 2
        assert payload["nestedEstimation"]["selection_mode"] == (
            "nested_outer_estimate"
        )
        assert payload["nestedEstimation"]["outer_fold_count"] == 2
        assert len(payload["nestedEstimation"]["selected_family_frequencies"]) >= 1
        assert len(payload["nestedEstimation"]["folds"]) == 2

        detail_response = client.get(
            "/v1/admin/model-selection-runs/selection-nested-estimation-v1",
            headers=headers,
        )
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["comparisonPolicy"]["selection_mode"] == (
            "nested_outer_estimate"
        )
        assert detail["nestedEstimation"]["final_candidate_signature"] != ""
        assert detail["nestedEstimation"]["final_candidate_outer_selection_rate"] >= 0


def test_tune_endpoint_can_require_nested_estimation_for_promotion(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={"version": "promotion-requires-nested-v1"},
            headers=headers,
        )
        assert export_response.status_code == 200

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-promotion-requires-nested-v1",
                "datasetVersion": "promotion-requires-nested-v1",
                "modelFamily": "xgboost",
                "requireNestedEstimationForPromotion": True,
                "requireLabelsDatasetForPromotion": False,
                "learningRates": [0.05],
                "estimatorCounts": [12],
                "maxDepths": [2],
                "minLeafSizes": [1],
                "earlyStoppingRounds": 4,
                "versionPrefix": "promotion-requires-nested-candidate",
                "promoteBest": True,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["promoted"] is False
        assert payload["promotionDecision"]["eligible"] is False
        assert (
            "nested_estimation_required_for_promotion"
            in payload["promotionDecision"]["blocking_reasons"]
        )


def test_tune_endpoint_nested_gate_can_block_for_outer_improvement_threshold(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={"version": "promotion-nested-threshold-v1"},
            headers=headers,
        )
        assert export_response.status_code == 200

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-promotion-nested-threshold-v1",
                "datasetVersion": "promotion-nested-threshold-v1",
                "modelFamilies": ["additive_spline", "xgboost"],
                "selectionMode": "nested_outer_estimate",
                "validationStrategy": "spatial_block_kfold",
                "validationFoldCount": 2,
                "nestedOuterFoldCount": 2,
                "requireNestedEstimationForPromotion": True,
                "requireLabelsDatasetForPromotion": False,
                "minNestedOuterValidationRmseImprovement": 1.0,
                "alphas": [0.75],
                "knotCounts": [3],
                "learningRates": [0.05],
                "estimatorCounts": [12],
                "maxDepths": [2],
                "minLeafSizes": [1],
                "earlyStoppingRounds": 4,
                "versionPrefix": "promotion-nested-threshold-candidate",
                "promoteBest": True,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["promoted"] is False
        assert payload["promotionDecision"]["eligible"] is False
        assert (
            "minimum_nested_outer_validation_rmse_improvement_not_met"
            in payload["promotionDecision"]["blocking_reasons"]
        )
        assert "comparison_vs_active" in payload["nestedEstimation"]
        assert (
            payload["promotionDecision"]["nested_outer_validation_rmse_improvement"]
            == payload["nestedEstimation"]["comparison_vs_active"][
                "validation_rmse_improvement"
            ]
        )


def test_tune_endpoint_nested_temporal_summary_exposes_latest_bucket_coverage(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    write_temporal_validation_dataset(datasets_path)

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-nested-temporal-summary-v1",
                "datasetVersion": "temporal-validation-dataset-v1",
                "modelFamily": "linear_ridge",
                "selectionMode": "nested_outer_estimate",
                "validationStrategy": "temporal_holdout_backtest",
                "validationFoldCount": 2,
                "nestedOuterFoldCount": 2,
                "alphas": [0.75],
                "versionPrefix": "nested-temporal-summary-candidate",
                "promoteBest": False,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        coverage = payload["nestedEstimation"]["temporal_cohort_coverage"]
        assert coverage["available"] is True
        assert coverage["bucket_count"] == 2
        assert coverage["latest_bucket"] == "bucket:2026-01-04"
        assert coverage["latest_fold_id"] != ""
        assert coverage["recent_window"]["available"] is True
        assert coverage["recent_window"]["resolved_window_size"] == 2
        assert coverage["recent_window"]["latest_bucket"] == "bucket:2026-01-04"


def test_tune_endpoint_nested_temporal_bucket_gate_can_block_promotion(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    write_temporal_validation_dataset(datasets_path)

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-nested-temporal-bucket-gate-v1",
                "datasetVersion": "temporal-validation-dataset-v1",
                "modelFamily": "linear_ridge",
                "selectionMode": "nested_outer_estimate",
                "validationStrategy": "temporal_holdout_backtest",
                "validationFoldCount": 2,
                "nestedOuterFoldCount": 2,
                "alphas": [0.75],
                "requireLabelsDatasetForPromotion": False,
                "requireNestedEstimationForPromotion": True,
                "minNestedTemporalOuterBucketCount": 3,
                "versionPrefix": "nested-temporal-bucket-gate-candidate",
                "promoteBest": True,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["promoted"] is False
        assert payload["promotionDecision"]["eligible"] is False
        assert (
            "minimum_nested_temporal_outer_bucket_count_not_met"
            in payload["promotionDecision"]["blocking_reasons"]
        )


def test_tune_endpoint_nested_temporal_latest_win_gate_blocks_when_unavailable(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={"version": "spatial-nested-temporal-unavailable-v1"},
            headers=headers,
        )
        assert export_response.status_code == 200

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-nested-temporal-unavailable-v1",
                "datasetVersion": "spatial-nested-temporal-unavailable-v1",
                "modelFamily": "xgboost",
                "selectionMode": "nested_outer_estimate",
                "validationStrategy": "spatial_block_kfold",
                "validationFoldCount": 2,
                "nestedOuterFoldCount": 2,
                "requireLabelsDatasetForPromotion": False,
                "requireNestedEstimationForPromotion": True,
                "requireNestedTemporalLatestWinForPromotion": True,
                "learningRates": [0.05],
                "estimatorCounts": [12],
                "maxDepths": [2],
                "minLeafSizes": [1],
                "earlyStoppingRounds": 4,
                "versionPrefix": "nested-temporal-unavailable-candidate",
                "promoteBest": True,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["promoted"] is False
        assert payload["promotionDecision"]["eligible"] is False
        assert (
            "nested_temporal_latest_win_unavailable"
            in payload["promotionDecision"]["blocking_reasons"]
        )


def test_tune_endpoint_nested_temporal_latest_rmse_threshold_can_block_promotion(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    write_temporal_validation_dataset(datasets_path)

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-nested-temporal-threshold-v1",
                "datasetVersion": "temporal-validation-dataset-v1",
                "modelFamily": "linear_ridge",
                "selectionMode": "nested_outer_estimate",
                "validationStrategy": "temporal_holdout_backtest",
                "validationFoldCount": 2,
                "nestedOuterFoldCount": 2,
                "alphas": [0.75],
                "requireLabelsDatasetForPromotion": False,
                "requireNestedEstimationForPromotion": True,
                "minNestedTemporalLatestValidationRmseImprovement": 10.0,
                "versionPrefix": "nested-temporal-threshold-candidate",
                "promoteBest": True,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["promoted"] is False
        assert payload["promotionDecision"]["eligible"] is False
        assert (
            "minimum_nested_temporal_latest_validation_rmse_improvement_not_met"
            in payload["promotionDecision"]["blocking_reasons"]
        )
        assert (
            payload["promotionDecision"][
                "nested_temporal_latest_validation_rmse_improvement"
            ]
            == payload["nestedEstimation"]["temporal_cohort_coverage"][
                "latest_fold_validation_rmse_improvement_vs_active"
            ]
        )


def test_tune_endpoint_nested_temporal_recent_average_threshold_can_block_promotion(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    write_temporal_validation_dataset(datasets_path)

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-nested-temporal-recent-threshold-v1",
                "datasetVersion": "temporal-validation-dataset-v1",
                "modelFamily": "linear_ridge",
                "selectionMode": "nested_outer_estimate",
                "validationStrategy": "temporal_holdout_backtest",
                "validationFoldCount": 2,
                "nestedOuterFoldCount": 2,
                "alphas": [0.75],
                "requireLabelsDatasetForPromotion": False,
                "requireNestedEstimationForPromotion": True,
                "nestedTemporalRecentWindowSize": 2,
                "minNestedTemporalRecentAverageValidationRmseImprovement": 10.0,
                "versionPrefix": "nested-temporal-recent-threshold-candidate",
                "promoteBest": True,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["promoted"] is False
        assert payload["promotionDecision"]["eligible"] is False
        assert (
            "minimum_nested_temporal_recent_average_validation_rmse_improvement_not_met"
            in payload["promotionDecision"]["blocking_reasons"]
        )
        assert (
            payload["promotionDecision"][
                "nested_temporal_recent_average_validation_rmse_improvement"
            ]
            == payload["nestedEstimation"]["temporal_cohort_coverage"][
                "recent_window"
            ]["average_validation_rmse_improvement_vs_active"]
        )


def test_tune_endpoint_nested_temporal_recent_win_rate_gate_blocks_when_unavailable(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={"version": "spatial-nested-temporal-recent-unavailable-v1"},
            headers=headers,
        )
        assert export_response.status_code == 200

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-nested-temporal-recent-unavailable-v1",
                "datasetVersion": "spatial-nested-temporal-recent-unavailable-v1",
                "modelFamily": "xgboost",
                "selectionMode": "nested_outer_estimate",
                "validationStrategy": "spatial_block_kfold",
                "validationFoldCount": 2,
                "nestedOuterFoldCount": 2,
                "requireLabelsDatasetForPromotion": False,
                "requireNestedEstimationForPromotion": True,
                "nestedTemporalRecentWindowSize": 2,
                "minNestedTemporalRecentWinRate": 0.5,
                "learningRates": [0.05],
                "estimatorCounts": [12],
                "maxDepths": [2],
                "minLeafSizes": [1],
                "earlyStoppingRounds": 4,
                "versionPrefix": "nested-temporal-recent-unavailable-candidate",
                "promoteBest": True,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["promoted"] is False
        assert payload["promotionDecision"]["eligible"] is False
        assert (
            "nested_temporal_recent_win_rate_unavailable"
            in payload["promotionDecision"]["blocking_reasons"]
        )


def test_tune_endpoint_exposes_slice_gate_summary(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={"version": "slice-summary-dataset-v1"},
            headers=headers,
        )
        assert export_response.status_code == 200

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-slice-summary-v1",
                "datasetVersion": "slice-summary-dataset-v1",
                "modelFamily": "xgboost",
                "learningRates": [0.05],
                "estimatorCounts": [12],
                "maxDepths": [2],
                "minLeafSizes": [1],
                "earlyStoppingRounds": 4,
                "versionPrefix": "slice-summary-candidate",
                "promoteBest": False,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        spatial_summary = payload["bestCandidateComparison"]["validation_slice_summary"][
            "spatial_block"
        ]
        temporal_summary = payload["bestCandidateComparison"]["validation_slice_summary"][
            "temporal_holdout_tag"
        ]
        assert spatial_summary["available"] is True
        assert temporal_summary["available"] is True
        assert spatial_summary["considered_slice_count"] > 0
        assert temporal_summary["considered_slice_count"] > 0
        assert spatial_summary["min_rows"] == 1


def test_tune_endpoint_slice_regression_gate_blocks_when_slice_rows_unavailable(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={"version": "slice-unavailable-dataset-v1"},
            headers=headers,
        )
        assert export_response.status_code == 200

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-slice-unavailable-v1",
                "datasetVersion": "slice-unavailable-dataset-v1",
                "modelFamily": "xgboost",
                "requireLabelsDatasetForPromotion": False,
                "sliceRegressionMinRows": 999,
                "maxSpatialSliceValidationRmseRegression": 0.5,
                "learningRates": [0.05],
                "estimatorCounts": [12],
                "maxDepths": [2],
                "minLeafSizes": [1],
                "earlyStoppingRounds": 4,
                "versionPrefix": "slice-unavailable-candidate",
                "promoteBest": True,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["promoted"] is False
        assert payload["promotionDecision"]["eligible"] is False
        assert (
            "spatial_slice_regression_unavailable"
            in payload["promotionDecision"]["blocking_reasons"]
        )
        assert (
            payload["promotionDecision"]["slice_gate_summary"]["spatial_block"][
                "available"
            ]
            is False
        )


def test_tune_endpoint_slice_regression_threshold_can_block_promotion(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        export_response = client.post(
            "/v1/admin/training-datasets/export",
            json={"version": "slice-threshold-dataset-v1"},
            headers=headers,
        )
        assert export_response.status_code == 200

        tune_response = client.post(
            "/v1/admin/models/tune",
            json={
                "version": "selection-slice-threshold-v1",
                "datasetVersion": "slice-threshold-dataset-v1",
                "modelFamily": "xgboost",
                "requireLabelsDatasetForPromotion": False,
                "sliceRegressionMinRows": 1,
                "maxSpatialSliceValidationRmseRegression": 0.05,
                "learningRates": [0.05],
                "estimatorCounts": [12],
                "maxDepths": [2],
                "minLeafSizes": [1],
                "earlyStoppingRounds": 4,
                "versionPrefix": "slice-threshold-candidate",
                "promoteBest": True,
            },
            headers=headers,
        )
        assert tune_response.status_code == 200
        payload = tune_response.json()
        assert payload["promoted"] is False
        assert payload["promotionDecision"]["eligible"] is False
        assert (
            "maximum_spatial_slice_validation_rmse_regression_not_met"
            in payload["promotionDecision"]["blocking_reasons"]
        )
        assert (
            payload["promotionDecision"]["slice_gate_summary"]["spatial_block"][
                "worst_validation_rmse_regression"
            ]
            > 0.05
        )


def test_modern_labels_benchmark_auto_exports_labels_dataset_and_runs_benchmark(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)
        latest_run = client.get("/v1/runs/latest", headers=headers).json()

        upsert_response = client.post(
            "/v1/admin/labels/upsert",
            json={
                "labels": [
                    {
                        "zoneId": "moc-01",
                        "observedAt": "2026-03-20T00:00:00Z",
                        "targetScore": 0.18,
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                    },
                    {
                        "zoneId": "moc-02",
                        "observedAt": "2026-03-21T00:00:00Z",
                        "targetScore": 0.39,
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                    },
                    {
                        "zoneId": "moc-03",
                        "observedAt": "2026-03-22T00:00:00Z",
                        "targetScore": 0.74,
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                    },
                    {
                        "zoneId": "moc-04",
                        "observedAt": "2026-03-23T00:00:00Z",
                        "targetScore": 0.91,
                        "source": "field_validation",
                        "featureRunId": latest_run["id"],
                    },
                ]
            },
            headers=headers,
        )
        assert upsert_response.status_code == 200

        response = client.post(
            "/v1/admin/models/benchmark/modern-labels",
            json={
                "datasetExportVersion": "modern-benchmark-labels-v1",
                "version": "selection-modern-benchmark-labels-v1",
                "maxLabels": 4,
                "promoteBest": False,
            },
            headers=headers,
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["preset"] == "modern_labels_benchmark_v1"
        assert payload["datasetResolution"]["source"] == "auto_exported"
        assert payload["datasetResolution"]["dataset_version"] == (
            "modern-benchmark-labels-v1"
        )
        assert payload["datasetExport"]["sourceMode"] == "labels"
        assert payload["datasetExport"]["datasetVersion"] == "modern-benchmark-labels-v1"
        assert payload["resolvedPolicy"]["validation_strategy"] == (
            "temporal_holdout_backtest"
        )
        assert payload["resolvedPolicy"]["selection_mode"] == (
            "nested_outer_estimate"
        )
        assert payload["resolvedPolicy"]["validation_fold_count"] == 2
        assert set(payload["resolvedPolicy"]["model_families"]) == {
            "linear_ridge",
            "beta_regression",
            "additive_spline",
            "xgboost",
        }
        assert payload["selection"]["datasetVersion"] == "modern-benchmark-labels-v1"
        assert payload["selection"]["selectionVersion"] == (
            "selection-modern-benchmark-labels-v1"
        )
        assert len(payload["selection"]["familyRollups"]) == 4


def test_modern_labels_benchmark_can_fallback_from_temporal_to_spatial_validation(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    write_spatial_only_labels_dataset(datasets_path)

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        response = client.post(
            "/v1/admin/models/benchmark/modern-labels",
            json={
                "datasetVersion": "spatial-only-labels-dataset-v1",
                "version": "selection-modern-spatial-fallback-v1",
                "promoteBest": False,
            },
            headers=headers,
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["datasetResolution"]["source"] == "provided"
        assert payload["datasetExport"] is None
        assert payload["resolvedPolicy"]["validation_strategy"] == "spatial_block_kfold"
        assert payload["resolvedPolicy"]["selection_mode"] == "single_stage"
        assert payload["resolvedPolicy"]["attempted_strategies"][0]["strategy"] == (
            "temporal_holdout_backtest"
        )
        assert payload["resolvedPolicy"]["attempted_strategies"][0]["available"] is False
        assert payload["resolvedPolicy"]["attempted_strategies"][1]["strategy"] == (
            "spatial_block_kfold"
        )
        assert payload["resolvedPolicy"]["attempted_strategies"][1]["available"] is True
        assert payload["resolvedPolicy"]["nested_feasibility"][0]["available"] is False
        assert payload["selection"]["datasetVersion"] == "spatial-only-labels-dataset-v1"
        assert len(payload["selection"]["familyRollups"]) == 4


def test_modern_labels_benchmark_review_returns_non_actionable_recommendation_without_review_task(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    write_spatial_only_labels_dataset(datasets_path)

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        response = client.post(
            "/v1/admin/models/benchmark/modern-labels/review",
            json={
                "datasetVersion": "spatial-only-labels-dataset-v1",
                "version": "selection-modern-benchmark-review-non-actionable-v1",
                "promoteBest": False,
            },
            headers=headers,
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["benchmark"]["preset"] == "modern_labels_benchmark_v1"
        assert payload["benchmark"]["selection"]["promotionDecision"]["eligible"] is False
        assert payload["recommendation"]["actionable"] is False
        assert (
            payload["recommendation"]["recommended_action"]
            == "no_action_required"
        )
        assert "promotion_decision_not_eligible" in (
            payload["recommendation"]["skipped_reasons"]
        )
        assert payload["createdAlerts"] == []
        assert payload["acknowledgedAlerts"] == []
        assert payload["reviewTask"] is None

        tasks_response = client.get(
            "/v1/admin/models/review-tasks",
            headers=headers,
        )
        assert tasks_response.status_code == 200
        assert tasks_response.json() == []


def test_modern_labels_benchmark_review_opens_promotion_review_when_actionable(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    from datetime import datetime, timezone

    from app.schemas.admin import (
        JobExecutionRead,
        RunModernLabelsBenchmarkResponse,
        TuneModelResponse,
    )
    from app.services import model_selection as model_selection_module

    now_utc = datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc)
    synthetic_job = JobExecutionRead(
        id=999,
        jobType="model_selection",
        status="completed",
        startedAt=now_utc,
        completedAt=now_utc,
        details={},
    )
    synthetic_selection = TuneModelResponse(
        job=synthetic_job,
        selectionVersion="selection-modern-benchmark-review-actionable-v1",
        artifactPath=str(selection_runs_path / "synthetic.json"),
        datasetVersion="synthetic-labels-dataset-v1",
        candidateCount=1,
        bestModelVersion="xgb-challenger-actionable-v1",
        promoted=False,
        activeModelVersion="seed-linear-v1",
        promotionDecision={
            "eligible": True,
            "promoted": False,
            "blocking_reasons": [],
            "active_model_version": "seed-linear-v1",
            "challenger_model_version": "xgb-challenger-actionable-v1",
        },
        bestCandidateComparison={"validation_rmse_delta": -0.05},
        familyRollups=[
            {
                "model_family": "xgboost",
                "candidate_count": 1,
                "best_model_version": "xgb-challenger-actionable-v1",
            }
        ],
        nestedEstimation={},
        candidates=[],
    )
    synthetic_benchmark = RunModernLabelsBenchmarkResponse(
        preset="modern_labels_benchmark_v1",
        datasetResolution={
            "source": "provided",
            "dataset_version": "synthetic-labels-dataset-v1",
        },
        datasetExport=None,
        resolvedPolicy={
            "preset": "modern_labels_benchmark_v1",
            "validation_strategy": "temporal_holdout_backtest",
            "selection_mode": "nested_outer_estimate",
        },
        selection=synthetic_selection,
    )

    original_run = model_selection_module.ModelSelectionService.run_modern_labels_benchmark

    def _patched_run(self, **kwargs):
        return synthetic_benchmark

    monkeypatch.setattr(
        model_selection_module.ModelSelectionService,
        "run_modern_labels_benchmark",
        _patched_run,
    )

    try:
        with create_test_client(tmp_path, monkeypatch) as client:
            headers = get_admin_headers(client)

            response = client.post(
                "/v1/admin/models/benchmark/modern-labels/review",
                json={
                    "datasetVersion": "synthetic-labels-dataset-v1",
                    "version": "selection-modern-benchmark-review-actionable-v1",
                    "promoteBest": False,
                    "notes": "auto benchmark from test",
                },
                headers=headers,
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["recommendation"]["actionable"] is True
            assert payload["recommendation"]["recommended_action"] == (
                "open_promotion_review_for_modern_labels_benchmark"
            )
            assert payload["recommendation"]["best_model_version"] == (
                "xgb-challenger-actionable-v1"
            )
            assert len(payload["createdAlerts"]) == 1
            assert payload["createdAlerts"][0]["eventType"] == (
                "model_modern_labels_benchmark_alert"
            )
            assert payload["createdAlerts"][0]["details"]["best_model_version"] == (
                "xgb-challenger-actionable-v1"
            )
            assert payload["createdAlerts"][0]["details"]["dataset_version"] == (
                "synthetic-labels-dataset-v1"
            )
            assert len(payload["acknowledgedAlerts"]) == 1
            assert payload["acknowledgedAlerts"][0]["status"] == "acknowledged"
            assert payload["reviewTask"] is not None
            assert payload["reviewTask"]["reviewType"] == "promotion_review"
            assert payload["reviewTask"]["status"] == "open"
            assert payload["reviewTask"]["candidateModelVersion"] == (
                "xgb-challenger-actionable-v1"
            )
            assert payload["reviewTask"]["activeModelVersion"] == "seed-linear-v1"
            assert payload["reviewTask"]["datasetVersion"] == (
                "synthetic-labels-dataset-v1"
            )
            assert payload["reviewTask"]["sourceEventType"] == (
                "model_modern_labels_benchmark_alert"
            )

            tasks_response = client.get(
                "/v1/admin/models/review-tasks",
                headers=headers,
            )
            assert tasks_response.status_code == 200
            tasks = tasks_response.json()
            assert len(tasks) == 1
            assert tasks[0]["id"] == payload["reviewTask"]["id"]
    finally:
        model_selection_module.ModelSelectionService.run_modern_labels_benchmark = (
            original_run
        )


def _build_synthetic_benchmark(
    *,
    best_model_version: str,
    active_model_version: str,
    eligible: bool,
    blocking_reasons: list[str] | None = None,
):
    from datetime import datetime, timezone

    from app.schemas.admin import (
        JobExecutionRead,
        RunModernLabelsBenchmarkResponse,
        TuneModelResponse,
    )

    now_utc = datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc)
    job = JobExecutionRead(
        id=1,
        jobType="model_selection",
        status="completed",
        startedAt=now_utc,
        completedAt=now_utc,
        details={},
    )
    selection = TuneModelResponse(
        job=job,
        selectionVersion="selection-unit-v1",
        artifactPath="/tmp/selection-unit-v1.json",
        datasetVersion="unit-labels-dataset-v1",
        candidateCount=1,
        bestModelVersion=best_model_version,
        promoted=False,
        activeModelVersion=active_model_version,
        promotionDecision={
            "eligible": eligible,
            "blocking_reasons": list(blocking_reasons or []),
            "active_model_version": active_model_version,
            "challenger_model_version": best_model_version,
        },
        bestCandidateComparison={},
        familyRollups=[],
        nestedEstimation={},
        candidates=[],
    )
    return RunModernLabelsBenchmarkResponse(
        preset="modern_labels_benchmark_v1",
        datasetResolution={
            "source": "provided",
            "dataset_version": "unit-labels-dataset-v1",
        },
        datasetExport=None,
        resolvedPolicy={},
        selection=selection,
    )


def test_build_modern_labels_benchmark_recommendation_flags_missing_best_candidate():
    from app.services.model_selection import ModelSelectionService

    benchmark = _build_synthetic_benchmark(
        best_model_version="",
        active_model_version="seed-linear-v1",
        eligible=False,
    )

    recommendation = (
        ModelSelectionService.build_modern_labels_benchmark_recommendation(
            benchmark
        )
    )

    assert recommendation["actionable"] is False
    assert recommendation["recommended_action"] == "no_action_required"
    assert recommendation["has_best_candidate"] is False
    assert "missing_best_candidate" in recommendation["skipped_reasons"]


def test_build_modern_labels_benchmark_recommendation_flags_challenger_matches_active():
    from app.services.model_selection import ModelSelectionService

    benchmark = _build_synthetic_benchmark(
        best_model_version="seed-linear-v1",
        active_model_version="seed-linear-v1",
        eligible=True,
    )

    recommendation = (
        ModelSelectionService.build_modern_labels_benchmark_recommendation(
            benchmark
        )
    )

    assert recommendation["actionable"] is False
    assert recommendation["best_candidate_differs_from_active"] is False
    assert "best_candidate_matches_active_model" in (
        recommendation["skipped_reasons"]
    )


def test_build_modern_labels_benchmark_recommendation_marks_actionable_when_all_conditions_met():
    from app.services.model_selection import ModelSelectionService

    benchmark = _build_synthetic_benchmark(
        best_model_version="xgb-challenger-v1",
        active_model_version="seed-linear-v1",
        eligible=True,
    )

    recommendation = (
        ModelSelectionService.build_modern_labels_benchmark_recommendation(
            benchmark
        )
    )

    assert recommendation["actionable"] is True
    assert recommendation["recommended_action"] == (
        "open_promotion_review_for_modern_labels_benchmark"
    )
    assert recommendation["skipped_reasons"] == []
    assert recommendation["best_model_version"] == "xgb-challenger-v1"
    assert recommendation["active_model_version"] == "seed-linear-v1"


def test_modern_labels_benchmark_review_skips_review_task_when_open_review_task_false(
    tmp_path, monkeypatch
):
    artifacts_path = tmp_path / "artifacts"
    datasets_path = tmp_path / "training-datasets"
    selection_runs_path = tmp_path / "selection-runs"
    monkeypatch.setenv("MODEL_ARTIFACTS_PATH", str(artifacts_path))
    monkeypatch.setenv("TRAINING_DATASETS_PATH", str(datasets_path))
    monkeypatch.setenv("MODEL_SELECTION_RUNS_PATH", str(selection_runs_path))

    write_spatial_only_labels_dataset(datasets_path)

    with create_test_client(tmp_path, monkeypatch) as client:
        headers = get_admin_headers(client)

        response = client.post(
            "/v1/admin/models/benchmark/modern-labels/review",
            json={
                "datasetVersion": "spatial-only-labels-dataset-v1",
                "version": "selection-modern-benchmark-review-disabled-v1",
                "promoteBest": False,
                "openReviewTask": False,
            },
            headers=headers,
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["createdAlerts"] == []
        assert payload["reviewTask"] is None
