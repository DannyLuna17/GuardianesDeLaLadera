from pathlib import Path

from app.ml.training import (
    build_seed_training_rows,
    export_additive_spline_artifact,
    export_beta_regression_artifact,
    export_seed_linear_artifact,
    export_xgboost_artifact,
)


def test_build_seed_training_rows_produces_latest_and_previous_samples():
    rows = build_seed_training_rows()

    assert len(rows) == 24
    assert {row.phase for row in rows} == {"latest", "previous"}
    zone_ids = {row.zone_id for row in rows}
    assert "moc-01" in zone_ids
    moc_latest = next(row for row in rows if row.zone_id == "moc-01" and row.phase == "latest")
    features = moc_latest.scoring_features()
    assert features["zone_event_count"] == 2.0
    assert features["intersecting_road_length_km"] == 70.0
    assert features["rain_overlay_peak_intensity"] == 3.0


def test_export_seed_linear_artifact_writes_trained_model(tmp_path: Path):
    artifact_path, artifact = export_seed_linear_artifact(
        version="test-trained-artifact",
        alpha=0.5,
        artifacts_path=tmp_path,
    )

    assert artifact_path.exists()
    assert artifact["artifact_type"] == "trained_linear_model"
    assert artifact["version"] == "test-trained-artifact"
    assert artifact["training"]["rows"] == 24
    assert "rain_72h" in artifact["coefficients"]
    assert "zone_event_count" in artifact["coefficients"]
    assert artifact["training"]["metrics"]["rmse"] >= 0.0
    assert artifact["training"]["splits"]["train_rows"] > 0
    assert artifact["training"]["splits"]["validation_rows"] > 0
    assert artifact["calibration"]["method"] in {"affine", "identity"}
    assert "validation_metrics_before" in artifact["calibration"]
    assert "validation_metrics_after" in artifact["calibration"]


def test_export_additive_spline_artifact_writes_interpretable_model(tmp_path: Path):
    artifact_path, artifact = export_additive_spline_artifact(
        version="test-additive-artifact",
        alpha=1.25,
        knot_count=3,
        artifacts_path=tmp_path,
    )

    assert artifact_path.exists()
    assert artifact["artifact_type"] == "additive_spline_model"
    assert artifact["model_family"] == "additive_spline"
    assert artifact["training"]["hyperparameters"]["alpha"] == 1.25
    assert artifact["training"]["hyperparameters"]["knot_count"] == 3
    assert artifact["basis_count"] >= len(artifact["feature_order"])
    assert "rain_72h" in artifact["feature_terms"]
    assert artifact["feature_importance"]["rain_72h"] >= 0.0


def test_export_beta_regression_artifact_writes_bounded_model(tmp_path: Path):
    artifact_path, artifact = export_beta_regression_artifact(
        version="test-beta-artifact",
        alpha=0.75,
        artifacts_path=tmp_path,
    )

    assert artifact_path.exists()
    assert artifact["artifact_type"] == "beta_regression_model"
    assert artifact["model_family"] == "beta_regression"
    assert artifact["training"]["hyperparameters"]["alpha"] == 0.75
    assert artifact["link_function"] == "logit"
    assert artifact["response_distribution"] == "beta"
    assert artifact["precision"] >= 2.0
    assert artifact["feature_importance"]["rain_72h"] >= 0.0


def test_export_xgboost_artifact_writes_boosted_model(tmp_path: Path):
    artifact_path, artifact = export_xgboost_artifact(
        version="test-xgboost-artifact",
        learning_rate=0.1,
        estimator_count=12,
        max_depth=3,
        min_leaf_size=1,
        early_stopping_rounds=4,
        artifacts_path=tmp_path,
    )

    assert artifact_path.exists()
    assert artifact["artifact_type"] == "xgboost_model"
    assert artifact["model_family"] == "xgboost"
    assert artifact["training"]["hyperparameters"]["learning_rate"] == 0.1
    assert artifact["training"]["hyperparameters"]["estimator_count"] == 12
    assert artifact["training"]["hyperparameters"]["subsample"] == 1.0
    assert artifact["effective_tree_count"] > 0
    assert artifact["feature_importance"]["rain_72h"] >= 0.0
