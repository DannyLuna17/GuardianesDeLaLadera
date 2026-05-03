from app.ml.datasets import build_seed_training_dataset
from app.ml.model_evaluations import build_model_evaluation, export_model_evaluation
from app.ml.model_registry import ModelRegistry
from app.ml.training import export_additive_spline_artifact
from app.ml.training import export_beta_regression_artifact
from app.ml.training import export_xgboost_artifact


def test_build_model_evaluation_exports_metrics_and_top_errors(tmp_path):
    artifact = ModelRegistry().load("trained-spatial-seed-v1")
    dataset = build_seed_training_dataset(version="evaluation-seed-dataset-v1")

    evaluation = build_model_evaluation(
        version="trained-spatial-seed-v1-on-seed-dataset-v1",
        artifact=artifact,
        dataset=dataset,
        top_error_count=4,
    )
    evaluation_path, saved_evaluation = export_model_evaluation(
        evaluation, evaluations_path=tmp_path
    )

    assert evaluation_path.exists()
    assert saved_evaluation["artifact_type"] == "model_evaluation"
    assert saved_evaluation["model_version"] == "trained-spatial-seed-v1"
    assert saved_evaluation["dataset_version"] == "evaluation-seed-dataset-v1"
    assert saved_evaluation["metrics"]["overall"]["rows"] == 24
    assert "calibrated_metrics" in saved_evaluation["metrics"]["validation"]
    assert "probability_metrics" in saved_evaluation["metrics"]["validation"]
    assert "raw_probability_metrics" in saved_evaluation["metrics"]["validation"]
    assert "risk_level_accuracy" in saved_evaluation["metrics"]["overall"]
    validation_probability = saved_evaluation["metrics"]["validation"][
        "probability_metrics"
    ]
    assert validation_probability["positive_threshold"] == 0.5
    assert validation_probability["brier_score"] >= 0.0
    assert validation_probability["ece"] >= 0.0
    diagnostics = saved_evaluation["diagnostics"]
    assert diagnostics["feature_importance"]["method"] == "absolute_linear_coefficient"
    assert diagnostics["feature_importance"]["top_features"]
    assert diagnostics["calibration_effect"]["validation_rmse_improvement"] > 0
    assert diagnostics["validation_slices"]["by_phase"]["latest"]["rows"] > 0
    assert diagnostics["validation_slices"]["by_spatial_block"]
    assert diagnostics["validation_slices"]["by_temporal_holdout_tag"]
    assert (
        "probability_metrics"
        in diagnostics["validation_slices"]["by_phase"]["latest"]
    )
    assert len(saved_evaluation["top_errors"]) == 4
    assert (
        saved_evaluation["top_errors"][0]["absError"]
        >= saved_evaluation["top_errors"][-1]["absError"]
    )


def test_build_model_evaluation_supports_additive_spline_artifacts(tmp_path):
    artifact_path, artifact = export_additive_spline_artifact(
        version="eval-additive-spline-v1",
        alpha=1.1,
        knot_count=3,
        artifacts_path=tmp_path,
    )
    dataset = build_seed_training_dataset(version="evaluation-additive-dataset-v1")

    evaluation = build_model_evaluation(
        version="eval-additive-spline-v1-on-seed-dataset-v1",
        artifact=artifact,
        dataset=dataset,
        top_error_count=3,
    )

    assert artifact_path.exists()
    assert evaluation["model_summary"]["artifact_type"] == "additive_spline_model"
    assert evaluation["model_summary"]["model_family"] == "additive_spline"
    assert evaluation["metrics"]["validation"]["calibrated_metrics"]["rmse"] >= 0.0
    diagnostics = evaluation["diagnostics"]
    assert (
        diagnostics["feature_importance"]["method"]
        == "mean_abs_additive_contribution"
    )
    assert diagnostics["feature_importance"]["top_features"]
    assert diagnostics["calibration_effect"]["method"] in {"affine", "identity"}
    assert diagnostics["validation_slices"]["by_spatial_block"]
    assert diagnostics["validation_slices"]["by_temporal_holdout_tag"]
    assert len(evaluation["top_errors"]) == 3


def test_build_model_evaluation_supports_beta_regression_artifacts(tmp_path):
    artifact_path, artifact = export_beta_regression_artifact(
        version="eval-beta-regression-v1",
        alpha=0.75,
        artifacts_path=tmp_path,
    )
    dataset = build_seed_training_dataset(version="evaluation-beta-dataset-v1")

    evaluation = build_model_evaluation(
        version="eval-beta-regression-v1-on-seed-dataset-v1",
        artifact=artifact,
        dataset=dataset,
        top_error_count=3,
    )

    assert artifact_path.exists()
    assert evaluation["model_summary"]["artifact_type"] == "beta_regression_model"
    assert evaluation["model_summary"]["model_family"] == "beta_regression"
    assert evaluation["metrics"]["validation"]["calibrated_metrics"]["rmse"] >= 0.0
    diagnostics = evaluation["diagnostics"]
    assert diagnostics["feature_importance"]["method"] == "absolute_logit_coefficient"
    assert diagnostics["feature_importance"]["top_features"]
    assert diagnostics["validation_slices"]["by_spatial_block"]
    assert diagnostics["validation_slices"]["by_temporal_holdout_tag"]
    assert len(evaluation["top_errors"]) == 3


def test_build_model_evaluation_supports_xgboost_artifacts(tmp_path):
    artifact_path, artifact = export_xgboost_artifact(
        version="eval-xgboost-v1",
        learning_rate=0.1,
        estimator_count=12,
        max_depth=3,
        min_leaf_size=1,
        early_stopping_rounds=4,
        artifacts_path=tmp_path,
    )
    dataset = build_seed_training_dataset(version="evaluation-xgboost-dataset-v1")

    evaluation = build_model_evaluation(
        version="eval-xgboost-v1-on-seed-dataset-v1",
        artifact=artifact,
        dataset=dataset,
        top_error_count=3,
    )

    assert artifact_path.exists()
    assert evaluation["model_summary"]["artifact_type"] == "xgboost_model"
    assert evaluation["model_summary"]["model_family"] == "xgboost"
    assert evaluation["metrics"]["validation"]["calibrated_metrics"]["rmse"] >= 0.0
    diagnostics = evaluation["diagnostics"]
    assert diagnostics["feature_importance"]["method"] == "gain"
    assert diagnostics["feature_importance"]["top_features"]
    assert diagnostics["validation_slices"]["by_spatial_block"]
    assert diagnostics["validation_slices"]["by_temporal_holdout_tag"]
    assert len(evaluation["top_errors"]) == 3
