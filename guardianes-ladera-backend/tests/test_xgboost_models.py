import json

import app.ml.xgboost_models as xgboost_models
from app.ml.xgboost_models import (
    predict_xgboost_model,
    train_xgboost_regressor,
)


def test_xgboost_regressor_fits_simple_nonlinear_signal():
    model = train_xgboost_regressor(
        feature_names=["x"],
        train_matrix=[[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]],
        train_targets=[0.05, 0.08, 0.1, 0.82, 0.88, 0.91],
        validation_matrix=[[0.15], [0.85]],
        validation_targets=[0.09, 0.86],
        learning_rate=0.2,
        estimator_count=16,
        max_depth=2,
        min_leaf_size=1,
        min_split_gain=0.0,
        early_stopping_rounds=4,
    )

    low_prediction = predict_xgboost_model(model, {"x": 0.1})
    high_prediction = predict_xgboost_model(model, {"x": 0.9})

    assert model["effective_tree_count"] > 0
    assert model["effective_tree_count"] <= model["requested_tree_count"]
    assert model["feature_importance"]["x"] > 0
    assert float(high_prediction) > float(low_prediction)


def test_xgboost_regressor_returns_feature_contributions():
    model = train_xgboost_regressor(
        feature_names=["rain_72h", "slope_deg"],
        train_matrix=[
            [50.0, 20.0],
            [80.0, 24.0],
            [120.0, 28.0],
            [160.0, 34.0],
            [200.0, 38.0],
        ],
        train_targets=[0.18, 0.25, 0.58, 0.82, 0.91],
        validation_matrix=[[90.0, 25.0], [180.0, 36.0]],
        validation_targets=[0.32, 0.87],
        learning_rate=0.15,
        estimator_count=12,
        max_depth=2,
        min_leaf_size=1,
        min_split_gain=0.0,
        early_stopping_rounds=3,
    )

    prediction, contributions, bias = predict_xgboost_model(
        model,
        {"rain_72h": 170.0, "slope_deg": 35.0},
        with_contributions=True,
    )

    total = bias + sum(float(value) for value in contributions.values())
    assert abs(float(prediction) - total) < 1e-5
    assert set(contributions) == {"rain_72h", "slope_deg"}


def test_xgboost_regressor_preserves_zero_best_iteration(monkeypatch):
    class FakeBooster:
        best_iteration = 0

        @staticmethod
        def save_config():
            return json.dumps(
                {"learner": {"learner_model_param": {"base_score": "0.5"}}}
            )

        @staticmethod
        def save_raw(*, raw_format):
            assert raw_format == "json"
            return b"{}"

        @staticmethod
        def get_score(*, importance_type):
            assert importance_type == "gain"
            return {"x": 1.0}

    def fake_train(*args, **kwargs):
        kwargs["evals_result"]["train"] = {"rmse": [0.12]}
        kwargs["evals_result"]["validation"] = {"rmse": [0.11]}
        return FakeBooster()

    monkeypatch.setattr(xgboost_models.xgb, "train", fake_train)

    model = train_xgboost_regressor(
        feature_names=["x"],
        train_matrix=[[0.0], [1.0]],
        train_targets=[0.1, 0.9],
        validation_matrix=[[0.5]],
        validation_targets=[0.5],
        estimator_count=24,
        early_stopping_rounds=4,
    )

    assert model["best_iteration"] == 0
    assert model["effective_tree_count"] == 1
    assert model["requested_tree_count"] == 24
