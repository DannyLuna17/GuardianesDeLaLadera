from app.ml.tree_boosting import (
    feature_gain_importance,
    predict_gradient_boosted_ensemble,
    train_gradient_boosted_regressor,
)


def test_gradient_boosted_regressor_fits_simple_nonlinear_signal():
    feature_names = ["x"]
    train_matrix = [[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]]
    train_targets = [0.05, 0.08, 0.1, 0.82, 0.88, 0.91]
    validation_matrix = [[0.15], [0.85]]
    validation_targets = [0.09, 0.86]

    model = train_gradient_boosted_regressor(
        feature_names=feature_names,
        train_matrix=train_matrix,
        train_targets=train_targets,
        validation_matrix=validation_matrix,
        validation_targets=validation_targets,
        learning_rate=0.2,
        estimator_count=20,
        max_depth=2,
        min_leaf_size=1,
        min_split_gain=0.0,
        early_stopping_rounds=4,
    )

    assert model["effective_tree_count"] > 0
    assert model["effective_tree_count"] <= 20
    assert model["feature_importance"]["x"] > 0

    low_prediction = predict_gradient_boosted_ensemble(
        {
            "base_score": model["base_score"],
            "learning_rate": model["learning_rate"],
            "feature_order": feature_names,
            "trees": model["trees"],
        },
        {"x": 0.1},
    )
    high_prediction = predict_gradient_boosted_ensemble(
        {
            "base_score": model["base_score"],
            "learning_rate": model["learning_rate"],
            "feature_order": feature_names,
            "trees": model["trees"],
        },
        {"x": 0.9},
    )

    assert float(high_prediction) > float(low_prediction)


def test_gradient_boosted_ensemble_returns_feature_contributions():
    ensemble = {
        "base_score": 0.5,
        "learning_rate": 0.3,
        "feature_order": ["rain_72h", "slope_deg"],
        "trees": [
            {
                "node_type": "split",
                "feature": "rain_72h",
                "feature_index": 0,
                "threshold": 100.0,
                "gain": 0.2,
                "left": {"node_type": "leaf", "value": -0.1},
                "right": {"node_type": "leaf", "value": 0.2},
            }
        ],
    }

    prediction, contributions = predict_gradient_boosted_ensemble(
        ensemble,
        {"rain_72h": 140.0, "slope_deg": 28.0},
        with_contributions=True,
    )

    assert round(float(prediction), 6) == 0.56
    assert contributions["rain_72h"] > 0
    assert contributions["slope_deg"] == 0.0
    assert feature_gain_importance(ensemble)["rain_72h"] == 0.2
