from app.ml.beta_regression import (
    predict_beta_regression_model,
    train_beta_regression_regressor,
)


def test_beta_regression_fits_simple_bounded_signal():
    model = train_beta_regression_regressor(
        feature_names=["x"],
        train_matrix=[[0.1], [0.2], [0.4], [0.6], [0.8], [0.9]],
        train_targets=[0.14, 0.19, 0.31, 0.58, 0.79, 0.86],
        alpha=0.5,
        max_iterations=200,
        learning_rate=0.2,
        tolerance=1e-6,
    )

    low_prediction = predict_beta_regression_model(model, {"x": 0.15})
    high_prediction = predict_beta_regression_model(model, {"x": 0.85})

    assert 0.0 < low_prediction < 1.0
    assert 0.0 < high_prediction < 1.0
    assert high_prediction > low_prediction
    assert model["precision"] >= 2.0


def test_beta_regression_returns_feature_contributions():
    model = train_beta_regression_regressor(
        feature_names=["x", "y"],
        train_matrix=[
            [0.1, 0.3],
            [0.2, 0.5],
            [0.4, 0.4],
            [0.6, 0.6],
            [0.8, 0.7],
            [0.9, 0.9],
        ],
        train_targets=[0.18, 0.26, 0.38, 0.57, 0.76, 0.88],
        alpha=0.75,
        max_iterations=250,
        learning_rate=0.2,
        tolerance=1e-6,
    )

    prediction, contributions, bias = predict_beta_regression_model(
        model,
        {"x": 0.7, "y": 0.8},
        with_contributions=True,
    )

    assert 0.0 < prediction < 1.0
    assert set(contributions) == {"x", "y"}
    assert isinstance(bias, float)
