from app.ml.additive_splines import (
    predict_additive_spline_regressor,
    train_additive_spline_regressor,
)


def test_additive_spline_regressor_fits_simple_monotonic_signal():
    model = train_additive_spline_regressor(
        feature_names=["x"],
        train_matrix=[[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]],
        train_targets=[0.05, 0.08, 0.1, 0.82, 0.88, 0.92],
        alpha=0.4,
        knot_count=2,
    )

    low_prediction = predict_additive_spline_regressor(model, {"x": 0.1})
    high_prediction = predict_additive_spline_regressor(model, {"x": 0.9})

    assert float(high_prediction) > float(low_prediction)
    assert model["feature_importance"]["x"] > 0
    assert model["feature_terms"]["x"]["direction"] in {"positive", "mixed"}
    assert model["basis_count"] == 3


def test_additive_spline_regressor_returns_feature_contributions():
    model = train_additive_spline_regressor(
        feature_names=["rain_72h", "slope_deg"],
        train_matrix=[
            [50.0, 20.0],
            [80.0, 24.0],
            [120.0, 28.0],
            [160.0, 34.0],
        ],
        train_targets=[0.18, 0.25, 0.58, 0.82],
        alpha=0.6,
        knot_count=1,
    )

    prediction, contributions = predict_additive_spline_regressor(
        model,
        {"rain_72h": 150.0, "slope_deg": 32.0},
        with_contributions=True,
    )

    baseline = float(model["intercept"])
    total = baseline + sum(float(value) for value in contributions.values())
    assert round(float(prediction) - total, 6) == 0.0
    assert set(contributions) == {"rain_72h", "slope_deg"}
