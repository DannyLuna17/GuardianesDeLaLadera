from app.ml.model_selection import build_model_selection_run
from app.ml.model_shadow import build_model_shadow_run


def test_model_selection_prefers_probability_tie_breakers_when_rmse_ties():
    candidates = [
        {
            "model_version": "candidate-a",
            "alpha": 0.75,
            "validation_rmse": 0.05,
            "validation_brier_score": 0.12,
            "validation_auprc": 0.62,
            "validation_recall": 0.7,
            "validation_mcc": 0.31,
            "validation_risk_level_accuracy": 0.9,
            "overall_rmse": 0.06,
        },
        {
            "model_version": "candidate-b",
            "alpha": 1.5,
            "validation_rmse": 0.05,
            "validation_brier_score": 0.08,
            "validation_auprc": 0.74,
            "validation_recall": 0.8,
            "validation_mcc": 0.45,
            "validation_risk_level_accuracy": 0.88,
            "overall_rmse": 0.06,
        },
    ]

    run = build_model_selection_run(
        version="selection-test-v1",
        dataset_version="dataset-v1",
        candidates=candidates,
        promoted=False,
        active_model_version="trained-spatial-seed-v1",
    )

    assert run["best_model_version"] == "candidate-b"
    assert run["candidates"][0]["rank"] == 1
    assert run["comparison_policy"]["tie_breakers"][:4] == [
        "validation_brier_score",
        "validation_auprc_desc",
        "validation_recall_desc",
        "validation_mcc_desc",
    ]


def test_model_shadow_prefers_probability_tie_breakers_when_rmse_ties():
    candidates = [
        {
            "model_version": "trained-spatial-seed-v1",
            "role": "active",
            "validation_rmse": 0.05,
            "validation_brier_score": 0.11,
            "validation_auprc": 0.63,
            "validation_recall": 0.71,
            "validation_mcc": 0.33,
            "validation_risk_level_accuracy": 0.89,
            "overall_rmse": 0.06,
        },
        {
            "model_version": "challenger-v1",
            "role": "challenger",
            "validation_rmse": 0.05,
            "validation_brier_score": 0.07,
            "validation_auprc": 0.79,
            "validation_recall": 0.81,
            "validation_mcc": 0.47,
            "validation_risk_level_accuracy": 0.88,
            "overall_rmse": 0.06,
        },
    ]

    run = build_model_shadow_run(
        version="shadow-test-v1",
        dataset_version="dataset-v1",
        dataset_context={"dataset_mode": "labels"},
        active_model_version="trained-spatial-seed-v1",
        candidates=candidates,
    )

    assert run["best_model_version"] == "challenger-v1"
    assert run["active_still_best"] is False
    assert run["comparison_policy"]["tie_breakers"][:4] == [
        "validation_brier_score",
        "validation_auprc_desc",
        "validation_recall_desc",
        "validation_mcc_desc",
    ]


def test_model_selection_supports_candidates_without_alpha():
    run = build_model_selection_run(
        version="selection-mixed-v1",
        dataset_version="dataset-v1",
        candidates=[
            {
                "model_version": "candidate-linear",
                "alpha": 0.75,
                "validation_rmse": 0.05,
                "validation_brier_score": 0.11,
                "validation_auprc": 0.65,
                "validation_recall": 0.75,
                "validation_mcc": 0.35,
                "validation_risk_level_accuracy": 0.9,
                "overall_rmse": 0.06,
            },
            {
                "model_version": "candidate-gbt",
                "alpha": None,
                "validation_rmse": 0.05,
                "validation_brier_score": 0.09,
                "validation_auprc": 0.78,
                "validation_recall": 0.8,
                "validation_mcc": 0.44,
                "validation_risk_level_accuracy": 0.88,
                "overall_rmse": 0.06,
            },
        ],
        promoted=False,
        active_model_version="trained-spatial-seed-v1",
    )

    assert run["best_model_version"] == "candidate-gbt"
