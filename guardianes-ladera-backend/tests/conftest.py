"""Shared pytest guardrails for the backend suite."""

from __future__ import annotations

import pytest


GENERATED_ML_OUTPUT_ENVS = {
    "MODEL_ARTIFACTS_PATH": "artifacts",
    "ACTIVE_MODEL_MANIFEST_PATH": "artifacts/active-model.json",
    "TRAINING_DATASETS_PATH": "training-datasets",
    "MODEL_EVALUATIONS_PATH": "evaluations",
    "MODEL_SELECTION_RUNS_PATH": "selection-runs",
    "MODEL_DRIFT_REPORTS_PATH": "drift-reports",
    "MODEL_SHADOW_RUNS_PATH": "shadow-runs",
}


@pytest.fixture(autouse=True)
def isolate_generated_ml_outputs(tmp_path, monkeypatch):
    """Route generated ML artifacts to per-test temp folders by default.

    Individual tests may still override these paths with their own
    ``monkeypatch.setenv`` calls. This fixture only provides a safe baseline
    so a missing override cannot write benchmark artifacts into ``app/ml``.
    """
    output_root = tmp_path / "generated-ml"
    for env_name, relative_path in GENERATED_ML_OUTPUT_ENVS.items():
        monkeypatch.setenv(env_name, str(output_root / relative_path))

    from app.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

