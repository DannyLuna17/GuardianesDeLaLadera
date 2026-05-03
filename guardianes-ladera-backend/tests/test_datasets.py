from pathlib import Path

from app.ml.datasets import (
    DATASET_ID,
    TrainingDatasetRegistry,
    build_dataset_context,
    build_training_dataset,
    export_seed_training_dataset,
    rows_from_dataset,
)
from app.ml.training import build_seed_training_rows


def test_export_seed_training_dataset_writes_versioned_dataset(tmp_path: Path):
    dataset_path, dataset = export_seed_training_dataset(
        version="test-training-dataset-v1",
        datasets_path=tmp_path,
    )

    assert dataset_path.exists()
    assert dataset["dataset_id"] == DATASET_ID
    assert dataset["artifact_type"] == "training_dataset"
    assert dataset["version"] == "test-training-dataset-v1"
    assert dataset["summary"]["rows"] == 24
    assert dataset["summary"]["splits"]["train_rows"] > 0
    assert dataset["summary"]["splits"]["validation_rows"] > 0
    assert dataset["summary"]["validation_policy"]["strategy"] == "deterministic_zone_hash_holdout"
    assert dataset["summary"]["sampling_policy"]["strategy"] == "bootstrap_zone_snapshot"
    assert "spatialBlockId" in dataset["summary"]["row_context_fields"]
    assert dataset["provenance"]["validation_policy"]["validation_bucket"] == 0
    assert dataset["rows"][0]["context"]["spatialBlockId"]
    assert dataset["rows"][0]["context"]["eventGroupId"]
    assert dataset["rows"][0]["context"]["temporalHoldoutTag"]
    assert "zone_event_count" in dataset["feature_order"]


def test_training_dataset_registry_lists_and_loads_versions(tmp_path: Path):
    export_seed_training_dataset(version="dataset-b", datasets_path=tmp_path)
    export_seed_training_dataset(version="dataset-a", datasets_path=tmp_path)

    registry = TrainingDatasetRegistry(datasets_path=tmp_path)
    versions = registry.list_versions()
    dataset = registry.load("dataset-a")
    rows = rows_from_dataset(dataset)

    assert versions == ["dataset-a", "dataset-b"]
    assert dataset["version"] == "dataset-a"
    assert len(rows) == 24
    assert rows[0].phase in {"latest", "previous"}


def test_build_training_dataset_respects_explicit_row_splits():
    rows = build_seed_training_rows()[:4]
    time_window = {
        "kind": "static",
        "reference_at": "2026-03-25T00:00:00+00:00",
        "start_at": "2026-03-25T00:00:00+00:00",
        "end_at": "2026-03-25T00:00:00+00:00",
        "span_days": 0,
    }
    context = build_dataset_context(
        dataset_mode="labels",
        dataset_family="labels:field_validation",
        time_window=time_window,
        source="field_validation",
        label_source_families=["field_validation"],
    )
    dataset = build_training_dataset(
        version="explicit-splits-v1",
        rows=rows,
        description="Dataset with explicit train/validation splits.",
        provenance={"exported_at": "2026-04-19T00:00:00+00:00", **context},
        row_contexts=[
            {"spatialBlockId": "a", "temporalHoldoutTag": "bucket:2026-03-25"},
            {"spatialBlockId": "a", "temporalHoldoutTag": "bucket:2026-03-25"},
            {"spatialBlockId": "b", "temporalHoldoutTag": "bucket:2026-03-26"},
            {"spatialBlockId": "b", "temporalHoldoutTag": "bucket:2026-03-26"},
        ],
        row_splits=["train", "train", "validation", "validation"],
        validation_policy_override={
            "strategy": "temporal_holdout_backtest",
            "unit": "temporalHoldoutTag",
        },
        summary_extra={
            "dataset_family": context["dataset_family"],
            "time_window": context["time_window"],
            "dataset_taxonomy": context["dataset_taxonomy"],
            "evaluation_cohort": context["evaluation_cohort"],
        },
    )

    assert dataset["summary"]["splits"]["train_rows"] == 2
    assert dataset["summary"]["splits"]["validation_rows"] == 2
    assert dataset["summary"]["validation_policy"]["strategy"] == (
        "temporal_holdout_backtest"
    )
    assert dataset["rows"][2]["split"] == "validation"
