from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import ApiError
from app.ml.features import SCORING_FEATURE_ORDER, ZoneFeatureSnapshot
from app.ml.training import (
    TRAINING_REFERENCE_AT,
    TrainingRow,
    build_seed_training_rows,
    split_training_rows,
)


DATASET_ID = "spatial-risk-training-dataset"


def _row_split(zone_id: str, validation_zone_ids: set[str]) -> str:
    return "validation" if zone_id in validation_zone_ids else "train"


def _spatial_block_id(zone_id: str) -> str:
    if "-" in zone_id:
        return zone_id.split("-", 1)[0]
    return zone_id


def _event_group_id(row: TrainingRow, context: dict[str, Any]) -> str:
    if context.get("labelId") is not None:
        return f"label:{context['labelId']}"
    if context.get("runId") is not None:
        return f"run:{context['runId']}"
    if context.get("featureRunId") is not None:
        return f"feature-run:{context['featureRunId']}"
    return f"phase:{row.phase}:zone:{row.zone_id}"


def _temporal_holdout_tag(
    *, split: str, row: TrainingRow, context: dict[str, Any]
) -> str:
    timestamp_candidates = [
        context.get("observedAt"),
        context.get("featureRunCompletedAt"),
        context.get("runCompletedAt"),
    ]
    parsed = _parse_iso_datetimes(
        [str(value) for value in timestamp_candidates if value is not None]
    )
    if parsed:
        return f"{split}:{parsed[0].date().isoformat()}"
    return f"{split}:{row.phase}"


def _enriched_row_context(
    *,
    row: TrainingRow,
    split: str,
    context: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized_context = dict(context or {})
    normalized_context.setdefault("spatialBlockId", _spatial_block_id(row.zone_id))
    normalized_context.setdefault(
        "eventGroupId",
        _event_group_id(row, normalized_context),
    )
    normalized_context.setdefault(
        "temporalHoldoutTag",
        _temporal_holdout_tag(split=split, row=row, context=normalized_context),
    )
    normalized_context.setdefault("splitUnit", "zone_id")
    normalized_context.setdefault("validationStrategy", "deterministic_zone_hash_holdout")
    return normalized_context


def _validation_policy() -> dict[str, Any]:
    return {
        "strategy": "deterministic_zone_hash_holdout",
        "unit": "zone_id",
        "bucket_count": 3,
        "validation_bucket": 0,
        "leakage_guard": "zone_level_entity_holdout",
    }


def _sampling_policy(dataset_mode: str) -> dict[str, Any]:
    if dataset_mode == "seed":
        return {
            "strategy": "bootstrap_zone_snapshot",
            "label_source": "synthetic_seed_scores",
            "feature_source": "seed_spatial_snapshot",
        }
    if dataset_mode == "operational":
        return {
            "strategy": "prediction_history_snapshot",
            "label_source": "historical_model_scores",
            "feature_source": "prediction_trace_feature_snapshot",
        }
    if dataset_mode == "labels":
        return {
            "strategy": "confirmed_outcome_labels_with_linked_features",
            "label_source": "governed_zone_outcome_labels",
            "feature_source": "linked_prediction_runs_or_backfill",
        }
    return {
        "strategy": "unknown",
        "label_source": "unknown",
        "feature_source": "unknown",
    }


def serialize_training_row(row: TrainingRow, *, split: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "zoneId": row.zone_id,
        "phase": row.phase,
        "split": split,
        "targetScore": row.target_score,
        "drivers": row.drivers,
        "featureSnapshot": row.feature_snapshot.as_dict(),
        "featureVector": row.scoring_features(),
    }
    if context:
        payload["context"] = context
    return payload


def feature_snapshot_from_mapping(item: dict[str, Any]) -> ZoneFeatureSnapshot:
    return ZoneFeatureSnapshot(
        municipality_event_count=int(item["municipality_event_count"]),
        zone_event_count=int(item["zone_event_count"]),
        recent_zone_event_count=int(item["recent_zone_event_count"]),
        intersecting_road_count=int(item["intersecting_road_count"]),
        intersecting_road_length_km=float(item["intersecting_road_length_km"]),
        rain_overlay_count=int(item["rain_overlay_count"]),
        rain_overlay_peak_intensity=int(item["rain_overlay_peak_intensity"]),
        rain_overlay_peak_label=item.get("rain_overlay_peak_label"),
    )


def deserialize_training_row(item: dict[str, Any]) -> TrainingRow:
    feature_snapshot = feature_snapshot_from_mapping(item["featureSnapshot"])
    return TrainingRow(
        zone_id=item["zoneId"],
        phase=item["phase"],
        target_score=float(item["targetScore"]),
        drivers=dict(item["drivers"]),
        feature_snapshot=feature_snapshot,
    )


def rows_from_dataset(dataset: dict[str, Any]) -> list[TrainingRow]:
    return [deserialize_training_row(item) for item in dataset["rows"]]


def _parse_iso_datetimes(values: list[str | None]) -> list[datetime]:
    timestamps: list[datetime] = []
    for value in values:
        if not value:
            continue
        normalized = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        timestamps.append(parsed)
    return timestamps


def _build_time_window(
    *,
    kind: str,
    timestamps: list[str | None] | None = None,
    fallback_reference: datetime | None = None,
) -> dict[str, Any]:
    parsed_timestamps = _parse_iso_datetimes(timestamps or [])
    if not parsed_timestamps and fallback_reference is not None:
        parsed_timestamps = [fallback_reference]
    if not parsed_timestamps:
        return {
            "kind": kind,
            "reference_at": None,
            "start_at": None,
            "end_at": None,
            "span_days": None,
        }
    start_at = min(parsed_timestamps)
    end_at = max(parsed_timestamps)
    return {
        "kind": kind,
        "reference_at": end_at.isoformat(),
        "start_at": start_at.isoformat(),
        "end_at": end_at.isoformat(),
        "span_days": (end_at.date() - start_at.date()).days,
    }


def _label_source_families(label_sources: list[str]) -> list[str]:
    families = {
        str(source).split(":", 1)[0].strip() for source in label_sources if str(source).strip()
    }
    return sorted(families)


def _dataset_family_parts(dataset_family: str | None) -> tuple[str, str | None]:
    normalized = str(dataset_family or "unknown").strip() or "unknown"
    family_root, _, family_variant = normalized.partition(":")
    return family_root or "unknown", family_variant or None


def _label_signal_kind(label_source_families: list[str]) -> str:
    if not label_source_families:
        return "unknown"
    signal_mapping = {
        "field_validation": "direct",
        "historical_event": "inventory",
        "ungrd": "proxy",
    }
    signal_kinds = {
        signal_mapping.get(str(family).strip(), "observed")
        for family in label_source_families
        if str(family).strip()
    }
    if not signal_kinds:
        return "unknown"
    if len(signal_kinds) == 1:
        return next(iter(signal_kinds))
    return "mixed"


def build_dataset_taxonomy(
    *,
    dataset_mode: str,
    dataset_family: str,
    source: str | None = None,
    label_source_families: list[str] | None = None,
) -> dict[str, Any]:
    family_root, family_variant = _dataset_family_parts(dataset_family)
    source_family = str(source or family_variant or dataset_mode).split(":", 1)[0].strip() or "unknown"
    source_families = sorted(
        {
            str(item).strip()
            for item in (label_source_families or [source_family])
            if str(item).strip()
        }
    )

    supervision_tier = "unknown"
    signal_type = "unknown"
    geographic_granularity = "zone"

    if dataset_mode == "seed":
        supervision_tier = "bootstrap"
        signal_type = "synthetic"
        geographic_granularity = "zone"
    elif dataset_mode == "operational":
        supervision_tier = "proxy"
        signal_type = "predicted"
        geographic_granularity = "zone"
    elif dataset_mode == "labels":
        supervision_tier = "observed"
        signal_type = _label_signal_kind(source_families)
        if signal_type == "proxy":
            geographic_granularity = "municipality"
        elif signal_type == "mixed":
            geographic_granularity = "mixed"
        else:
            geographic_granularity = "zone"

    taxonomy_group = (
        f"{dataset_mode}:{supervision_tier}:{geographic_granularity}:{signal_type}"
    )
    return {
        "family_root": family_root,
        "family_variant": family_variant,
        "source": source,
        "source_family": source_family,
        "source_families": source_families,
        "supervision_tier": supervision_tier,
        "signal_type": signal_type,
        "geographic_granularity": geographic_granularity,
        "taxonomy_group": taxonomy_group,
        "stability_group": taxonomy_group,
    }


def _parse_time_window_reference(time_window: dict[str, Any]) -> datetime | None:
    candidates = [
        time_window.get("reference_at"),
        time_window.get("end_at"),
        time_window.get("start_at"),
    ]
    parsed_candidates = _parse_iso_datetimes([value for value in candidates if value])
    if not parsed_candidates:
        return None
    return parsed_candidates[0]


def _next_month_start(moment: datetime) -> datetime:
    if moment.month == 12:
        return datetime(moment.year + 1, 1, 1, tzinfo=timezone.utc)
    return datetime(moment.year, moment.month + 1, 1, tzinfo=timezone.utc)


def _next_quarter_start(moment: datetime) -> datetime:
    quarter = ((moment.month - 1) // 3) + 1
    if quarter == 4:
        return datetime(moment.year + 1, 1, 1, tzinfo=timezone.utc)
    month = quarter * 3 + 1
    return datetime(moment.year, month, 1, tzinfo=timezone.utc)


def _bucket_bounds(reference_at: datetime, bucket_type: str) -> tuple[datetime, datetime, str, int]:
    if bucket_type == "static":
        start_at = datetime(
            reference_at.year,
            reference_at.month,
            reference_at.day,
            tzinfo=timezone.utc,
        )
        end_at = start_at + timedelta(days=1) - timedelta(seconds=1)
        return start_at, end_at, start_at.date().isoformat(), start_at.toordinal()

    if bucket_type == "month":
        start_at = datetime(reference_at.year, reference_at.month, 1, tzinfo=timezone.utc)
        end_at = _next_month_start(start_at) - timedelta(seconds=1)
        bucket_label = f"{reference_at.year:04d}-{reference_at.month:02d}"
        bucket_index = reference_at.year * 12 + reference_at.month
        return start_at, end_at, bucket_label, bucket_index

    if bucket_type == "quarter":
        quarter = ((reference_at.month - 1) // 3) + 1
        start_month = 3 * (quarter - 1) + 1
        start_at = datetime(reference_at.year, start_month, 1, tzinfo=timezone.utc)
        end_at = _next_quarter_start(start_at) - timedelta(seconds=1)
        bucket_label = f"{reference_at.year:04d}-Q{quarter}"
        bucket_index = reference_at.year * 4 + quarter
        return start_at, end_at, bucket_label, bucket_index

    start_at = datetime(reference_at.year, 1, 1, tzinfo=timezone.utc)
    end_at = datetime(reference_at.year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(
        seconds=1
    )
    bucket_label = f"{reference_at.year:04d}"
    return start_at, end_at, bucket_label, reference_at.year


def _select_cohort_bucket_type(
    *,
    dataset_mode: str,
    time_window: dict[str, Any],
    dataset_taxonomy: dict[str, Any],
) -> str:
    span_days = time_window.get("span_days")
    span_days = int(span_days) if isinstance(span_days, int | float) else None
    signal_type = str(dataset_taxonomy.get("signal_type") or "unknown")

    if dataset_mode == "seed":
        return "static"
    if dataset_mode == "operational":
        if span_days is not None and span_days > 62:
            return "quarter"
        return "month"
    if dataset_mode == "labels":
        if signal_type == "direct":
            if span_days is not None and span_days > 62:
                return "quarter"
            return "month"
        if signal_type in {"inventory", "proxy", "mixed"}:
            if span_days is not None and span_days > 180:
                return "year"
            return "quarter"
        if span_days is not None and span_days > 120:
            return "quarter"
        return "month"
    if span_days is not None and span_days > 120:
        return "quarter"
    return "month"


def build_evaluation_cohort(
    *,
    dataset_mode: str,
    dataset_taxonomy: dict[str, Any],
    time_window: dict[str, Any],
) -> dict[str, Any]:
    reference_at = _parse_time_window_reference(time_window)
    if reference_at is None:
        return {
            "cohort_key": None,
            "cohort_group": None,
            "bucket_type": None,
            "bucket_label": None,
            "bucket_index": None,
            "bucket_start_at": None,
            "bucket_end_at": None,
            "reference_at": None,
        }

    bucket_type = _select_cohort_bucket_type(
        dataset_mode=dataset_mode,
        time_window=time_window,
        dataset_taxonomy=dataset_taxonomy,
    )
    bucket_start_at, bucket_end_at, bucket_label, bucket_index = _bucket_bounds(
        reference_at, bucket_type
    )
    cohort_group = f"{dataset_taxonomy.get('taxonomy_group') or 'unknown'}:{bucket_type}"
    return {
        "cohort_key": f"{cohort_group}:{bucket_label}",
        "cohort_group": cohort_group,
        "bucket_type": bucket_type,
        "bucket_label": bucket_label,
        "bucket_index": bucket_index,
        "bucket_start_at": bucket_start_at.isoformat(),
        "bucket_end_at": bucket_end_at.isoformat(),
        "reference_at": reference_at.isoformat(),
    }


def build_dataset_context(
    *,
    dataset_mode: str,
    dataset_family: str,
    time_window: dict[str, Any],
    source: str | None = None,
    label_source_families: list[str] | None = None,
) -> dict[str, Any]:
    dataset_taxonomy = build_dataset_taxonomy(
        dataset_mode=dataset_mode,
        dataset_family=dataset_family,
        source=source,
        label_source_families=label_source_families,
    )
    evaluation_cohort = build_evaluation_cohort(
        dataset_mode=dataset_mode,
        dataset_taxonomy=dataset_taxonomy,
        time_window=time_window,
    )
    return {
        "dataset_mode": dataset_mode,
        "dataset_family": dataset_family,
        "time_window": time_window,
        "source": source,
        "label_source_families": label_source_families or [],
        "dataset_taxonomy": dataset_taxonomy,
        "evaluation_cohort": evaluation_cohort,
    }


def normalize_dataset_context(context: dict[str, Any]) -> dict[str, Any]:
    dataset_mode = str(
        context.get("dataset_mode") or context.get("source") or "unknown"
    )
    dataset_family = str(context.get("dataset_family") or "unknown")
    time_window = dict(context.get("time_window") or {})
    source = context.get("source")

    explicit_taxonomy = dict(context.get("dataset_taxonomy") or {})
    label_source_families = list(
        context.get("label_source_families")
        or explicit_taxonomy.get("source_families")
        or []
    )
    computed_taxonomy = build_dataset_taxonomy(
        dataset_mode=dataset_mode,
        dataset_family=dataset_family,
        source=source,
        label_source_families=label_source_families,
    )
    dataset_taxonomy = {**computed_taxonomy, **explicit_taxonomy}

    explicit_cohort = dict(context.get("evaluation_cohort") or {})
    computed_cohort = build_evaluation_cohort(
        dataset_mode=dataset_mode,
        dataset_taxonomy=dataset_taxonomy,
        time_window=time_window,
    )
    evaluation_cohort = {**computed_cohort, **explicit_cohort}

    return {
        "dataset_mode": dataset_mode,
        "dataset_family": dataset_family,
        "time_window": time_window,
        "source": source,
        "label_source_families": label_source_families,
        "dataset_taxonomy": dataset_taxonomy,
        "evaluation_cohort": evaluation_cohort,
    }


def build_training_dataset(
    *,
    version: str,
    rows: list[TrainingRow],
    description: str,
    provenance: dict[str, Any],
    row_contexts: list[dict[str, Any] | None] | None = None,
    row_splits: list[str] | None = None,
    validation_policy_override: dict[str, Any] | None = None,
    summary_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ApiError(400, "training_dataset_empty", "The requested training dataset has no rows to export.")

    if row_splits is not None and len(row_splits) != len(rows):
        raise ApiError(
            400,
            "training_dataset_split_mismatch",
            "Explicit row_splits must align one-to-one with rows.",
        )
    if row_splits is not None and any(split not in {"train", "validation"} for split in row_splits):
        raise ApiError(
            400,
            "training_dataset_split_invalid",
            "Explicit row_splits may only contain 'train' or 'validation'.",
        )

    resolved_row_splits = list(row_splits or [])
    _, validation_rows = split_training_rows(rows)
    validation_zone_ids = {row.zone_id for row in validation_rows}
    zone_ids = sorted({row.zone_id for row in rows})
    phase_breakdown: dict[str, int] = {}
    serialized_rows: list[dict[str, Any]] = []
    dataset_mode = str(provenance.get("dataset_mode") or provenance.get("source") or "unknown")
    validation_policy = dict(validation_policy_override or _validation_policy())
    sampling_policy = _sampling_policy(dataset_mode)
    resolved_train_rows = 0
    resolved_validation_rows = 0

    for index, row in enumerate(rows):
        phase_breakdown[row.phase] = phase_breakdown.get(row.phase, 0) + 1
        split = (
            resolved_row_splits[index]
            if row_splits is not None
            else _row_split(row.zone_id, validation_zone_ids)
        )
        if split == "validation":
            resolved_validation_rows += 1
        else:
            resolved_train_rows += 1
        context = _enriched_row_context(
            row=row,
            split=split,
            context=row_contexts[index] if row_contexts is not None else None,
        )
        serialized_rows.append(
            serialize_training_row(
                row,
                split=split,
                context=context,
            )
        )

    row_context_fields = sorted(
        {
            key
            for item in serialized_rows
            for key in (item.get("context") or {})
        }
    )
    summary = {
        "rows": len(rows),
        "zones": len(zone_ids),
        "splits": {
            "train_rows": resolved_train_rows,
            "validation_rows": resolved_validation_rows,
        },
        "phase_breakdown": phase_breakdown,
        "zone_ids": zone_ids,
        "validation_policy": validation_policy,
        "sampling_policy": sampling_policy,
        "row_context_fields": row_context_fields,
    }
    if summary_extra:
        summary.update(summary_extra)

    provenance_payload = {
        **provenance,
        "validation_policy": validation_policy,
        "sampling_policy": sampling_policy,
    }
    return {
        "dataset_id": DATASET_ID,
        "version": version,
        "artifact_type": "training_dataset",
        "description": description,
        "label_name": "target_score",
        "feature_order": SCORING_FEATURE_ORDER,
        "rows": serialized_rows,
        "summary": summary,
        "provenance": provenance_payload,
    }


def build_seed_training_dataset(
    *,
    version: str,
    rows: list[TrainingRow] | None = None,
) -> dict[str, Any]:
    rows = rows or build_seed_training_rows()
    dataset_family = "seed:frontend_seed_bootstrap"
    time_window = _build_time_window(
        kind="bootstrap_reference",
        fallback_reference=TRAINING_REFERENCE_AT,
    )
    context = build_dataset_context(
        dataset_mode="seed",
        dataset_family=dataset_family,
        time_window=time_window,
        source="frontend_seed_bootstrap",
    )
    return build_training_dataset(
        version=version,
        rows=rows,
        description="Versioned training dataset exported from the current bootstrap spatial features and labels.",
        provenance={
            "exported_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            **context,
        },
        summary_extra={
            "dataset_family": context["dataset_family"],
            "time_window": context["time_window"],
            "dataset_taxonomy": context["dataset_taxonomy"],
            "evaluation_cohort": context["evaluation_cohort"],
        },
    )


def build_operational_training_dataset(
    *,
    version: str,
    rows: list[TrainingRow],
    row_contexts: list[dict[str, Any] | None],
    run_ids: list[int],
    model_versions: list[str],
    skipped_predictions: int,
) -> dict[str, Any]:
    dataset_family = "operational:prediction_history"
    time_window = _build_time_window(
        kind="operational_run_history",
        timestamps=[(item or {}).get("runCompletedAt") for item in row_contexts],
    )
    context = build_dataset_context(
        dataset_mode="operational",
        dataset_family=dataset_family,
        time_window=time_window,
        source="operational_prediction_history",
    )
    return build_training_dataset(
        version=version,
        rows=rows,
        description="Versioned training dataset exported from persisted prediction runs and explanation traces.",
        provenance={
            "exported_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "run_ids": run_ids,
            "model_versions": model_versions,
            "skipped_predictions": skipped_predictions,
            "feature_source": "zone_explanation.trace.feature_snapshot",
            "label_source": "zone_predictions.risk_score",
            **context,
        },
        row_contexts=row_contexts,
        summary_extra={
            "dataset_family": context["dataset_family"],
            "time_window": context["time_window"],
            "dataset_taxonomy": context["dataset_taxonomy"],
            "evaluation_cohort": context["evaluation_cohort"],
            "runs": len(run_ids),
            "run_ids": run_ids,
            "model_versions": model_versions,
            "skipped_predictions": skipped_predictions,
        },
    )


def build_labeled_training_dataset(
    *,
    version: str,
    rows: list[TrainingRow],
    row_contexts: list[dict[str, Any] | None],
    label_ids: list[int],
    label_sources: list[str],
    matched_predictions: int,
    unresolved_labels: int,
) -> dict[str, Any]:
    label_source_families = _label_source_families(label_sources)
    family_suffix = (
        label_source_families[0]
        if len(label_source_families) == 1
        else "mixed"
        if label_source_families
        else "unknown"
    )
    dataset_family = f"labels:{family_suffix}"
    time_window = _build_time_window(
        kind="observed_outcomes",
        timestamps=[(item or {}).get("observedAt") for item in row_contexts],
    )
    context = build_dataset_context(
        dataset_mode="labels",
        dataset_family=dataset_family,
        time_window=time_window,
        source="governed_zone_outcome_labels",
        label_source_families=label_source_families,
    )
    return build_training_dataset(
        version=version,
        rows=rows,
        description="Versioned supervised training dataset exported from governed zone outcome labels.",
        provenance={
            "exported_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "label_ids": label_ids,
            "label_sources": label_sources,
            "matched_predictions": matched_predictions,
            "unresolved_labels": unresolved_labels,
            "feature_source": "linked prediction runs and explanation traces",
            "label_source": "zone_outcome_labels.target_score",
            **context,
        },
        row_contexts=row_contexts,
        summary_extra={
            "dataset_family": context["dataset_family"],
            "label_source_families": label_source_families,
            "time_window": context["time_window"],
            "dataset_taxonomy": context["dataset_taxonomy"],
            "evaluation_cohort": context["evaluation_cohort"],
            "labels": len(label_ids),
            "label_ids": label_ids,
            "label_sources": label_sources,
            "matched_predictions": matched_predictions,
            "unresolved_labels": unresolved_labels,
        },
    )


def export_training_dataset(dataset: dict[str, Any], *, datasets_path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    settings = get_settings()
    target_directory = datasets_path or settings.resolved_training_datasets_path
    target_directory.mkdir(parents=True, exist_ok=True)
    dataset_path = target_directory / f"{dataset['version']}.json"
    dataset_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    return dataset_path, dataset


def export_seed_training_dataset(
    *,
    version: str,
    rows: list[TrainingRow] | None = None,
    datasets_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    dataset = build_seed_training_dataset(version=version, rows=rows)
    return export_training_dataset(dataset, datasets_path=datasets_path)


class TrainingDatasetRegistry:
    def __init__(self, datasets_path: Path | None = None) -> None:
        settings = get_settings()
        self.datasets_path = datasets_path or settings.resolved_training_datasets_path

    @lru_cache
    def load(self, version: str) -> dict[str, Any]:
        dataset_path = self.datasets_path / f"{version}.json"
        if not dataset_path.exists():
            raise ApiError(404, "training_dataset_not_found", f"Training dataset '{version}' was not found.")
        with dataset_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def list_versions(self) -> list[str]:
        return sorted(path.stem for path in self.datasets_path.glob("*.json"))
