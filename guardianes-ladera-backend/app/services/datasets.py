from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import joinedload, selectinload
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import ApiError
from app.ml.features import ZoneFeatureBuilder
from app.ml.datasets import (
    TrainingDatasetRegistry,
    build_labeled_training_dataset,
    build_operational_training_dataset,
    export_seed_training_dataset,
    export_training_dataset,
    feature_snapshot_from_mapping,
)
from app.ml.training import TrainingRow
from app.models import (
    JobExecution,
    PredictionRun,
    Zone,
    ZoneExplanation,
    ZoneOutcomeLabel,
    ZonePrediction,
)
from app.schemas.admin import (
    ExportTrainingDatasetResponse,
    JobExecutionRead,
    TrainingDatasetDetailRead,
    TrainingDatasetRowPreviewRead,
    TrainingDatasetSummaryRead,
)
from app.services.labels import OutcomeLabelService


DEFAULT_TRAINING_DATASET_VERSION = "training-spatial-seed-v1"
DEFAULT_OPERATIONAL_TRAINING_DATASET_VERSION = "training-operational-history-v1"
DEFAULT_LABELED_TRAINING_DATASET_VERSION = "training-supervised-labels-v1"


class TrainingDatasetService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.settings = get_settings()
        self.registry = TrainingDatasetRegistry()
        self.feature_builder = ZoneFeatureBuilder(session)

    @staticmethod
    def _job_read(job: JobExecution) -> JobExecutionRead:
        return JobExecutionRead(
            id=job.id,
            jobType=job.job_type,
            status=job.status,
            startedAt=job.started_at,
            completedAt=job.completed_at,
            details=job.details or {},
        )

    def _artifact_path(self, version: str) -> Path:
        return self.settings.resolved_training_datasets_path / f"{version}.json"

    @staticmethod
    def _summary_read(version: str, dataset: dict) -> TrainingDatasetSummaryRead:
        summary = dataset.get("summary", {})
        splits = summary.get("splits", {})
        provenance = dataset.get("provenance", {})
        return TrainingDatasetSummaryRead(
            version=version,
            datasetId=dataset.get("dataset_id", "unknown"),
            artifactType=dataset.get("artifact_type", "training_dataset"),
            description=dataset.get("description"),
            rows=summary.get("rows", len(dataset.get("rows", []))),
            zones=summary.get("zones", 0),
            featureCount=len(dataset.get("feature_order", [])),
            trainRows=splits.get("train_rows", 0),
            validationRows=splits.get("validation_rows", 0),
            provenanceSource=provenance.get("source"),
            exportedAt=provenance.get("exported_at"),
        )

    @staticmethod
    def _sample_rows(dataset: dict, *, sample_size: int) -> list[TrainingDatasetRowPreviewRead]:
        sample_rows: list[TrainingDatasetRowPreviewRead] = []
        for item in dataset.get("rows", [])[:sample_size]:
            sample_rows.append(
                TrainingDatasetRowPreviewRead(
                    zoneId=item["zoneId"],
                    phase=item["phase"],
                    split=item["split"],
                    targetScore=item["targetScore"],
                    featureVector=item["featureVector"],
                    context=item.get("context") or {},
                )
            )
        return sample_rows

    def export_seed_dataset(
        self,
        *,
        version: str | None = None,
        origin: str = "manual",
    ) -> ExportTrainingDatasetResponse:
        target_version = version or DEFAULT_TRAINING_DATASET_VERSION
        return self.export_dataset(version=target_version, source_mode="seed", origin=origin)

    @staticmethod
    def _as_utc(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    def _list_operational_runs(self, *, run_ids: list[int] | None, max_runs: int) -> list[PredictionRun]:
        statement = (
            select(PredictionRun)
            .where(PredictionRun.status == "completed")
            .options(
                selectinload(PredictionRun.predictions).selectinload(ZonePrediction.explanation),
                selectinload(PredictionRun.predictions)
                .joinedload(ZonePrediction.zone)
                .joinedload(Zone.municipality),
            )
            .order_by(PredictionRun.completed_at.desc())
        )
        if run_ids:
            statement = statement.where(PredictionRun.id.in_(run_ids))
        else:
            statement = statement.limit(max_runs)

        runs = list(self.session.scalars(statement).unique().all())
        if run_ids:
            runs_by_id = {run.id: run for run in runs}
            missing_run_ids = [run_id for run_id in run_ids if run_id not in runs_by_id]
            if missing_run_ids:
                missing_display = ", ".join(str(run_id) for run_id in missing_run_ids)
                raise ApiError(
                    404,
                    "training_dataset_run_not_found",
                    f"Operational dataset export could not find completed runs: {missing_display}.",
                )
            return [runs_by_id[run_id] for run_id in run_ids]
        return runs

    def _resolve_prediction_for_label(self, label: ZoneOutcomeLabel) -> ZonePrediction | None:
        statement = (
            select(ZonePrediction)
            .where(ZonePrediction.zone_id == label.zone_id)
            .options(
                joinedload(ZonePrediction.run),
                joinedload(ZonePrediction.zone).joinedload(Zone.municipality),
                joinedload(ZonePrediction.explanation),
            )
        )
        if label.feature_run_id is not None:
            statement = statement.where(ZonePrediction.run_id == label.feature_run_id)
        else:
            statement = (
                statement.join(ZonePrediction.run)
                .where(PredictionRun.completed_at <= label.observed_at)
                .order_by(PredictionRun.completed_at.desc())
            )
        statement = statement.limit(1)
        return self.session.scalar(statement)

    def _list_governed_labels(
        self,
        *,
        label_ids: list[int] | None,
        label_sources: list[str] | None,
        max_labels: int,
        observed_after: datetime | None,
        observed_before: datetime | None,
    ) -> list[ZoneOutcomeLabel]:
        statement = (
            select(ZoneOutcomeLabel)
            .where(ZoneOutcomeLabel.status == "confirmed")
            .options(
                joinedload(ZoneOutcomeLabel.zone).joinedload(Zone.municipality),
                joinedload(ZoneOutcomeLabel.feature_run),
            )
            .order_by(ZoneOutcomeLabel.observed_at.desc(), ZoneOutcomeLabel.id.desc())
        )
        if label_ids:
            statement = statement.where(ZoneOutcomeLabel.id.in_(label_ids))
        else:
            statement = statement.limit(max_labels)
        if label_sources:
            statement = statement.where(ZoneOutcomeLabel.source.in_(label_sources))
        if observed_after is not None:
            statement = statement.where(ZoneOutcomeLabel.observed_at >= observed_after)
        if observed_before is not None:
            statement = statement.where(ZoneOutcomeLabel.observed_at <= observed_before)

        labels = list(self.session.scalars(statement).unique().all())
        labels = [
            label
            for label in labels
            if OutcomeLabelService.effective_training_eligibility_status(label) == "eligible"
        ]
        if label_ids:
            labels_by_id = {label.id: label for label in labels}
            missing_label_ids = [label_id for label_id in label_ids if label_id not in labels_by_id]
            if missing_label_ids:
                missing_display = ", ".join(str(label_id) for label_id in missing_label_ids)
                raise ApiError(
                    404,
                    "training_dataset_label_not_found",
                    f"Supervised dataset export could not find confirmed and training-eligible labels: {missing_display}.",
                )
            return [labels_by_id[label_id] for label_id in label_ids]
        return labels

    def _build_operational_dataset(self, *, version: str, run_ids: list[int] | None, max_runs: int) -> tuple[Path, dict]:
        runs = self._list_operational_runs(run_ids=run_ids, max_runs=max_runs)
        rows: list[TrainingRow] = []
        row_contexts: list[dict] = []
        model_versions: set[str] = set()
        skipped_predictions = 0
        backfilled_predictions = 0

        for run in runs:
            model_versions.add(run.model_version)
            predictions = sorted(run.predictions, key=lambda item: item.zone_id)
            for prediction in predictions:
                explanation = prediction.explanation
                trace = explanation.trace if isinstance(explanation, ZoneExplanation) and explanation.trace else {}
                feature_snapshot_payload = trace.get("feature_snapshot")
                feature_snapshot_source = "trace"
                if isinstance(feature_snapshot_payload, dict):
                    feature_snapshot = feature_snapshot_from_mapping(feature_snapshot_payload)
                else:
                    if prediction.zone is None:
                        skipped_predictions += 1
                        continue
                    feature_snapshot = self.feature_builder.build_for_zone(
                        prediction.zone,
                        as_of=self._as_utc(run.completed_at),
                    )
                    feature_snapshot_source = "backfill"
                    backfilled_predictions += 1
                rows.append(
                    TrainingRow(
                        zone_id=prediction.zone_id,
                        phase=f"run-{run.id}",
                        target_score=float(prediction.risk_score),
                        drivers=dict(prediction.drivers),
                        feature_snapshot=feature_snapshot,
                    )
                )
                row_contexts.append(
                    {
                        "runId": run.id,
                        "runStartedAt": self._as_utc(run.started_at).isoformat() if self._as_utc(run.started_at) else None,
                        "runCompletedAt": self._as_utc(run.completed_at).isoformat() if self._as_utc(run.completed_at) else None,
                        "modelVersion": run.model_version,
                        "predictionId": prediction.id,
                        "confidence": prediction.confidence,
                        "trend": prediction.trend,
                        "riskDelta": round(float(prediction.risk_delta), 3),
                        "featureSnapshotSource": feature_snapshot_source,
                    }
                )

        dataset = build_operational_training_dataset(
            version=version,
            rows=rows,
            row_contexts=row_contexts,
            run_ids=[run.id for run in runs],
            model_versions=sorted(model_versions),
            skipped_predictions=skipped_predictions,
        )
        dataset["summary"]["backfilled_predictions"] = backfilled_predictions
        dataset["provenance"]["backfilled_predictions"] = backfilled_predictions
        return export_training_dataset(dataset)

    def _build_labeled_dataset(
        self,
        *,
        version: str,
        label_ids: list[int] | None,
        label_sources: list[str] | None,
        max_labels: int,
        observed_after: datetime | None,
        observed_before: datetime | None,
    ) -> tuple[Path, dict]:
        labels = self._list_governed_labels(
            label_ids=label_ids,
            label_sources=label_sources,
            max_labels=max_labels,
            observed_after=observed_after,
            observed_before=observed_before,
        )
        rows: list[TrainingRow] = []
        row_contexts: list[dict] = []
        unresolved_labels = 0
        backfilled_predictions = 0

        for label in labels:
            prediction = self._resolve_prediction_for_label(label)
            if prediction is None or prediction.run is None:
                unresolved_labels += 1
                continue

            explanation = prediction.explanation
            trace = explanation.trace if isinstance(explanation, ZoneExplanation) and explanation.trace else {}
            feature_snapshot_payload = trace.get("feature_snapshot")
            feature_snapshot_source = "trace"
            if isinstance(feature_snapshot_payload, dict):
                feature_snapshot = feature_snapshot_from_mapping(feature_snapshot_payload)
            else:
                if prediction.zone is None:
                    unresolved_labels += 1
                    continue
                feature_snapshot = self.feature_builder.build_for_zone(
                    prediction.zone,
                    as_of=self._as_utc(prediction.run.completed_at),
                )
                feature_snapshot_source = "backfill"
                backfilled_predictions += 1

            rows.append(
                TrainingRow(
                    zone_id=label.zone_id,
                    phase=f"label-{label.id}",
                    target_score=float(label.target_score),
                    drivers=dict(prediction.drivers),
                    feature_snapshot=feature_snapshot,
                )
            )
            row_contexts.append(
                {
                    "labelId": label.id,
                    "observedAt": self._as_utc(label.observed_at).isoformat() if self._as_utc(label.observed_at) else None,
                    "labelSource": label.source,
                    "labelStatus": label.status,
                    "featureRunId": prediction.run_id,
                    "featureRunCompletedAt": self._as_utc(prediction.run.completed_at).isoformat()
                    if self._as_utc(prediction.run.completed_at)
                    else None,
                    "predictionId": prediction.id,
                    "modelVersion": prediction.run.model_version,
                    "featureSnapshotSource": feature_snapshot_source,
                }
            )

        dataset = build_labeled_training_dataset(
            version=version,
            rows=rows,
            row_contexts=row_contexts,
            label_ids=[label.id for label in labels],
            label_sources=sorted({label.source for label in labels}),
            matched_predictions=len(rows),
            unresolved_labels=unresolved_labels,
        )
        dataset["summary"]["backfilled_predictions"] = backfilled_predictions
        dataset["provenance"]["backfilled_predictions"] = backfilled_predictions
        return export_training_dataset(dataset)

    def export_dataset(
        self,
        *,
        version: str | None = None,
        source_mode: str = "seed",
        run_ids: list[int] | None = None,
        max_runs: int = 5,
        label_ids: list[int] | None = None,
        label_sources: list[str] | None = None,
        max_labels: int = 100,
        observed_after: datetime | None = None,
        observed_before: datetime | None = None,
        origin: str = "manual",
    ) -> ExportTrainingDatasetResponse:
        normalized_source_mode = source_mode.lower().strip()
        if normalized_source_mode not in {"seed", "operational", "labels"}:
            raise ApiError(
                400,
                "invalid_training_dataset_source_mode",
                f"Unsupported training dataset source mode: {source_mode}",
            )
        if self.settings.real_data_only and normalized_source_mode == "seed":
            raise ApiError(
                409,
                "seed_dataset_export_disabled",
                "Seed-backed training dataset export is disabled because REAL_DATA_ONLY is enabled. Export operational or labels datasets instead.",
            )
        if normalized_source_mode != "operational" and run_ids:
            raise ApiError(
                400,
                "training_dataset_run_ids_not_supported",
                "Run IDs are only supported for operational training-dataset exports.",
            )
        if normalized_source_mode != "labels" and (label_ids or label_sources):
            raise ApiError(
                400,
                "training_dataset_label_filters_not_supported",
                "Label filters are only supported for labels-mode training-dataset exports.",
            )
        if normalized_source_mode != "labels" and (
            observed_after is not None or observed_before is not None
        ):
            raise ApiError(
                400,
                "training_dataset_observed_window_not_supported",
                "ObservedAt filters are only supported for labels-mode training-dataset exports.",
            )

        if normalized_source_mode == "seed":
            target_version = version or DEFAULT_TRAINING_DATASET_VERSION
        elif normalized_source_mode == "operational":
            target_version = version or DEFAULT_OPERATIONAL_TRAINING_DATASET_VERSION
        else:
            target_version = version or DEFAULT_LABELED_TRAINING_DATASET_VERSION

        started_at = datetime.now(timezone.utc).replace(microsecond=0)
        job = JobExecution(
            job_type="training_dataset_export",
            status="running",
            started_at=started_at,
            details={
                "version": target_version,
                "origin": origin,
                "source_mode": normalized_source_mode,
                "run_ids": run_ids or [],
                "max_runs": max_runs,
                "label_ids": label_ids or [],
                "label_sources": label_sources or [],
                "max_labels": max_labels,
                "observed_after": observed_after.isoformat() if observed_after else None,
                "observed_before": observed_before.isoformat() if observed_before else None,
            },
        )
        self.session.add(job)
        self.session.flush()

        try:
            if normalized_source_mode == "seed":
                dataset_path, dataset = export_seed_training_dataset(version=target_version)
            elif normalized_source_mode == "operational":
                dataset_path, dataset = self._build_operational_dataset(
                    version=target_version,
                    run_ids=run_ids,
                    max_runs=max_runs,
                )
            else:
                dataset_path, dataset = self._build_labeled_dataset(
                    version=target_version,
                    label_ids=label_ids,
                    label_sources=label_sources,
                    max_labels=max_labels,
                    observed_after=observed_after,
                    observed_before=observed_before,
                )
        except Exception as exc:
            completed_at = datetime.now(timezone.utc).replace(microsecond=0)
            job.status = "failed"
            job.completed_at = completed_at
            job.details = {
                "version": target_version,
                "origin": origin,
                "source_mode": normalized_source_mode,
                "run_ids": run_ids or [],
                "max_runs": max_runs,
                "label_ids": label_ids or [],
                "label_sources": label_sources or [],
                "max_labels": max_labels,
                "observed_after": observed_after.isoformat() if observed_after else None,
                "observed_before": observed_before.isoformat() if observed_before else None,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            self.session.commit()
            raise

        summary = dataset.get("summary", {})
        splits = summary.get("splits", {})
        split_counts = {
            "train": int(splits.get("train_rows", 0)),
            "validation": int(splits.get("validation_rows", 0)),
        }
        completed_at = datetime.now(timezone.utc).replace(microsecond=0)
        job.status = "completed"
        job.completed_at = completed_at
        job.details = {
            "version": target_version,
            "origin": origin,
            "source_mode": normalized_source_mode,
            "dataset_path": str(dataset_path),
            "rows": int(summary.get("rows", len(dataset.get("rows", [])))),
            "split_counts": split_counts,
            "run_count": summary.get("runs"),
            "run_ids": summary.get("run_ids", []),
            "label_count": summary.get("labels"),
            "label_ids": summary.get("label_ids", []),
        }
        self.session.commit()
        self.session.refresh(job)
        self.registry.load.cache_clear()

        return ExportTrainingDatasetResponse(
            job=self._job_read(job),
            datasetVersion=target_version,
            datasetPath=str(dataset_path),
            sourceMode=normalized_source_mode,
            runCount=summary.get("runs"),
            labelCount=summary.get("labels"),
            rows=int(summary.get("rows", len(dataset.get("rows", [])))),
            featureOrder=dataset.get("feature_order", []),
            splitCounts=split_counts,
        )

    def list_datasets(self) -> list[TrainingDatasetSummaryRead]:
        datasets: list[TrainingDatasetSummaryRead] = []
        for version in self.registry.list_versions():
            dataset = self.registry.load(version)
            datasets.append(self._summary_read(version, dataset))
        return sorted(datasets, key=lambda item: item.version)

    def get_dataset(self, version: str, *, sample_size: int = 5) -> TrainingDatasetDetailRead:
        dataset = self.registry.load(version)
        return TrainingDatasetDetailRead(
            version=version,
            datasetId=dataset.get("dataset_id", "unknown"),
            artifactType=dataset.get("artifact_type", "training_dataset"),
            description=dataset.get("description"),
            artifactPath=str(self._artifact_path(version)),
            labelName=dataset.get("label_name", "target_score"),
            featureOrder=dataset.get("feature_order", []),
            summary=dataset.get("summary", {}),
            provenance=dataset.get("provenance", {}),
            sampleRows=self._sample_rows(dataset, sample_size=sample_size),
        )
