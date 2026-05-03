from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload, selectinload

from app.core.config import get_settings
from app.core.exceptions import ApiError
from app.db.bootstrap import (
    risk_level_from_score,
    risk_text_from_level,
    trend_from_delta,
)
from app.ml.features import ZoneFeatureBuilder
from app.ml.inference import BaselineInferenceService
from app.ml.model_registry import ModelRegistry
from app.models import (
    HistoricalEvent,
    JobExecution,
    MunicipalityRainPoint,
    PredictionRun,
    SourceCatalog,
    SourceSyncEvent,
    UngrdRecord,
    Zone,
    ZoneExplanation,
    ZonePrediction,
)
from app.schemas.admin import (
    JobExecutionRead,
    RefreshExplanationResponse,
    TriggerRunResponse,
)
from app.schemas.dashboard import RunSummaryRead
from app.services.dashboard import DashboardService
from app.services.explanation_builder import (
    build_driver_chips,
    build_suggestions,
    build_summary,
)
from app.services.operational_scoring import (
    OPERATIONAL_MODEL_VERSION,
    OperationalRiskScoringService,
)
from app.services.structural_catalog import ensure_real_data_structural_catalog


class RunService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.settings = get_settings()
        ensure_real_data_structural_catalog(session, for_api=True)
        self.dashboard_service = DashboardService(session)
        self.inference_service = BaselineInferenceService()
        self.model_registry = ModelRegistry()
        self.feature_builder = ZoneFeatureBuilder(session)
        self.operational_scorer = OperationalRiskScoringService(session)

    @staticmethod
    def _as_utc(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    def _ensure_runtime_run_allowed(self, run: PredictionRun) -> PredictionRun:
        if not self.settings.real_data_only:
            return run
        if self._is_legacy_run(run):
            raise ApiError(
                409,
                "legacy_prediction_run_blocked",
                "Legacy seed or synthetic prediction runs cannot be reused while REAL_DATA_ONLY is enabled.",
            )
        return run

    @staticmethod
    def _is_legacy_run(run: PredictionRun) -> bool:
        note = (run.notes or "").lower()
        return any(token in note for token in ("semilla", "seed", "synthetic"))

    def _latest_run(self) -> PredictionRun:
        statement = (
            select(PredictionRun)
            .options(
                selectinload(PredictionRun.predictions)
                .joinedload(ZonePrediction.zone)
                .joinedload(Zone.municipality),
                selectinload(PredictionRun.predictions)
                .joinedload(ZonePrediction.zone)
                .selectinload(Zone.road_segments),
                selectinload(PredictionRun.predictions).selectinload(
                    ZonePrediction.explanation
                ),
            )
            .order_by(PredictionRun.completed_at.desc())
            .limit(1)
        )
        run = self.session.scalar(statement)
        if run is None:
            raise ApiError(404, "run_not_found", "No prediction run is available yet.")
        return self._ensure_runtime_run_allowed(run)

    def _latest_non_legacy_run(self) -> PredictionRun | None:
        statement = (
            select(PredictionRun)
            .options(
                selectinload(PredictionRun.predictions)
                .joinedload(ZonePrediction.zone)
                .joinedload(Zone.municipality),
                selectinload(PredictionRun.predictions)
                .joinedload(ZonePrediction.zone)
                .selectinload(Zone.road_segments),
                selectinload(PredictionRun.predictions).selectinload(
                    ZonePrediction.explanation
                ),
            )
            .order_by(PredictionRun.completed_at.desc())
        )
        for run in self.session.scalars(statement).all():
            if not self._is_legacy_run(run):
                return run
        return None

    def _list_sources(self) -> list[SourceCatalog]:
        statement = (
            select(SourceCatalog)
            .options(joinedload(SourceCatalog.sync_status))
            .order_by(SourceCatalog.id)
        )
        return list(self.session.scalars(statement).all())

    def _latest_sync_event(self, source_id: str) -> SourceSyncEvent | None:
        statement = (
            select(SourceSyncEvent)
            .where(SourceSyncEvent.source_id == source_id)
            .order_by(SourceSyncEvent.completed_at.desc(), SourceSyncEvent.id.desc())
            .limit(1)
        )
        return self.session.scalar(statement)

    def _ensure_real_operational_sources_ready(self) -> None:
        invalid_sources: list[str] = []
        for source_id in ("IDEAM", "SGC", "UNGRD"):
            latest_event = self._latest_sync_event(source_id)
            if latest_event is None:
                continue
            if latest_event.transport == "seed" or "seed" in latest_event.adapter_key.lower():
                invalid_sources.append(source_id)
        if invalid_sources:
            invalid_display = ", ".join(invalid_sources)
            raise ApiError(
                409,
                "legacy_operational_data_blocked",
                "Legacy seed-backed operational data is still present for: "
                f"{invalid_display}. Purge the runtime tables and re-run official ingestion before scoring.",
            )

    def _source_snapshot(self) -> dict[str, str]:
        now = datetime.now(timezone.utc)
        snapshot: dict[str, str] = {}
        for source in self._list_sources():
            updated_at = (
                self._as_utc(source.sync_status.last_success_at)
                if source.sync_status
                else None
            )
            if source.category in {"historico", "infraestructura"}:
                snapshot[source.id] = "Estatico"
            elif updated_at is None:
                snapshot[source.id] = "Desactualizado"
            else:
                minutes = int((now - updated_at).total_seconds() // 60)
                if minutes <= 30:
                    snapshot[source.id] = "Fresco"
                elif minutes <= 180:
                    snapshot[source.id] = "Retrasado"
                else:
                    snapshot[source.id] = "Desactualizado"
        return snapshot

    def _refresh_dynamic_sources(self, run_index: int) -> None:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        source_offsets = {
            "IDEAM": max(3, 8 + (run_index % 4) * 2),
            "NASA": max(10, 20 + (run_index % 3) * 6),
            "UNGRD": 70 if run_index % 2 == 0 else 145,
            "SENTINEL": 150 if run_index % 3 == 0 else 235,
        }
        for source in self._list_sources():
            if (
                source.category in {"historico", "infraestructura"}
                or source.sync_status is None
            ):
                continue
            offset = source_offsets.get(source.id, 90)
            source.sync_status.last_synced_at = now - timedelta(minutes=offset)
            source.sync_status.last_success_at = now - timedelta(minutes=offset)
            source.sync_status.status_note = f"Corrida sintetica #{run_index}."

    def _create_explanation(
        self,
        prediction: ZonePrediction,
        snapshot: dict[str, str],
        event_count: int,
        extra_trace: dict | None = None,
    ) -> ZoneExplanation:
        zone = prediction.zone
        risk_level = risk_level_from_score(prediction.risk_score)
        stale_sources = [
            source_id
            for source_id, status in snapshot.items()
            if status in {"Retrasado", "Desactualizado"}
        ]
        trace = {
            "model_version": prediction.run.model_version,
            "risk_level": risk_level,
            "event_count": event_count,
            "generation_mode": "template",
        }
        if extra_trace:
            trace.update(extra_trace)
        data_warnings = [
            f"Fuente {source_id} con frescura imperfecta en esta corrida."
            for source_id in stale_sources[:3]
        ]
        if trace.get("missing_susceptibility_baselines"):
            data_warnings.append(
                "No hay baselines oficiales de pendiente, geologia, suelo o cobertura para esta zona."
            )
        return ZoneExplanation(
            prediction=prediction,
            mode="template",
            summary=build_summary(
                zone.name,
                zone.municipality.name,
                risk_text_from_level(risk_level),
                prediction.drivers,
                event_count,
            ),
            driver_chips=build_driver_chips(prediction.drivers),
            suggestions=build_suggestions(
                zone.name,
                zone.municipality.name,
                risk_level,
                [segment.name for segment in zone.road_segments],
            ),
            data_warnings=data_warnings,
            trace=trace,
            generated_at=prediction.created_at,
        )

    def _job_read(self, job: JobExecutionRead | JobExecution) -> JobExecutionRead:
        if isinstance(job, JobExecutionRead):
            return job
        return JobExecutionRead(
            id=job.id,
            jobType=job.job_type,
            status=job.status,
            startedAt=self._as_utc(job.started_at),
            completedAt=self._as_utc(job.completed_at),
            details=job.details or {},
        )

    def trigger_run(
        self,
        note: str | None = None,
        origin: str = "manual",
        generate_explanations: bool = True,
    ) -> TriggerRunResponse:
        if self.settings.real_data_only:
            return self._trigger_real_data_run(
                note=note,
                origin=origin,
                generate_explanations=generate_explanations,
            )
        latest_run = self._latest_run()
        previous_predictions = {
            prediction.zone_id: prediction for prediction in latest_run.predictions
        }
        run_index = latest_run.id + 1
        started_at = datetime.now(timezone.utc).replace(microsecond=0)

        job = JobExecution(
            job_type="prediction_run",
            status="running",
            started_at=started_at,
            details={
                "requested_note": note,
                "origin": origin,
                "generate_explanations": generate_explanations,
            },
        )
        self.session.add(job)
        self.session.flush()

        self._refresh_dynamic_sources(run_index)
        snapshot = self._source_snapshot()
        partial_data = any(
            status in {"Retrasado", "Desactualizado"} for status in snapshot.values()
        )

        run = PredictionRun(
            started_at=started_at,
            completed_at=started_at,
            status="completed",
            model_version=self.model_registry.active_version(),
            partial_data=partial_data,
            notes=note or f"Triggered synthetic run #{run_index}.",
        )
        self.session.add(run)
        self.session.flush()

        zone_statement = (
            select(Zone)
            .options(joinedload(Zone.municipality), selectinload(Zone.road_segments))
            .where(Zone.is_active.is_(True))
            .order_by(Zone.id)
        )
        zones = list(self.session.scalars(zone_statement).all())
        generated_predictions: list[ZonePrediction] = []

        for zone in zones:
            previous = previous_predictions.get(zone.id)
            if previous is None:
                continue
            feature_snapshot = self.feature_builder.build_for_zone(
                zone, as_of=started_at
            )
            inference_result = self.inference_service.predict(
                zone_id=zone.id,
                previous_drivers=previous.drivers,
                run_index=run_index,
                snapshot=snapshot,
                feature_snapshot=feature_snapshot,
            )
            delta = round(inference_result.score - previous.risk_score, 3)
            prediction = ZonePrediction(
                run=run,
                zone=zone,
                risk_score=inference_result.score,
                confidence=inference_result.confidence,
                drivers=inference_result.drivers,
                risk_delta=delta,
                trend=trend_from_delta(delta),
                source_snapshot=snapshot,
                created_at=started_at,
            )
            self.session.add(prediction)
            self.session.flush()
            if generate_explanations:
                self.session.add(
                    self._create_explanation(
                        prediction,
                        snapshot,
                        feature_snapshot.zone_event_count
                        or feature_snapshot.municipality_event_count,
                        extra_trace=inference_result.trace,
                    )
                )
            generated_predictions.append(prediction)

        run.completed_at = datetime.now(timezone.utc).replace(microsecond=0)
        job.status = "completed"
        job.completed_at = run.completed_at
        job.details = {
            "run_id": run.id,
            "zones_scored": len(generated_predictions),
            "partial_data": partial_data,
            "note": run.notes,
            "origin": origin,
            "generate_explanations": generate_explanations,
        }
        self.session.commit()
        self.session.refresh(job)
        self.session.refresh(run)

        run_summary = self.dashboard_service.get_run_detail(run.id)
        return TriggerRunResponse(
            job=self._job_read(job),
            run=RunSummaryRead.model_validate(run_summary.model_dump()),
        )

    def _trigger_real_data_run(
        self,
        *,
        note: str | None,
        origin: str,
        generate_explanations: bool,
    ) -> TriggerRunResponse:
        self._ensure_real_operational_sources_ready()
        has_operational_data = any(
            self.session.scalar(select(model.id).limit(1)) is not None
            for model in (MunicipalityRainPoint, HistoricalEvent, UngrdRecord)
        )
        if not has_operational_data:
            raise ApiError(
                409,
                "operational_data_unavailable",
                "No operational data is available yet. Trigger official ingestion before running scoring.",
            )

        previous_run = self._latest_non_legacy_run()
        previous_predictions = (
            {prediction.zone_id: prediction for prediction in previous_run.predictions}
            if previous_run is not None
            else {}
        )
        started_at = datetime.now(timezone.utc).replace(microsecond=0)

        job = JobExecution(
            job_type="prediction_run",
            status="running",
            started_at=started_at,
            details={
                "requested_note": note,
                "origin": origin,
                "generate_explanations": generate_explanations,
                "mode": "real_data_only",
            },
        )
        self.session.add(job)
        self.session.flush()

        snapshot = self._source_snapshot()
        partial_data = any(
            status in {"Retrasado", "Desactualizado"} for status in snapshot.values()
        )
        run = PredictionRun(
            started_at=started_at,
            completed_at=started_at,
            status="completed",
            model_version=OPERATIONAL_MODEL_VERSION,
            partial_data=partial_data,
            notes=note or "Triggered real-data operational run.",
        )
        self.session.add(run)
        self.session.flush()

        zone_statement = (
            select(Zone)
            .options(joinedload(Zone.municipality), selectinload(Zone.road_segments))
            .where(Zone.is_active.is_(True))
            .order_by(Zone.id)
        )
        zones = list(self.session.scalars(zone_statement).all())
        generated_predictions: list[ZonePrediction] = []

        for zone in zones:
            scoring_result = self.operational_scorer.score_zone(
                zone=zone,
                source_snapshot=snapshot,
                as_of=started_at,
            )
            previous = previous_predictions.get(zone.id)
            previous_score = previous.risk_score if previous is not None else scoring_result.score
            delta = round(scoring_result.score - previous_score, 3)
            prediction = ZonePrediction(
                run=run,
                zone=zone,
                risk_score=scoring_result.score,
                confidence=scoring_result.confidence,
                drivers=scoring_result.drivers,
                risk_delta=delta,
                trend=trend_from_delta(delta),
                source_snapshot=snapshot,
                created_at=started_at,
            )
            self.session.add(prediction)
            self.session.flush()
            if generate_explanations:
                feature_snapshot = scoring_result.trace.get("feature_snapshot") or {}
                event_count = int(
                    feature_snapshot.get("zone_event_count")
                    or feature_snapshot.get("municipality_event_count")
                    or 0
                )
                self.session.add(
                    self._create_explanation(
                        prediction,
                        snapshot,
                        event_count,
                        extra_trace=scoring_result.trace,
                    )
                )
            generated_predictions.append(prediction)

        run.completed_at = datetime.now(timezone.utc).replace(microsecond=0)
        job.status = "completed"
        job.completed_at = run.completed_at
        job.details = {
            "run_id": run.id,
            "zones_scored": len(generated_predictions),
            "partial_data": partial_data,
            "note": run.notes,
            "origin": origin,
            "generate_explanations": generate_explanations,
            "mode": "real_data_only",
        }
        self.session.commit()
        self.session.refresh(job)
        self.session.refresh(run)

        run_summary = self.dashboard_service.get_run_detail(run.id)
        return TriggerRunResponse(
            job=self._job_read(job),
            run=RunSummaryRead.model_validate(run_summary.model_dump()),
        )

    def refresh_explanations(
        self, run_id: int | None = None, origin: str = "manual"
    ) -> RefreshExplanationResponse:
        run = (
            self._latest_run()
            if run_id is None
            else self.session.scalar(
                select(PredictionRun).where(PredictionRun.id == run_id)
            )
        )
        if run is None:
            raise ApiError(404, "run_not_found", "No prediction run is available yet.")
        self._ensure_runtime_run_allowed(run)

        statement = (
            select(ZonePrediction)
            .where(ZonePrediction.run_id == run.id)
            .options(
                joinedload(ZonePrediction.zone).joinedload(Zone.municipality),
                joinedload(ZonePrediction.zone).selectinload(Zone.road_segments),
                joinedload(ZonePrediction.explanation),
            )
        )
        predictions = list(self.session.scalars(statement).all())
        snapshot = self._source_snapshot()
        started_at = datetime.now(timezone.utc).replace(microsecond=0)

        job = JobExecution(
            job_type="explanation_refresh",
            status="running",
            started_at=started_at,
            details={"run_id": run.id, "origin": origin},
        )
        self.session.add(job)
        self.session.flush()

        refreshed = 0
        for prediction in predictions:
            feature_snapshot = self.feature_builder.build_for_zone(
                prediction.zone, as_of=started_at
            )
            previous_trace = (
                prediction.explanation.trace
                if prediction.explanation is not None
                else None
            )
            if prediction.explanation is not None:
                self.session.delete(prediction.explanation)
                self.session.flush()
            self.session.add(
                self._create_explanation(
                    prediction,
                    snapshot,
                    feature_snapshot.zone_event_count
                    or feature_snapshot.municipality_event_count,
                    extra_trace={
                        **(previous_trace or {}),
                        "feature_snapshot": feature_snapshot.as_dict(),
                        "uses_spatial_features": True,
                    },
                )
            )
            refreshed += 1

        completed_at = datetime.now(timezone.utc).replace(microsecond=0)
        job.status = "completed"
        job.completed_at = completed_at
        job.details = {"run_id": run.id, "refreshed_count": refreshed, "origin": origin}
        self.session.commit()
        self.session.refresh(job)

        return RefreshExplanationResponse(
            job=self._job_read(job),
            refreshedCount=refreshed,
            runId=run.id,
        )

    def list_jobs(self) -> list[JobExecutionRead]:
        statement = select(JobExecution).order_by(JobExecution.started_at.desc())
        return [self._job_read(item) for item in self.session.scalars(statement).all()]
