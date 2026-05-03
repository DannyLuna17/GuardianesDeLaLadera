from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.core.exceptions import ApiError
from app.integrations.base import SyncResult
from app.integrations.registry import build_adapter, list_supported_sources
from app.models import JobExecution, SourceCatalog, SourceSyncEvent, SourceSyncStatus
from app.schemas.admin import IngestionSourceRead, SourceSyncEventRead, TriggerIngestionResponse
from app.services.structural_catalog import ensure_real_data_structural_catalog

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self, session: Session) -> None:
        self.session = session
        ensure_real_data_structural_catalog(session, for_api=True)

    def _list_sources(self) -> list[SourceCatalog]:
        statement = select(SourceCatalog).options(joinedload(SourceCatalog.sync_status)).order_by(SourceCatalog.id)
        return list(self.session.scalars(statement).all())

    def _job_read(self, job: JobExecution) -> dict:
        return {
            "id": job.id,
            "jobType": job.job_type,
            "status": job.status,
            "startedAt": job.started_at,
            "completedAt": job.completed_at,
            "details": job.details or {},
        }

    def _validate_selected_sources(self, source_ids: list[str]) -> list[str]:
        selected_sources = list(dict.fromkeys(source_ids))
        supported_sources = set(list_supported_sources())
        unsupported = [source_id for source_id in selected_sources if source_id not in supported_sources]
        if unsupported:
            unsupported_display = ", ".join(unsupported)
            raise ApiError(400, "unsupported_source", f"Unsupported ingestion source(s): {unsupported_display}.")

        existing_source_ids = {source.id for source in self._list_sources()}
        missing_catalog = [source_id for source_id in selected_sources if source_id not in existing_source_ids]
        if missing_catalog:
            missing_display = ", ".join(missing_catalog)
            raise ApiError(404, "source_not_found", f"Source catalog entries are missing for: {missing_display}.")

        return selected_sources

    def _run_adapter(self, source_id: str) -> SyncResult:
        adapter = build_adapter(source_id)
        try:
            return adapter.sync(self.session)
        except Exception as exc:
            logger.exception(
                "Ingestion adapter failed",
                extra={"source_id": source_id, "adapter": type(adapter).__name__},
            )
            adapter_key = getattr(adapter, "adapter_key", f"{getattr(adapter, 'transport', 'adapter')}.{source_id.lower()}")
            return SyncResult(
                source_id=source_id,
                processed_records=0,
                status="failed",
                message=str(exc) or f"{source_id} synchronization failed.",
                adapter_key=adapter_key,
                transport=getattr(adapter, "transport", "unknown"),
                details={"error_type": type(exc).__name__},
            )

    @staticmethod
    def _detail_datetime(result: SyncResult, key: str) -> datetime | None:
        value = result.details.get(key)
        if value in (None, ""):
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    def _update_sync_status(self, source_id: str, result: SyncResult, synced_at: datetime) -> None:
        source = self.session.scalar(select(SourceCatalog).where(SourceCatalog.id == source_id))
        if source is None:
            raise ApiError(404, "source_not_found", f"Source '{source_id}' does not exist.")
        status = source.sync_status
        if status is None:
            status = SourceSyncStatus(source=source)
            self.session.add(status)
        provider_updated_at = self._detail_datetime(result, "provider_updated_at")
        status.last_synced_at = synced_at
        status.last_success_at = (
            provider_updated_at or synced_at
            if result.status == "completed"
            else status.last_success_at
        )
        status.last_error = None if result.status == "completed" else result.message
        status.status_note = result.message

    def _record_sync_event(
        self,
        source_id: str,
        result: SyncResult,
        origin: str,
        started_at: datetime,
        completed_at: datetime,
    ) -> None:
        source = self.session.scalar(select(SourceCatalog).where(SourceCatalog.id == source_id))
        if source is None:
            raise ApiError(404, "source_not_found", f"Source '{source_id}' does not exist.")
        self.session.add(
            SourceSyncEvent(
                source=source,
                origin=origin,
                adapter_key=result.adapter_key,
                transport=result.transport,
                status=result.status,
                processed_records=result.processed_records,
                started_at=started_at,
                completed_at=completed_at,
                message=result.message,
                details=result.details,
            )
        )

    def _ingestion_source_read(self, result: SyncResult) -> IngestionSourceRead:
        return IngestionSourceRead(
            sourceId=result.source_id,
            processedRecords=result.processed_records,
            adapterKey=result.adapter_key,
            transport=result.transport,
            status=result.status,
            message=result.message,
            details=result.details,
        )

    def _sync_event_read(self, event: SourceSyncEvent) -> SourceSyncEventRead:
        return SourceSyncEventRead(
            id=event.id,
            sourceId=event.source_id,
            sourceLabel=event.source.label,
            origin=event.origin,
            adapterKey=event.adapter_key,
            transport=event.transport,
            status=event.status,
            processedRecords=event.processed_records,
            startedAt=event.started_at,
            completedAt=event.completed_at,
            message=event.message,
            details=event.details or {},
        )

    def sync_sources(
        self,
        source_ids: list[str] | None = None,
        origin: str = "manual",
        note: str | None = None,
    ) -> TriggerIngestionResponse:
        selected_sources = self._validate_selected_sources(source_ids or ["IDEAM", "SGC", "UNGRD"])
        started_at = datetime.now(timezone.utc).replace(microsecond=0)

        job = JobExecution(
            job_type="ingestion_sync",
            status="running",
            started_at=started_at,
            details={"sources": selected_sources, "origin": origin, "note": note},
        )
        self.session.add(job)
        self.session.flush()

        synced_sources: list[IngestionSourceRead] = []
        failed_count = 0
        for source_id in selected_sources:
            logger.info("Starting source ingestion: %s (origin=%s)", source_id, origin)
            source_started_at = datetime.now(timezone.utc).replace(microsecond=0)
            result = self._run_adapter(source_id)
            source_completed_at = datetime.now(timezone.utc).replace(microsecond=0)
            self._update_sync_status(source_id, result, source_completed_at)
            self._record_sync_event(source_id, result, origin, source_started_at, source_completed_at)
            synced_sources.append(self._ingestion_source_read(result))
            logger.info(
                "Finished source ingestion: %s status=%s processed=%s transport=%s adapter=%s",
                source_id,
                result.status,
                result.processed_records,
                result.transport,
                result.adapter_key,
            )
            if result.status != "completed":
                failed_count += 1

        completed_at = datetime.now(timezone.utc).replace(microsecond=0)
        if failed_count == 0:
            job.status = "completed"
        elif failed_count == len(synced_sources):
            job.status = "failed"
        else:
            job.status = "completed_with_errors"
        job.completed_at = completed_at
        job.details = {
            "sources": selected_sources,
            "origin": origin,
            "note": note,
            "synced_count": len(synced_sources),
            "failed_count": failed_count,
        }
        self.session.commit()
        self.session.refresh(job)

        return TriggerIngestionResponse(
            job=self._job_read(job),
            syncedSources=synced_sources,
        )

    def list_sync_events(self, source_id: str | None = None, limit: int = 20) -> list[SourceSyncEventRead]:
        statement = select(SourceSyncEvent).options(joinedload(SourceSyncEvent.source))
        if source_id:
            statement = statement.where(SourceSyncEvent.source_id == source_id)
        statement = statement.order_by(SourceSyncEvent.started_at.desc(), SourceSyncEvent.id.desc()).limit(limit)
        return [self._sync_event_read(event) for event in self.session.scalars(statement).all()]
