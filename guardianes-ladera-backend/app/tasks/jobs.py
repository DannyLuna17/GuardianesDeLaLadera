from __future__ import annotations

import logging

from app.core.config import get_settings
from app.db.session import session_scope
from app.services.ingestion import IngestionService
from app.services.labels import OutcomeLabelService
from app.services.notification_service import AdminNotificationService
from app.services.model_monitoring_cycle import ModelMonitoringCycleService
from app.services.pipeline import PipelineService
from app.services.runs import RunService

logger = logging.getLogger(__name__)


def run_ingestion_cycle(
    sources: list[str] | None = None, note: str | None = None
) -> dict:
    with session_scope() as session:
        service = IngestionService(session)
        response = service.sync_sources(
            source_ids=sources,
            origin="scheduler",
            note=note or "Scheduled ingestion sync.",
        )
        logger.info(
            "Scheduled ingestion cycle completed",
            extra={"sources": sources or ["IDEAM", "SGC", "UNGRD"]},
        )
        return response.model_dump(by_alias=True)


def run_prediction_cycle(note: str | None = None) -> dict:
    with session_scope() as session:
        service = RunService(session)
        response = service.trigger_run(
            note=note or "Scheduled prediction run.",
            origin="scheduler",
        )
        logger.info(
            "Scheduled prediction run completed", extra={"run_id": response.run.id}
        )
        return response.model_dump(by_alias=True)


def run_operational_cycle(
    sources: list[str] | None = None, note: str | None = None
) -> dict:
    with session_scope() as session:
        service = PipelineService(session)
        response = service.trigger_full_pipeline(
            sources=sources,
            note=note or "Scheduled operational pipeline.",
            origin="scheduler",
        )
        logger.info(
            "Scheduled operational pipeline completed",
            extra={"pipeline_job_id": response.job.id, "run_id": response.run.run.id},
        )
        return response.model_dump(by_alias=True)


def refresh_latest_explanations() -> dict:
    with session_scope() as session:
        service = RunService(session)
        response = service.refresh_explanations(origin="scheduler")
        logger.info(
            "Scheduled explanation refresh completed",
            extra={
                "run_id": response.run_id,
                "refreshed_count": response.refreshed_count,
            },
        )
        return response.model_dump(by_alias=True)


def run_training_release_sla_cycle(note: str | None = None) -> dict:
    with session_scope() as session:
        service = OutcomeLabelService(session)
        response = service.run_training_release_sla_scan(
            note=note or "Automated scheduler release SLA scan.",
            origin="scheduler",
        )
        logger.info(
            "Scheduled training release SLA scan completed",
            extra={
                "job_id": response.job.id,
                "escalated_count": response.escalated_count,
                "notification_count": response.notification_count,
            },
        )
        return response.model_dump(by_alias=True)


def run_training_release_reassignment_cycle(note: str | None = None) -> dict:
    with session_scope() as session:
        service = OutcomeLabelService(session)
        response = service.run_training_release_reassignment_scan(
            note=note or "Automated scheduler release reassignment scan.",
            origin="scheduler",
        )
        logger.info(
            "Scheduled training release reassignment scan completed",
            extra={
                "job_id": response.job.id,
                "reassigned_count": response.reassigned_count,
                "notification_count": response.notification_count,
            },
        )
        return response.model_dump(by_alias=True)


def run_notification_ack_cycle(note: str | None = None) -> dict:
    with session_scope() as session:
        service = AdminNotificationService(session)
        response = service.run_notification_ack_scan(
            note=note or "Automated scheduler notification ack scan.",
            origin="scheduler",
        )
        logger.info(
            "Scheduled notification acknowledgement scan completed",
            extra={
                "job_id": response.job.id,
                "source_count": response.source_count,
                "reminded_count": response.reminded_count,
            },
        )
        return response.model_dump(by_alias=True)


def run_notification_delivery_retry_cycle(note: str | None = None) -> dict:
    with session_scope() as session:
        service = AdminNotificationService(session)
        response = service.run_notification_delivery_retry_scan(
            note=note or "Automated scheduler notification delivery retry scan.",
            origin="scheduler",
        )
        logger.info(
            "Scheduled notification delivery retry scan completed",
            extra={
                "job_id": response.job.id,
                "candidate_count": response.candidate_count,
                "retried_count": response.retried_count,
                "skipped_count": response.skipped_count,
            },
        )
        return response.model_dump(by_alias=True)


def run_notification_delivery_failure_cycle(note: str | None = None) -> dict:
    with session_scope() as session:
        service = AdminNotificationService(session)
        response = service.run_notification_delivery_failure_scan(
            note=note or "Automated scheduler notification delivery failure scan.",
            origin="scheduler",
        )
        logger.info(
            "Scheduled notification delivery failure scan completed",
            extra={
                "job_id": response.job.id,
                "candidate_count": response.candidate_count,
                "alerted_count": response.alerted_count,
                "resolved_alert_count": response.resolved_alert_count,
                "skipped_count": response.skipped_count,
            },
        )
        return response.model_dump(by_alias=True)


def run_model_monitoring_cycle(note: str | None = None) -> dict:
    with session_scope() as session:
        settings = get_settings()
        service = ModelMonitoringCycleService(session)
        response = service.run_monitoring_cycle(
            drift_top_error_count=settings.model_monitoring_drift_top_error_count,
            shadow_top_error_count=settings.model_monitoring_shadow_top_error_count,
            shadow_max_candidates=settings.model_monitoring_shadow_max_candidates,
            note=note or "Automated scheduler model monitoring cycle.",
            origin="scheduler",
        )
        logger.info(
            "Scheduled model monitoring cycle completed",
            extra={
                "job_id": response.job.id,
                "dataset_version": response.dataset_version,
                "active_model_version": response.active_model_version,
                "skipped": response.skipped,
            },
        )
        return response.model_dump(by_alias=True)
