from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import get_settings

from app.core.exceptions import ApiError
from app.ml.datasets import TrainingDatasetRegistry, normalize_dataset_context
from app.ml.model_registry import ModelRegistry
from app.models import JobExecution, NotificationEvent
from app.schemas.admin import (
    JobExecutionRead,
    NotificationEventRead,
    ScanModelMonitoringResponse,
)
from app.services.model_drift import ModelDriftService
from app.services.model_shadow import ModelShadowService
from app.services.notifications import NotificationService

MODEL_MONITORING_DRIFT_ALERT_EVENT_TYPE = "model_monitoring_drift_alert"
MODEL_MONITORING_SHADOW_ALERT_EVENT_TYPE = "model_monitoring_shadow_alert"


class ModelMonitoringCycleService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.settings = get_settings()
        self.dataset_registry = TrainingDatasetRegistry()
        self.model_registry = ModelRegistry()
        self.drift_service = ModelDriftService(session)
        self.shadow_service = ModelShadowService(session)
        self.notification_service = NotificationService(session)

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

    def _latest_labels_dataset_version(self) -> str:
        candidates: list[tuple[str, str]] = []
        for version in self.dataset_registry.list_versions():
            dataset = self.dataset_registry.load(version)
            dataset_context = normalize_dataset_context(dataset.get("provenance") or {})
            if dataset_context.get("dataset_mode") != "labels":
                continue
            exported_at = str((dataset.get("provenance") or {}).get("exported_at") or "")
            candidates.append((exported_at, version))

        if not candidates:
            raise ApiError(
                404,
                "model_monitoring_labels_dataset_not_found",
                "No label-backed training dataset is available for automated model monitoring.",
            )

        candidates.sort(reverse=True)
        return candidates[0][1]

    def _alert_targets(self) -> list[dict]:
        usernames = (
            list(self.settings.notification_model_monitoring_usernames)
            or list(self.settings.notification_release_ops_usernames)
            or (
                [self.settings.seed_admin_username]
                if self.settings.seed_admin_username
                else []
            )
        )
        normalized: list[dict] = []
        seen: set[str] = set()
        for index, username in enumerate(usernames):
            normalized_username = str(username).strip()
            if not normalized_username or normalized_username in seen:
                continue
            seen.add(normalized_username)
            normalized.append(
                {
                    "username": normalized_username,
                    "routing_audience": "model_monitoring_watch",
                    "routing_reason": "predictive_model_monitoring",
                    "is_primary": index == 0,
                }
            )
        return normalized

    def _open_alerts(
        self, *, event_type: str, model_version: str
    ) -> list[NotificationEvent]:
        statement = (
            select(NotificationEvent)
            .options(selectinload(NotificationEvent.delivery_attempts))
            .where(
                NotificationEvent.event_type == event_type,
                NotificationEvent.status == "open",
            )
            .order_by(NotificationEvent.created_at.asc(), NotificationEvent.id.asc())
        )
        return [
            notification
            for notification in self.session.scalars(statement).all()
            if (notification.details or {}).get("model_version") == model_version
        ]

    def _resolve_alerts(
        self,
        alerts: list[NotificationEvent],
        *,
        resolved_at: datetime,
        dataset_version: str | None,
        origin: str,
        resolution_reason: str,
    ) -> int:
        resolved_by = f"{origin}:model_monitoring_cycle"
        resolved_count = 0
        for alert in alerts:
            if alert.status != "open":
                continue
            details = dict(alert.details or {})
            details["resolved_at"] = resolved_at.isoformat()
            details["resolved_by"] = resolved_by
            details["resolution_reason"] = resolution_reason
            details["resolution_dataset_version"] = dataset_version
            alert.details = details
            alert.status = "resolved"
            alert.acknowledged_at = resolved_at
            alert.acknowledged_by = resolved_by
            resolved_count += 1
        return resolved_count

    def _update_open_alerts(
        self,
        alerts: list[NotificationEvent],
        *,
        severity: str,
        title: str,
        message: str,
        details: dict,
        observed_at: datetime,
    ) -> list[NotificationEvent]:
        updated: list[NotificationEvent] = []
        delivery_channels = self.notification_service._delivery_channels_for_severity(
            severity
        )
        for alert in alerts:
            alert.severity = severity
            alert.channel = delivery_channels[0] if delivery_channels else "in_app"
            alert.delivery_channels = delivery_channels
            alert.title = title
            alert.message = message
            merged_details = dict(alert.details or {})
            merged_details.update(details)
            merged_details["last_observed_at"] = observed_at.isoformat()
            alert.details = merged_details
            updated.append(alert)
        return updated

    def _drift_alert_payload(
        self,
        *,
        active_model_version: str,
        dataset_version: str,
        drift_response,
    ) -> tuple[bool, str, str, str, dict]:
        severity = str(drift_response.severity or "unavailable")
        should_alert = severity in {"warning", "critical"} and bool(
            drift_response.drift_detected
        )
        summary = drift_response.drift_summary or {}
        rmse_delta = summary.get("validation_rmse_delta")
        accuracy_delta = summary.get("validation_risk_level_accuracy_delta")
        title = f"Model drift detected for {active_model_version}"
        message = (
            f"Active model {active_model_version} shows {severity} drift on {dataset_version}. "
            + "Action: review the latest labeled cohort before further promotion decisions."
        )
        details = {
            "model_version": active_model_version,
            "dataset_version": dataset_version,
            "drift_version": drift_response.drift_version,
            "drift_severity": severity,
            "drift_detected": drift_response.drift_detected,
            "validation_rmse_delta": rmse_delta,
            "validation_risk_level_accuracy_delta": accuracy_delta,
            "baseline": drift_response.baseline or {},
            "current": drift_response.current or {},
            "drift_summary": summary,
            "summary": message,
            "recommended_action": (
                "Review the fresh labels cohort, compare it to the current baseline, and decide whether retraining or rollback review is needed."
            ),
        }
        return should_alert, severity, title, message, details

    def _shadow_alert_payload(
        self,
        *,
        active_model_version: str,
        dataset_version: str,
        shadow_response,
    ) -> tuple[bool, str, str, str, dict]:
        recommendation = shadow_response.recommendation or {}
        status = str(recommendation.get("status") or "unknown")
        should_alert = (
            status == "review_challenger"
            and str(shadow_response.best_model_version) != active_model_version
        )
        title = f"Challenger review suggested for {active_model_version}"
        message = (
            f"Shadow evaluation on {dataset_version} found challenger {shadow_response.best_model_version} ahead of active model {active_model_version}. "
            + "Action: review the challenger before any promotion decision."
        )
        top_candidate = next(
            (
                candidate
                for candidate in shadow_response.candidates
                if candidate.rank == 1
            ),
            None,
        )
        details = {
            "model_version": active_model_version,
            "dataset_version": dataset_version,
            "shadow_version": shadow_response.shadow_version,
            "shadow_status": status,
            "best_model_version": shadow_response.best_model_version,
            "active_still_best": shadow_response.active_still_best,
            "recommendation": recommendation,
            "top_candidate": top_candidate.model_dump(by_alias=True)
            if top_candidate is not None
            else None,
            "summary": message,
            "recommended_action": (
                "Inspect the challenger diagnostics, compare it to the champion on the labeled cohort, and decide whether to open a promotion review."
            ),
        }
        return should_alert, "warning", title, message, details

    def _apply_alert_policy(
        self,
        *,
        event_type: str,
        should_alert: bool,
        severity: str,
        title: str,
        message: str,
        details: dict,
        active_model_version: str,
        dataset_version: str | None,
        observed_at: datetime,
        origin: str,
        resolution_reason: str,
    ) -> tuple[list[NotificationEventRead], int, int, int]:
        if not self.settings.enable_model_monitoring_alerts:
            return [], 0, 0, 0

        open_alerts = self._open_alerts(
            event_type=event_type,
            model_version=active_model_version,
        )
        if not should_alert:
            resolved_count = self._resolve_alerts(
                open_alerts,
                resolved_at=observed_at,
                dataset_version=dataset_version,
                origin=origin,
                resolution_reason=resolution_reason,
            )
            return [], 0, 0, resolved_count

        if open_alerts:
            updated = self._update_open_alerts(
                open_alerts,
                severity=severity,
                title=title,
                message=message,
                details=details,
                observed_at=observed_at,
            )
            return (
                [self.notification_service._read(alert) for alert in updated],
                0,
                len(updated),
                0,
            )

        created = self.notification_service.create_routed_events(
            event_type=event_type,
            severity=severity,
            title=title,
            message=message,
            targets=self._alert_targets(),
            details=details,
            template_key=event_type,
        )
        return (
            [self.notification_service._read(alert) for alert in created],
            len(created),
            0,
            0,
        )

    def run_monitoring_cycle(
        self,
        *,
        dataset_version: str | None = None,
        drift_top_error_count: int = 10,
        shadow_top_error_count: int = 5,
        shadow_max_candidates: int = 4,
        note: str | None = None,
        origin: str = "manual",
    ) -> ScanModelMonitoringResponse:
        started_at = datetime.now(timezone.utc).replace(microsecond=0)
        active_model_version = self.model_registry.active_version()
        job = JobExecution(
            job_type="model_monitoring_cycle",
            status="running",
            started_at=started_at,
            details={
                "origin": origin,
                "note": note,
                "active_model_version": active_model_version,
                "requested_dataset_version": dataset_version,
                "drift_top_error_count": drift_top_error_count,
                "shadow_top_error_count": shadow_top_error_count,
                "shadow_max_candidates": shadow_max_candidates,
            },
        )
        self.session.add(job)
        self.session.flush()

        try:
            resolved_dataset_version = dataset_version or self._latest_labels_dataset_version()
        except ApiError as exc:
            if exc.code != "model_monitoring_labels_dataset_not_found":
                completed_at = datetime.now(timezone.utc).replace(microsecond=0)
                job.status = "failed"
                job.completed_at = completed_at
                job.details = {
                    **(job.details or {}),
                    "error": exc.message,
                    "error_code": exc.code,
                    "error_type": type(exc).__name__,
                }
                self.session.commit()
                raise

            completed_at = datetime.now(timezone.utc).replace(microsecond=0)
            job.status = "skipped"
            job.completed_at = completed_at
            job.details = {
                **(job.details or {}),
                "reason": exc.message,
                "reason_code": exc.code,
            }
            self.session.commit()
            self.session.refresh(job)
            return ScanModelMonitoringResponse(
                job=self._job_read(job),
                activeModelVersion=active_model_version,
                datasetVersion=None,
                skipped=True,
                reason=exc.message,
                drift=None,
                shadow=None,
            )

        try:
            drift_response = self.drift_service.scan_model_drift(
                model_version=active_model_version,
                dataset_version=resolved_dataset_version,
                top_error_count=drift_top_error_count,
                origin=origin,
            )
            shadow_response = self.shadow_service.scan_shadow_run(
                dataset_version=resolved_dataset_version,
                max_candidates=shadow_max_candidates,
                top_error_count=shadow_top_error_count,
                origin=origin,
            )
        except Exception as exc:
            completed_at = datetime.now(timezone.utc).replace(microsecond=0)
            job.status = "failed"
            job.completed_at = completed_at
            job.details = {
                **(job.details or {}),
                "dataset_version": resolved_dataset_version,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            self.session.commit()
            raise

        completed_at = datetime.now(timezone.utc).replace(microsecond=0)
        alert_observed_at = completed_at
        created_alert_count = 0
        updated_alert_count = 0
        resolved_alert_count = 0
        touched_alerts: list[NotificationEventRead] = []

        (
            drift_should_alert,
            drift_alert_severity,
            drift_alert_title,
            drift_alert_message,
            drift_alert_details,
        ) = self._drift_alert_payload(
            active_model_version=active_model_version,
            dataset_version=resolved_dataset_version,
            drift_response=drift_response,
        )
        drift_alerts, drift_created, drift_updated, drift_resolved = (
            self._apply_alert_policy(
                event_type=MODEL_MONITORING_DRIFT_ALERT_EVENT_TYPE,
                should_alert=drift_should_alert,
                severity=drift_alert_severity,
                title=drift_alert_title,
                message=drift_alert_message,
                details=drift_alert_details,
                active_model_version=active_model_version,
                dataset_version=resolved_dataset_version,
                observed_at=alert_observed_at,
                origin=origin,
                resolution_reason="drift_no_longer_detected",
            )
        )
        created_alert_count += drift_created
        updated_alert_count += drift_updated
        resolved_alert_count += drift_resolved
        touched_alerts.extend(drift_alerts)

        (
            shadow_should_alert,
            shadow_alert_severity,
            shadow_alert_title,
            shadow_alert_message,
            shadow_alert_details,
        ) = self._shadow_alert_payload(
            active_model_version=active_model_version,
            dataset_version=resolved_dataset_version,
            shadow_response=shadow_response,
        )
        shadow_alerts, shadow_created, shadow_updated, shadow_resolved = (
            self._apply_alert_policy(
                event_type=MODEL_MONITORING_SHADOW_ALERT_EVENT_TYPE,
                should_alert=shadow_should_alert,
                severity=shadow_alert_severity,
                title=shadow_alert_title,
                message=shadow_alert_message,
                details=shadow_alert_details,
                active_model_version=active_model_version,
                dataset_version=resolved_dataset_version,
                observed_at=alert_observed_at,
                origin=origin,
                resolution_reason="shadow_recommendation_cleared",
            )
        )
        created_alert_count += shadow_created
        updated_alert_count += shadow_updated
        resolved_alert_count += shadow_resolved
        touched_alerts.extend(shadow_alerts)

        job.status = "completed"
        job.completed_at = completed_at
        job.details = {
            **(job.details or {}),
            "dataset_version": resolved_dataset_version,
            "drift_job_id": drift_response.job.id,
            "drift_version": drift_response.drift_version,
            "drift_severity": drift_response.severity,
            "shadow_job_id": shadow_response.job.id,
            "shadow_version": shadow_response.shadow_version,
            "shadow_status": (shadow_response.recommendation or {}).get("status"),
            "shadow_best_model_version": shadow_response.best_model_version,
            "created_alert_count": created_alert_count,
            "updated_alert_count": updated_alert_count,
            "resolved_alert_count": resolved_alert_count,
        }
        self.session.commit()
        self.session.refresh(job)

        return ScanModelMonitoringResponse(
            job=self._job_read(job),
            activeModelVersion=active_model_version,
            datasetVersion=resolved_dataset_version,
            skipped=False,
            reason=None,
            createdAlertCount=created_alert_count,
            updatedAlertCount=updated_alert_count,
            resolvedAlertCount=resolved_alert_count,
            alerts=touched_alerts,
            drift=drift_response,
            shadow=shadow_response,
        )
