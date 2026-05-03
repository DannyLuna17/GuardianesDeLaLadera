from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.core.config import get_settings
from app.core.exceptions import ApiError
from app.db.bootstrap import risk_level_from_score
from app.models import (
    HistoricalEvent,
    JobExecution,
    PredictionRun,
    UserAccount,
    Zone,
    ZoneOutcomeLabel,
    ZonePrediction,
)
from app.repositories.dashboard import DashboardRepository
from app.schemas.admin import (
    AssignTrainingReleaseResponse,
    AssignOutcomeLabelsResponse,
    AcknowledgeNotificationsResponse,
    EscalateTrainingReleaseResponse,
    FieldValidationObservationWrite,
    ImportFieldValidationLabelsResponse,
    ImportHistoricalLabelsResponse,
    ImportUngrdLabelsResponse,
    JobExecutionRead,
    NotificationDeliveryAttemptRead,
    NotificationDeliverySummaryRead,
    NotificationEventRead,
    OutcomeLabelRead,
    OutcomeLabelReleaseQueueRead,
    OutcomeLabelReviewQueueRead,
    OutcomeLabelWrite,
    ReassignTrainingReleaseResponse,
    RequestTrainingReleaseResponse,
    ReviewTrainingReleaseResponse,
    RetryNotificationDeliveryResponse,
    ReviewOutcomeLabelsResponse,
    TriggerTrainingReleaseReassignmentScanResponse,
    TriggerTrainingReleaseSlaScanResponse,
    TriggerNotificationAckScanResponse,
    TriggerNotificationDeliveryFailureScanResponse,
    TriggerNotificationDeliveryRetryScanResponse,
    UpdateTrainingEligibilityResponse,
    UpsertOutcomeLabelsResponse,
)
from app.services.notifications import NotificationService


DEFAULT_HISTORICAL_SEVERITY_SCORES = {
    "alta": 0.88,
    "media": 0.63,
    "baja": 0.36,
}

DEFAULT_FIELD_VALIDATION_SEVERITY_SCORES = {
    "alta": 0.9,
    "high": 0.9,
    "media": 0.64,
    "medium": 0.64,
    "baja": 0.38,
    "low": 0.38,
}

SOURCE_AWARE_DEDUP_PREFIXES = (
    "historical_event:",
    "ungrd_record:",
    "field_validation:",
)

DEFAULT_UNGRD_SUMMARY_SCORES = {
    "default": 0.56,
    "critical": 0.84,
    "high": 0.74,
    "medium": 0.58,
    "low": 0.42,
}

REVIEW_QUEUE_STATUSES = ("draft", "needs_revision")
DEFAULT_TRAINING_ELIGIBILITY_BY_STATUS = {
    "draft": "pending_review",
    "needs_revision": "pending_review",
    "confirmed": "eligible",
    "rejected": "ineligible",
}
TRAINING_RELEASE_PENDING_STATUS = "pending"
TRAINING_RELEASE_ESCALATED_STATUS = "escalated"


class OutcomeLabelService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.settings = get_settings()
        self.repository = DashboardRepository(session)
        self.notification_service = NotificationService(session)

    @staticmethod
    def _as_utc(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _normalize_observed_at(value: datetime) -> datetime:
        normalized = (
            value.astimezone(timezone.utc)
            if value.tzinfo
            else value.replace(tzinfo=timezone.utc)
        )
        return normalized.replace(microsecond=0)

    def _get_zone(self, zone_id: str) -> Zone:
        statement = (
            select(Zone)
            .where(Zone.id == zone_id)
            .options(joinedload(Zone.municipality))
        )
        zone = self.session.scalar(statement)
        if zone is None:
            raise ApiError(404, "zone_not_found", f"Zone '{zone_id}' was not found.")
        return zone

    def _resolve_feature_run(
        self,
        *,
        zone_id: str,
        observed_at: datetime,
        feature_run_id: int | None,
        allow_latest_fallback: bool = False,
    ) -> PredictionRun:
        statement = (
            select(PredictionRun)
            .join(PredictionRun.predictions)
            .where(
                ZonePrediction.zone_id == zone_id, PredictionRun.status == "completed"
            )
        )
        if feature_run_id is not None:
            statement = statement.where(PredictionRun.id == feature_run_id)
        else:
            statement = statement.where(
                PredictionRun.completed_at <= observed_at
            ).order_by(PredictionRun.completed_at.desc())
        statement = statement.limit(1)
        run = self.session.scalar(statement)
        if run is None and allow_latest_fallback:
            fallback_statement = (
                select(PredictionRun)
                .join(PredictionRun.predictions)
                .where(
                    ZonePrediction.zone_id == zone_id,
                    PredictionRun.status == "completed",
                )
                .order_by(PredictionRun.completed_at.desc())
                .limit(1)
            )
            run = self.session.scalar(fallback_statement)
        if run is None:
            if feature_run_id is not None:
                raise ApiError(
                    404,
                    "label_feature_run_not_found",
                    f"Completed run '{feature_run_id}' does not exist for zone '{zone_id}'.",
                )
            raise ApiError(
                400,
                "label_feature_run_resolution_failed",
                f"No completed run was found for zone '{zone_id}' at or before the label timestamp.",
            )
        return run

    @staticmethod
    def _normalize_datetime(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        normalized = (
            value.astimezone(timezone.utc)
            if value.tzinfo
            else value.replace(tzinfo=timezone.utc)
        )
        return normalized.replace(microsecond=0)

    @classmethod
    def _default_training_eligibility_for_status(cls, status: str) -> str:
        return DEFAULT_TRAINING_ELIGIBILITY_BY_STATUS.get(status, "pending_review")

    @classmethod
    def effective_training_eligibility_status(cls, label: ZoneOutcomeLabel) -> str:
        return (
            label.training_eligibility_status
            or cls._default_training_eligibility_for_status(label.status)
        )

    def _review_readiness(self, label: ZoneOutcomeLabel) -> dict:
        evidence = label.evidence or {}
        source = (label.source or "").lower()
        import_mode = str(evidence.get("import_mode") or "").lower()

        checks: list[tuple[str, bool]]
        if import_mode == "field_validation" or source.startswith("field_validation:"):
            checks = [
                ("observation_id", bool(evidence.get("observation_id"))),
                ("observer", bool(evidence.get("observer"))),
                ("site_visit_id", bool(evidence.get("site_visit_id"))),
                ("team_id", bool(evidence.get("team_id"))),
                ("location_notes", bool(evidence.get("location_notes"))),
                (
                    "gps_accuracy_meters",
                    evidence.get("gps_accuracy_meters") is not None,
                ),
                (
                    "media_or_attachment",
                    bool(
                        (evidence.get("media_refs") or [])
                        or (evidence.get("attachment_refs") or [])
                    ),
                ),
            ]
        elif import_mode == "historical_event" or source.startswith(
            "historical_event:"
        ):
            checks = [
                ("event_id", bool(evidence.get("event_id"))),
                ("event_source", bool(evidence.get("event_source"))),
                ("event_type", bool(evidence.get("event_type"))),
                ("event_severity", bool(evidence.get("event_severity"))),
            ]
        elif import_mode == "ungrd_record" or source.startswith("ungrd_record:"):
            checks = [
                ("record_id", bool(evidence.get("record_id"))),
                ("record_summary", bool(evidence.get("record_summary"))),
                ("municipality", bool(evidence.get("municipality"))),
                (
                    "zone_rank_within_record",
                    evidence.get("zone_rank_within_record") is not None,
                ),
            ]
        else:
            checks = [
                ("source", bool(label.source)),
                ("feature_run_id", label.feature_run_id is not None),
            ]

        completed = sum(1 for _, is_present in checks if is_present)
        score = round(completed / len(checks), 3) if checks else 1.0
        missing_fields = [
            field_name for field_name, is_present in checks if not is_present
        ]
        normalized_due_at = self._normalize_datetime(label.review_due_at)
        now = datetime.now(timezone.utc).replace(microsecond=0)
        is_queue_status = label.status in REVIEW_QUEUE_STATUSES
        return {
            "score": score,
            "missing_fields": missing_fields,
            "ready_for_review": is_queue_status and not missing_fields,
            "is_overdue": is_queue_status
            and normalized_due_at is not None
            and normalized_due_at < now,
        }

    def _training_release_overdue(self, label: ZoneOutcomeLabel) -> bool:
        normalized_due_at = self._normalize_datetime(label.training_release_due_at)
        if (
            label.training_release_status != TRAINING_RELEASE_PENDING_STATUS
            or normalized_due_at is None
        ):
            return False
        return normalized_due_at < datetime.now(timezone.utc).replace(microsecond=0)

    @staticmethod
    def _training_release_escalated(label: ZoneOutcomeLabel) -> bool:
        return (
            label.training_release_status == TRAINING_RELEASE_PENDING_STATUS
            and label.training_release_escalation_status
            == TRAINING_RELEASE_ESCALATED_STATUS
        )

    def _label_read(self, label: ZoneOutcomeLabel) -> OutcomeLabelRead:
        zone = label.zone
        municipality_name = zone.municipality.name if zone and zone.municipality else ""
        feature_run_completed_at = (
            label.feature_run.completed_at if label.feature_run else None
        )
        readiness = self._review_readiness(label)
        return OutcomeLabelRead(
            id=label.id,
            zoneId=label.zone_id,
            zoneName=zone.name if zone else label.zone_id,
            municipality=municipality_name,
            observedAt=label.observed_at,
            targetScore=round(label.target_score, 3),
            targetRiskLevel=risk_level_from_score(label.target_score),
            source=label.source,
            status=label.status,
            featureRunId=label.feature_run_id,
            featureRunCompletedAt=feature_run_completed_at,
            notes=label.notes,
            evidence=label.evidence or {},
            assignedReviewer=label.assigned_reviewer,
            assignedAt=label.assigned_at,
            reviewDueAt=label.review_due_at,
            trainingEligibilityStatus=self.effective_training_eligibility_status(label),
            trainingEligibilityUpdatedAt=label.training_eligibility_updated_at,
            trainingEligibilityUpdatedBy=label.training_eligibility_updated_by,
            trainingEligibilityNotes=label.training_eligibility_notes,
            trainingReleaseStatus=label.training_release_status,
            trainingReleaseCriteria=label.training_release_criteria or [],
            trainingReleaseRequestedAt=label.training_release_requested_at,
            trainingReleaseRequestedBy=label.training_release_requested_by,
            trainingReleaseRequestedNotes=label.training_release_requested_notes,
            trainingReleaseReviewedAt=label.training_release_reviewed_at,
            trainingReleaseReviewedBy=label.training_release_reviewed_by,
            trainingReleaseReviewNotes=label.training_release_review_notes,
            trainingReleaseAssignedReviewer=label.training_release_assigned_reviewer,
            trainingReleaseAssignedAt=label.training_release_assigned_at,
            trainingReleaseDueAt=label.training_release_due_at,
            trainingReleaseIsOverdue=self._training_release_overdue(label),
            trainingReleaseEscalationStatus=label.training_release_escalation_status,
            trainingReleaseEscalationLevel=label.training_release_escalation_level,
            trainingReleaseEscalatedAt=label.training_release_escalated_at,
            trainingReleaseEscalatedBy=label.training_release_escalated_by,
            trainingReleaseEscalationReason=label.training_release_escalation_reason,
            trainingReleaseIsEscalated=self._training_release_escalated(label),
            evidenceCompletenessScore=readiness["score"],
            missingEvidenceFields=readiness["missing_fields"],
            readyForReview=readiness["ready_for_review"],
            isOverdue=readiness["is_overdue"],
            reviewedAt=label.reviewed_at,
            reviewedBy=label.reviewed_by,
            reviewNotes=label.review_notes,
            createdAt=label.created_at,
            updatedAt=label.updated_at,
        )

    def list_labels(
        self,
        *,
        zone_id: str | None = None,
        source: str | None = None,
        status: str | None = None,
        training_eligibility_status: str | None = None,
        training_release_status: str | None = None,
        training_release_escalation_status: str | None = None,
        limit: int = 100,
    ) -> list[OutcomeLabelRead]:
        statement = (
            select(ZoneOutcomeLabel)
            .options(
                joinedload(ZoneOutcomeLabel.zone).joinedload(Zone.municipality),
                joinedload(ZoneOutcomeLabel.feature_run),
            )
            .order_by(ZoneOutcomeLabel.observed_at.desc(), ZoneOutcomeLabel.id.desc())
        )
        if zone_id:
            statement = statement.where(ZoneOutcomeLabel.zone_id == zone_id)
        if source:
            statement = statement.where(ZoneOutcomeLabel.source == source)
        if status:
            statement = statement.where(ZoneOutcomeLabel.status == status)
        if training_eligibility_status is None:
            statement = statement.limit(limit)
        labels = list(self.session.scalars(statement).unique().all())
        label_reads = [self._label_read(label) for label in labels]
        if training_eligibility_status:
            normalized_status = training_eligibility_status.strip().lower()
            label_reads = [
                label
                for label in label_reads
                if label.training_eligibility_status.lower() == normalized_status
            ]
        if training_release_status:
            normalized_release_status = training_release_status.strip().lower()
            label_reads = [
                label
                for label in label_reads
                if (label.training_release_status or "").lower()
                == normalized_release_status
            ]
        if training_release_escalation_status:
            normalized_escalation_status = (
                training_release_escalation_status.strip().lower()
            )
            label_reads = [
                label
                for label in label_reads
                if (label.training_release_escalation_status or "").lower()
                == normalized_escalation_status
            ]
        if (
            training_eligibility_status
            or training_release_status
            or training_release_escalation_status
        ):
            label_reads = label_reads[:limit]
        return label_reads

    def list_notifications(
        self,
        *,
        status: str | None = None,
        severity: str | None = None,
        target_username: str | None = None,
        event_type: str | None = None,
        channel: str | None = None,
        delivery_status: str | None = None,
        overdue_only: bool = False,
        limit: int = 100,
    ) -> list[NotificationEventRead]:
        return self.notification_service.list_notifications(
            status=status,
            severity=severity,
            target_username=target_username,
            event_type=event_type,
            channel=channel,
            delivery_status=delivery_status,
            overdue_only=overdue_only,
            limit=limit,
        )

    def list_notification_delivery_attempts(
        self,
        *,
        notification_id: int | None = None,
        channel: str | None = None,
        status: str | None = None,
        target_username: str | None = None,
        provider_name: str | None = None,
        failure_classification: str | None = None,
        delivery_origin: str | None = None,
        limit: int = 100,
    ) -> list[NotificationDeliveryAttemptRead]:
        return self.notification_service.list_delivery_attempts(
            notification_id=notification_id,
            channel=channel,
            status=status,
            target_username=target_username,
            provider_name=provider_name,
            failure_classification=failure_classification,
            delivery_origin=delivery_origin,
            limit=limit,
        )

    def get_notification_delivery_summary(self) -> NotificationDeliverySummaryRead:
        return self.notification_service.get_delivery_summary()

    def acknowledge_notifications(
        self,
        *,
        notification_ids: list[int],
        acknowledged_by: str,
    ) -> AcknowledgeNotificationsResponse:
        return self.notification_service.acknowledge_notifications(
            notification_ids=notification_ids,
            acknowledged_by=acknowledged_by,
        )

    def retry_notification_delivery(
        self,
        *,
        notification_ids: list[int],
        triggered_by: str,
        channels: list[str] | None = None,
        note: str | None = None,
        origin: str = "manual",
    ) -> RetryNotificationDeliveryResponse:
        return self.notification_service.retry_delivery(
            notification_ids=notification_ids,
            triggered_by=triggered_by,
            channels=channels,
            note=note,
            origin=origin,
        )

    def run_notification_ack_scan(
        self,
        *,
        max_notifications: int = 100,
        note: str | None = None,
        origin: str = "manual",
    ) -> TriggerNotificationAckScanResponse:
        return self.notification_service.run_ack_deadline_scan(
            max_notifications=max_notifications,
            note=note,
            origin=origin,
        )

    def run_notification_delivery_retry_scan(
        self,
        *,
        max_notifications: int = 100,
        note: str | None = None,
        origin: str = "manual",
    ) -> TriggerNotificationDeliveryRetryScanResponse:
        return self.notification_service.run_delivery_retry_scan(
            max_notifications=max_notifications,
            note=note,
            origin=origin,
        )

    def run_notification_delivery_failure_scan(
        self,
        *,
        max_notifications: int = 100,
        note: str | None = None,
        origin: str = "manual",
    ) -> TriggerNotificationDeliveryFailureScanResponse:
        return self.notification_service.run_delivery_failure_scan(
            max_notifications=max_notifications,
            note=note,
            origin=origin,
        )

    def run_training_release_sla_scan(
        self,
        *,
        max_labels: int = 100,
        note: str | None = None,
        origin: str = "manual",
    ) -> TriggerTrainingReleaseSlaScanResponse:
        started_at = datetime.now(timezone.utc).replace(microsecond=0)
        job = JobExecution(
            job_type="training_release_sla_scan",
            status="running",
            started_at=started_at,
            details={"origin": origin, "max_labels": max_labels, "note": note},
        )
        self.session.add(job)
        self.session.flush()

        statement = (
            select(ZoneOutcomeLabel)
            .where(
                ZoneOutcomeLabel.training_release_status
                == TRAINING_RELEASE_PENDING_STATUS,
                ZoneOutcomeLabel.training_release_due_at.is_not(None),
                ZoneOutcomeLabel.training_release_due_at < started_at,
            )
            .options(
                joinedload(ZoneOutcomeLabel.zone).joinedload(Zone.municipality),
                joinedload(ZoneOutcomeLabel.feature_run),
            )
            .order_by(
                ZoneOutcomeLabel.training_release_due_at.asc(),
                ZoneOutcomeLabel.id.asc(),
            )
            .limit(max_labels)
        )
        candidates = list(self.session.scalars(statement).unique().all())
        target_labels = [
            label
            for label in candidates
            if label.training_release_escalation_status
            != TRAINING_RELEASE_ESCALATED_STATUS
        ]

        escalated_labels: list[OutcomeLabelRead] = []
        notification_count = 0
        if target_labels:
            escalation_response = self.escalate_training_release(
                label_ids=[label.id for label in target_labels],
                escalation_reason=note
                or "Automatic SLA escalation: pending release review is overdue.",
                escalated_by=f"system:{origin}",
                escalation_level=self.settings.training_release_auto_escalation_level,
                notification_event_type="training_release_auto_escalation",
                notification_severity="critical",
            )
            escalated_labels = escalation_response.labels
            notification_count = escalation_response.escalated_count

        completed_at = datetime.now(timezone.utc).replace(microsecond=0)
        job.status = "completed"
        job.completed_at = completed_at
        job.details = {
            "origin": origin,
            "max_labels": max_labels,
            "note": note,
            "candidates": len(candidates),
            "escalated_count": len(escalated_labels),
            "notification_count": notification_count,
            "auto_escalation_level": self.settings.training_release_auto_escalation_level,
        }
        self.session.commit()
        self.session.refresh(job)

        return TriggerTrainingReleaseSlaScanResponse(
            job=self._job_read(job),
            escalatedCount=len(escalated_labels),
            notificationCount=notification_count,
            labels=escalated_labels,
        )

    def _get_reviewer_user(self, username: str) -> UserAccount:
        statement = select(UserAccount).where(
            UserAccount.username == username, UserAccount.is_active.is_(True)
        )
        user = self.session.scalar(statement)
        if user is None:
            raise ApiError(
                404,
                "reviewer_not_found",
                f"Active reviewer '{username}' was not found.",
            )
        return user

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

    @staticmethod
    def _normalize_ungrd_scores(
        overrides: dict[str, float] | None = None,
    ) -> dict[str, float]:
        scores = dict(DEFAULT_UNGRD_SUMMARY_SCORES)
        if overrides:
            for key, value in overrides.items():
                scores[key.strip().lower()] = float(value)
        return scores

    @staticmethod
    def _ungrd_record_observed_at(record_date: datetime | None) -> datetime:
        if record_date is None:
            return datetime.now(timezone.utc).replace(microsecond=0)
        if (
            hasattr(record_date, "year")
            and hasattr(record_date, "month")
            and hasattr(record_date, "day")
        ):
            return datetime(
                record_date.year,
                record_date.month,
                record_date.day,
                12,
                0,
                tzinfo=timezone.utc,
            )
        return datetime.now(timezone.utc).replace(microsecond=0)

    @classmethod
    def _score_ungrd_summary(
        cls, summary: str, overrides: dict[str, float] | None = None
    ) -> float:
        text = summary.lower()
        scores = cls._normalize_ungrd_scores(overrides)
        score = scores["default"]

        if any(
            token in text
            for token in ["movimientos en masa", "remocion en masa", "desliz"]
        ):
            score = max(score, scores["high"])
        if any(
            token in text
            for token in [
                "afectacion vial",
                "carretera",
                "corredor rural",
                "invias",
                "via ",
            ]
        ):
            score = max(score, scores["high"] + 0.04)
        if any(
            token in text
            for token in ["comites de riesgo", "incremento", "saturacion de suelos"]
        ):
            score = max(score, scores["high"])
        if any(
            token in text
            for token in ["riesgo medio", "taludes", "verificacion tecnica"]
        ):
            score = max(score, scores["medium"])
        if any(
            token in text
            for token in ["menores", "seguimiento", "vigilancia", "parcial"]
        ):
            score = min(score, scores["medium"])

        if "critica" in text or "evacuacion" in text:
            score = max(score, scores["critical"])

        return max(0.0, min(1.0, round(score, 3)))

    @staticmethod
    def _merge_notes(existing: str | None, incoming: str | None) -> str | None:
        if existing and incoming:
            return f"{existing}\n{incoming}"
        return incoming or existing

    @staticmethod
    def _is_source_aware_import_source(source: str) -> bool:
        return source.startswith(SOURCE_AWARE_DEDUP_PREFIXES)

    def _find_existing_label_for_upsert(
        self,
        *,
        zone_id: str,
        observed_at: datetime,
        source: str,
    ) -> ZoneOutcomeLabel | None:
        exact_statement = select(ZoneOutcomeLabel).where(
            ZoneOutcomeLabel.zone_id == zone_id,
            ZoneOutcomeLabel.observed_at == observed_at,
            ZoneOutcomeLabel.source == source,
        )
        existing = self.session.scalar(exact_statement)
        if existing is not None:
            return existing
        if not self._is_source_aware_import_source(source):
            return None
        return self.session.scalar(
            select(ZoneOutcomeLabel).where(
                ZoneOutcomeLabel.zone_id == zone_id,
                ZoneOutcomeLabel.source == source,
            )
        )

    @staticmethod
    def _append_review_history(
        evidence: dict,
        *,
        reviewer_username: str,
        decision: str,
        review_notes: str | None,
        reviewed_at: datetime,
    ) -> dict:
        updated_evidence = dict(evidence or {})
        history = list(updated_evidence.get("review_history") or [])
        history.append(
            {
                "reviewed_at": reviewed_at.isoformat(),
                "reviewed_by": reviewer_username,
                "decision": decision,
                "review_notes": review_notes,
            }
        )
        updated_evidence["review_history"] = history
        return updated_evidence

    @staticmethod
    def _append_assignment_history(
        evidence: dict,
        *,
        assigned_by: str,
        reviewer_username: str,
        assignment_notes: str | None,
        assigned_at: datetime,
        review_due_at: datetime | None,
    ) -> dict:
        updated_evidence = dict(evidence or {})
        history = list(updated_evidence.get("assignment_history") or [])
        history.append(
            {
                "assigned_at": assigned_at.isoformat(),
                "assigned_by": assigned_by,
                "reviewer_username": reviewer_username,
                "assignment_notes": assignment_notes,
                "review_due_at": review_due_at.isoformat() if review_due_at else None,
            }
        )
        updated_evidence["assignment_history"] = history
        updated_evidence["last_assigned_reviewer"] = reviewer_username
        return updated_evidence

    @staticmethod
    def _append_training_eligibility_history(
        evidence: dict,
        *,
        updated_by: str,
        training_eligibility_status: str,
        notes: str | None,
        updated_at: datetime,
    ) -> dict:
        updated_evidence = dict(evidence or {})
        history = list(updated_evidence.get("training_eligibility_history") or [])
        history.append(
            {
                "updated_at": updated_at.isoformat(),
                "updated_by": updated_by,
                "training_eligibility_status": training_eligibility_status,
                "notes": notes,
            }
        )
        updated_evidence["training_eligibility_history"] = history
        updated_evidence["training_eligibility_status"] = training_eligibility_status
        return updated_evidence

    @staticmethod
    def _append_training_release_history(
        evidence: dict,
        *,
        action: str,
        acted_by: str,
        criteria: list[str] | None,
        notes: str | None,
        acted_at: datetime,
        decision: str | None = None,
    ) -> dict:
        updated_evidence = dict(evidence or {})
        history = list(updated_evidence.get("training_release_history") or [])
        history.append(
            {
                "action": action,
                "acted_at": acted_at.isoformat(),
                "acted_by": acted_by,
                "criteria": list(criteria or []),
                "notes": notes,
                "decision": decision,
            }
        )
        updated_evidence["training_release_history"] = history
        return updated_evidence

    @staticmethod
    def _append_training_release_assignment_history(
        evidence: dict,
        *,
        assigned_by: str,
        reviewer_username: str,
        assignment_notes: str | None,
        assigned_at: datetime,
        review_due_at: datetime | None,
    ) -> dict:
        updated_evidence = dict(evidence or {})
        history = list(
            updated_evidence.get("training_release_assignment_history") or []
        )
        history.append(
            {
                "assigned_at": assigned_at.isoformat(),
                "assigned_by": assigned_by,
                "reviewer_username": reviewer_username,
                "assignment_notes": assignment_notes,
                "review_due_at": review_due_at.isoformat() if review_due_at else None,
            }
        )
        updated_evidence["training_release_assignment_history"] = history
        updated_evidence["last_training_release_assigned_reviewer"] = reviewer_username
        return updated_evidence

    @staticmethod
    def _append_training_release_reassignment_history(
        evidence: dict,
        *,
        previous_reviewer_username: str | None,
        reviewer_username: str,
        reassigned_by: str,
        reassignment_reason: str,
        reassigned_at: datetime,
        review_due_at: datetime | None,
    ) -> dict:
        updated_evidence = dict(evidence or {})
        history = list(
            updated_evidence.get("training_release_reassignment_history") or []
        )
        history.append(
            {
                "reassigned_at": reassigned_at.isoformat(),
                "reassigned_by": reassigned_by,
                "previous_reviewer_username": previous_reviewer_username,
                "reviewer_username": reviewer_username,
                "reassignment_reason": reassignment_reason,
                "review_due_at": review_due_at.isoformat() if review_due_at else None,
            }
        )
        updated_evidence["training_release_reassignment_history"] = history
        updated_evidence["last_training_release_reassigned_reviewer"] = (
            reviewer_username
        )
        return updated_evidence

    @staticmethod
    def _append_training_release_escalation_history(
        evidence: dict,
        *,
        escalated_by: str,
        escalation_reason: str,
        escalation_level: int,
        escalated_at: datetime,
    ) -> dict:
        updated_evidence = dict(evidence or {})
        history = list(
            updated_evidence.get("training_release_escalation_history") or []
        )
        history.append(
            {
                "escalated_at": escalated_at.isoformat(),
                "escalated_by": escalated_by,
                "escalation_reason": escalation_reason,
                "escalation_level": escalation_level,
            }
        )
        updated_evidence["training_release_escalation_history"] = history
        updated_evidence["last_training_release_escalation_level"] = escalation_level
        return updated_evidence

    @staticmethod
    def _notification_target_for_release(label: ZoneOutcomeLabel) -> str | None:
        return (
            label.training_release_assigned_reviewer
            or label.training_release_requested_by
        )

    @staticmethod
    def _dedupe_notification_targets(targets: list[dict]) -> list[dict]:
        normalized_targets: list[dict] = []
        seen_usernames: set[str] = set()
        primary_assigned = False
        for target in targets:
            username = str(target.get("username") or "").strip()
            if not username or username in seen_usernames:
                continue
            seen_usernames.add(username)
            is_primary = bool(target.get("is_primary", False))
            if is_primary and primary_assigned:
                is_primary = False
            if is_primary:
                primary_assigned = True
            normalized_targets.append(
                {
                    "username": username,
                    "routing_audience": str(target.get("routing_audience") or "direct"),
                    "routing_reason": str(target.get("routing_reason") or "direct"),
                    "is_primary": is_primary,
                }
            )
        if normalized_targets and not primary_assigned:
            normalized_targets[0]["is_primary"] = True
        return normalized_targets

    def _release_assignment_notification_targets(
        self, reviewer_username: str
    ) -> list[dict]:
        return self._dedupe_notification_targets(
            [
                {
                    "username": reviewer_username,
                    "routing_audience": "assigned_reviewer",
                    "routing_reason": "release_assignment",
                    "is_primary": True,
                }
            ]
        )

    def _release_escalation_notification_targets(
        self, label: ZoneOutcomeLabel
    ) -> list[dict]:
        targets: list[dict] = []
        assigned_reviewer = (label.training_release_assigned_reviewer or "").strip()
        if assigned_reviewer:
            targets.append(
                {
                    "username": assigned_reviewer,
                    "routing_audience": "assigned_reviewer",
                    "routing_reason": "active_release_reviewer",
                    "is_primary": True,
                }
            )
        requester = (label.training_release_requested_by or "").strip()
        if requester and self.settings.notification_escalation_include_requester:
            targets.append(
                {
                    "username": requester,
                    "routing_audience": "requester_copy",
                    "routing_reason": "release_request_requester",
                    "is_primary": not targets,
                }
            )
        for username in self.settings.notification_release_ops_usernames:
            targets.append(
                {
                    "username": username,
                    "routing_audience": "ops_watch",
                    "routing_reason": "configured_ops_watcher",
                    "is_primary": False,
                }
            )
        return self._dedupe_notification_targets(targets)

    def _release_resolution_notification_targets(
        self,
        *,
        requester_username: str | None,
        assigned_reviewer_username: str | None,
    ) -> list[dict]:
        targets: list[dict] = []
        requester = (requester_username or "").strip()
        if requester:
            targets.append(
                {
                    "username": requester,
                    "routing_audience": "requester",
                    "routing_reason": "release_request_requester",
                    "is_primary": True,
                }
            )
        assigned_reviewer = (assigned_reviewer_username or "").strip()
        if (
            assigned_reviewer
            and self.settings.notification_resolution_copy_assigned_reviewer
        ):
            targets.append(
                {
                    "username": assigned_reviewer,
                    "routing_audience": "assigned_reviewer_copy",
                    "routing_reason": "active_release_reviewer",
                    "is_primary": not targets,
                }
            )
        return self._dedupe_notification_targets(targets)

    def _release_reassignment_notification_targets(
        self,
        *,
        reviewer_username: str,
        previous_reviewer_username: str | None,
    ) -> list[dict]:
        targets: list[dict] = [
            {
                "username": reviewer_username,
                "routing_audience": "assigned_reviewer",
                "routing_reason": "release_reassignment_target",
                "is_primary": True,
            }
        ]
        previous_reviewer = (previous_reviewer_username or "").strip()
        if (
            previous_reviewer
            and self.settings.notification_reassignment_copy_previous_reviewer
        ):
            targets.append(
                {
                    "username": previous_reviewer,
                    "routing_audience": "previous_reviewer_copy",
                    "routing_reason": "release_reassignment_previous_reviewer",
                    "is_primary": False,
                }
            )
        return self._dedupe_notification_targets(targets)

    def _create_release_assignment_notification(
        self, label: ZoneOutcomeLabel, reviewer_username: str
    ) -> None:
        zone_name = label.zone.name if label.zone is not None else label.zone_id
        due_at = self._normalize_datetime(label.training_release_due_at)
        due_display = due_at.isoformat() if due_at else None
        self.notification_service.create_routed_events(
            event_type="training_release_assignment",
            severity="info",
            title=f"Release review assigned for {zone_name}",
            message=(
                f"Governed label {label.id} for {zone_name} now requires second-pass release review by "
                f"{reviewer_username}."
                + (f" Review before {due_display}." if due_display else "")
                + " Action: inspect evidence completeness and approve or reject the release request."
            ),
            targets=self._release_assignment_notification_targets(reviewer_username),
            related_label_id=label.id,
            details={
                "label_id": label.id,
                "zone_id": label.zone_id,
                "zone_name": zone_name,
                "training_release_due_at": due_display,
                "summary": "Release review assigned",
                "recommended_action": "Review the governed label evidence and resolve the release request before the due date.",
            },
            template_key="release_assignment_review",
        )

    def _create_release_escalation_notification(
        self,
        label: ZoneOutcomeLabel,
        *,
        event_type: str,
        severity: str,
        escalation_reason: str,
        escalation_level: int,
        escalated_by: str,
    ) -> None:
        zone_name = label.zone.name if label.zone is not None else label.zone_id
        self.notification_service.create_routed_events(
            event_type=event_type,
            severity=severity,
            title=f"Release review escalated for {zone_name}",
            message=(
                f"Governed label {label.id} for {zone_name} was escalated to level {escalation_level}. "
                f"Reason: {escalation_reason}. Action: prioritize review and document the release decision."
            ),
            targets=self._release_escalation_notification_targets(label),
            related_label_id=label.id,
            details={
                "label_id": label.id,
                "zone_id": label.zone_id,
                "zone_name": zone_name,
                "escalation_level": escalation_level,
                "escalated_by": escalated_by,
                "escalation_reason": escalation_reason,
                "summary": "Release review escalated",
                "recommended_action": "Prioritize this label in the release queue and resolve the escalation path.",
            },
            template_key="release_escalation_notice",
        )

    def _create_release_resolution_notification(
        self,
        label: ZoneOutcomeLabel,
        *,
        decision: str,
        reviewed_by: str,
        assigned_reviewer_username: str | None = None,
    ) -> None:
        zone_name = label.zone.name if label.zone is not None else label.zone_id
        targets = self._release_resolution_notification_targets(
            requester_username=label.training_release_requested_by,
            assigned_reviewer_username=assigned_reviewer_username,
        )
        if not targets:
            return
        outcome_text = (
            "returned to training eligibility"
            if decision == "approved"
            else "remains on hold"
        )
        self.notification_service.create_routed_events(
            event_type="training_release_resolution",
            severity="info" if decision == "approved" else "warning",
            title=f"Release review resolved for {zone_name}",
            message=(
                f"Governed label {label.id} release review was {decision} by {reviewed_by}; the label "
                f"{outcome_text}. Action: verify the downstream training-governance state."
            ),
            targets=targets,
            related_label_id=label.id,
            details={
                "label_id": label.id,
                "zone_id": label.zone_id,
                "zone_name": zone_name,
                "decision": decision,
                "reviewed_by": reviewed_by,
                "summary": "Release review resolved",
                "recommended_action": "Confirm that the resulting training-governance state matches the release decision.",
            },
            template_key="release_resolution_notice",
        )

    def _create_release_reassignment_notification(
        self,
        label: ZoneOutcomeLabel,
        *,
        reviewer_username: str,
        previous_reviewer_username: str | None,
        reassignment_reason: str,
    ) -> None:
        zone_name = label.zone.name if label.zone is not None else label.zone_id
        due_at = self._normalize_datetime(label.training_release_due_at)
        due_display = due_at.isoformat() if due_at else None
        self.notification_service.create_routed_events(
            event_type="training_release_reassignment",
            severity="warning",
            title=f"Release review reassigned for {zone_name}",
            message=(
                f"Governed label {label.id} for {zone_name} was reassigned to {reviewer_username}."
                + (
                    f" Previous reviewer: {previous_reviewer_username}."
                    if previous_reviewer_username
                    else ""
                )
                + f" Reason: {reassignment_reason}."
                + (f" New due date: {due_display}." if due_display else "")
                + " Action: confirm ownership and continue the release review."
            ),
            targets=self._release_reassignment_notification_targets(
                reviewer_username=reviewer_username,
                previous_reviewer_username=previous_reviewer_username,
            ),
            related_label_id=label.id,
            details={
                "label_id": label.id,
                "zone_id": label.zone_id,
                "zone_name": zone_name,
                "reviewer_username": reviewer_username,
                "previous_reviewer_username": previous_reviewer_username,
                "reassignment_reason": reassignment_reason,
                "training_release_due_at": due_display,
                "summary": "Release review reassigned",
                "recommended_action": "Confirm the new reviewer assignment and continue the release workflow.",
            },
            template_key="release_reassignment_notice",
        )

    @staticmethod
    def _normalized_release_criteria(criteria: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in criteria:
            value = item.strip()
            if value and value not in normalized:
                normalized.append(value)
        return normalized

    @staticmethod
    def _clear_training_release_fields(label: ZoneOutcomeLabel) -> None:
        label.training_release_status = None
        label.training_release_criteria = None
        label.training_release_requested_at = None
        label.training_release_requested_by = None
        label.training_release_requested_notes = None
        label.training_release_reviewed_at = None
        label.training_release_reviewed_by = None
        label.training_release_review_notes = None
        label.training_release_assigned_reviewer = None
        label.training_release_assigned_at = None
        label.training_release_due_at = None
        label.training_release_escalation_status = None
        label.training_release_escalation_level = None
        label.training_release_escalated_at = None
        label.training_release_escalated_by = None
        label.training_release_escalation_reason = None

    @staticmethod
    def _merge_operator_evidence(existing: dict | None, incoming: dict | None) -> dict:
        merged = dict(existing or {})
        merged.update(incoming or {})
        for key in (
            "review_history",
            "assignment_history",
            "training_eligibility_history",
            "training_release_history",
            "training_release_assignment_history",
            "training_release_reassignment_history",
            "training_release_escalation_history",
            "last_assigned_reviewer",
            "training_eligibility_status",
            "last_training_release_assigned_reviewer",
            "last_training_release_reassigned_reviewer",
            "last_training_release_escalation_level",
        ):
            if existing and key in existing:
                merged[key] = existing[key]
        return merged

    @staticmethod
    def _normalize_field_validation_scores(
        overrides: dict[str, float] | None = None,
    ) -> dict[str, float]:
        scores = dict(DEFAULT_FIELD_VALIDATION_SEVERITY_SCORES)
        if overrides:
            for key, value in overrides.items():
                scores[key.strip().lower()] = float(value)
        return scores

    @staticmethod
    def _normalize_severity_scores(
        overrides: dict[str, float] | None = None,
    ) -> dict[str, float]:
        scores = dict(DEFAULT_HISTORICAL_SEVERITY_SCORES)
        if overrides:
            for key, value in overrides.items():
                scores[key.strip().lower()] = float(value)
        return scores

    @staticmethod
    def _historical_label_observed_at(event: HistoricalEvent) -> datetime:
        return datetime(
            event.date.year,
            event.date.month,
            event.date.day,
            12,
            0,
            tzinfo=timezone.utc,
        )

    def import_historical_event_labels(
        self,
        *,
        municipality: str | None = None,
        zone_id: str | None = None,
        event_ids: list[str] | None = None,
        event_source: str | None = None,
        status: str = "draft",
        max_events: int = 100,
        severity_score_overrides: dict[str, float] | None = None,
    ) -> ImportHistoricalLabelsResponse:
        severity_scores = self._normalize_severity_scores(severity_score_overrides)
        target_event_ids = set(event_ids or [])
        normalized_event_source = event_source.lower().strip() if event_source else None

        if zone_id:
            zone = self._get_zone(zone_id)
            zones = [zone]
        else:
            zones = self.repository.list_zones()
            if municipality:
                zones = [
                    zone
                    for zone in zones
                    if zone.municipality.name.lower() == municipality.lower()
                ]

        prepared_labels: list[OutcomeLabelWrite] = []
        imported_event_ids: list[str] = []
        skipped_count = 0

        for zone in zones:
            events = self.repository.list_historical_events_for_zone(zone.id)
            for event in events:
                if target_event_ids and event.id not in target_event_ids:
                    continue
                if (
                    normalized_event_source
                    and event.source.lower() != normalized_event_source
                ):
                    continue
                if len(prepared_labels) >= max_events:
                    break

                observed_at = self._historical_label_observed_at(event)
                try:
                    feature_run = self._resolve_feature_run(
                        zone_id=zone.id,
                        observed_at=observed_at,
                        feature_run_id=None,
                        allow_latest_fallback=True,
                    )
                except ApiError:
                    skipped_count += 1
                    continue

                severity_key = event.severity.strip().lower()
                target_score = severity_scores.get(severity_key)
                if target_score is None:
                    skipped_count += 1
                    continue

                feature_run_completed_at = self._as_utc(feature_run.completed_at)
                used_fallback_run = (
                    feature_run_completed_at > observed_at
                    if feature_run_completed_at is not None
                    else False
                )
                prepared_labels.append(
                    OutcomeLabelWrite(
                        zoneId=zone.id,
                        observedAt=observed_at,
                        targetScore=target_score,
                        source=f"historical_event:{event.id}",
                        featureRunId=feature_run.id,
                        status=status,
                        notes=f"Imported from historical event {event.id} ({event.type}, {event.severity}).",
                        evidence={
                            "import_mode": "historical_event",
                            "dedup_strategy": "zone_source",
                            "event_id": event.id,
                            "event_source": event.source,
                            "event_type": event.type,
                            "event_severity": event.severity,
                            "municipality": zone.municipality.name,
                            "zone_id": zone.id,
                            "feature_run_resolution": (
                                "latest_available_backfill"
                                if used_fallback_run
                                else "historical_match"
                            ),
                        },
                    )
                )
                imported_event_ids.append(event.id)

            if len(prepared_labels) >= max_events:
                break

        if not prepared_labels:
            return ImportHistoricalLabelsResponse(
                createdCount=0,
                updatedCount=0,
                skippedCount=skipped_count,
                importedEventIds=[],
                labels=[],
            )

        upsert_response = self.upsert_labels(prepared_labels)
        return ImportHistoricalLabelsResponse(
            createdCount=upsert_response.created_count,
            updatedCount=upsert_response.updated_count,
            skippedCount=skipped_count,
            importedEventIds=imported_event_ids,
            labels=upsert_response.labels,
        )

    def import_ungrd_record_labels(
        self,
        *,
        municipality: str | None = None,
        zone_id: str | None = None,
        record_ids: list[str] | None = None,
        status: str = "draft",
        max_records: int = 50,
        max_zones_per_record: int = 2,
        summary_score_overrides: dict[str, float] | None = None,
    ) -> ImportUngrdLabelsResponse:
        target_record_ids = set(record_ids or [])
        records = self.repository.list_ungrd_records()
        if municipality:
            records = [
                record
                for record in records
                if record.municipality.name.lower() == municipality.lower()
            ]
        if target_record_ids:
            records = [record for record in records if record.id in target_record_ids]
        records = records[:max_records]

        selected_zone = self._get_zone(zone_id) if zone_id else None
        prepared_labels: list[OutcomeLabelWrite] = []
        imported_record_ids: list[str] = []
        skipped_count = 0

        for record in records:
            if selected_zone is not None:
                if selected_zone.municipality_id != record.municipality_id:
                    skipped_count += 1
                    continue
                candidate_zone_predictions = []
            else:
                candidate_zone_predictions = (
                    self.repository.list_latest_zone_predictions(
                        municipality=record.municipality.name
                    )[:max_zones_per_record]
                )

            if selected_zone is not None:
                candidate_zones = [selected_zone]
            else:
                candidate_zones = [
                    prediction.zone for prediction in candidate_zone_predictions
                ]

            if not candidate_zones:
                skipped_count += 1
                continue

            observed_at = self._historical_label_observed_at(record)
            base_score = self._score_ungrd_summary(
                record.summary, summary_score_overrides
            )
            imported_record_ids.append(record.id)

            for index, zone in enumerate(candidate_zones, start=1):
                try:
                    feature_run = self._resolve_feature_run(
                        zone_id=zone.id,
                        observed_at=observed_at,
                        feature_run_id=None,
                        allow_latest_fallback=True,
                    )
                except ApiError:
                    skipped_count += 1
                    continue

                feature_run_completed_at = self._as_utc(feature_run.completed_at)
                used_fallback_run = (
                    feature_run_completed_at > observed_at
                    if feature_run_completed_at is not None
                    else False
                )
                adjusted_score = base_score
                if selected_zone is None and len(candidate_zones) > 1:
                    adjusted_score = round(
                        max(0.0, min(1.0, base_score - ((index - 1) * 0.06))), 3
                    )

                source_id = f"ungrd_record:{record.id}"
                notes = (
                    f"Imported from UNGRD record {record.id} for municipality {record.municipality.name}. "
                    f"Summary: {record.summary}"
                )
                prepared_labels.append(
                    OutcomeLabelWrite(
                        zoneId=zone.id,
                        observedAt=observed_at,
                        targetScore=adjusted_score,
                        source=source_id,
                        featureRunId=feature_run.id,
                        status=status,
                        notes=notes,
                        evidence={
                            "import_mode": "ungrd_record",
                            "dedup_strategy": "zone_source",
                            "record_id": record.id,
                            "record_summary": record.summary,
                            "municipality": record.municipality.name,
                            "zone_id": zone.id,
                            "zone_rank_within_record": index,
                            "max_zones_per_record": max_zones_per_record,
                            "feature_run_resolution": (
                                "latest_available_backfill"
                                if used_fallback_run
                                else "historical_match"
                            ),
                        },
                    )
                )

        if not prepared_labels:
            return ImportUngrdLabelsResponse(
                createdCount=0,
                updatedCount=0,
                skippedCount=skipped_count,
                importedRecordIds=[],
                labels=[],
            )

        upsert_response = self.upsert_labels(prepared_labels)
        return ImportUngrdLabelsResponse(
            createdCount=upsert_response.created_count,
            updatedCount=upsert_response.updated_count,
            skippedCount=skipped_count,
            importedRecordIds=imported_record_ids,
            labels=upsert_response.labels,
        )

    def import_field_validation_labels(
        self,
        *,
        observations: list[FieldValidationObservationWrite],
        severity_score_overrides: dict[str, float] | None = None,
    ) -> ImportFieldValidationLabelsResponse:
        severity_scores = self._normalize_field_validation_scores(
            severity_score_overrides
        )
        prepared_labels: list[OutcomeLabelWrite] = []
        imported_observation_ids: list[str] = []
        skipped_count = 0

        for observation in observations:
            zone = self._get_zone(observation.zone_id)
            observed_at = self._normalize_observed_at(observation.observed_at)
            try:
                feature_run = self._resolve_feature_run(
                    zone_id=zone.id,
                    observed_at=observed_at,
                    feature_run_id=observation.feature_run_id,
                    allow_latest_fallback=observation.feature_run_id is None,
                )
            except ApiError:
                skipped_count += 1
                continue

            target_score = observation.target_score
            severity_key = (
                observation.severity.strip().lower() if observation.severity else None
            )
            if target_score is None:
                if severity_key is None:
                    skipped_count += 1
                    continue
                target_score = severity_scores.get(severity_key)
                if target_score is None:
                    skipped_count += 1
                    continue

            feature_run_completed_at = self._as_utc(feature_run.completed_at)
            used_fallback_run = (
                feature_run_completed_at > observed_at
                if feature_run_completed_at is not None
                else False
            )
            source_id = f"field_validation:{observation.observation_id}"
            prepared_labels.append(
                OutcomeLabelWrite(
                    zoneId=zone.id,
                    observedAt=observed_at,
                    targetScore=round(float(target_score), 3),
                    source=source_id,
                    featureRunId=feature_run.id,
                    status=observation.status,
                    notes=observation.notes
                    or f"Imported from field validation observation {observation.observation_id}.",
                    evidence={
                        "import_mode": "field_validation",
                        "dedup_strategy": "zone_source",
                        "observation_id": observation.observation_id,
                        "observer": observation.observer,
                        "site_visit_id": observation.site_visit_id,
                        "team_id": observation.team_id,
                        "media_refs": observation.media_refs or [],
                        "attachment_refs": observation.attachment_refs or [],
                        "gps_accuracy_meters": observation.gps_accuracy_meters,
                        "location_notes": observation.location_notes,
                        "zone_id": zone.id,
                        "municipality": zone.municipality.name,
                        "severity": observation.severity,
                        "feature_run_resolution": (
                            "latest_available_backfill"
                            if used_fallback_run
                            else "direct_or_historical_match"
                        ),
                        **observation.evidence,
                    },
                )
            )
            imported_observation_ids.append(observation.observation_id)

        if not prepared_labels:
            return ImportFieldValidationLabelsResponse(
                createdCount=0,
                updatedCount=0,
                skippedCount=skipped_count,
                importedObservationIds=[],
                labels=[],
            )

        upsert_response = self.upsert_labels(prepared_labels)
        return ImportFieldValidationLabelsResponse(
            createdCount=upsert_response.created_count,
            updatedCount=upsert_response.updated_count,
            skippedCount=skipped_count,
            importedObservationIds=imported_observation_ids,
            labels=upsert_response.labels,
        )

    def review_labels(
        self,
        *,
        label_ids: list[int],
        decision: str,
        reviewer_username: str,
        review_notes: str | None = None,
    ) -> ReviewOutcomeLabelsResponse:
        statement = (
            select(ZoneOutcomeLabel)
            .where(ZoneOutcomeLabel.id.in_(label_ids))
            .options(
                joinedload(ZoneOutcomeLabel.zone).joinedload(Zone.municipality),
                joinedload(ZoneOutcomeLabel.feature_run),
            )
        )
        labels = list(self.session.scalars(statement).unique().all())
        labels_by_id = {label.id: label for label in labels}
        missing_label_ids = [
            label_id for label_id in label_ids if label_id not in labels_by_id
        ]
        if missing_label_ids:
            missing_display = ", ".join(str(label_id) for label_id in missing_label_ids)
            raise ApiError(
                404, "label_not_found", f"Governed labels not found: {missing_display}."
            )

        reviewed_at = datetime.now(timezone.utc).replace(microsecond=0)
        ordered_labels = [labels_by_id[label_id] for label_id in label_ids]

        if decision == "confirmed":
            incomplete_labels: list[str] = []
            for label in ordered_labels:
                readiness = self._review_readiness(label)
                if readiness["missing_fields"]:
                    incomplete_labels.append(
                        f"{label.id} ({', '.join(readiness['missing_fields'])})"
                    )
            if incomplete_labels:
                raise ApiError(
                    400,
                    "label_evidence_incomplete",
                    "Cannot confirm labels with incomplete review evidence: "
                    + "; ".join(incomplete_labels)
                    + ".",
                )

        for label in ordered_labels:
            label.status = decision
            label.reviewed_at = reviewed_at
            label.reviewed_by = reviewer_username
            label.review_notes = review_notes
            label.notes = self._merge_notes(label.notes, review_notes)
            label.evidence = self._append_review_history(
                label.evidence or {},
                reviewer_username=reviewer_username,
                decision=decision,
                review_notes=review_notes,
                reviewed_at=reviewed_at,
            )
            label.training_eligibility_status = (
                self._default_training_eligibility_for_status(decision)
            )
            label.training_eligibility_updated_at = reviewed_at
            label.training_eligibility_updated_by = reviewer_username
            label.training_eligibility_notes = (
                f"Auto-updated from review decision: {decision}."
            )
            label.evidence = self._append_training_eligibility_history(
                label.evidence,
                updated_by=reviewer_username,
                training_eligibility_status=label.training_eligibility_status,
                notes=label.training_eligibility_notes,
                updated_at=reviewed_at,
            )
            self._clear_training_release_fields(label)
            if decision in {"confirmed", "rejected"}:
                label.assigned_reviewer = None
                label.assigned_at = None
                label.review_due_at = None
            label.updated_at = reviewed_at

        self.session.commit()
        return ReviewOutcomeLabelsResponse(
            reviewedCount=len(ordered_labels),
            labels=[self._label_read(label) for label in ordered_labels],
        )

    def assign_labels(
        self,
        *,
        label_ids: list[int],
        reviewer_username: str,
        assigned_by: str,
        review_due_at: datetime | None = None,
        assignment_notes: str | None = None,
    ) -> AssignOutcomeLabelsResponse:
        self._get_reviewer_user(reviewer_username)
        normalized_due_at = self._normalize_datetime(review_due_at)

        statement = (
            select(ZoneOutcomeLabel)
            .where(ZoneOutcomeLabel.id.in_(label_ids))
            .options(
                joinedload(ZoneOutcomeLabel.zone).joinedload(Zone.municipality),
                joinedload(ZoneOutcomeLabel.feature_run),
            )
        )
        labels = list(self.session.scalars(statement).unique().all())
        labels_by_id = {label.id: label for label in labels}
        missing_label_ids = [
            label_id for label_id in label_ids if label_id not in labels_by_id
        ]
        if missing_label_ids:
            missing_display = ", ".join(str(label_id) for label_id in missing_label_ids)
            raise ApiError(
                404, "label_not_found", f"Governed labels not found: {missing_display}."
            )

        ordered_labels = [labels_by_id[label_id] for label_id in label_ids]
        invalid_labels = [
            str(label.id)
            for label in ordered_labels
            if label.status not in REVIEW_QUEUE_STATUSES
        ]
        if invalid_labels:
            raise ApiError(
                400,
                "label_assignment_not_allowed",
                "Only draft or needs_revision labels can be assigned for review: "
                + ", ".join(invalid_labels)
                + ".",
            )

        assigned_at = datetime.now(timezone.utc).replace(microsecond=0)
        for label in ordered_labels:
            label.assigned_reviewer = reviewer_username
            label.assigned_at = assigned_at
            label.review_due_at = normalized_due_at
            label.evidence = self._append_assignment_history(
                label.evidence or {},
                assigned_by=assigned_by,
                reviewer_username=reviewer_username,
                assignment_notes=assignment_notes,
                assigned_at=assigned_at,
                review_due_at=normalized_due_at,
            )
            label.updated_at = assigned_at

        self.session.commit()
        return AssignOutcomeLabelsResponse(
            assignedCount=len(ordered_labels),
            labels=[self._label_read(label) for label in ordered_labels],
        )

    def update_training_eligibility(
        self,
        *,
        label_ids: list[int],
        training_eligibility_status: str,
        updated_by: str,
        notes: str | None = None,
    ) -> UpdateTrainingEligibilityResponse:
        statement = (
            select(ZoneOutcomeLabel)
            .where(ZoneOutcomeLabel.id.in_(label_ids))
            .options(
                joinedload(ZoneOutcomeLabel.zone).joinedload(Zone.municipality),
                joinedload(ZoneOutcomeLabel.feature_run),
            )
        )
        labels = list(self.session.scalars(statement).unique().all())
        labels_by_id = {label.id: label for label in labels}
        missing_label_ids = [
            label_id for label_id in label_ids if label_id not in labels_by_id
        ]
        if missing_label_ids:
            missing_display = ", ".join(str(label_id) for label_id in missing_label_ids)
            raise ApiError(
                404, "label_not_found", f"Governed labels not found: {missing_display}."
            )

        ordered_labels = [labels_by_id[label_id] for label_id in label_ids]
        invalid_labels = [
            str(label.id) for label in ordered_labels if label.status != "confirmed"
        ]
        if invalid_labels:
            raise ApiError(
                400,
                "training_eligibility_update_not_allowed",
                "Only confirmed labels can have training eligibility updated: "
                + ", ".join(invalid_labels)
                + ".",
            )

        updated_at = datetime.now(timezone.utc).replace(microsecond=0)
        for label in ordered_labels:
            current_status = self.effective_training_eligibility_status(label)
            if current_status == "hold" and training_eligibility_status == "eligible":
                raise ApiError(
                    400,
                    "training_release_request_required",
                    "Held labels must go through the release-request workflow before they can return to eligible.",
                )
            label.training_eligibility_status = training_eligibility_status
            label.training_eligibility_updated_at = updated_at
            label.training_eligibility_updated_by = updated_by
            label.training_eligibility_notes = notes
            label.evidence = self._append_training_eligibility_history(
                label.evidence or {},
                updated_by=updated_by,
                training_eligibility_status=training_eligibility_status,
                notes=notes,
                updated_at=updated_at,
            )
            self._clear_training_release_fields(label)
            label.updated_at = updated_at

        self.session.commit()
        return UpdateTrainingEligibilityResponse(
            updatedCount=len(ordered_labels),
            labels=[self._label_read(label) for label in ordered_labels],
        )

    def request_training_release(
        self,
        *,
        label_ids: list[int],
        release_criteria: list[str],
        requested_by: str,
        notes: str | None = None,
    ) -> RequestTrainingReleaseResponse:
        normalized_criteria = self._normalized_release_criteria(release_criteria)
        if not normalized_criteria:
            raise ApiError(
                400,
                "training_release_criteria_required",
                "At least one release criterion is required.",
            )

        statement = (
            select(ZoneOutcomeLabel)
            .where(ZoneOutcomeLabel.id.in_(label_ids))
            .options(
                joinedload(ZoneOutcomeLabel.zone).joinedload(Zone.municipality),
                joinedload(ZoneOutcomeLabel.feature_run),
            )
        )
        labels = list(self.session.scalars(statement).unique().all())
        labels_by_id = {label.id: label for label in labels}
        missing_label_ids = [
            label_id for label_id in label_ids if label_id not in labels_by_id
        ]
        if missing_label_ids:
            missing_display = ", ".join(str(label_id) for label_id in missing_label_ids)
            raise ApiError(
                404, "label_not_found", f"Governed labels not found: {missing_display}."
            )

        ordered_labels = [labels_by_id[label_id] for label_id in label_ids]
        invalid_labels = [
            str(label.id)
            for label in ordered_labels
            if label.status != "confirmed"
            or self.effective_training_eligibility_status(label) != "hold"
        ]
        if invalid_labels:
            raise ApiError(
                400,
                "training_release_request_not_allowed",
                "Only confirmed labels currently on hold can request release: "
                + ", ".join(invalid_labels)
                + ".",
            )

        pending_labels = [
            str(label.id)
            for label in ordered_labels
            if label.training_release_status == TRAINING_RELEASE_PENDING_STATUS
        ]
        if pending_labels:
            raise ApiError(
                400,
                "training_release_request_already_pending",
                "Release review is already pending for labels: "
                + ", ".join(pending_labels)
                + ".",
            )

        requested_at = datetime.now(timezone.utc).replace(microsecond=0)
        for label in ordered_labels:
            label.training_release_status = TRAINING_RELEASE_PENDING_STATUS
            label.training_release_criteria = normalized_criteria
            label.training_release_requested_at = requested_at
            label.training_release_requested_by = requested_by
            label.training_release_requested_notes = notes
            label.training_release_reviewed_at = None
            label.training_release_reviewed_by = None
            label.training_release_review_notes = None
            label.training_release_assigned_reviewer = None
            label.training_release_assigned_at = None
            label.training_release_due_at = None
            label.training_release_escalation_status = None
            label.training_release_escalation_level = None
            label.training_release_escalated_at = None
            label.training_release_escalated_by = None
            label.training_release_escalation_reason = None
            label.evidence = self._append_training_release_history(
                label.evidence or {},
                action="request",
                acted_by=requested_by,
                criteria=normalized_criteria,
                notes=notes,
                acted_at=requested_at,
            )
            label.updated_at = requested_at

        self.session.commit()
        return RequestTrainingReleaseResponse(
            requestedCount=len(ordered_labels),
            labels=[self._label_read(label) for label in ordered_labels],
        )

    def assign_training_release(
        self,
        *,
        label_ids: list[int],
        reviewer_username: str,
        assigned_by: str,
        review_due_at: datetime | None = None,
        assignment_notes: str | None = None,
    ) -> AssignTrainingReleaseResponse:
        self._get_reviewer_user(reviewer_username)
        normalized_due_at = self._normalize_datetime(review_due_at)

        statement = (
            select(ZoneOutcomeLabel)
            .where(ZoneOutcomeLabel.id.in_(label_ids))
            .options(
                joinedload(ZoneOutcomeLabel.zone).joinedload(Zone.municipality),
                joinedload(ZoneOutcomeLabel.feature_run),
            )
        )
        labels = list(self.session.scalars(statement).unique().all())
        labels_by_id = {label.id: label for label in labels}
        missing_label_ids = [
            label_id for label_id in label_ids if label_id not in labels_by_id
        ]
        if missing_label_ids:
            missing_display = ", ".join(str(label_id) for label_id in missing_label_ids)
            raise ApiError(
                404, "label_not_found", f"Governed labels not found: {missing_display}."
            )

        ordered_labels = [labels_by_id[label_id] for label_id in label_ids]
        invalid_labels = [
            str(label.id)
            for label in ordered_labels
            if label.training_release_status != TRAINING_RELEASE_PENDING_STATUS
        ]
        if invalid_labels:
            raise ApiError(
                400,
                "training_release_assignment_not_allowed",
                "Only labels with a pending release request can be assigned for release review: "
                + ", ".join(invalid_labels)
                + ".",
            )

        assigned_at = datetime.now(timezone.utc).replace(microsecond=0)
        for label in ordered_labels:
            label.training_release_assigned_reviewer = reviewer_username
            label.training_release_assigned_at = assigned_at
            label.training_release_due_at = normalized_due_at
            label.evidence = self._append_training_release_assignment_history(
                label.evidence or {},
                assigned_by=assigned_by,
                reviewer_username=reviewer_username,
                assignment_notes=assignment_notes,
                assigned_at=assigned_at,
                review_due_at=normalized_due_at,
            )
            self._create_release_assignment_notification(label, reviewer_username)
            label.updated_at = assigned_at

        self.session.commit()
        return AssignTrainingReleaseResponse(
            assignedCount=len(ordered_labels),
            labels=[self._label_read(label) for label in ordered_labels],
        )

    def reassign_training_release(
        self,
        *,
        label_ids: list[int],
        reviewer_username: str,
        reassigned_by: str,
        reassignment_reason: str,
        review_due_at: datetime | None = None,
    ) -> ReassignTrainingReleaseResponse:
        self._get_reviewer_user(reviewer_username)
        normalized_due_at = self._normalize_datetime(review_due_at)
        normalized_reason = reassignment_reason.strip()
        if not normalized_reason:
            raise ApiError(
                400,
                "training_release_reassignment_reason_required",
                "A non-empty reassignment reason is required.",
            )

        statement = (
            select(ZoneOutcomeLabel)
            .where(ZoneOutcomeLabel.id.in_(label_ids))
            .options(
                joinedload(ZoneOutcomeLabel.zone).joinedload(Zone.municipality),
                joinedload(ZoneOutcomeLabel.feature_run),
            )
        )
        labels = list(self.session.scalars(statement).unique().all())
        labels_by_id = {label.id: label for label in labels}
        missing_label_ids = [
            label_id for label_id in label_ids if label_id not in labels_by_id
        ]
        if missing_label_ids:
            missing_display = ", ".join(str(label_id) for label_id in missing_label_ids)
            raise ApiError(
                404, "label_not_found", f"Governed labels not found: {missing_display}."
            )

        ordered_labels = [labels_by_id[label_id] for label_id in label_ids]
        invalid_labels = [
            str(label.id)
            for label in ordered_labels
            if label.training_release_status != TRAINING_RELEASE_PENDING_STATUS
        ]
        if invalid_labels:
            raise ApiError(
                400,
                "training_release_reassignment_not_allowed",
                "Only labels with a pending release request can be reassigned: "
                + ", ".join(invalid_labels)
                + ".",
            )

        reassigned_at = datetime.now(timezone.utc).replace(microsecond=0)
        for label in ordered_labels:
            previous_reviewer = label.training_release_assigned_reviewer
            label.training_release_assigned_reviewer = reviewer_username
            label.training_release_assigned_at = reassigned_at
            if normalized_due_at is not None:
                label.training_release_due_at = normalized_due_at
            label.evidence = self._append_training_release_reassignment_history(
                label.evidence or {},
                previous_reviewer_username=previous_reviewer,
                reviewer_username=reviewer_username,
                reassigned_by=reassigned_by,
                reassignment_reason=normalized_reason,
                reassigned_at=reassigned_at,
                review_due_at=label.training_release_due_at,
            )
            self._create_release_reassignment_notification(
                label,
                reviewer_username=reviewer_username,
                previous_reviewer_username=previous_reviewer,
                reassignment_reason=normalized_reason,
            )
            label.updated_at = reassigned_at

        self.session.commit()
        return ReassignTrainingReleaseResponse(
            reassignedCount=len(ordered_labels),
            labels=[self._label_read(label) for label in ordered_labels],
        )

    def escalate_training_release(
        self,
        *,
        label_ids: list[int],
        escalation_reason: str,
        escalated_by: str,
        escalation_level: int | None = None,
        notification_event_type: str = "training_release_escalation",
        notification_severity: str = "warning",
    ) -> EscalateTrainingReleaseResponse:
        statement = (
            select(ZoneOutcomeLabel)
            .where(ZoneOutcomeLabel.id.in_(label_ids))
            .options(
                joinedload(ZoneOutcomeLabel.zone).joinedload(Zone.municipality),
                joinedload(ZoneOutcomeLabel.feature_run),
            )
        )
        labels = list(self.session.scalars(statement).unique().all())
        labels_by_id = {label.id: label for label in labels}
        missing_label_ids = [
            label_id for label_id in label_ids if label_id not in labels_by_id
        ]
        if missing_label_ids:
            missing_display = ", ".join(str(label_id) for label_id in missing_label_ids)
            raise ApiError(
                404, "label_not_found", f"Governed labels not found: {missing_display}."
            )

        ordered_labels = [labels_by_id[label_id] for label_id in label_ids]
        invalid_labels = [
            str(label.id)
            for label in ordered_labels
            if label.training_release_status != TRAINING_RELEASE_PENDING_STATUS
        ]
        if invalid_labels:
            raise ApiError(
                400,
                "training_release_escalation_not_allowed",
                "Only labels with a pending release request can be escalated: "
                + ", ".join(invalid_labels)
                + ".",
            )

        normalized_reason = escalation_reason.strip()
        if not normalized_reason:
            raise ApiError(
                400,
                "training_release_escalation_reason_required",
                "A non-empty escalation reason is required.",
            )

        escalated_at = datetime.now(timezone.utc).replace(microsecond=0)
        for label in ordered_labels:
            current_level = label.training_release_escalation_level or 0
            resolved_level = (
                escalation_level if escalation_level is not None else current_level + 1
            )
            label.training_release_escalation_status = TRAINING_RELEASE_ESCALATED_STATUS
            label.training_release_escalation_level = resolved_level
            label.training_release_escalated_at = escalated_at
            label.training_release_escalated_by = escalated_by
            label.training_release_escalation_reason = normalized_reason
            label.evidence = self._append_training_release_escalation_history(
                label.evidence or {},
                escalated_by=escalated_by,
                escalation_reason=normalized_reason,
                escalation_level=resolved_level,
                escalated_at=escalated_at,
            )
            self._create_release_escalation_notification(
                label,
                event_type=notification_event_type,
                severity=notification_severity,
                escalation_reason=normalized_reason,
                escalation_level=resolved_level,
                escalated_by=escalated_by,
            )
            label.updated_at = escalated_at

        self.session.commit()
        return EscalateTrainingReleaseResponse(
            escalatedCount=len(ordered_labels),
            labels=[self._label_read(label) for label in ordered_labels],
        )

    def run_training_release_reassignment_scan(
        self,
        *,
        max_labels: int = 100,
        note: str | None = None,
        origin: str = "manual",
    ) -> TriggerTrainingReleaseReassignmentScanResponse:
        started_at = datetime.now(timezone.utc).replace(microsecond=0)
        job = JobExecution(
            job_type="training_release_reassignment_scan",
            status="running",
            started_at=started_at,
            details={"origin": origin, "max_labels": max_labels, "note": note},
        )
        self.session.add(job)
        self.session.flush()

        target_reviewer = self.settings.training_release_auto_reassign_reviewer
        self._get_reviewer_user(target_reviewer)
        next_due_at = started_at + timedelta(
            hours=self.settings.training_release_auto_reassign_due_in_hours
        )

        statement = (
            select(ZoneOutcomeLabel)
            .where(
                ZoneOutcomeLabel.training_release_status
                == TRAINING_RELEASE_PENDING_STATUS,
                ZoneOutcomeLabel.training_release_escalation_status
                == TRAINING_RELEASE_ESCALATED_STATUS,
                ZoneOutcomeLabel.training_release_due_at.is_not(None),
                ZoneOutcomeLabel.training_release_due_at < started_at,
            )
            .options(
                joinedload(ZoneOutcomeLabel.zone).joinedload(Zone.municipality),
                joinedload(ZoneOutcomeLabel.feature_run),
            )
            .order_by(
                ZoneOutcomeLabel.training_release_due_at.asc(),
                ZoneOutcomeLabel.id.asc(),
            )
            .limit(max_labels)
        )
        candidates = list(self.session.scalars(statement).unique().all())
        target_labels = [
            label
            for label in candidates
            if label.training_release_assigned_reviewer != target_reviewer
        ]

        reassigned_labels: list[OutcomeLabelRead] = []
        notification_count = 0
        if target_labels:
            reassignment_response = self.reassign_training_release(
                label_ids=[label.id for label in target_labels],
                reviewer_username=target_reviewer,
                reassigned_by=f"system:{origin}",
                reassignment_reason=note
                or "Automatic reassignment after overdue escalated release review.",
                review_due_at=next_due_at,
            )
            reassigned_labels = reassignment_response.labels
            notification_count = reassignment_response.reassigned_count

        completed_at = datetime.now(timezone.utc).replace(microsecond=0)
        job.status = "completed"
        job.completed_at = completed_at
        job.details = {
            "origin": origin,
            "max_labels": max_labels,
            "note": note,
            "candidates": len(candidates),
            "reassigned_count": len(reassigned_labels),
            "notification_count": notification_count,
            "target_reviewer": target_reviewer,
            "due_in_hours": self.settings.training_release_auto_reassign_due_in_hours,
        }
        self.session.commit()
        self.session.refresh(job)

        return TriggerTrainingReleaseReassignmentScanResponse(
            job=self._job_read(job),
            reassignedCount=len(reassigned_labels),
            notificationCount=notification_count,
            labels=reassigned_labels,
        )

    def review_training_release(
        self,
        *,
        label_ids: list[int],
        decision: str,
        reviewed_by: str,
        notes: str | None = None,
    ) -> ReviewTrainingReleaseResponse:
        statement = (
            select(ZoneOutcomeLabel)
            .where(ZoneOutcomeLabel.id.in_(label_ids))
            .options(
                joinedload(ZoneOutcomeLabel.zone).joinedload(Zone.municipality),
                joinedload(ZoneOutcomeLabel.feature_run),
            )
        )
        labels = list(self.session.scalars(statement).unique().all())
        labels_by_id = {label.id: label for label in labels}
        missing_label_ids = [
            label_id for label_id in label_ids if label_id not in labels_by_id
        ]
        if missing_label_ids:
            missing_display = ", ".join(str(label_id) for label_id in missing_label_ids)
            raise ApiError(
                404, "label_not_found", f"Governed labels not found: {missing_display}."
            )

        ordered_labels = [labels_by_id[label_id] for label_id in label_ids]
        invalid_labels = [
            str(label.id)
            for label in ordered_labels
            if label.training_release_status != TRAINING_RELEASE_PENDING_STATUS
        ]
        if invalid_labels:
            raise ApiError(
                400,
                "training_release_review_not_allowed",
                "Only labels with a pending release request can be reviewed: "
                + ", ".join(invalid_labels)
                + ".",
            )

        reviewed_at = datetime.now(timezone.utc).replace(microsecond=0)
        for label in ordered_labels:
            resolved_assigned_reviewer = label.training_release_assigned_reviewer
            label.training_release_status = decision
            label.training_release_reviewed_at = reviewed_at
            label.training_release_reviewed_by = reviewed_by
            label.training_release_review_notes = notes
            label.evidence = self._append_training_release_history(
                label.evidence or {},
                action="review",
                acted_by=reviewed_by,
                criteria=label.training_release_criteria or [],
                notes=notes,
                acted_at=reviewed_at,
                decision=decision,
            )
            if decision == "approved":
                label.training_eligibility_status = "eligible"
                label.training_eligibility_updated_at = reviewed_at
                label.training_eligibility_updated_by = reviewed_by
                label.training_eligibility_notes = (
                    "Released from hold after approved release review."
                )
                label.evidence = self._append_training_eligibility_history(
                    label.evidence,
                    updated_by=reviewed_by,
                    training_eligibility_status="eligible",
                    notes=label.training_eligibility_notes,
                    updated_at=reviewed_at,
                )
            label.training_release_assigned_reviewer = None
            label.training_release_assigned_at = None
            label.training_release_due_at = None
            label.training_release_escalation_status = None
            label.training_release_escalation_level = None
            label.training_release_escalated_at = None
            label.training_release_escalated_by = None
            label.training_release_escalation_reason = None
            self._create_release_resolution_notification(
                label,
                decision=decision,
                reviewed_by=reviewed_by,
                assigned_reviewer_username=resolved_assigned_reviewer,
            )
            label.updated_at = reviewed_at

        self.session.commit()
        return ReviewTrainingReleaseResponse(
            reviewedCount=len(ordered_labels),
            labels=[self._label_read(label) for label in ordered_labels],
        )

    def list_training_release_queue(
        self,
        *,
        assigned_reviewer: str | None = None,
        overdue_only: bool = False,
        escalated_only: bool = False,
        limit: int = 100,
    ) -> OutcomeLabelReleaseQueueRead:
        statement = (
            select(ZoneOutcomeLabel)
            .where(
                ZoneOutcomeLabel.training_release_status
                == TRAINING_RELEASE_PENDING_STATUS
            )
            .options(
                joinedload(ZoneOutcomeLabel.zone).joinedload(Zone.municipality),
                joinedload(ZoneOutcomeLabel.feature_run),
            )
            .order_by(
                ZoneOutcomeLabel.training_release_requested_at.desc(),
                ZoneOutcomeLabel.id.desc(),
            )
        )
        if assigned_reviewer:
            statement = statement.where(
                ZoneOutcomeLabel.training_release_assigned_reviewer == assigned_reviewer
            )

        labels = list(self.session.scalars(statement).unique().all())
        label_reads = [self._label_read(label) for label in labels]
        if overdue_only:
            label_reads = [
                label for label in label_reads if label.training_release_is_overdue
            ]
        if escalated_only:
            label_reads = [
                label for label in label_reads if label.training_release_is_escalated
            ]
        label_reads = label_reads[:limit]

        return OutcomeLabelReleaseQueueRead(
            total=len(label_reads),
            assignedCount=sum(
                1 for label in label_reads if label.training_release_assigned_reviewer
            ),
            unassignedCount=sum(
                1
                for label in label_reads
                if not label.training_release_assigned_reviewer
            ),
            overdueCount=sum(
                1 for label in label_reads if label.training_release_is_overdue
            ),
            escalatedCount=sum(
                1 for label in label_reads if label.training_release_is_escalated
            ),
            labels=label_reads,
        )

    def list_review_queue(
        self,
        *,
        assigned_reviewer: str | None = None,
        ready_for_review: bool | None = None,
        limit: int = 100,
    ) -> OutcomeLabelReviewQueueRead:
        statement = (
            select(ZoneOutcomeLabel)
            .where(ZoneOutcomeLabel.status.in_(REVIEW_QUEUE_STATUSES))
            .options(
                joinedload(ZoneOutcomeLabel.zone).joinedload(Zone.municipality),
                joinedload(ZoneOutcomeLabel.feature_run),
            )
            .order_by(ZoneOutcomeLabel.updated_at.desc(), ZoneOutcomeLabel.id.desc())
        )
        if assigned_reviewer:
            statement = statement.where(
                ZoneOutcomeLabel.assigned_reviewer == assigned_reviewer
            )

        labels = list(self.session.scalars(statement).unique().all())
        label_reads = [self._label_read(label) for label in labels]
        if ready_for_review is not None:
            label_reads = [
                label
                for label in label_reads
                if label.ready_for_review is ready_for_review
            ]
        label_reads = label_reads[:limit]

        return OutcomeLabelReviewQueueRead(
            total=len(label_reads),
            readyCount=sum(1 for label in label_reads if label.ready_for_review),
            assignedCount=sum(1 for label in label_reads if label.assigned_reviewer),
            overdueCount=sum(1 for label in label_reads if label.is_overdue),
            labels=label_reads,
        )

    def upsert_labels(
        self, items: list[OutcomeLabelWrite]
    ) -> UpsertOutcomeLabelsResponse:
        created_count = 0
        updated_count = 0
        label_ids: list[int] = []

        for item in items:
            observed_at = self._normalize_observed_at(item.observed_at)
            self._get_zone(item.zone_id)
            feature_run = self._resolve_feature_run(
                zone_id=item.zone_id,
                observed_at=observed_at,
                feature_run_id=item.feature_run_id,
            )

            existing = self._find_existing_label_for_upsert(
                zone_id=item.zone_id,
                observed_at=observed_at,
                source=item.source,
            )
            now = datetime.now(timezone.utc).replace(microsecond=0)

            if existing is None:
                label = ZoneOutcomeLabel(
                    zone_id=item.zone_id,
                    feature_run_id=feature_run.id,
                    observed_at=observed_at,
                    target_score=item.target_score,
                    source=item.source,
                    status=item.status,
                    notes=item.notes,
                    evidence=item.evidence,
                    assigned_reviewer=None,
                    assigned_at=None,
                    review_due_at=None,
                    training_eligibility_status=self._default_training_eligibility_for_status(
                        item.status
                    ),
                    training_eligibility_updated_at=now,
                    training_eligibility_updated_by=None,
                    training_eligibility_notes=None,
                    training_release_status=None,
                    training_release_criteria=None,
                    training_release_requested_at=None,
                    training_release_requested_by=None,
                    training_release_requested_notes=None,
                    training_release_reviewed_at=None,
                    training_release_reviewed_by=None,
                    training_release_review_notes=None,
                    training_release_assigned_reviewer=None,
                    training_release_assigned_at=None,
                    training_release_due_at=None,
                    training_release_escalation_status=None,
                    training_release_escalation_level=None,
                    training_release_escalated_at=None,
                    training_release_escalated_by=None,
                    training_release_escalation_reason=None,
                    reviewed_at=None,
                    reviewed_by=None,
                    review_notes=None,
                    created_at=now,
                    updated_at=now,
                )
                self.session.add(label)
                self.session.flush()
                created_count += 1
            else:
                preserve_training_eligibility = (
                    existing.status == item.status == "confirmed"
                    and existing.training_eligibility_status is not None
                    and existing.training_eligibility_updated_by is not None
                )
                existing.observed_at = observed_at
                existing.feature_run_id = feature_run.id
                existing.target_score = item.target_score
                existing.status = item.status
                existing.notes = item.notes
                existing.evidence = self._merge_operator_evidence(
                    existing.evidence, item.evidence
                )
                if not preserve_training_eligibility:
                    existing.training_eligibility_status = (
                        self._default_training_eligibility_for_status(item.status)
                    )
                    existing.training_eligibility_updated_at = now
                    existing.training_eligibility_updated_by = None
                    existing.training_eligibility_notes = None
                    self._clear_training_release_fields(existing)
                if item.status == "draft":
                    existing.reviewed_at = None
                    existing.reviewed_by = None
                    existing.review_notes = None
                existing.updated_at = now
                label = existing
                updated_count += 1

            label_ids.append(label.id)

        self.session.commit()

        statement = (
            select(ZoneOutcomeLabel)
            .where(ZoneOutcomeLabel.id.in_(label_ids))
            .options(
                joinedload(ZoneOutcomeLabel.zone).joinedload(Zone.municipality),
                joinedload(ZoneOutcomeLabel.feature_run),
            )
            .order_by(ZoneOutcomeLabel.observed_at.desc(), ZoneOutcomeLabel.id.desc())
        )
        labels = list(self.session.scalars(statement).unique().all())
        return UpsertOutcomeLabelsResponse(
            createdCount=created_count,
            updatedCount=updated_count,
            labels=[self._label_read(label) for label in labels],
        )
