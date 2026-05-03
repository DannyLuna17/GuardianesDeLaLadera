from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.exceptions import ApiError
from app.models import ModelReviewTask, NotificationEvent
from app.schemas.admin import ModelReviewTaskRead, OpenModelReviewTasksResponse

PREDICTIVE_REVIEW_TYPES_BY_ALERT = {
    "model_monitoring_shadow_alert": {"promotion_review"},
    "model_monitoring_drift_alert": {"rollback_review", "retraining_review"},
    "model_modern_labels_benchmark_alert": {"promotion_review"},
}
ALLOWED_DECISIONS_BY_REVIEW_TYPE = {
    "promotion_review": {
        "approve_promotion_review",
        "reject_promotion_review",
    },
    "rollback_review": {
        "approve_rollback_review",
        "reject_rollback_review",
    },
    "retraining_review": {
        "approve_retraining_review",
        "reject_retraining_review",
    },
}
ACTION_GUARDRAILS = {
    "promotion": {
        "review_type": "promotion_review",
        "required_decision": "approve_promotion_review",
        "enforce_dataset_match": False,
    },
    "rollback": {
        "review_type": "rollback_review",
        "required_decision": "approve_rollback_review",
        "enforce_dataset_match": False,
    },
    "retraining": {
        "review_type": "retraining_review",
        "required_decision": "approve_retraining_review",
        "enforce_dataset_match": False,
    },
}
TERMINAL_MODEL_REVIEW_TASK_STATUSES = {"resolved", "cancelled"}
ACTIVE_MODEL_REVIEW_TASK_STATUSES = {"open", "in_progress"}


class ModelReviewTaskService:
    def __init__(self, session: Session) -> None:
        self.session = session

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc).replace(microsecond=0)

    @staticmethod
    def _task_read(task: ModelReviewTask) -> ModelReviewTaskRead:
        return ModelReviewTaskRead(
            id=task.id,
            reviewType=task.review_type,
            status=task.status,
            sourceNotificationId=task.source_notification_id,
            sourceEventType=task.source_event_type,
            sourceAlertSeverity=task.source_alert_severity,
            sourceAlertStatus=task.source_alert_status,
            activeModelVersion=task.active_model_version,
            candidateModelVersion=task.candidate_model_version,
            datasetVersion=task.dataset_version,
            title=task.title,
            summary=task.summary,
            recommendedAction=task.recommended_action,
            assignedReviewer=task.assigned_reviewer,
            dueAt=task.due_at,
            decision=task.decision,
            resolutionNotes=task.resolution_notes,
            details=task.details or {},
            createdAt=task.created_at,
            createdBy=task.created_by,
            updatedAt=task.updated_at,
            updatedBy=task.updated_by,
            resolvedAt=task.resolved_at,
            resolvedBy=task.resolved_by,
        )

    def _get_notifications(self, notification_ids: list[int]) -> list[NotificationEvent]:
        requested_ids = list(dict.fromkeys(notification_ids))
        notifications = self.session.scalars(
            select(NotificationEvent)
            .options(selectinload(NotificationEvent.delivery_attempts))
            .where(NotificationEvent.id.in_(requested_ids))
        ).all()
        found_by_id = {notification.id: notification for notification in notifications}
        missing_ids = [notification_id for notification_id in requested_ids if notification_id not in found_by_id]
        if missing_ids:
            raise ApiError(
                404,
                "model_review_source_notification_not_found",
                f"Predictive review alert notifications not found: {missing_ids}.",
            )
        return [found_by_id[notification_id] for notification_id in requested_ids]

    def _ensure_review_type_allowed(
        self, *, notification: NotificationEvent, review_type: str
    ) -> None:
        allowed_types = PREDICTIVE_REVIEW_TYPES_BY_ALERT.get(notification.event_type)
        if allowed_types is None:
            raise ApiError(
                400,
                "model_review_source_not_supported",
                "Only predictive monitoring drift and shadow alerts can open model review tasks.",
            )
        if review_type not in allowed_types:
            allowed_display = ", ".join(sorted(allowed_types))
            raise ApiError(
                400,
                "model_review_type_not_allowed_for_alert",
                f"Review type '{review_type}' is not allowed for alert '{notification.event_type}'. Allowed types: {allowed_display}.",
            )
        if notification.status == "open":
            raise ApiError(
                400,
                "model_review_alert_not_acknowledged",
                "Open predictive alerts must be acknowledged before they can open a model review task.",
            )

    def _active_task_for_notification(
        self, *, notification_id: int, review_type: str
    ) -> ModelReviewTask | None:
        statement = (
            select(ModelReviewTask)
            .where(
                ModelReviewTask.source_notification_id == notification_id,
                ModelReviewTask.review_type == review_type,
                ModelReviewTask.status.in_(ACTIVE_MODEL_REVIEW_TASK_STATUSES),
            )
            .order_by(ModelReviewTask.created_at.desc(), ModelReviewTask.id.desc())
        )
        return self.session.scalars(statement).first()

    @staticmethod
    def _title_for_task(
        *, review_type: str, active_model_version: str, candidate_model_version: str | None
    ) -> str:
        if review_type == "promotion_review":
            subject = candidate_model_version or active_model_version
            return f"Promotion review for {subject}"
        if review_type == "rollback_review":
            return f"Rollback review for {active_model_version}"
        return f"Retraining review for {active_model_version}"

    @staticmethod
    def _summary_for_task(
        *,
        review_type: str,
        notification: NotificationEvent,
        active_model_version: str,
        candidate_model_version: str | None,
        dataset_version: str | None,
    ) -> str:
        if review_type == "promotion_review":
            return (
                f"Review whether challenger {candidate_model_version or 'unknown'} should replace active model "
                f"{active_model_version} on labeled dataset {dataset_version or 'unknown'}."
            )
        if review_type == "rollback_review":
            return (
                f"Review whether active model {active_model_version} should be rolled back after predictive drift findings "
                f"on labeled dataset {dataset_version or 'unknown'}."
            )
        return (
            f"Review whether active model {active_model_version} should be retrained after predictive monitoring findings "
            f"on labeled dataset {dataset_version or 'unknown'}."
        )

    @staticmethod
    def _history_entry(
        *, action: str, actor: str, at: datetime, details: dict | None = None
    ) -> dict:
        payload = {"action": action, "actor": actor, "at": at.isoformat()}
        if details:
            payload["details"] = details
        return payload

    def _append_history(
        self, *, task: ModelReviewTask, action: str, actor: str, at: datetime, details: dict | None = None
    ) -> None:
        payload = dict(task.details or {})
        history = list(payload.get("history") or [])
        history.append(self._history_entry(action=action, actor=actor, at=at, details=details))
        payload["history"] = history
        task.details = payload

    @staticmethod
    def _allowed_decisions(review_type: str) -> set[str]:
        return ALLOWED_DECISIONS_BY_REVIEW_TYPE.get(review_type, set())

    def _task_snapshot_from_notification(
        self,
        *,
        notification: NotificationEvent,
        review_type: str,
        notes: str | None,
    ) -> dict:
        notification_details = dict(notification.details or {})
        return {
            "source_alert": {
                "notification_id": notification.id,
                "event_type": notification.event_type,
                "status": notification.status,
                "severity": notification.severity,
                "title": notification.title,
                "message": notification.message,
                "created_at": notification.created_at.isoformat(),
                "acknowledged_at": notification.acknowledged_at.isoformat()
                if notification.acknowledged_at
                else None,
                "acknowledged_by": notification.acknowledged_by,
                "details": notification_details,
            },
            "operator_note": notes,
            "review_type": review_type,
        }

    def open_review_tasks_from_alerts(
        self,
        *,
        notification_ids: list[int],
        review_type: str,
        opened_by: str,
        assigned_reviewer: str | None = None,
        due_at: datetime | None = None,
        notes: str | None = None,
    ) -> OpenModelReviewTasksResponse:
        notifications = self._get_notifications(notification_ids)
        ordered_tasks: list[ModelReviewTask] = []
        created_count = 0
        skipped_count = 0
        now = self._now()

        for notification in notifications:
            self._ensure_review_type_allowed(
                notification=notification, review_type=review_type
            )
            existing_task = self._active_task_for_notification(
                notification_id=notification.id, review_type=review_type
            )
            if existing_task is not None:
                ordered_tasks.append(existing_task)
                skipped_count += 1
                continue

            alert_details = dict(notification.details or {})
            active_model_version = str(alert_details.get("model_version") or "unknown")
            candidate_model_version = alert_details.get("best_model_version")
            dataset_version = alert_details.get("dataset_version")
            recommended_action = alert_details.get("recommended_action")
            task = ModelReviewTask(
                review_type=review_type,
                status="open",
                source_notification_id=notification.id,
                source_event_type=notification.event_type,
                source_alert_severity=notification.severity,
                source_alert_status=notification.status,
                active_model_version=active_model_version,
                candidate_model_version=(
                    str(candidate_model_version)
                    if candidate_model_version is not None
                    else None
                ),
                dataset_version=(
                    str(dataset_version) if dataset_version is not None else None
                ),
                title=self._title_for_task(
                    review_type=review_type,
                    active_model_version=active_model_version,
                    candidate_model_version=(
                        str(candidate_model_version)
                        if candidate_model_version is not None
                        else None
                    ),
                ),
                summary=self._summary_for_task(
                    review_type=review_type,
                    notification=notification,
                    active_model_version=active_model_version,
                    candidate_model_version=(
                        str(candidate_model_version)
                        if candidate_model_version is not None
                        else None
                    ),
                    dataset_version=(
                        str(dataset_version) if dataset_version is not None else None
                    ),
                ),
                recommended_action=(
                    str(recommended_action) if recommended_action is not None else None
                ),
                assigned_reviewer=assigned_reviewer,
                due_at=due_at,
                details=self._task_snapshot_from_notification(
                    notification=notification,
                    review_type=review_type,
                    notes=notes,
                ),
                created_at=now,
                created_by=opened_by,
                updated_at=now,
                updated_by=opened_by,
            )
            self.session.add(task)
            self.session.flush()
            self._append_history(
                task=task,
                action="created",
                actor=opened_by,
                at=now,
                details={
                    "source_notification_id": notification.id,
                    "assigned_reviewer": assigned_reviewer,
                    "due_at": due_at.isoformat() if due_at else None,
                },
            )

            notification_payload = dict(notification.details or {})
            linked_task_ids = [
                int(value)
                for value in list(notification_payload.get("model_review_task_ids") or [])
                if isinstance(value, int)
            ]
            if task.id not in linked_task_ids:
                linked_task_ids.append(task.id)
            notification_payload["model_review_task_ids"] = linked_task_ids
            notification_payload["last_model_review_task_type"] = review_type
            notification_payload["last_model_review_task_opened_at"] = now.isoformat()
            notification_payload["last_model_review_task_opened_by"] = opened_by
            notification.details = notification_payload

            ordered_tasks.append(task)
            created_count += 1

        self.session.commit()
        return OpenModelReviewTasksResponse(
            createdCount=created_count,
            skippedCount=skipped_count,
            tasks=[self._task_read(task) for task in ordered_tasks],
        )

    def _get_task(self, task_id: int) -> ModelReviewTask:
        statement = (
            select(ModelReviewTask)
            .options(selectinload(ModelReviewTask.source_notification))
            .where(ModelReviewTask.id == task_id)
        )
        task = self.session.scalars(statement).first()
        if task is None:
            raise ApiError(
                404,
                "model_review_task_not_found",
                f"Model review task {task_id} was not found.",
            )
        return task

    def list_review_tasks(
        self,
        *,
        review_type: str | None = None,
        status: str | None = None,
        assigned_reviewer: str | None = None,
        source_notification_id: int | None = None,
        active_model_version: str | None = None,
        candidate_model_version: str | None = None,
        limit: int = 100,
    ) -> list[ModelReviewTaskRead]:
        statement = (
            select(ModelReviewTask)
            .options(selectinload(ModelReviewTask.source_notification))
            .order_by(ModelReviewTask.created_at.desc(), ModelReviewTask.id.desc())
            .limit(limit)
        )
        if review_type:
            statement = statement.where(ModelReviewTask.review_type == review_type)
        if status:
            statement = statement.where(ModelReviewTask.status == status)
        if assigned_reviewer:
            statement = statement.where(
                ModelReviewTask.assigned_reviewer == assigned_reviewer
            )
        if source_notification_id is not None:
            statement = statement.where(
                ModelReviewTask.source_notification_id == source_notification_id
            )
        if active_model_version:
            statement = statement.where(
                ModelReviewTask.active_model_version == active_model_version
            )
        if candidate_model_version:
            statement = statement.where(
                ModelReviewTask.candidate_model_version == candidate_model_version
            )
        tasks = self.session.scalars(statement).all()
        return [self._task_read(task) for task in tasks]

    def get_review_task(self, task_id: int) -> ModelReviewTaskRead:
        return self._task_read(self._get_task(task_id))

    def update_review_task(
        self,
        *,
        task_id: int,
        updated_by: str,
        status: str,
        assigned_reviewer: str | None,
        due_at: datetime | None,
        decision: str | None,
        notes: str | None,
        provided_fields: set[str],
    ) -> ModelReviewTaskRead:
        task = self._get_task(task_id)
        if task.status in TERMINAL_MODEL_REVIEW_TASK_STATUSES and status != task.status:
            raise ApiError(
                400,
                "model_review_task_terminal",
                "Resolved or cancelled model review tasks cannot be reopened.",
            )
        if status == "resolved" and not decision:
            raise ApiError(
                400,
                "model_review_task_decision_required",
                "Resolved model review tasks require a decision.",
            )
        if status in TERMINAL_MODEL_REVIEW_TASK_STATUSES and not notes:
            raise ApiError(
                400,
                "model_review_task_notes_required",
                "Resolved or cancelled model review tasks require notes.",
            )
        if status != "resolved" and "decision" in provided_fields and decision is not None:
            raise ApiError(
                400,
                "model_review_task_decision_requires_resolution",
                "A model review task decision can only be set when resolving the task.",
            )
        if status == "resolved":
            allowed_decisions = self._allowed_decisions(task.review_type)
            if allowed_decisions and decision not in allowed_decisions:
                allowed_display = ", ".join(sorted(allowed_decisions))
                raise ApiError(
                    400,
                    "model_review_task_invalid_decision",
                    f"Decision '{decision}' is not allowed for review type '{task.review_type}'. Allowed decisions: {allowed_display}.",
                )

        now = self._now()
        change_details: dict[str, object] = {"previous_status": task.status, "next_status": status}

        if task.status != status:
            task.status = status
        if "assigned_reviewer" in provided_fields:
            task.assigned_reviewer = assigned_reviewer
            change_details["assigned_reviewer"] = assigned_reviewer
        if "due_at" in provided_fields:
            task.due_at = due_at
            change_details["due_at"] = due_at.isoformat() if due_at else None

        if status in TERMINAL_MODEL_REVIEW_TASK_STATUSES:
            task.resolved_at = now
            task.resolved_by = updated_by
            task.resolution_notes = notes
            change_details["resolution_notes"] = notes
            if status == "resolved":
                task.decision = decision
                change_details["decision"] = decision
        else:
            if "notes" in provided_fields:
                payload = dict(task.details or {})
                payload["latest_note"] = notes
                task.details = payload
                change_details["note"] = notes

        task.updated_at = now
        task.updated_by = updated_by
        self._append_history(
            task=task,
            action="status_update",
            actor=updated_by,
            at=now,
            details=change_details,
        )
        self.session.commit()
        return self._task_read(task)

    def validate_action_guardrail(
        self,
        *,
        task_id: int,
        action_type: str,
        candidate_model_version: str | None = None,
        active_model_version: str | None = None,
        dataset_version: str | None = None,
    ) -> ModelReviewTask:
        task = self._get_task(task_id)
        guardrail = ACTION_GUARDRAILS.get(action_type)
        if guardrail is None:
            raise ApiError(
                400,
                "model_review_action_not_supported",
                f"Unsupported governed action '{action_type}'.",
            )
        if task.review_type != guardrail["review_type"]:
            raise ApiError(
                409,
                "model_review_task_wrong_type",
                f"Task {task_id} is '{task.review_type}' and cannot govern '{action_type}'.",
            )
        if task.status != "resolved":
            raise ApiError(
                409,
                "model_review_task_not_resolved",
                f"Task {task_id} must be resolved before it can govern '{action_type}'.",
            )
        if task.decision != guardrail["required_decision"]:
            raise ApiError(
                409,
                "model_review_task_not_approved",
                f"Task {task_id} does not carry the required approval decision for '{action_type}'.",
            )
        if (
            candidate_model_version is not None
            and task.candidate_model_version is not None
            and task.candidate_model_version != candidate_model_version
        ):
            raise ApiError(
                409,
                "model_review_task_candidate_mismatch",
                f"Task {task_id} was approved for candidate '{task.candidate_model_version}', not '{candidate_model_version}'.",
            )
        if (
            active_model_version is not None
            and task.active_model_version != active_model_version
        ):
            raise ApiError(
                409,
                "model_review_task_active_model_mismatch",
                f"Task {task_id} was approved for active model '{task.active_model_version}', not '{active_model_version}'.",
            )
        if (
            bool(guardrail.get("enforce_dataset_match"))
            and dataset_version is not None
            and task.dataset_version is not None
            and task.dataset_version != dataset_version
        ):
            raise ApiError(
                409,
                "model_review_task_dataset_mismatch",
                f"Task {task_id} was approved for dataset '{task.dataset_version}', not '{dataset_version}'.",
            )
        return task

    def record_governed_action(
        self,
        *,
        task_id: int,
        action_type: str,
        actor: str,
        outcome: dict,
    ) -> ModelReviewTaskRead:
        task = self._get_task(task_id)
        now = self._now()
        payload = dict(task.details or {})
        governed_actions = list(payload.get("governed_actions") or [])
        governed_actions.append(
            {
                "action": action_type,
                "actor": actor,
                "at": now.isoformat(),
                "outcome": outcome,
            }
        )
        payload["governed_actions"] = governed_actions
        payload["last_governed_action"] = action_type
        payload["last_governed_action_at"] = now.isoformat()
        payload["last_governed_action_by"] = actor
        task.details = payload
        task.updated_at = now
        task.updated_by = actor
        self._append_history(
            task=task,
            action="governed_action_recorded",
            actor=actor,
            at=now,
            details={"action_type": action_type, "outcome": outcome},
        )
        self.session.commit()
        return self._task_read(task)
