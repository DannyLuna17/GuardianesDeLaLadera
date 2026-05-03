from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import get_settings
from app.core.exceptions import ApiError
from app.models import JobExecution, NotificationDeliveryAttempt, NotificationEvent
from app.schemas.admin import (
    AcknowledgeNotificationsResponse,
    JobExecutionRead,
    NotificationDeliveryAttemptRead,
    NotificationDeliverySummaryRead,
    NotificationEventRead,
    RetryNotificationDeliveryResponse,
    TriggerNotificationAckScanResponse,
    TriggerNotificationDeliveryFailureScanResponse,
    TriggerNotificationDeliveryRetryScanResponse,
)
from app.services.notification_delivery import build_notification_delivery_adapter

DELIVERY_FAILURE_ALERT_EVENT_TYPE = "notification_delivery_failure_alert"


class NotificationService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.settings = get_settings()

    @staticmethod
    def _normalize_datetime(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).replace(microsecond=0)

    @staticmethod
    def _is_default_in_app_only(channels: list[str]) -> bool:
        return [channel.strip().lower() for channel in channels] == ["in_app"]

    @staticmethod
    def _non_real_data_default_channels(severity: str) -> list[str]:
        normalized = severity.lower().strip()
        if normalized == "critical":
            return ["in_app", "email_stub", "ops_webhook_stub"]
        if normalized == "warning":
            return ["in_app", "email_stub"]
        return ["in_app"]

    def _delivery_channels_for_severity(self, severity: str) -> list[str]:
        normalized = severity.lower().strip()
        configured: list[str]
        if normalized == "critical":
            configured = list(self.settings.notification_delivery_channels_critical)
        elif normalized == "warning":
            configured = list(self.settings.notification_delivery_channels_warning)
        else:
            configured = list(self.settings.notification_delivery_channels_info)
        if (
            not self.settings.real_data_only
            and self._is_default_in_app_only(configured)
        ):
            return self._non_real_data_default_channels(normalized)
        return configured

    def _ack_due_at_for_severity(self, created_at: datetime, severity: str) -> datetime:
        normalized = severity.lower().strip()
        if normalized == "critical":
            hours = self.settings.notification_ack_deadline_hours_critical
        elif normalized == "warning":
            hours = self.settings.notification_ack_deadline_hours_warning
        else:
            hours = self.settings.notification_ack_deadline_hours_info
        return (created_at + timedelta(hours=hours)).replace(microsecond=0)

    def _is_ack_overdue(self, notification: NotificationEvent) -> bool:
        normalized_due_at = self._normalize_datetime(notification.ack_due_at)
        if normalized_due_at is None or notification.acknowledged_at is not None:
            return False
        return normalized_due_at < datetime.now(timezone.utc).replace(microsecond=0)

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
    def _severity_to_reminder_severity(severity: str) -> str:
        normalized = severity.lower().strip()
        if normalized == "critical":
            return "critical"
        return "warning"

    def _reminder_severity_for_sequence(
        self, source_severity: str, reminder_sequence: int
    ) -> str:
        if (
            reminder_sequence
            >= self.settings.notification_ack_reminder_escalate_after_count
        ):
            return "critical"
        return self._severity_to_reminder_severity(source_severity)

    def _reminder_channels_for_sequence(
        self, source_severity: str, reminder_sequence: int
    ) -> list[str]:
        reminder_severity = self._reminder_severity_for_sequence(
            source_severity, reminder_sequence
        )
        configured: list[str]
        if reminder_severity == "critical":
            configured = list(self.settings.notification_ack_reminder_channels_critical)
        else:
            configured = list(self.settings.notification_ack_reminder_channels_warning)
        if (
            not self.settings.real_data_only
            and self._is_default_in_app_only(configured)
        ):
            return self._non_real_data_default_channels(reminder_severity)
        return configured

    @staticmethod
    def _configured_channels(notification: NotificationEvent) -> list[str]:
        return list(
            notification.delivery_channels
            or ([notification.channel] if notification.channel else [])
        )

    def _retryable_failure_classifications(self) -> set[str]:
        configured = {
            item.lower()
            for item in self.settings.notification_retryable_failure_classifications
        }
        if not self.settings.real_data_only:
            configured.add("stubbed_transient_failure")
        return configured

    @staticmethod
    def _normalized_routing_targets(targets: list[dict] | None) -> list[dict]:
        normalized_targets: list[dict] = []
        seen_usernames: set[str] = set()
        for index, target in enumerate(targets or []):
            username = str(target.get("username") or "").strip()
            if not username or username in seen_usernames:
                continue
            seen_usernames.add(username)
            normalized_targets.append(
                {
                    "username": username,
                    "routing_audience": str(target.get("routing_audience") or "direct"),
                    "routing_reason": str(target.get("routing_reason") or "direct"),
                    "is_primary": bool(target.get("is_primary", index == 0)),
                }
            )
        if normalized_targets and not any(
            target["is_primary"] for target in normalized_targets
        ):
            normalized_targets[0]["is_primary"] = True
        primary_username = None
        for target in normalized_targets:
            if target["is_primary"]:
                primary_username = target["username"]
                break
        if primary_username is None and normalized_targets:
            normalized_targets[0]["is_primary"] = True
            primary_username = normalized_targets[0]["username"]
        if primary_username is not None:
            for target in normalized_targets:
                target["is_primary"] = target["username"] == primary_username
        return normalized_targets

    def _latest_delivery_attempt_by_channel(
        self,
        notification: NotificationEvent,
    ) -> dict[str, NotificationDeliveryAttempt]:
        latest: dict[str, NotificationDeliveryAttempt] = {}
        for attempt in notification.delivery_attempts or []:
            previous = latest.get(attempt.channel)
            attempt_key = (
                self._normalize_datetime(attempt.attempted_at)
                or datetime.min.replace(tzinfo=timezone.utc),
                attempt.id,
            )
            if previous is None:
                latest[attempt.channel] = attempt
                continue
            previous_key = (
                self._normalize_datetime(previous.attempted_at)
                or datetime.min.replace(tzinfo=timezone.utc),
                previous.id,
            )
            if attempt_key >= previous_key:
                latest[attempt.channel] = attempt
        return latest

    def _delivery_summary(self, notification: NotificationEvent) -> dict:
        configured_channels = self._configured_channels(notification)
        latest_by_channel = self._latest_delivery_attempt_by_channel(notification)
        configured_channel_set = set(configured_channels)
        completed_channels = {
            channel
            for channel, attempt in latest_by_channel.items()
            if attempt.status == "completed"
        }
        failed_channels = {
            channel
            for channel, attempt in latest_by_channel.items()
            if attempt.status == "failed"
        }
        attempted_channels = set(latest_by_channel)

        if not configured_channels:
            delivery_status = "not_configured"
        elif not attempted_channels:
            delivery_status = "pending"
        elif attempted_channels != configured_channel_set:
            delivery_status = (
                "partial_failure" if failed_channels else "partial_pending"
            )
        elif completed_channels == configured_channel_set:
            delivery_status = "delivered"
        elif failed_channels == configured_channel_set:
            delivery_status = "failed"
        elif failed_channels and completed_channels:
            delivery_status = "partial_failure"
        else:
            delivery_status = "pending"

        last_attempt_at = None
        if notification.delivery_attempts:
            last_attempt_at = max(
                (
                    self._normalize_datetime(attempt.attempted_at)
                    for attempt in notification.delivery_attempts
                    if attempt.attempted_at is not None
                ),
                default=None,
            )

        return {
            "delivery_status": delivery_status,
            "delivery_attempt_count": len(notification.delivery_attempts or []),
            "failed_delivery_count": len(failed_channels),
            "last_delivery_attempt_at": last_attempt_at,
        }

    def _attempt_count_for_channel(
        self, notification: NotificationEvent, channel: str
    ) -> int:
        return sum(
            1
            for attempt in notification.delivery_attempts or []
            if attempt.channel == channel
        )

    def _delivery_failure_watch_targets(
        self, source_notification: NotificationEvent
    ) -> list[dict]:
        configured_watchers = (
            list(self.settings.notification_delivery_failure_watch_usernames)
            or list(self.settings.notification_release_ops_usernames)
            or (
                [self.settings.seed_admin_username]
                if self.settings.seed_admin_username
                else []
            )
        )
        primary_username = (
            configured_watchers[0]
            if configured_watchers
            else source_notification.target_username
        )
        targets: list[dict] = []
        for username in configured_watchers:
            targets.append(
                {
                    "username": username,
                    "routing_audience": "delivery_ops_watch",
                    "routing_reason": "notification_delivery_failure_monitor",
                    "is_primary": username == primary_username,
                }
            )
        if not targets and source_notification.target_username:
            targets.append(
                {
                    "username": source_notification.target_username,
                    "routing_audience": "source_notification_owner",
                    "routing_reason": "notification_delivery_failure_fallback",
                    "is_primary": True,
                }
            )
        return targets

    @staticmethod
    def _attempt_provider_name(attempt: NotificationDeliveryAttempt) -> str | None:
        return attempt.details.get("provider_name") if attempt.details else None

    @staticmethod
    def _attempt_provider_status(attempt: NotificationDeliveryAttempt) -> str | None:
        return attempt.details.get("provider_status") if attempt.details else None

    @staticmethod
    def _attempt_failure_classification(
        attempt: NotificationDeliveryAttempt,
    ) -> str | None:
        return (
            attempt.details.get("failure_classification") if attempt.details else None
        )

    @staticmethod
    def _attempt_retryable(attempt: NotificationDeliveryAttempt) -> bool:
        return bool((attempt.details or {}).get("retryable", False))

    @staticmethod
    def _attempt_delivery_origin(attempt: NotificationDeliveryAttempt) -> str | None:
        return attempt.details.get("delivery_origin") if attempt.details else None

    @staticmethod
    def _attempt_payload_preview(attempt: NotificationDeliveryAttempt) -> dict | None:
        value = attempt.details.get("payload_preview") if attempt.details else None
        return dict(value) if isinstance(value, dict) else value

    @staticmethod
    def _attempt_provider_receipt(attempt: NotificationDeliveryAttempt) -> dict | None:
        value = attempt.details.get("provider_receipt") if attempt.details else None
        return dict(value) if isinstance(value, dict) else value

    def _attempt_is_retryable_failure(
        self, attempt: NotificationDeliveryAttempt
    ) -> bool:
        if attempt.status != "failed":
            return False
        failure_classification = (
            (self._attempt_failure_classification(attempt) or "").strip().lower()
        )
        retryable_failure_classifications = self._retryable_failure_classifications()
        if (
            failure_classification
            and failure_classification not in retryable_failure_classifications
        ):
            return False
        return self._attempt_retryable(attempt)

    def _delivery_failure_state(self, notification: NotificationEvent) -> dict:
        latest_by_channel = self._latest_delivery_attempt_by_channel(notification)
        failed_channels: list[str] = []
        retryable_failed_channels: list[str] = []
        retryable_alertable_channels: list[str] = []
        non_retryable_failed_channels: list[str] = []
        max_attempt_reached_channels: list[str] = []
        attempt_counts_by_channel: dict[str, int] = {}
        provider_names_by_channel: dict[str, str] = {}
        failure_classifications_by_channel: dict[str, str] = {}
        last_failed_attempt_at: datetime | None = None

        for channel in self._configured_channels(notification):
            latest_attempt = latest_by_channel.get(channel)
            if latest_attempt is None or latest_attempt.status != "failed":
                continue

            failed_channels.append(channel)
            attempt_count = self._attempt_count_for_channel(notification, channel)
            attempt_counts_by_channel[channel] = attempt_count
            provider_names_by_channel[channel] = (
                self._attempt_provider_name(latest_attempt)
                or latest_attempt.adapter_key
            )
            failure_classifications_by_channel[channel] = (
                self._attempt_failure_classification(latest_attempt) or "unknown"
            )

            normalized_attempted_at = self._normalize_datetime(
                latest_attempt.attempted_at
            )
            if normalized_attempted_at is not None and (
                last_failed_attempt_at is None
                or normalized_attempted_at > last_failed_attempt_at
            ):
                last_failed_attempt_at = normalized_attempted_at

            if self._attempt_is_retryable_failure(latest_attempt):
                retryable_failed_channels.append(channel)
                if (
                    attempt_count
                    >= self.settings.notification_delivery_failure_alert_after_attempts
                ):
                    retryable_alertable_channels.append(channel)
            else:
                non_retryable_failed_channels.append(channel)

            if (
                attempt_count
                >= self.settings.notification_delivery_retry_max_attempts_per_channel
            ):
                max_attempt_reached_channels.append(channel)

        reason_codes: list[str] = []
        if non_retryable_failed_channels:
            reason_codes.append("non_retryable_failure")
        if max_attempt_reached_channels:
            reason_codes.append("max_attempts_reached")
        if retryable_alertable_channels:
            reason_codes.append("repeated_retryable_failure")

        delivery_status = self._delivery_summary(notification)["delivery_status"]
        severity = (
            "critical"
            if (non_retryable_failed_channels or max_attempt_reached_channels)
            else "warning"
        )
        return {
            "delivery_status": delivery_status,
            "failed_channels": failed_channels,
            "retryable_failed_channels": retryable_failed_channels,
            "retryable_alertable_channels": retryable_alertable_channels,
            "non_retryable_failed_channels": non_retryable_failed_channels,
            "max_attempt_reached_channels": max_attempt_reached_channels,
            "attempt_counts_by_channel": attempt_counts_by_channel,
            "provider_names_by_channel": provider_names_by_channel,
            "failure_classifications_by_channel": failure_classifications_by_channel,
            "last_failed_attempt_at": last_failed_attempt_at,
            "reason_codes": reason_codes,
            "is_problematic": bool(reason_codes),
            "alert_severity": severity,
        }

    def _resolve_delivery_failure_alerts(
        self,
        alerts: list[NotificationEvent],
        *,
        resolved_at: datetime,
        origin: str,
        note: str | None,
        source_notification: NotificationEvent,
    ) -> int:
        resolved_by = f"{origin}:notification_delivery_failure_scan"
        resolved_count = 0
        for alert in alerts:
            if alert.status != "open":
                continue
            details = dict(alert.details or {})
            details["resolved_at"] = resolved_at.isoformat()
            details["resolved_by"] = resolved_by
            details["resolution_note"] = note
            details["resolution_delivery_status"] = self._delivery_summary(
                source_notification
            )["delivery_status"]
            alert.details = details
            alert.status = "resolved"
            alert.acknowledged_at = resolved_at
            alert.acknowledged_by = resolved_by
            resolved_count += 1
        return resolved_count

    def _retryable_failed_channels(
        self,
        notification: NotificationEvent,
        *,
        started_at: datetime,
    ) -> list[str]:
        latest_by_channel = self._latest_delivery_attempt_by_channel(notification)
        retryable: list[str] = []
        backoff_cutoff = started_at - timedelta(
            minutes=self.settings.notification_delivery_retry_backoff_minutes
        )
        for channel in self._configured_channels(notification):
            latest_attempt = latest_by_channel.get(channel)
            if latest_attempt is None or latest_attempt.status != "failed":
                continue
            failure_classification = (
                (self._attempt_failure_classification(latest_attempt) or "")
                .strip()
                .lower()
            )
            retryable_failure_classifications = (
                self._retryable_failure_classifications()
            )
            if (
                failure_classification
                and failure_classification not in retryable_failure_classifications
            ):
                continue
            if failure_classification and not self._attempt_retryable(latest_attempt):
                continue
            if (
                self._attempt_count_for_channel(notification, channel)
                >= self.settings.notification_delivery_retry_max_attempts_per_channel
            ):
                continue
            latest_attempted_at = self._normalize_datetime(latest_attempt.attempted_at)
            if latest_attempted_at is not None and latest_attempted_at > backoff_cutoff:
                continue
            retryable.append(channel)
        return retryable

    def _attempt_read(
        self, attempt: NotificationDeliveryAttempt
    ) -> NotificationDeliveryAttemptRead:
        notification = attempt.notification
        return NotificationDeliveryAttemptRead(
            id=attempt.id,
            notificationId=attempt.notification_event_id,
            eventType=notification.event_type if notification is not None else "",
            channel=attempt.channel,
            adapterKey=attempt.adapter_key,
            providerName=self._attempt_provider_name(attempt),
            providerStatus=self._attempt_provider_status(attempt),
            status=attempt.status,
            failureClassification=self._attempt_failure_classification(attempt),
            retryable=self._attempt_retryable(attempt),
            deliveryOrigin=self._attempt_delivery_origin(attempt),
            payloadPreview=self._attempt_payload_preview(attempt),
            providerReceipt=self._attempt_provider_receipt(attempt),
            targetUsername=notification.target_username
            if notification is not None
            else None,
            relatedLabelId=notification.related_label_id
            if notification is not None
            else None,
            attemptedAt=attempt.attempted_at,
            completedAt=attempt.completed_at,
            deliveryReference=attempt.delivery_reference,
            errorMessage=attempt.error_message,
            details=attempt.details or {},
        )

    def _read(self, notification: NotificationEvent) -> NotificationEventRead:
        delivery_summary = self._delivery_summary(notification)
        return NotificationEventRead(
            id=notification.id,
            eventType=notification.event_type,
            severity=notification.severity,
            status=notification.status,
            channel=notification.channel,
            deliveryChannels=self._configured_channels(notification),
            deliveryStatus=delivery_summary["delivery_status"],
            deliveryAttemptCount=delivery_summary["delivery_attempt_count"],
            failedDeliveryCount=delivery_summary["failed_delivery_count"],
            lastDeliveryAttemptAt=delivery_summary["last_delivery_attempt_at"],
            title=notification.title,
            message=notification.message,
            targetUsername=notification.target_username,
            relatedLabelId=notification.related_label_id,
            details=notification.details or {},
            createdAt=notification.created_at,
            ackDueAt=notification.ack_due_at,
            reminderCount=notification.reminder_count,
            lastReminderAt=notification.last_reminder_at,
            isAckOverdue=self._is_ack_overdue(notification),
            acknowledgedAt=notification.acknowledged_at,
            acknowledgedBy=notification.acknowledged_by,
        )

    def _record_delivery_attempt(
        self,
        *,
        notification: NotificationEvent,
        channel: str,
        adapter_key: str,
        status: str,
        attempted_at: datetime,
        completed_at: datetime | None,
        delivery_reference: str | None,
        error_message: str | None,
        details: dict | None,
    ) -> NotificationDeliveryAttempt:
        attempt = NotificationDeliveryAttempt(
            notification=notification,
            channel=channel,
            adapter_key=adapter_key,
            status=status,
            attempted_at=attempted_at,
            completed_at=completed_at,
            delivery_reference=delivery_reference,
            error_message=error_message,
            details=details or {},
        )
        self.session.add(attempt)
        self.session.flush()
        return attempt

    def _deliver_notification(
        self,
        notification: NotificationEvent,
        *,
        channels: list[str],
        delivery_origin: str,
        delivery_note: str | None = None,
    ) -> list[NotificationDeliveryAttempt]:
        attempts: list[NotificationDeliveryAttempt] = []
        for channel in channels:
            attempted_at = datetime.now(timezone.utc).replace(microsecond=0)
            try:
                adapter = build_notification_delivery_adapter(channel, self.settings)
                result = adapter.deliver(notification)
                details = dict(result.details or {})
                details["provider_name"] = result.provider_name
                details["provider_status"] = result.provider_status
                details["failure_classification"] = result.failure_classification
                details["retryable"] = result.retryable
                details["payload_preview"] = result.payload_preview
                details["provider_receipt"] = result.provider_receipt
                details["delivery_origin"] = delivery_origin
                if delivery_note:
                    details["delivery_note"] = delivery_note
                attempt = self._record_delivery_attempt(
                    notification=notification,
                    channel=channel,
                    adapter_key=adapter.adapter_key,
                    status=result.status,
                    attempted_at=attempted_at,
                    completed_at=attempted_at
                    if result.status in {"completed", "failed"}
                    else None,
                    delivery_reference=result.delivery_reference,
                    error_message=result.error_message,
                    details=details,
                )
            except Exception as exc:
                attempt = self._record_delivery_attempt(
                    notification=notification,
                    channel=channel,
                    adapter_key=f"notifications.unsupported:{channel}",
                    status="failed",
                    attempted_at=attempted_at,
                    completed_at=attempted_at,
                    delivery_reference=None,
                    error_message=str(exc),
                    details={
                        "provider_name": channel,
                        "provider_status": "configuration_error",
                        "failure_classification": "configuration_error",
                        "retryable": False,
                        "delivery_origin": delivery_origin,
                        "delivery_note": delivery_note,
                        "exception_type": exc.__class__.__name__,
                        "provider_receipt": {
                            "receipt_type": "unsupported_channel",
                            "provider_status": "configuration_error",
                            "provider_code": "UNSUPPORTED_CHANNEL",
                        },
                    },
                )
            attempts.append(attempt)
        return attempts

    def create_event(
        self,
        *,
        event_type: str,
        severity: str,
        title: str,
        message: str,
        target_username: str | None = None,
        related_label_id: int | None = None,
        details: dict | None = None,
        delivery_channels: list[str] | None = None,
        ack_due_at: datetime | None = None,
    ) -> NotificationEvent:
        created_at = datetime.now(timezone.utc).replace(microsecond=0)
        resolved_channels = delivery_channels or self._delivery_channels_for_severity(
            severity
        )
        resolved_ack_due_at = self._normalize_datetime(
            ack_due_at
        ) or self._ack_due_at_for_severity(created_at, severity)
        notification = NotificationEvent(
            event_type=event_type,
            severity=severity,
            status="open",
            channel=resolved_channels[0] if resolved_channels else "in_app",
            delivery_channels=resolved_channels,
            title=title,
            message=message,
            target_username=target_username,
            related_label_id=related_label_id,
            details=details or {},
            created_at=created_at,
            ack_due_at=resolved_ack_due_at,
            reminder_count=0,
            last_reminder_at=None,
            acknowledged_at=None,
            acknowledged_by=None,
        )
        self.session.add(notification)
        self.session.flush()
        self._deliver_notification(
            notification, channels=resolved_channels, delivery_origin="initial"
        )
        return notification

    def create_routed_events(
        self,
        *,
        event_type: str,
        severity: str,
        title: str,
        message: str,
        targets: list[dict] | None,
        related_label_id: int | None = None,
        details: dict | None = None,
        delivery_channels: list[str] | None = None,
        ack_due_at: datetime | None = None,
        template_key: str | None = None,
        template_version: str = "v2",
    ) -> list[NotificationEvent]:
        normalized_targets = self._normalized_routing_targets(targets)
        notifications: list[NotificationEvent] = []
        for index, target in enumerate(normalized_targets, start=1):
            routed_details = dict(details or {})
            routed_details["template_key"] = template_key or event_type
            routed_details["template_version"] = template_version
            routed_details["routing"] = {
                "target_username": target["username"],
                "routing_audience": target["routing_audience"],
                "routing_reason": target["routing_reason"],
                "is_primary": target["is_primary"],
                "target_index": index,
                "target_count": len(normalized_targets),
            }
            notification = self.create_event(
                event_type=event_type,
                severity=severity,
                title=title,
                message=message,
                target_username=target["username"],
                related_label_id=related_label_id,
                details=routed_details,
                delivery_channels=delivery_channels,
                ack_due_at=ack_due_at,
            )
            notifications.append(notification)
        return notifications

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
        statement = (
            select(NotificationEvent)
            .options(selectinload(NotificationEvent.delivery_attempts))
            .order_by(NotificationEvent.created_at.desc(), NotificationEvent.id.desc())
        )
        if status:
            statement = statement.where(NotificationEvent.status == status)
        if severity:
            statement = statement.where(NotificationEvent.severity == severity)
        if target_username:
            statement = statement.where(
                NotificationEvent.target_username == target_username
            )
        if event_type:
            statement = statement.where(NotificationEvent.event_type == event_type)
        if not (channel or delivery_status or overdue_only):
            statement = statement.limit(limit)
        notifications = list(self.session.scalars(statement).all())
        notification_reads = [
            self._read(notification) for notification in notifications
        ]
        if channel:
            normalized_channel = channel.strip().lower()
            notification_reads = [
                notification
                for notification in notification_reads
                if normalized_channel
                in {item.lower() for item in notification.delivery_channels}
            ]
        if delivery_status:
            normalized_delivery_status = delivery_status.strip().lower()
            notification_reads = [
                notification
                for notification in notification_reads
                if notification.delivery_status.lower() == normalized_delivery_status
            ]
        if overdue_only:
            notification_reads = [
                notification
                for notification in notification_reads
                if notification.is_ack_overdue
            ]
        if channel or delivery_status or overdue_only:
            notification_reads = notification_reads[:limit]
        return notification_reads

    def list_delivery_attempts(
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
        statement = (
            select(NotificationDeliveryAttempt)
            .options(selectinload(NotificationDeliveryAttempt.notification))
            .order_by(
                NotificationDeliveryAttempt.attempted_at.desc(),
                NotificationDeliveryAttempt.id.desc(),
            )
        )
        if notification_id is not None:
            statement = statement.where(
                NotificationDeliveryAttempt.notification_event_id == notification_id
            )
        if channel:
            statement = statement.where(NotificationDeliveryAttempt.channel == channel)
        if status:
            statement = statement.where(NotificationDeliveryAttempt.status == status)
        if target_username:
            statement = statement.join(NotificationDeliveryAttempt.notification).where(
                NotificationEvent.target_username == target_username
            )
        attempts = list(self.session.scalars(statement).all())
        attempt_reads = [self._attempt_read(attempt) for attempt in attempts]
        if provider_name:
            normalized_provider_name = provider_name.strip().lower()
            attempt_reads = [
                attempt
                for attempt in attempt_reads
                if (attempt.provider_name or "").lower() == normalized_provider_name
            ]
        if failure_classification:
            normalized_failure_classification = failure_classification.strip().lower()
            attempt_reads = [
                attempt
                for attempt in attempt_reads
                if (attempt.failure_classification or "").lower()
                == normalized_failure_classification
            ]
        if delivery_origin:
            normalized_delivery_origin = delivery_origin.strip().lower()
            attempt_reads = [
                attempt
                for attempt in attempt_reads
                if (attempt.delivery_origin or "").lower() == normalized_delivery_origin
            ]
        return attempt_reads[:limit]

    def get_delivery_summary(self) -> NotificationDeliverySummaryRead:
        statement = (
            select(NotificationEvent)
            .options(selectinload(NotificationEvent.delivery_attempts))
            .order_by(NotificationEvent.created_at.desc(), NotificationEvent.id.desc())
        )
        notifications = list(self.session.scalars(statement).all())

        delivery_status_counts: dict[str, int] = {}
        severity_counts: dict[str, int] = {}
        channel_failure_counts: dict[str, int] = {}
        provider_failure_counts: dict[str, int] = {}
        failure_classification_counts: dict[str, int] = {}
        notifications_with_failures = 0
        retryable_failure_notification_count = 0
        non_retryable_failure_notification_count = 0
        max_attempt_reached_notification_count = 0
        ack_overdue_count = 0
        active_alert_count = 0
        oldest_outstanding_failure_at: datetime | None = None

        for notification in notifications:
            delivery_status = self._delivery_summary(notification)["delivery_status"]
            delivery_status_counts[delivery_status] = (
                delivery_status_counts.get(delivery_status, 0) + 1
            )
            severity_key = notification.severity.lower().strip()
            severity_counts[severity_key] = severity_counts.get(severity_key, 0) + 1

            if self._is_ack_overdue(notification):
                ack_overdue_count += 1
            if (
                notification.event_type == DELIVERY_FAILURE_ALERT_EVENT_TYPE
                and notification.status == "open"
            ):
                active_alert_count += 1

            failure_state = self._delivery_failure_state(notification)
            if not failure_state["failed_channels"]:
                continue

            notifications_with_failures += 1
            if failure_state["retryable_failed_channels"]:
                retryable_failure_notification_count += 1
            if failure_state["non_retryable_failed_channels"]:
                non_retryable_failure_notification_count += 1
            if failure_state["max_attempt_reached_channels"]:
                max_attempt_reached_notification_count += 1

            latest_failed_at = failure_state["last_failed_attempt_at"]
            if latest_failed_at is not None and (
                oldest_outstanding_failure_at is None
                or latest_failed_at < oldest_outstanding_failure_at
            ):
                oldest_outstanding_failure_at = latest_failed_at

            for channel in failure_state["failed_channels"]:
                channel_failure_counts[channel] = (
                    channel_failure_counts.get(channel, 0) + 1
                )
                provider_name = failure_state["provider_names_by_channel"].get(
                    channel, "unknown"
                )
                provider_failure_counts[provider_name] = (
                    provider_failure_counts.get(provider_name, 0) + 1
                )
                failure_classification = failure_state[
                    "failure_classifications_by_channel"
                ].get(channel, "unknown")
                failure_classification_counts[failure_classification] = (
                    failure_classification_counts.get(failure_classification, 0) + 1
                )

        return NotificationDeliverySummaryRead(
            totalNotifications=len(notifications),
            openNotifications=sum(
                1 for notification in notifications if notification.status == "open"
            ),
            acknowledgedNotifications=sum(
                1
                for notification in notifications
                if notification.status == "acknowledged"
            ),
            resolvedNotifications=sum(
                1 for notification in notifications if notification.status == "resolved"
            ),
            deliveryStatusCounts=delivery_status_counts,
            severityCounts=severity_counts,
            channelFailureCounts=channel_failure_counts,
            providerFailureCounts=provider_failure_counts,
            failureClassificationCounts=failure_classification_counts,
            notificationsWithFailures=notifications_with_failures,
            retryableFailureNotificationCount=retryable_failure_notification_count,
            nonRetryableFailureNotificationCount=non_retryable_failure_notification_count,
            maxAttemptReachedNotificationCount=max_attempt_reached_notification_count,
            ackOverdueCount=ack_overdue_count,
            activeAlertCount=active_alert_count,
            oldestOutstandingFailureAt=oldest_outstanding_failure_at,
        )

    def acknowledge_notifications(
        self,
        *,
        notification_ids: list[int],
        acknowledged_by: str,
    ) -> AcknowledgeNotificationsResponse:
        statement = (
            select(NotificationEvent)
            .where(NotificationEvent.id.in_(notification_ids))
            .options(selectinload(NotificationEvent.delivery_attempts))
        )
        notifications = list(self.session.scalars(statement).all())
        notifications_by_id = {
            notification.id: notification for notification in notifications
        }
        missing_ids = [
            notification_id
            for notification_id in notification_ids
            if notification_id not in notifications_by_id
        ]
        if missing_ids:
            missing_display = ", ".join(
                str(notification_id) for notification_id in missing_ids
            )
            raise ApiError(
                404,
                "notification_not_found",
                f"Notifications not found: {missing_display}.",
            )

        acknowledged_at = datetime.now(timezone.utc).replace(microsecond=0)
        ordered_notifications = [
            notifications_by_id[notification_id] for notification_id in notification_ids
        ]
        for notification in ordered_notifications:
            notification.status = "acknowledged"
            notification.acknowledged_at = acknowledged_at
            notification.acknowledged_by = acknowledged_by

        self.session.commit()
        return AcknowledgeNotificationsResponse(
            acknowledgedCount=len(ordered_notifications),
            notifications=[
                self._read(notification) for notification in ordered_notifications
            ],
        )

    def retry_delivery(
        self,
        *,
        notification_ids: list[int],
        triggered_by: str,
        channels: list[str] | None = None,
        note: str | None = None,
        origin: str = "manual",
    ) -> RetryNotificationDeliveryResponse:
        started_at = datetime.now(timezone.utc).replace(microsecond=0)
        job = JobExecution(
            job_type="notification_delivery_retry",
            status="running",
            started_at=started_at,
            details={
                "origin": origin,
                "triggered_by": triggered_by,
                "note": note,
                "requested_channels": channels or [],
            },
        )
        self.session.add(job)
        self.session.flush()

        statement = (
            select(NotificationEvent)
            .where(NotificationEvent.id.in_(notification_ids))
            .options(selectinload(NotificationEvent.delivery_attempts))
        )
        notifications = list(self.session.scalars(statement).all())
        notifications_by_id = {
            notification.id: notification for notification in notifications
        }
        missing_ids = [
            notification_id
            for notification_id in notification_ids
            if notification_id not in notifications_by_id
        ]
        if missing_ids:
            missing_display = ", ".join(
                str(notification_id) for notification_id in missing_ids
            )
            raise ApiError(
                404,
                "notification_not_found",
                f"Notifications not found: {missing_display}.",
            )

        requested_channels = None
        if channels:
            requested_channels = {
                channel.strip().lower() for channel in channels if channel.strip()
            }
            if not requested_channels:
                requested_channels = None

        attempts: list[NotificationDeliveryAttempt] = []
        retried_count = 0
        skipped_count = 0
        ordered_notifications = [
            notifications_by_id[notification_id] for notification_id in notification_ids
        ]
        for notification in ordered_notifications:
            configured_channels = self._configured_channels(notification)
            latest_by_channel = self._latest_delivery_attempt_by_channel(notification)
            if requested_channels is None:
                target_channels = [
                    channel
                    for channel in configured_channels
                    if channel in latest_by_channel
                    and latest_by_channel[channel].status == "failed"
                ]
            else:
                target_channels = [
                    channel
                    for channel in configured_channels
                    if channel.lower() in requested_channels
                ]

            if not target_channels:
                skipped_count += 1
                continue

            attempts.extend(
                self._deliver_notification(
                    notification,
                    channels=target_channels,
                    delivery_origin="retry",
                    delivery_note=note,
                )
            )
            retried_count += len(target_channels)

        completed_at = datetime.now(timezone.utc).replace(microsecond=0)
        job.status = "completed"
        job.completed_at = completed_at
        job.details = {
            "origin": origin,
            "triggered_by": triggered_by,
            "note": note,
            "requested_notification_count": len(notification_ids),
            "requested_channels": sorted(requested_channels)
            if requested_channels
            else [],
            "retried_count": retried_count,
            "skipped_count": skipped_count,
        }
        self.session.commit()
        self.session.refresh(job)

        return RetryNotificationDeliveryResponse(
            job=self._job_read(job),
            retriedCount=retried_count,
            skippedCount=skipped_count,
            attempts=[self._attempt_read(attempt) for attempt in attempts],
        )

    def run_delivery_retry_scan(
        self,
        *,
        max_notifications: int = 100,
        note: str | None = None,
        origin: str = "manual",
    ) -> TriggerNotificationDeliveryRetryScanResponse:
        started_at = datetime.now(timezone.utc).replace(microsecond=0)
        job = JobExecution(
            job_type="notification_delivery_retry_scan",
            status="running",
            started_at=started_at,
            details={
                "origin": origin,
                "max_notifications": max_notifications,
                "note": note,
            },
        )
        self.session.add(job)
        self.session.flush()

        statement = (
            select(NotificationEvent)
            .options(selectinload(NotificationEvent.delivery_attempts))
            .where(NotificationEvent.status == "open")
            .order_by(NotificationEvent.created_at.asc(), NotificationEvent.id.asc())
            .limit(max_notifications)
        )
        candidate_notifications = list(self.session.scalars(statement).all())

        attempts: list[NotificationDeliveryAttempt] = []
        retried_count = 0
        skipped_count = 0
        eligible_candidates = 0
        for notification in candidate_notifications:
            target_channels = self._retryable_failed_channels(
                notification, started_at=started_at
            )
            if not target_channels:
                skipped_count += 1
                continue
            eligible_candidates += 1
            attempts.extend(
                self._deliver_notification(
                    notification,
                    channels=target_channels,
                    delivery_origin="retry_scan",
                    delivery_note=note,
                )
            )
            retried_count += len(target_channels)

        completed_at = datetime.now(timezone.utc).replace(microsecond=0)
        job.status = "completed"
        job.completed_at = completed_at
        job.details = {
            "origin": origin,
            "max_notifications": max_notifications,
            "note": note,
            "candidate_count": eligible_candidates,
            "retried_count": retried_count,
            "skipped_count": skipped_count,
            "retry_backoff_minutes": self.settings.notification_delivery_retry_backoff_minutes,
            "retry_max_attempts_per_channel": self.settings.notification_delivery_retry_max_attempts_per_channel,
        }
        self.session.commit()
        self.session.refresh(job)

        return TriggerNotificationDeliveryRetryScanResponse(
            job=self._job_read(job),
            candidateCount=eligible_candidates,
            retriedCount=retried_count,
            skippedCount=skipped_count,
            attempts=[self._attempt_read(attempt) for attempt in attempts],
        )

    def run_delivery_failure_scan(
        self,
        *,
        max_notifications: int = 100,
        note: str | None = None,
        origin: str = "manual",
    ) -> TriggerNotificationDeliveryFailureScanResponse:
        started_at = datetime.now(timezone.utc).replace(microsecond=0)
        job = JobExecution(
            job_type="notification_delivery_failure_scan",
            status="running",
            started_at=started_at,
            details={
                "origin": origin,
                "max_notifications": max_notifications,
                "note": note,
            },
        )
        self.session.add(job)
        self.session.flush()

        source_statement = (
            select(NotificationEvent)
            .options(selectinload(NotificationEvent.delivery_attempts))
            .where(NotificationEvent.status == "open")
            .order_by(NotificationEvent.created_at.asc(), NotificationEvent.id.asc())
        )
        source_notifications = [
            notification
            for notification in self.session.scalars(source_statement).all()
            if notification.event_type != DELIVERY_FAILURE_ALERT_EVENT_TYPE
        ][:max_notifications]

        active_alert_statement = (
            select(NotificationEvent)
            .options(selectinload(NotificationEvent.delivery_attempts))
            .where(
                NotificationEvent.event_type == DELIVERY_FAILURE_ALERT_EVENT_TYPE,
                NotificationEvent.status == "open",
            )
        )
        active_alerts = list(self.session.scalars(active_alert_statement).all())
        active_alerts_by_source: dict[int, list[NotificationEvent]] = {}
        for alert in active_alerts:
            source_notification_id = (alert.details or {}).get("source_notification_id")
            if isinstance(source_notification_id, int):
                active_alerts_by_source.setdefault(source_notification_id, []).append(
                    alert
                )

        alert_notifications: list[NotificationEvent] = []
        candidate_count = 0
        alerted_count = 0
        skipped_count = 0
        resolved_alert_count = 0

        for notification in source_notifications:
            failure_state = self._delivery_failure_state(notification)
            existing_alerts = active_alerts_by_source.get(notification.id, [])
            if not failure_state["is_problematic"]:
                if existing_alerts:
                    resolved_alert_count += self._resolve_delivery_failure_alerts(
                        existing_alerts,
                        resolved_at=started_at,
                        origin=origin,
                        note=note,
                        source_notification=notification,
                    )
                continue

            candidate_count += 1
            if existing_alerts:
                skipped_count += 1
                continue

            failed_channels_display = ", ".join(failure_state["failed_channels"])
            reason_display = ", ".join(
                code.replace("_", " ") for code in failure_state["reason_codes"]
            )
            summary = f"Notification {notification.id} is failing delivery on {failed_channels_display}."
            if reason_display:
                summary = f"{summary} Detected reasons: {reason_display}."
            recommended_action = (
                "Review the latest provider receipt and failure classification, then fix non-retryable issues "
                "or allow the retry policy to continue."
            )
            alerts = self.create_routed_events(
                event_type=DELIVERY_FAILURE_ALERT_EVENT_TYPE,
                severity=failure_state["alert_severity"],
                title="Notification delivery requires attention",
                message=summary,
                targets=self._delivery_failure_watch_targets(notification),
                related_label_id=notification.related_label_id,
                details={
                    "source_notification_id": notification.id,
                    "source_event_type": notification.event_type,
                    "source_target_username": notification.target_username,
                    "source_delivery_status": failure_state["delivery_status"],
                    "failed_channels": failure_state["failed_channels"],
                    "retryable_failed_channels": failure_state[
                        "retryable_failed_channels"
                    ],
                    "non_retryable_failed_channels": failure_state[
                        "non_retryable_failed_channels"
                    ],
                    "max_attempt_reached_channels": failure_state[
                        "max_attempt_reached_channels"
                    ],
                    "attempt_counts_by_channel": failure_state[
                        "attempt_counts_by_channel"
                    ],
                    "provider_names_by_channel": failure_state[
                        "provider_names_by_channel"
                    ],
                    "failure_classifications_by_channel": failure_state[
                        "failure_classifications_by_channel"
                    ],
                    "last_failed_attempt_at": failure_state[
                        "last_failed_attempt_at"
                    ].isoformat()
                    if failure_state["last_failed_attempt_at"]
                    else None,
                    "alert_reason_codes": failure_state["reason_codes"],
                    "scan_origin": origin,
                    "scan_note": note,
                    "summary": summary,
                    "recommended_action": recommended_action,
                },
                template_key="notification_delivery_failure_alert",
            )
            alert_notifications.extend(alerts)
            alerted_count += len(alerts)

        completed_at = datetime.now(timezone.utc).replace(microsecond=0)
        job.status = "completed"
        job.completed_at = completed_at
        job.details = {
            "origin": origin,
            "max_notifications": max_notifications,
            "note": note,
            "candidate_count": candidate_count,
            "alerted_count": alerted_count,
            "skipped_count": skipped_count,
            "resolved_alert_count": resolved_alert_count,
            "alert_after_attempts": self.settings.notification_delivery_failure_alert_after_attempts,
            "retry_max_attempts_per_channel": self.settings.notification_delivery_retry_max_attempts_per_channel,
        }
        self.session.commit()
        self.session.refresh(job)

        return TriggerNotificationDeliveryFailureScanResponse(
            job=self._job_read(job),
            candidateCount=candidate_count,
            alertedCount=alerted_count,
            skippedCount=skipped_count,
            resolvedAlertCount=resolved_alert_count,
            alerts=[self._read(notification) for notification in alert_notifications],
        )

    def run_ack_deadline_scan(
        self,
        *,
        max_notifications: int = 100,
        note: str | None = None,
        origin: str = "manual",
    ) -> TriggerNotificationAckScanResponse:
        started_at = datetime.now(timezone.utc).replace(microsecond=0)
        job = JobExecution(
            job_type="notification_ack_deadline_scan",
            status="running",
            started_at=started_at,
            details={
                "origin": origin,
                "max_notifications": max_notifications,
                "note": note,
            },
        )
        self.session.add(job)
        self.session.flush()

        statement = (
            select(NotificationEvent)
            .options(selectinload(NotificationEvent.delivery_attempts))
            .where(
                NotificationEvent.status == "open",
                NotificationEvent.ack_due_at.is_not(None),
                NotificationEvent.ack_due_at < started_at,
                NotificationEvent.reminder_count
                < self.settings.notification_ack_reminder_max_count,
            )
            .order_by(NotificationEvent.ack_due_at.asc(), NotificationEvent.id.asc())
            .limit(max_notifications)
        )
        source_notifications = list(self.session.scalars(statement).all())
        reminder_notifications: list[NotificationEvent] = []
        escalated_reminder_count = 0
        for source_notification in source_notifications:
            reminder_sequence = source_notification.reminder_count + 1
            reminder_severity = self._reminder_severity_for_sequence(
                source_notification.severity, reminder_sequence
            )
            reminder_channels = self._reminder_channels_for_sequence(
                source_notification.severity, reminder_sequence
            )
            if reminder_severity == "critical":
                escalated_reminder_count += 1
            reminder_notification = self.create_event(
                event_type="notification_ack_deadline_reminder",
                severity=reminder_severity,
                title=f"Acknowledgement overdue: {source_notification.title}",
                message=(
                    f"Notification {source_notification.id} remains unacknowledged past its deadline."
                ),
                target_username=source_notification.target_username,
                related_label_id=source_notification.related_label_id,
                delivery_channels=reminder_channels,
                details={
                    "source_notification_id": source_notification.id,
                    "source_event_type": source_notification.event_type,
                    "source_severity": source_notification.severity,
                    "original_ack_due_at": source_notification.ack_due_at.isoformat()
                    if source_notification.ack_due_at
                    else None,
                    "scan_origin": origin,
                    "scan_note": note,
                    "reminder_sequence": reminder_sequence,
                    "reminder_delivery_channels": reminder_channels,
                    "reminder_severity": reminder_severity,
                },
            )
            source_notification.reminder_count += 1
            source_notification.last_reminder_at = started_at
            reminder_notifications.append(reminder_notification)

        completed_at = datetime.now(timezone.utc).replace(microsecond=0)
        job.status = "completed"
        job.completed_at = completed_at
        job.details = {
            "origin": origin,
            "max_notifications": max_notifications,
            "note": note,
            "source_count": len(source_notifications),
            "reminded_count": len(reminder_notifications),
            "reminder_max_count": self.settings.notification_ack_reminder_max_count,
            "escalated_reminder_count": escalated_reminder_count,
            "reminder_escalate_after_count": self.settings.notification_ack_reminder_escalate_after_count,
        }
        self.session.commit()
        self.session.refresh(job)

        return TriggerNotificationAckScanResponse(
            job=self._job_read(job),
            sourceCount=len(source_notifications),
            remindedCount=len(reminder_notifications),
            notifications=[
                self._read(notification) for notification in reminder_notifications
            ],
        )
