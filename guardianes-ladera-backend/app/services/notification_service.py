"""Facade service for notification admin operations.

This module provides ``AdminNotificationService``, a thin facade that
exposes the notification-related endpoints previously routed through
``OutcomeLabelService``.  Internally it delegates to the existing
``NotificationService`` (``app.services.notifications``), keeping all
business logic in one place.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.schemas.admin import (
    AcknowledgeNotificationsResponse,
    NotificationDeliveryAttemptRead,
    NotificationDeliverySummaryRead,
    NotificationEventRead,
    RetryNotificationDeliveryResponse,
    TriggerNotificationAckScanResponse,
    TriggerNotificationDeliveryFailureScanResponse,
    TriggerNotificationDeliveryRetryScanResponse,
)
from app.services.notifications import NotificationService


class AdminNotificationService:
    """Admin-facing notification operations.

    Each public method mirrors the signature that ``OutcomeLabelService``
    previously exposed so that the ``admin.py`` router can switch its
    dependency without changing any call-sites.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.notification_service = NotificationService(session)

    # ------------------------------------------------------------------
    # Listing / querying
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Scans
    # ------------------------------------------------------------------

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
