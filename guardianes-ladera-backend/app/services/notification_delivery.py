from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.core.config import Settings
from app.services.notification_payloads import (
    build_email_payload_preview,
    build_in_app_payload_preview,
    build_webhook_payload_preview,
)

if TYPE_CHECKING:
    from app.models import NotificationEvent


@dataclass(slots=True)
class NotificationDeliveryResult:
    status: str
    delivery_reference: str | None
    details: dict
    provider_name: str
    provider_status: str
    failure_classification: str | None = None
    retryable: bool = False
    payload_preview: dict | None = None
    provider_receipt: dict | None = None
    error_message: str | None = None


class BaseNotificationDeliveryAdapter:
    channel = "unknown"
    adapter_key = "unknown.adapter"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def deliver(self, notification: NotificationEvent) -> NotificationDeliveryResult:
        raise NotImplementedError

    def _should_fail(self) -> bool:
        return self.channel.lower() in {item.lower() for item in self.settings.notification_stub_fail_channels}


class InAppNotificationDeliveryAdapter(BaseNotificationDeliveryAdapter):
    channel = "in_app"
    adapter_key = "notifications.in_app"

    def deliver(self, notification: NotificationEvent) -> NotificationDeliveryResult:
        return NotificationDeliveryResult(
            status="completed",
            delivery_reference=f"in-app:{notification.id}",
            provider_name="in_app_store",
            provider_status="stored",
            retryable=False,
            payload_preview=build_in_app_payload_preview(notification),
            provider_receipt={
                "receipt_type": "persistence",
                "provider_status": "stored",
            },
            details={
                "delivery_mode": "persisted_in_app_notification",
                "target_username": notification.target_username,
            },
        )


class EmailStubNotificationDeliveryAdapter(BaseNotificationDeliveryAdapter):
    channel = "email_stub"
    adapter_key = "notifications.email_stub"

    def deliver(self, notification: NotificationEvent) -> NotificationDeliveryResult:
        if self._should_fail():
            return NotificationDeliveryResult(
                status="failed",
                delivery_reference=None,
                provider_name="email_stub",
                provider_status="temporary_failure",
                failure_classification="stubbed_transient_failure",
                retryable=True,
                payload_preview=build_email_payload_preview(notification),
                provider_receipt={
                    "receipt_type": "provider_stub_failure",
                    "provider_status": "temporary_failure",
                    "provider_code": "EMAIL_STUB_TEMPFAIL",
                },
                details={
                    "recipient": notification.target_username,
                    "provider": "email_stub",
                },
                error_message="Stubbed email delivery failure for configured channel.",
            )
        return NotificationDeliveryResult(
            status="completed",
            delivery_reference=f"email-stub:{notification.id}:{notification.target_username or 'unrouted'}",
            provider_name="email_stub",
            provider_status="accepted",
            retryable=False,
            payload_preview=build_email_payload_preview(notification),
            provider_receipt={
                "receipt_type": "provider_stub_ack",
                "provider_status": "accepted",
                "provider_code": "EMAIL_STUB_ACCEPTED",
            },
            details={
                "recipient": notification.target_username,
                "provider": "email_stub",
                "subject_preview": notification.title[:120],
            },
        )


class OpsWebhookStubNotificationDeliveryAdapter(BaseNotificationDeliveryAdapter):
    channel = "ops_webhook_stub"
    adapter_key = "notifications.ops_webhook_stub"

    def deliver(self, notification: NotificationEvent) -> NotificationDeliveryResult:
        if self._should_fail():
            return NotificationDeliveryResult(
                status="failed",
                delivery_reference=None,
                provider_name="ops_webhook_stub",
                provider_status="temporary_failure",
                failure_classification="stubbed_transient_failure",
                retryable=True,
                payload_preview=build_webhook_payload_preview(notification),
                provider_receipt={
                    "receipt_type": "provider_stub_failure",
                    "provider_status": "temporary_failure",
                    "provider_code": "WEBHOOK_STUB_TEMPFAIL",
                },
                details={
                    "target": "ops_webhook_stub",
                    "event_type": notification.event_type,
                },
                error_message="Stubbed webhook delivery failure for configured channel.",
            )
        return NotificationDeliveryResult(
            status="completed",
            delivery_reference=f"ops-webhook-stub:{notification.id}",
            provider_name="ops_webhook_stub",
            provider_status="accepted",
            retryable=False,
            payload_preview=build_webhook_payload_preview(notification),
            provider_receipt={
                "receipt_type": "provider_stub_ack",
                "provider_status": "accepted",
                "provider_code": "WEBHOOK_STUB_ACCEPTED",
            },
            details={
                "target": "ops_webhook_stub",
                "event_type": notification.event_type,
                "severity": notification.severity,
            },
        )


def build_notification_delivery_adapter(channel: str, settings: Settings) -> BaseNotificationDeliveryAdapter:
    normalized = channel.strip().lower()
    adapter_map = {
        "in_app": InAppNotificationDeliveryAdapter,
        "email_stub": EmailStubNotificationDeliveryAdapter,
        "ops_webhook_stub": OpsWebhookStubNotificationDeliveryAdapter,
    }
    adapter_class = adapter_map.get(normalized)
    if adapter_class is None:
        raise ValueError(f"Unsupported notification delivery channel '{channel}'.")
    return adapter_class(settings)
