from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models import NotificationEvent


def _routing(notification: NotificationEvent) -> dict:
    routing = notification.details.get("routing") if notification.details else None
    return dict(routing or {})


def build_in_app_payload_preview(notification: NotificationEvent) -> dict:
    routing = _routing(notification)
    return {
        "channel": "in_app",
        "title": notification.title,
        "message": notification.message,
        "target_username": notification.target_username,
        "template_key": notification.details.get("template_key") if notification.details else None,
        "routing_audience": routing.get("routing_audience"),
    }


def build_email_payload_preview(notification: NotificationEvent) -> dict:
    routing = _routing(notification)
    subject = f"[{notification.severity.upper()}] {notification.title}"
    body_lines = [
        notification.message,
        "",
        f"Event type: {notification.event_type}",
    ]
    if notification.related_label_id is not None:
        body_lines.append(f"Related label: {notification.related_label_id}")
    if routing.get("routing_audience"):
        body_lines.append(f"Audience: {routing['routing_audience']}")
    template_key = notification.details.get("template_key") if notification.details else None
    if template_key:
        body_lines.append(f"Template: {template_key}")
    return {
        "channel": "email_stub",
        "to": notification.target_username,
        "subject": subject[:160],
        "body_preview": "\n".join(body_lines)[:500],
        "template_key": template_key,
        "routing_audience": routing.get("routing_audience"),
    }


def build_webhook_payload_preview(notification: NotificationEvent) -> dict:
    routing = _routing(notification)
    return {
        "channel": "ops_webhook_stub",
        "webhook_body": {
            "eventType": notification.event_type,
            "severity": notification.severity,
            "title": notification.title,
            "message": notification.message,
            "targetUsername": notification.target_username,
            "relatedLabelId": notification.related_label_id,
            "templateKey": notification.details.get("template_key") if notification.details else None,
            "routingAudience": routing.get("routing_audience"),
        }
    }
