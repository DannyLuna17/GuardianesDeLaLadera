"""Date parsing helpers shared by standalone pipeline scripts."""

from __future__ import annotations

from datetime import datetime, timezone


def parse_iso_datetime(value: datetime | str | None) -> datetime | None:
    """Parse an ISO datetime, accepting ``Z`` and assuming UTC for naive values."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value)
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed

