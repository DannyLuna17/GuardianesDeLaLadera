"""String normalization helpers shared by standalone pipeline scripts."""

from __future__ import annotations

import hashlib
import re
import unicodedata


def normalize_admin_name(raw: str | None) -> str:
    """Return an ASCII uppercase key for Colombian admin names."""
    if not raw:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(raw))
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return ascii_only.strip().upper()


def slugify(text: str) -> str:
    """Return a lowercase ASCII slug, falling back to ``unknown``."""
    normalized = (
        unicodedata.normalize("NFKD", text or "")
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    return normalized or "unknown"


def truncate_with_hash(value: str, *, limit: int) -> str:
    """Truncate while keeping a short deterministic suffix for uniqueness."""
    if len(value) <= limit:
        return value
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:6]
    return f"{value[: limit - 7]}-{digest}"

