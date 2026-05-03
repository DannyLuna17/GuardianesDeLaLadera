from __future__ import annotations

from app.core.config import get_settings
from app.core.exceptions import ApiError
from app.integrations.provider_adapters import HTTP_ADAPTERS
from app.integrations.seed_adapters import ADAPTERS as SEED_ADAPTERS


def list_supported_sources() -> list[str]:
    return sorted(set(SEED_ADAPTERS) | set(HTTP_ADAPTERS))


def resolve_transport(source_id: str) -> str:
    settings = get_settings()
    configured_transport = settings.transport_for_source(source_id)
    if configured_transport == "auto":
        if settings.real_data_only:
            if settings.source_base_url(source_id):
                return "http"
            raise ApiError(
                409,
                "real_data_source_not_configured",
                f"Source '{source_id}' requires a configured base URL because REAL_DATA_ONLY is enabled.",
            )
        return "seed"
    if configured_transport == "seed" and settings.real_data_only:
        raise ApiError(
            409,
            "seed_transport_disabled",
            f"Seed transport is disabled for source '{source_id}' because REAL_DATA_ONLY is enabled.",
        )
    return configured_transport


def build_adapter(source_id: str):
    settings = get_settings()
    effective_transport = resolve_transport(source_id)
    if effective_transport == "http":
        if not settings.source_base_url(source_id):
            raise ApiError(
                409,
                "real_data_source_not_configured",
                f"Source '{source_id}' requires a configured base URL for HTTP ingestion.",
            )
        adapter_cls = HTTP_ADAPTERS.get(source_id)
    else:
        adapter_cls = SEED_ADAPTERS.get(source_id)
    if adapter_cls is None:
        raise RuntimeError(f"No adapter is registered for source '{source_id}' using transport '{effective_transport}'.")
    return adapter_cls()
