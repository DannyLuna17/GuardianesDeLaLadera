from __future__ import annotations

from datetime import datetime
from typing import Any


def _coalesce(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def _find_list(payload: Any, *keys: str) -> list[Any] | None:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None

    search_keys = list(keys) + ["items", "records", "results", "events", "municipalities", "data"]
    for key in search_keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested = _find_list(value, *keys)
            if nested is not None:
                return nested
    return None


def _find_dict(payload: Any, *keys: str) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        for key in ("data", "payload", "result"):
            value = payload.get(key)
            if isinstance(value, dict):
                nested = _find_dict(value, *keys)
                if nested is not None:
                    return nested
    return None


def _request_meta(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        value = payload.get("request_meta")
        if isinstance(value, dict):
            return value
    return {}


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _normalize_time_label(value: Any) -> str:
    if value is None or value == "":
        raise RuntimeError("Rain series point is missing a time label.")
    label = str(value).strip()
    if label.lower().startswith("h"):
        return label
    try:
        parsed = datetime.fromisoformat(label.replace("Z", "+00:00"))
    except ValueError:
        return label
    return parsed.strftime("%Y-%m-%d %H:%M")


def _normalize_date(value: Any) -> str:
    if value is None or value == "":
        raise RuntimeError("Event record is missing a date value.")
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        year, month, day = value[:3]
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text[:10]
    return parsed.date().isoformat()


def _normalize_coords(item: dict[str, Any]) -> list[float]:
    location = item.get("location")
    if isinstance(location, dict):
        lat = _coalesce(location, "lat", "latitude")
        lon = _coalesce(location, "lon", "lng", "longitude")
        if lat is not None and lon is not None:
            return [float(lat), float(lon)]

    geometry = item.get("geometry")
    if isinstance(geometry, dict):
        coords = geometry.get("coordinates")
        if isinstance(coords, list) and len(coords) >= 2:
            return [float(coords[1]), float(coords[0])]

    coords = item.get("coords")
    if isinstance(coords, list) and len(coords) >= 2:
        return [float(coords[0]), float(coords[1])]

    coords = item.get("coordinates")
    if isinstance(coords, list) and len(coords) >= 2:
        return [float(coords[0]), float(coords[1])]

    raise RuntimeError("Event record is missing coordinate data.")


def parse_ideam_payload(payload: Any) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rain_series = payload.get("rainSeries") if isinstance(payload, dict) else None
    if isinstance(rain_series, dict):
        return rain_series, {
            "payload_mode": "normalized_http",
            "municipality_count": len(rain_series),
        }

    municipalities = _find_list(payload, "municipalities", "seriesByMunicipality")
    if municipalities is None:
        raise RuntimeError("IDEAM HTTP payload must contain a 'rainSeries' object or municipality rain-series list.")

    normalized: dict[str, list[dict[str, Any]]] = {}
    total_points = 0
    for municipality_entry in municipalities:
        if not isinstance(municipality_entry, dict):
            raise RuntimeError("IDEAM municipality entry must be an object.")
        municipality_name = _coalesce(
            municipality_entry,
            "municipality",
            "municipio",
            "name",
            "nombre",
        )
        if municipality_name is None:
            raise RuntimeError("IDEAM municipality entry is missing the municipality name.")

        series = _find_list(
            municipality_entry,
            "series",
            "rainSeries",
            "timeline",
            "measurements",
            "mediciones",
        )
        if series is None:
            raise RuntimeError(f"IDEAM municipality '{municipality_name}' is missing its rain series collection.")

        normalized_points: list[dict[str, Any]] = []
        for point in series:
            if not isinstance(point, dict):
                raise RuntimeError(f"IDEAM municipality '{municipality_name}' contains an invalid rain series point.")
            range_block = point.get("forecast_range") or point.get("forecastRange")
            forecast_low = _coalesce(point, "forecastLow", "forecast_low", "forecastMin", "forecast_min")
            forecast_high = _coalesce(point, "forecastHigh", "forecast_high", "forecastMax", "forecast_max")
            if isinstance(range_block, dict):
                forecast_low = forecast_low if forecast_low is not None else _coalesce(range_block, "min", "low")
                forecast_high = forecast_high if forecast_high is not None else _coalesce(range_block, "max", "high")
            forecast_low_value = _as_float(forecast_low)
            forecast_high_value = _as_float(forecast_high)
            forecast_range = _as_float(_coalesce(point, "forecastRange", "forecast_range_value"))
            if forecast_range is None and forecast_low_value is not None and forecast_high_value is not None:
                forecast_range = round(forecast_high_value - forecast_low_value, 3)

            normalized_points.append(
                {
                    "time": _normalize_time_label(
                        _coalesce(point, "time", "label", "timestamp", "datetime", "fechaHora", "fecha_hora")
                    ),
                    "observed": _as_float(
                        _coalesce(
                            point,
                            "observed",
                            "observed_mm",
                            "observation",
                            "observation_mm",
                            "precipitationObserved",
                        )
                    ),
                    "forecast": _as_float(
                        _coalesce(point, "forecast", "forecast_mm", "prediction", "predicted_mm")
                    ),
                    "forecastLow": forecast_low_value,
                    "forecastHigh": forecast_high_value,
                    "forecastRange": forecast_range,
                }
            )
            total_points += 1
        normalized[str(municipality_name)] = normalized_points

    return normalized, {
        "payload_mode": "provider_http_municipalities",
        "municipality_count": len(normalized),
        "point_count": total_points,
    }


def parse_sgc_payload(payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    request_meta = _request_meta(payload)
    historical_events = payload.get("historicalEvents") if isinstance(payload, dict) else None
    if isinstance(historical_events, list):
        return historical_events, {
            "payload_mode": "normalized_http",
            "record_count": len(historical_events),
        }

    if isinstance(payload, dict) and "content" in payload and not isinstance(
        payload.get("content"), list
    ):
        raise RuntimeError("SGC official payload must expose a 'content' list.")

    official_events = payload.get("content") if isinstance(payload, dict) else None
    if isinstance(official_events, list):
        normalized_by_id: dict[str, dict[str, Any]] = {}
        skipped_records = 0
        for item in official_events:
            if not isinstance(item, dict):
                raise RuntimeError("SGC event entry must be an object.")
            event_id = _coalesce(item, "id")
            municipality_block = item.get("municipio")
            municipality = (
                municipality_block.get("nombre")
                if isinstance(municipality_block, dict)
                else municipality_block
            )
            latitude = _coalesce(item, "lgLatitud", "latitud")
            longitude = _coalesce(item, "lgLongitud", "longitud")
            if municipality is None or latitude is None or longitude is None:
                skipped_records += 1
                continue
            movement_type = item.get("tipoMovimiento")
            movement_type_value = (
                movement_type.get("valor")
                if isinstance(movement_type, dict)
                else movement_type if isinstance(movement_type, str) else None
            )
            classification = item.get("clasificacionMovimiento")
            classification_value = (
                classification.get("valor")
                if isinstance(classification, dict)
                else classification if isinstance(classification, str) else None
            )
            severity = "Media"
            erosion_state = item.get("estadoErosion")
            if isinstance(erosion_state, dict):
                erosion_value = str(erosion_state.get("valor") or "").lower()
                if erosion_value.startswith("alta"):
                    severity = "Alta"
                elif erosion_value.startswith("baja"):
                    severity = "Baja"
            if event_id is None:
                continue
            normalized_by_id[str(event_id)] = {
                "id": str(event_id),
                "municipality": str(municipality),
                "date": _normalize_date(
                    _coalesce(item, "drFechaEvento", "fechaEvento", "fechaActualizacion")
                ),
                "severity": severity,
                "type": str(
                    movement_type_value
                    or classification_value
                    or "Movimiento en masa"
                ),
                "coords": [float(latitude), float(longitude)],
                "source": "SGC",
            }
        detail_meta = {
            key: value
            for key, value in request_meta.items()
            if key != "provider"
        }

        normalized = list(normalized_by_id.values())

        return normalized, {
            "payload_mode": "official_simma_api",
            "record_count": len(normalized),
            "total_elements": payload.get("totalElements"),
            "skipped_records": skipped_records,
            **detail_meta,
        }

    events = _find_list(payload, "events", "records", "movements")
    if events is None:
        raise RuntimeError("SGC HTTP payload must contain a 'historicalEvents' list or provider event collection.")

    normalized: list[dict[str, Any]] = []
    for item in events:
        if not isinstance(item, dict):
            raise RuntimeError("SGC event entry must be an object.")
        event_id = _coalesce(item, "id", "event_id", "eventId", "codigo", "code")
        municipality = _coalesce(item, "municipality", "municipio", "municipality_name", "location_name")
        if event_id is None or municipality is None:
            raise RuntimeError("SGC event entry is missing the id or municipality.")
        normalized.append(
            {
                "id": str(event_id),
                "municipality": str(municipality),
                "date": _normalize_date(_coalesce(item, "date", "fecha", "event_date", "occurred_at")),
                "severity": str(_coalesce(item, "severity", "nivel", "threat_level", "impact") or "Media").title(),
                "type": str(_coalesce(item, "type", "movement_type", "fenomeno", "event_type") or "Deslizamiento"),
                "coords": _normalize_coords(item),
                "source": str(_coalesce(item, "source", "fuente") or "SGC"),
            }
        )

    return normalized, {
        "payload_mode": "provider_http_events",
        "record_count": len(normalized),
    }


def parse_ungrd_payload(payload: Any) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    request_meta = _request_meta(payload)
    ungrd_records = payload.get("ungrdRecords") if isinstance(payload, dict) else None
    if isinstance(ungrd_records, dict):
        return ungrd_records, {
            "payload_mode": "normalized_http",
            "municipality_count": len(ungrd_records),
        }

    if (
        isinstance(payload, dict)
        and request_meta.get("provider") == "official_socrata_api"
        and "records" in payload
        and not isinstance(payload.get("records"), list)
    ):
        raise RuntimeError("UNGRD official payload must expose a 'records' list.")

    official_records = None
    if isinstance(payload, list):
        official_records = payload
    elif (
        isinstance(payload, dict)
        and request_meta.get("provider") == "official_socrata_api"
    ):
        official_records = _find_list(payload, "records")

    if official_records is not None:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in official_records:
            if not isinstance(item, dict):
                raise RuntimeError("UNGRD record entry must be an object.")
            municipality = _coalesce(item, "municipio", "municipality")
            record_id = _coalesce(item, ":id", "id", "record_id", "recordId")
            event_name = _coalesce(item, "evento", "event", "type") or "Evento UNGRD"
            if municipality is None or record_id is None:
                raise RuntimeError("UNGRD record entry is missing the municipality or id.")
            people = _coalesce(item, "personas", "familias")
            summary = str(event_name)
            if people not in (None, ""):
                summary = f"{summary}. Personas/familias reportadas: {people}."
            grouped.setdefault(str(municipality), []).append(
                {
                    "id": str(record_id),
                    "municipality": str(municipality),
                    "date": _normalize_date(_coalesce(item, "fecha", "date")),
                    "summary": summary,
                }
            )

        return grouped, {
            "payload_mode": "official_socrata_api",
            "municipality_count": len(grouped),
            "record_count": sum(len(records) for records in grouped.values()),
            **{
                key: value
                for key, value in request_meta.items()
                if key != "provider"
            },
        }

    records = _find_list(payload, "records", "events", "reports")
    if records is None:
        raise RuntimeError("UNGRD HTTP payload must contain an 'ungrdRecords' object or provider records collection.")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        if not isinstance(item, dict):
            raise RuntimeError("UNGRD record entry must be an object.")
        record_id = _coalesce(item, "id", "record_id", "recordId", "codigo", "code")
        municipality = _coalesce(item, "municipality", "municipio", "municipality_name", "location_name")
        summary = _coalesce(item, "summary", "resumen", "description", "detalle")
        if record_id is None or municipality is None or summary is None:
            raise RuntimeError("UNGRD record entry is missing the id, municipality, or summary.")
        municipality_name = str(municipality)
        grouped.setdefault(municipality_name, []).append(
            {
                "id": str(record_id),
                "municipality": municipality_name,
                "date": _normalize_date(_coalesce(item, "date", "fecha", "reported_at", "created_at")),
                "summary": str(summary),
            }
        )

    return grouped, {
        "payload_mode": "provider_http_records",
        "municipality_count": len(grouped),
        "record_count": sum(len(records) for records in grouped.values()),
    }
