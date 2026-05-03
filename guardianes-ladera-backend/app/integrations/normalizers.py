from __future__ import annotations

from datetime import date

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.text import normalize_lookup_key
from app.db.spatial import point_geometry_value, session_dialect_name
from app.models import HistoricalEvent, Municipality, MunicipalityRainPoint, UngrdRecord


def municipalities_by_name(session: Session) -> dict[str, Municipality]:
    return {
        normalize_lookup_key(municipality.name): municipality
        for municipality in session.scalars(select(Municipality)).all()
    }


def resolve_municipality(
    municipalities: dict[str, Municipality], municipality_name: str
) -> Municipality:
    municipality = municipalities.get(normalize_lookup_key(municipality_name))
    if municipality is None:
        raise KeyError(f"Municipality '{municipality_name}' is not configured.")
    return municipality


def sync_rain_series(session: Session, rain_series: dict[str, list[dict]]) -> int:
    municipalities = municipalities_by_name(session)
    total = 0
    for municipality_name, points in rain_series.items():
        municipality = resolve_municipality(municipalities, municipality_name)
        session.execute(
            delete(MunicipalityRainPoint).where(MunicipalityRainPoint.municipality_id == municipality.id)
        )
        for index, point in enumerate(points):
            session.add(
                MunicipalityRainPoint(
                    municipality=municipality,
                    time_label=point["time"],
                    observed=point.get("observed"),
                    forecast=point.get("forecast"),
                    forecast_low=point.get("forecastLow"),
                    forecast_high=point.get("forecastHigh"),
                    forecast_range=point.get("forecastRange"),
                    sort_order=index,
                )
            )
            total += 1
    return total


def sync_historical_events(session: Session, events: list[dict]) -> int:
    municipalities = municipalities_by_name(session)
    dialect_name = session_dialect_name(session)
    total = 0
    for item in events:
        municipality = resolve_municipality(municipalities, item["municipality"])
        existing = session.get(HistoricalEvent, item["id"])
        if existing is None:
            session.add(
                HistoricalEvent(
                    id=item["id"],
                    municipality=municipality,
                    date=date.fromisoformat(item["date"]),
                    severity=item["severity"],
                    type=item["type"],
                    coords=item["coords"],
                    coords_geom=point_geometry_value(item["coords"], dialect_name),
                    source=item["source"],
                )
            )
        else:
            existing.municipality = municipality
            existing.date = date.fromisoformat(item["date"])
            existing.severity = item["severity"]
            existing.type = item["type"]
            existing.coords = item["coords"]
            existing.coords_geom = point_geometry_value(item["coords"], dialect_name)
            existing.source = item["source"]
        total += 1
    return total


def sync_ungrd_records(session: Session, ungrd_records: dict[str, list[dict]]) -> int:
    municipalities = municipalities_by_name(session)
    total = 0
    for municipality_name, records in ungrd_records.items():
        municipality = resolve_municipality(municipalities, municipality_name)
        for record in records:
            existing = session.get(UngrdRecord, record["id"])
            if existing is None:
                session.add(
                    UngrdRecord(
                        id=record["id"],
                        municipality=municipality,
                        date=date.fromisoformat(record["date"]),
                        summary=record["summary"],
                    )
                )
            else:
                existing.municipality = municipality
                existing.date = date.fromisoformat(record["date"])
                existing.summary = record["summary"]
            total += 1
    return total
