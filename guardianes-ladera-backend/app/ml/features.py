from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.models import Zone
from app.repositories.dashboard import DashboardRepository


RAIN_INTENSITY_SCORES = {
    "baja": 1,
    "media": 2,
    "alta": 3,
}

SCORING_FEATURE_ORDER = [
    "rain_72h",
    "rain_24h",
    "slope_deg",
    "municipality_event_count",
    "zone_event_count",
    "recent_zone_event_count",
    "intersecting_road_count",
    "intersecting_road_length_km",
    "rain_overlay_count",
    "rain_overlay_peak_intensity",
    "deforestation_proxy",
]


@dataclass(frozen=True)
class ZoneFeatureSnapshot:
    municipality_event_count: int
    zone_event_count: int
    recent_zone_event_count: int
    intersecting_road_count: int
    intersecting_road_length_km: float
    rain_overlay_count: int
    rain_overlay_peak_intensity: int
    rain_overlay_peak_label: str | None

    def as_dict(self) -> dict:
        return asdict(self)


class ZoneFeatureBuilder:
    def __init__(self, session: Session) -> None:
        self.repository = DashboardRepository(session)
        self._municipality_event_count_cache: dict[str, int] = {}

    def _municipality_event_count(self, municipality_name: str) -> int:
        cache_key = municipality_name.lower()
        if cache_key not in self._municipality_event_count_cache:
            self._municipality_event_count_cache[cache_key] = len(
                self.repository.list_historical_events(municipality=municipality_name)
            )
        return self._municipality_event_count_cache[cache_key]

    def build_for_zone(self, zone: Zone, as_of: datetime | None = None) -> ZoneFeatureSnapshot:
        as_of = as_of or datetime.now(timezone.utc)
        lookback_date = (as_of - timedelta(days=365 * 3)).date()

        municipality_event_count = self._municipality_event_count(zone.municipality.name)
        zone_events = self.repository.list_historical_events_for_zone(zone.id)
        intersecting_road_segments = self.repository.list_spatial_road_segments_for_zone(zone.id)
        rain_overlays = self.repository.list_rain_overlays_for_zone(zone.id)

        rain_overlay_peak_label = None
        rain_overlay_peak_intensity = 0
        if rain_overlays:
            rain_overlay_peak_label = max(
                (overlay.intensity for overlay in rain_overlays),
                key=lambda item: RAIN_INTENSITY_SCORES.get(item.lower(), 0),
            )
            rain_overlay_peak_intensity = RAIN_INTENSITY_SCORES.get(rain_overlay_peak_label.lower(), 0)

        return ZoneFeatureSnapshot(
            municipality_event_count=municipality_event_count,
            zone_event_count=len(zone_events),
            recent_zone_event_count=sum(1 for event in zone_events if event.date >= lookback_date),
            intersecting_road_count=len(intersecting_road_segments),
            intersecting_road_length_km=round(sum(segment.length_km for segment in intersecting_road_segments), 2),
            rain_overlay_count=len(rain_overlays),
            rain_overlay_peak_intensity=rain_overlay_peak_intensity,
            rain_overlay_peak_label=rain_overlay_peak_label,
        )


def build_scoring_feature_vector(drivers: dict, feature_snapshot: ZoneFeatureSnapshot) -> dict[str, float]:
    return {
        "rain_72h": float(drivers["rain_72h"]),
        "rain_24h": float(drivers["rain_24h"]),
        "slope_deg": float(drivers["slope_deg"]),
        "municipality_event_count": float(feature_snapshot.municipality_event_count),
        "zone_event_count": float(feature_snapshot.zone_event_count),
        "recent_zone_event_count": float(feature_snapshot.recent_zone_event_count),
        "intersecting_road_count": float(feature_snapshot.intersecting_road_count),
        "intersecting_road_length_km": float(feature_snapshot.intersecting_road_length_km),
        "rain_overlay_count": float(feature_snapshot.rain_overlay_count),
        "rain_overlay_peak_intensity": float(feature_snapshot.rain_overlay_peak_intensity),
        "deforestation_proxy": float(drivers.get("deforestation_proxy") or 0.0),
    }
