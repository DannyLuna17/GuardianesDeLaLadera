from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.db.bootstrap import clamp
from app.ml.features import ZoneFeatureBuilder, ZoneFeatureSnapshot
from app.models import MunicipalityRainPoint, UngrdRecord, Zone
from app.repositories.dashboard import DashboardRepository


OPERATIONAL_MODEL_ID = "operational-real-data-heuristic"
OPERATIONAL_MODEL_VERSION = "operational-real-data-v1"


@dataclass(frozen=True)
class OperationalScoringResult:
    score: float
    confidence: str
    drivers: dict
    trace: dict


class OperationalRiskScoringService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.repository = DashboardRepository(session)
        self.feature_builder = ZoneFeatureBuilder(session)
        self._rain_cache: dict[str, list[MunicipalityRainPoint]] = {}
        self._ungrd_cache: dict[str, list[UngrdRecord]] = {}

    @staticmethod
    def _normalize_dt(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @staticmethod
    def _parse_time_label(label: str) -> datetime | None:
        text = str(label).strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    def _rain_points_for_municipality(self, municipality_name: str) -> list[MunicipalityRainPoint]:
        cache_key = municipality_name.lower()
        if cache_key not in self._rain_cache:
            self._rain_cache[cache_key] = [
                point
                for point in self.repository.list_rain_points()
                if point.municipality.name.lower() == municipality_name.lower()
            ]
        return self._rain_cache[cache_key]

    def _ungrd_records_for_municipality(self, municipality_name: str) -> list[UngrdRecord]:
        cache_key = municipality_name.lower()
        if cache_key not in self._ungrd_cache:
            self._ungrd_cache[cache_key] = [
                record
                for record in self.repository.list_ungrd_records()
                if record.municipality.name.lower() == municipality_name.lower()
            ]
        return self._ungrd_cache[cache_key]

    def _rain_totals(self, municipality_name: str) -> tuple[int, int, datetime | None, int]:
        observed_points: list[tuple[datetime, float]] = []
        for point in self._rain_points_for_municipality(municipality_name):
            point_time = self._parse_time_label(point.time_label)
            if point_time is None:
                continue
            value = point.observed if point.observed is not None else point.forecast
            if value is None:
                continue
            observed_points.append((point_time, float(value)))

        observed_points.sort(key=lambda item: item[0], reverse=True)
        if not observed_points:
            return 0, 0, None, 0

        latest_points = observed_points[:3]
        rain_24h = round(latest_points[0][1])
        rain_72h = round(sum(value for _, value in latest_points[:3]))
        return rain_24h, rain_72h, latest_points[0][0], len(observed_points)

    def _recent_ungrd_count(self, municipality_name: str, as_of: datetime) -> int:
        lookback_date = (as_of - timedelta(days=365)).date()
        return sum(
            1
            for record in self._ungrd_records_for_municipality(municipality_name)
            if record.date >= lookback_date
        )

    @staticmethod
    def _confidence_from_sources(
        snapshot: dict[str, str],
        *,
        rain_points: int,
        municipality_events: int,
        ungrd_records: int,
    ) -> str:
        score = 0
        ideam_status = snapshot.get("IDEAM")
        if rain_points > 0:
            if ideam_status == "Fresco":
                score += 2
            elif ideam_status == "Retrasado":
                score += 1

        for source_id, availability in (
            ("SGC", municipality_events),
            ("UNGRD", ungrd_records),
        ):
            if availability <= 0:
                continue
            status = snapshot.get(source_id)
            if status in {"Fresco", "Estatico"}:
                score += 2
            elif status == "Retrasado":
                score += 1

        if score >= 5:
            return "Alta"
        if score >= 3:
            return "Media"
        return "Baja"

    @staticmethod
    def _component(value: float, normalizer: float, weight: float) -> float:
        return min(value / normalizer, 1.0) * weight

    def score_zone(
        self,
        *,
        zone: Zone,
        source_snapshot: dict[str, str],
        as_of: datetime | None = None,
    ) -> OperationalScoringResult:
        as_of = self._normalize_dt(as_of) or datetime.now(timezone.utc)
        feature_snapshot = self.feature_builder.build_for_zone(zone, as_of=as_of)
        rain_24h, rain_72h, rain_as_of, rain_point_count = self._rain_totals(
            zone.municipality.name
        )
        ungrd_recent_count = self._recent_ungrd_count(zone.municipality.name, as_of)

        components = {
            "rain_72h": self._component(float(rain_72h), 150.0, 0.24),
            "rain_24h": self._component(float(rain_24h), 80.0, 0.16),
            "zone_event_count": self._component(
                float(feature_snapshot.zone_event_count), 4.0, 0.12
            ),
            "recent_zone_event_count": self._component(
                float(feature_snapshot.recent_zone_event_count), 2.0, 0.12
            ),
            "municipality_event_count": self._component(
                float(feature_snapshot.municipality_event_count), 25.0, 0.06
            ),
            "intersecting_road_length_km": self._component(
                float(feature_snapshot.intersecting_road_length_km), 8.0, 0.06
            ),
            "intersecting_road_count": self._component(
                float(feature_snapshot.intersecting_road_count), 3.0, 0.04
            ),
            "recent_ungrd_records": self._component(
                float(ungrd_recent_count), 2.0, 0.04
            ),
            "population_exposure": self._component(
                float(zone.exposure.get("population_estimate") or 0), 4000.0, 0.02
            ),
            "household_exposure": self._component(
                float(zone.exposure.get("households_estimate") or 0), 1000.0, 0.02
            ),
        }

        raw_score = 0.04 + sum(components.values())
        bounded_score = round(clamp(raw_score, 0.04, 0.92), 3)
        confidence = self._confidence_from_sources(
            source_snapshot,
            rain_points=rain_point_count,
            municipality_events=feature_snapshot.municipality_event_count,
            ungrd_records=len(self._ungrd_records_for_municipality(zone.municipality.name)),
        )

        drivers = {
            "rain_6h": None,
            "rain_24h": rain_24h,
            "rain_72h": rain_72h,
            "slope_deg": None,
            "geology_class": None,
            "soil_class": None,
            "deforestation_proxy": None,
        }
        trace = {
            "model_id": OPERATIONAL_MODEL_ID,
            "model_version": OPERATIONAL_MODEL_VERSION,
            "artifact_type": "operational_heuristic",
            "generation_mode": "real_data_only",
            "feature_snapshot": feature_snapshot.as_dict(),
            "component_scores": {key: round(value, 3) for key, value in components.items()},
            "raw_model_score": round(raw_score, 6),
            "uses_spatial_features": True,
            "rain_point_count": rain_point_count,
            "rain_data_as_of": rain_as_of.isoformat() if rain_as_of else None,
            "recent_ungrd_records": ungrd_recent_count,
            "source_snapshot": source_snapshot,
            "missing_susceptibility_baselines": [
                "slope_deg",
                "geology_class",
                "soil_class",
                "deforestation_proxy",
            ],
        }
        return OperationalScoringResult(
            score=bounded_score,
            confidence=confidence,
            drivers=drivers,
            trace=trace,
        )
