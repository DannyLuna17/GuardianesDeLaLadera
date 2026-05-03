from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload, selectinload

from app.db.spatial import session_dialect_name
from app.db.spatial_filters import (
    BoundingBox,
    linestring_intersects_bbox,
    linestring_intersects_polygon,
    overlay_bounds_intersect_bbox,
    overlay_bounds_intersect_polygon,
    point_within_bbox,
    point_within_polygon,
    polygon_intersects_bbox,
)
from app.models import (
    HistoricalEvent,
    Municipality,
    MunicipalityRainPoint,
    PredictionRun,
    RainOverlay,
    RoadSegment,
    SourceCatalog,
    UngrdRecord,
    Zone,
    ZonePrediction,
)


class DashboardRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    @property
    def dialect_name(self) -> str:
        return session_dialect_name(self.session)

    def list_municipalities(self) -> list[Municipality]:
        statement = select(Municipality).order_by(Municipality.name)
        return list(self.session.scalars(statement).all())

    def list_zones(self) -> list[Zone]:
        statement = (
            select(Zone)
            .options(
                joinedload(Zone.municipality),
                selectinload(Zone.road_segments),
            )
            .order_by(Zone.name)
        )
        return list(self.session.scalars(statement).all())

    def get_zone(self, zone_id: str) -> Zone | None:
        statement = (
            select(Zone)
            .where(Zone.id == zone_id)
            .options(
                joinedload(Zone.municipality),
                selectinload(Zone.road_segments),
            )
        )
        return self.session.scalar(statement)

    def list_road_segments(
        self,
        municipality: str | None = None,
        bounds: BoundingBox | None = None,
    ) -> list[RoadSegment]:
        statement = select(RoadSegment).options(joinedload(RoadSegment.municipality))
        if municipality:
            statement = statement.join(RoadSegment.municipality).where(func.lower(Municipality.name) == municipality.lower())
        if bounds and self.dialect_name == "postgresql":
            statement = statement.where(func.ST_Intersects(RoadSegment.coords_geom, bounds.to_postgis_envelope()))
        statement = statement.order_by(RoadSegment.name)
        segments = list(self.session.scalars(statement).unique().all())
        if bounds and self.dialect_name != "postgresql":
            segments = [segment for segment in segments if linestring_intersects_bbox(segment.coords, bounds)]
        return segments

    def list_historical_events(
        self,
        municipality: str | None = None,
        bounds: BoundingBox | None = None,
    ) -> list[HistoricalEvent]:
        statement = select(HistoricalEvent).options(joinedload(HistoricalEvent.municipality))
        if municipality:
            statement = statement.join(HistoricalEvent.municipality).where(
                func.lower(Municipality.name) == municipality.lower()
            )
        if bounds and self.dialect_name == "postgresql":
            statement = statement.where(func.ST_Intersects(HistoricalEvent.coords_geom, bounds.to_postgis_envelope()))
        statement = statement.order_by(HistoricalEvent.date.desc())
        events = list(self.session.scalars(statement).unique().all())
        if bounds and self.dialect_name != "postgresql":
            events = [event for event in events if point_within_bbox(event.coords, bounds)]
        return events

    def list_historical_events_for_zone(self, zone_id: str) -> list[HistoricalEvent]:
        if self.dialect_name == "postgresql":
            statement = (
                select(HistoricalEvent)
                .join(Zone, Zone.id == zone_id)
                .options(joinedload(HistoricalEvent.municipality))
                .where(
                    HistoricalEvent.municipality_id == Zone.municipality_id,
                    func.ST_Covers(Zone.polygon_geom, HistoricalEvent.coords_geom),
                )
                .order_by(HistoricalEvent.date.desc())
            )
            return list(self.session.scalars(statement).unique().all())

        zone = self.get_zone(zone_id)
        if zone is None:
            return []
        statement = (
            select(HistoricalEvent)
            .where(HistoricalEvent.municipality_id == zone.municipality_id)
            .options(joinedload(HistoricalEvent.municipality))
            .order_by(HistoricalEvent.date.desc())
        )
        events = list(self.session.scalars(statement).all())
        return [event for event in events if point_within_polygon(event.coords, zone.polygon)]

    def list_ungrd_records(self) -> list[UngrdRecord]:
        statement = select(UngrdRecord).options(joinedload(UngrdRecord.municipality)).order_by(UngrdRecord.date.desc())
        return list(self.session.scalars(statement).all())

    def list_sources(self) -> list[SourceCatalog]:
        statement = select(SourceCatalog).options(joinedload(SourceCatalog.sync_status)).order_by(SourceCatalog.id)
        return list(self.session.scalars(statement).all())

    def list_rain_points(self) -> list[MunicipalityRainPoint]:
        statement = (
            select(MunicipalityRainPoint)
            .options(joinedload(MunicipalityRainPoint.municipality))
            .order_by(MunicipalityRainPoint.municipality_id, MunicipalityRainPoint.sort_order)
        )
        return list(self.session.scalars(statement).all())

    def list_rain_overlays(
        self,
        municipality: str | None = None,
        bounds: BoundingBox | None = None,
    ) -> list[RainOverlay]:
        statement = select(RainOverlay).options(joinedload(RainOverlay.municipality))
        if municipality:
            statement = statement.join(RainOverlay.municipality).where(func.lower(Municipality.name) == municipality.lower())
        if bounds and self.dialect_name == "postgresql":
            statement = statement.where(func.ST_Intersects(RainOverlay.bounds_geom, bounds.to_postgis_envelope()))
        statement = statement.order_by(RainOverlay.id)
        overlays = list(self.session.scalars(statement).unique().all())
        if bounds and self.dialect_name != "postgresql":
            overlays = [overlay for overlay in overlays if overlay_bounds_intersect_bbox(overlay.bounds, bounds)]
        return overlays

    def list_spatial_road_segments_for_zone(self, zone_id: str) -> list[RoadSegment]:
        if self.dialect_name == "postgresql":
            statement = (
                select(RoadSegment)
                .join(Zone, Zone.id == zone_id)
                .options(joinedload(RoadSegment.municipality))
                .where(
                    RoadSegment.municipality_id == Zone.municipality_id,
                    func.ST_Intersects(RoadSegment.coords_geom, Zone.polygon_geom),
                )
                .order_by(RoadSegment.name)
            )
            return list(self.session.scalars(statement).unique().all())

        zone = self.get_zone(zone_id)
        if zone is None:
            return []
        statement = (
            select(RoadSegment)
            .where(RoadSegment.municipality_id == zone.municipality_id)
            .options(joinedload(RoadSegment.municipality))
            .order_by(RoadSegment.name)
        )
        segments = list(self.session.scalars(statement).all())
        return [segment for segment in segments if linestring_intersects_polygon(segment.coords, zone.polygon)]

    def list_rain_overlays_for_zone(self, zone_id: str) -> list[RainOverlay]:
        if self.dialect_name == "postgresql":
            statement = (
                select(RainOverlay)
                .join(Zone, Zone.id == zone_id)
                .options(joinedload(RainOverlay.municipality))
                .where(
                    RainOverlay.municipality_id == Zone.municipality_id,
                    func.ST_Intersects(RainOverlay.bounds_geom, Zone.polygon_geom),
                )
                .order_by(RainOverlay.id)
            )
            return list(self.session.scalars(statement).unique().all())

        zone = self.get_zone(zone_id)
        if zone is None:
            return []
        statement = (
            select(RainOverlay)
            .where(RainOverlay.municipality_id == zone.municipality_id)
            .options(joinedload(RainOverlay.municipality))
            .order_by(RainOverlay.id)
        )
        overlays = list(self.session.scalars(statement).all())
        return [overlay for overlay in overlays if overlay_bounds_intersect_polygon(overlay.bounds, zone.polygon)]

    def get_latest_run_id(self) -> int | None:
        statement = select(PredictionRun.id).order_by(PredictionRun.completed_at.desc()).limit(1)
        return self.session.scalar(statement)

    def list_latest_zone_predictions(
        self,
        municipality: str | None = None,
        zone_type: str | None = None,
        min_risk_score: float | None = None,
        bounds: BoundingBox | None = None,
    ) -> list[ZonePrediction]:
        latest_run_id = self.get_latest_run_id()
        if latest_run_id is None:
            return []

        statement = (
            select(ZonePrediction)
            .where(ZonePrediction.run_id == latest_run_id)
            .join(ZonePrediction.zone)
            .options(
                joinedload(ZonePrediction.zone).joinedload(Zone.municipality),
                joinedload(ZonePrediction.zone).selectinload(Zone.road_segments),
                joinedload(ZonePrediction.explanation),
            )
        )
        if municipality:
            statement = statement.join(Zone.municipality).where(func.lower(Municipality.name) == municipality.lower())
        if zone_type:
            statement = statement.where(func.lower(Zone.type) == zone_type.lower())
        if min_risk_score is not None:
            statement = statement.where(ZonePrediction.risk_score >= min_risk_score)
        if bounds and self.dialect_name == "postgresql":
            statement = statement.where(func.ST_Intersects(Zone.polygon_geom, bounds.to_postgis_envelope()))
        statement = statement.order_by(ZonePrediction.risk_score.desc())
        predictions = list(self.session.scalars(statement).unique().all())
        if bounds and self.dialect_name != "postgresql":
            predictions = [
                prediction
                for prediction in predictions
                if polygon_intersects_bbox(prediction.zone.polygon, bounds)
            ]
        return predictions

    def get_latest_run(self) -> PredictionRun | None:
        statement = (
            select(PredictionRun)
            .options(
                selectinload(PredictionRun.predictions)
                .joinedload(ZonePrediction.zone)
                .joinedload(Zone.municipality),
                selectinload(PredictionRun.predictions)
                .joinedload(ZonePrediction.zone)
                .selectinload(Zone.road_segments),
                selectinload(PredictionRun.predictions).selectinload(ZonePrediction.explanation),
            )
            .order_by(PredictionRun.completed_at.desc())
            .limit(1)
        )
        return self.session.scalar(statement)

    def get_run(self, run_id: int) -> PredictionRun | None:
        statement = (
            select(PredictionRun)
            .where(PredictionRun.id == run_id)
            .options(
                selectinload(PredictionRun.predictions)
                .joinedload(ZonePrediction.zone)
                .joinedload(Zone.municipality),
                selectinload(PredictionRun.predictions)
                .joinedload(ZonePrediction.zone)
                .selectinload(Zone.road_segments),
                selectinload(PredictionRun.predictions).selectinload(ZonePrediction.explanation),
            )
        )
        return self.session.scalar(statement)

    def get_prediction_for_zone(self, zone_id: str, run_id: int) -> ZonePrediction | None:
        statement = (
            select(ZonePrediction)
            .where(ZonePrediction.zone_id == zone_id, ZonePrediction.run_id == run_id)
            .options(
                joinedload(ZonePrediction.zone).joinedload(Zone.municipality),
                joinedload(ZonePrediction.zone).selectinload(Zone.road_segments),
                joinedload(ZonePrediction.explanation),
            )
        )
        return self.session.scalar(statement)
