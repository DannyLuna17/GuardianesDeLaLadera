from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.v1.dependencies.spatial import get_bounding_box
from app.db.session import get_db
from app.db.spatial_filters import BoundingBox
from app.schemas.dashboard import (
    DashboardBootstrapRead,
    DashboardSummaryRead,
    HistoricalEventRead,
    MunicipalityRead,
    RainOverlayRead,
    RainPointRead,
    RoadSegmentRead,
    RunDetailRead,
    RunSummaryRead,
    SourceCatalogRead,
    SourceStatusRead,
    UngrdRecordRead,
    ZoneExplanationRead,
    ZoneRead,
    ZoneSpatialSummaryRead,
)
from app.services.dashboard import DashboardService

router = APIRouter(prefix="/v1", tags=["dashboard"])


def get_dashboard_service(session: Session = Depends(get_db)) -> DashboardService:
    return DashboardService(session)


@router.get("/municipalities", response_model=list[MunicipalityRead])
def list_municipalities(service: DashboardService = Depends(get_dashboard_service)):
    return service.list_municipalities()


@router.get("/zones", response_model=list[ZoneRead])
def list_zones(
    municipality: str | None = None,
    zone_type: str | None = Query(default=None, alias="zoneType"),
    min_risk_level: str | None = Query(default=None, alias="minRiskLevel"),
    bounds: BoundingBox | None = Depends(get_bounding_box),
    service: DashboardService = Depends(get_dashboard_service),
):
    return service.list_zones(
        municipality=municipality,
        zone_type=zone_type,
        min_risk_level=min_risk_level,
        bounds=bounds,
    )


@router.get("/zones/{zone_id}", response_model=ZoneRead)
def get_zone(zone_id: str, service: DashboardService = Depends(get_dashboard_service)):
    return service.get_zone(zone_id)


@router.get("/zones/{zone_id}/rain-series", response_model=list[RainPointRead])
def get_zone_rain_series(zone_id: str, service: DashboardService = Depends(get_dashboard_service)):
    return service.rain_series_for_zone(zone_id)


@router.get("/zones/{zone_id}/roads", response_model=list[RoadSegmentRead])
def get_zone_roads(zone_id: str, service: DashboardService = Depends(get_dashboard_service)):
    return service.list_road_segments_for_zone(zone_id)


@router.get("/zones/{zone_id}/events", response_model=list[HistoricalEventRead])
def get_zone_events(zone_id: str, service: DashboardService = Depends(get_dashboard_service)):
    return service.list_zone_events(zone_id)


@router.get("/zones/{zone_id}/explanation", response_model=ZoneExplanationRead)
def get_zone_explanation(zone_id: str, service: DashboardService = Depends(get_dashboard_service)):
    return service.get_zone_explanation(zone_id)


@router.get("/zones/{zone_id}/spatial-summary", response_model=ZoneSpatialSummaryRead)
def get_zone_spatial_summary(zone_id: str, service: DashboardService = Depends(get_dashboard_service)):
    return service.get_zone_spatial_summary(zone_id)


@router.get("/historical-events", response_model=list[HistoricalEventRead])
def list_historical_events(
    municipality: str | None = None,
    bounds: BoundingBox | None = Depends(get_bounding_box),
    service: DashboardService = Depends(get_dashboard_service),
):
    return service.list_historical_events(municipality=municipality, bounds=bounds)


@router.get("/ungrd-records", response_model=list[UngrdRecordRead])
def list_ungrd_records(
    municipality: str | None = None,
    service: DashboardService = Depends(get_dashboard_service),
):
    return service.list_ungrd_records(municipality=municipality)


@router.get("/road-segments", response_model=list[RoadSegmentRead])
def list_road_segments(
    municipality: str | None = None,
    bounds: BoundingBox | None = Depends(get_bounding_box),
    service: DashboardService = Depends(get_dashboard_service),
):
    return service.list_road_segments(municipality=municipality, bounds=bounds)


@router.get("/source-catalog", response_model=list[SourceCatalogRead])
def list_source_catalog(service: DashboardService = Depends(get_dashboard_service)):
    return service.list_source_catalog()


@router.get("/source-status", response_model=list[SourceStatusRead])
def list_source_status(service: DashboardService = Depends(get_dashboard_service)):
    return service.list_source_status()


@router.get("/rain-overlays", response_model=dict[str, list[RainOverlayRead]])
def list_rain_overlays(
    municipality: str | None = None,
    bounds: BoundingBox | None = Depends(get_bounding_box),
    service: DashboardService = Depends(get_dashboard_service),
):
    return service.rain_overlays_by_municipality(municipality=municipality, bounds=bounds)


@router.get("/runs/latest", response_model=RunSummaryRead)
def get_latest_run(service: DashboardService = Depends(get_dashboard_service)):
    return service.get_latest_run_summary()


@router.get("/runs/{run_id}", response_model=RunDetailRead)
def get_run(run_id: int, service: DashboardService = Depends(get_dashboard_service)):
    return service.get_run_detail(run_id)


@router.get("/dashboard/summary", response_model=DashboardSummaryRead)
def get_dashboard_summary(service: DashboardService = Depends(get_dashboard_service)):
    return service.get_dashboard_summary()


@router.get("/dashboard/bootstrap", response_model=DashboardBootstrapRead)
def get_dashboard_bootstrap(service: DashboardService = Depends(get_dashboard_service)):
    return service.get_dashboard_bootstrap()
