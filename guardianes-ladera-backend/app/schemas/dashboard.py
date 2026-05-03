from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


RiskLevel = Literal["Verde", "Amarillo", "Naranja", "Rojo"]
Confidence = Literal["Baja", "Media", "Alta"]


class SchemaBase(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        protected_namespaces=(),
    )


class MunicipalityRead(SchemaBase):
    id: str
    name: str
    center: list[float]
    zoom: int


class ZoneDriversRead(SchemaBase):
    rain_6h: int | None = None
    rain_24h: int | None = None
    rain_72h: int | None = None
    slope_deg: int | None = None
    geology_class: str | None = None
    soil_class: str | None = None
    deforestation_proxy: float | None = None


class ZoneExposureRead(SchemaBase):
    population_estimate: int | None = None
    households_estimate: int | None = None


class ZoneAssetsRead(SchemaBase):
    road_segment_ids: list[str]


class ZoneRead(SchemaBase):
    id: str
    name: str
    municipality: str
    municipality_id: str = Field(alias="municipalityId")
    type: str
    centroid: list[float]
    polygon: list[list[float]]
    risk_score: float = Field(alias="riskScore")
    risk_level: RiskLevel = Field(alias="riskLevel")
    confidence: Confidence
    drivers: ZoneDriversRead
    exposure: ZoneExposureRead
    assets: ZoneAssetsRead
    last_updated: datetime = Field(alias="lastUpdated")
    risk_delta: float = Field(alias="riskDelta")
    trend: str


class RoadSegmentRead(SchemaBase):
    id: str
    name: str
    municipality: str
    municipality_id: str = Field(alias="municipalityId")
    coords: list[list[float]]
    risk_level: str = Field(alias="riskLevel")
    length_km: float
    note: str


class RainPointRead(SchemaBase):
    time: str
    observed: float | None = None
    forecast: float | None = None
    forecastLow: float | None = None
    forecastHigh: float | None = None
    forecastRange: float | None = None


class HistoricalEventRead(SchemaBase):
    id: str
    municipality: str
    date: date
    severity: str
    type: str
    coords: list[float]
    source: str


class UngrdRecordRead(SchemaBase):
    id: str
    municipality: str
    date: date
    summary: str


class SourceCatalogRead(SchemaBase):
    id: str
    label: str
    category: str


class SourceStatusRead(SchemaBase):
    id: str
    label: str
    category: str
    status: str
    minutes: int | None = None
    updated_at: datetime | None = Field(default=None, alias="updatedAt")
    note: str | None = None


class RainOverlayRead(SchemaBase):
    bounds: list[list[float]]
    intensity: str


class ZoneExplanationRead(SchemaBase):
    zone_id: str = Field(alias="zoneId")
    run_id: int = Field(alias="runId")
    mode: str
    summary: str
    driver_chips: list[str] = Field(alias="driverChips")
    suggestions: list[str]
    data_warnings: list[str] = Field(alias="dataWarnings")
    trace: dict
    generated_at: datetime = Field(alias="generatedAt")


class ZoneSpatialSummaryRead(SchemaBase):
    zone_id: str = Field(alias="zoneId")
    municipality: str
    historical_event_count: int = Field(alias="historicalEventCount")
    historical_event_ids: list[str] = Field(alias="historicalEventIds")
    severity_breakdown: dict[str, int] = Field(alias="severityBreakdown")
    intersecting_road_segment_count: int = Field(alias="intersectingRoadSegmentCount")
    intersecting_road_segment_ids: list[str] = Field(alias="intersectingRoadSegmentIds")
    intersecting_road_length_km: float = Field(alias="intersectingRoadLengthKm")
    rain_overlay_count: int = Field(alias="rainOverlayCount")
    rain_overlay_intensities: list[str] = Field(alias="rainOverlayIntensities")


class RunSummaryRead(SchemaBase):
    id: int
    status: str
    model_version: str = Field(alias="modelVersion")
    partial_data: bool = Field(alias="partialData")
    started_at: datetime = Field(alias="startedAt")
    completed_at: datetime = Field(alias="completedAt")
    zones_monitored: int = Field(alias="zonesMonitored")
    high_risk_count: int = Field(alias="highRiskCount")
    freshness_percent: int = Field(alias="freshnessPercent")
    active_sources_count: int = Field(alias="activeSourcesCount")
    total_sources_count: int = Field(alias="totalSourcesCount")


class RunDetailRead(RunSummaryRead):
    zones: list[ZoneRead]


class DataProvenanceItemRead(SchemaBase):
    key: str
    label: str
    state: str
    summary: str
    detail: str | None = None


class DataProvenanceRead(SchemaBase):
    real_data_only: bool = Field(alias="realDataOnly")
    mock_data_present: bool = Field(alias="mockDataPresent")
    items: list[DataProvenanceItemRead]


class DashboardKpiRead(SchemaBase):
    areas_monitored: int = Field(alias="areasMonitored")
    high_risk_count: int = Field(alias="highRiskCount")
    freshness_percent: int = Field(alias="freshnessPercent")
    partial_data: bool = Field(alias="partialData")


class DashboardSummaryRead(SchemaBase):
    latest_run: RunSummaryRead = Field(alias="latestRun")
    municipalities: list[MunicipalityRead]
    top_risk_zones: list[ZoneRead] = Field(alias="topRiskZones")
    source_status: list[SourceStatusRead] = Field(alias="sourceStatus")
    kpis: DashboardKpiRead


class DashboardBootstrapRead(SchemaBase):
    municipalities: list[MunicipalityRead]
    zones: list[ZoneRead]
    road_segments: list[RoadSegmentRead] = Field(alias="roadSegments")
    rain_series: dict[str, list[RainPointRead]] = Field(alias="rainSeries")
    historical_events: list[HistoricalEventRead] = Field(alias="historicalEvents")
    ungrd_records: dict[str, list[UngrdRecordRead]] = Field(alias="ungrdRecords")
    source_catalog: list[SourceCatalogRead] = Field(alias="sourceCatalog")
    source_status: list[SourceStatusRead] = Field(alias="sourceStatus")
    rain_overlays: dict[str, list[RainOverlayRead]] = Field(alias="rainOverlays")
    risk_colors: dict[str, str] = Field(alias="riskColors")
    risk_order: dict[str, int] = Field(alias="riskOrder")
    latest_run: RunSummaryRead = Field(alias="latestRun")
    data_provenance: DataProvenanceRead = Field(alias="dataProvenance")
