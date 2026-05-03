from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import ApiError
from app.db.bootstrap import risk_level_from_score
from app.db.spatial_filters import BoundingBox
from app.models import PredictionRun, SourceSyncEvent, ZonePrediction
from app.repositories.dashboard import DashboardRepository
from app.schemas.dashboard import (
    DataProvenanceItemRead,
    DataProvenanceRead,
    DashboardBootstrapRead,
    DashboardKpiRead,
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
    ZoneAssetsRead,
    ZoneDriversRead,
    ZoneExplanationRead,
    ZoneExposureRead,
    ZoneRead,
    ZoneSpatialSummaryRead,
)
from app.services.structural_catalog import (
    collect_structural_catalog_violations,
    ensure_real_data_structural_catalog,
)


RISK_COLORS = {
    "Verde": "#2e7d32",
    "Amarillo": "#f9a825",
    "Naranja": "#ef6c00",
    "Rojo": "#c62828",
}
RISK_ORDER = {"Verde": 0, "Amarillo": 1, "Naranja": 2, "Rojo": 3}
RISK_LEVEL_MIN_SCORE = {"Verde": 0.0, "Amarillo": 0.25, "Naranja": 0.5, "Rojo": 0.75}
RAIN_INTENSITY_ORDER = {"baja": 0, "media": 1, "alta": 2}


class DashboardService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.settings = get_settings()
        self.repository = DashboardRepository(session)
        ensure_real_data_structural_catalog(session, for_api=True)

    def list_municipalities(self) -> list[MunicipalityRead]:
        return [MunicipalityRead.model_validate(item) for item in self.repository.list_municipalities()]

    def list_source_catalog(self) -> list[SourceCatalogRead]:
        sources = self.repository.list_sources()
        return [SourceCatalogRead(id=source.id, label=source.label, category=source.category) for source in sources]

    @staticmethod
    def _as_utc(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    def list_source_status(self) -> list[SourceStatusRead]:
        now = datetime.now(timezone.utc)
        payload: list[SourceStatusRead] = []
        for source in self.repository.list_sources():
            status = "Desactualizado"
            minutes: int | None = None
            note = source.sync_status.status_note if source.sync_status else None
            updated_at = self._as_utc(source.sync_status.last_success_at) if source.sync_status else None
            if source.category in {"historico", "infraestructura"}:
                status = "Estatico"
            elif updated_at is not None:
                minutes = int((now - updated_at).total_seconds() // 60)
                if minutes <= 30:
                    status = "Fresco"
                elif minutes <= 180:
                    status = "Retrasado"
                else:
                    status = "Desactualizado"
            payload.append(
                SourceStatusRead(
                    id=source.id,
                    label=source.label,
                    category=source.category,
                    status=status,
                    minutes=minutes,
                    updatedAt=updated_at,
                    note=note,
                )
            )
        return payload

    def _latest_sync_event(self, source_id: str) -> SourceSyncEvent | None:
        statement = (
            select(SourceSyncEvent)
            .where(SourceSyncEvent.source_id == source_id)
            .order_by(SourceSyncEvent.completed_at.desc(), SourceSyncEvent.id.desc())
            .limit(1)
        )
        return self.session.scalar(statement)

    @staticmethod
    def _provenance_item(
        *,
        key: str,
        label: str,
        state: str,
        summary: str,
        detail: str | None = None,
    ) -> DataProvenanceItemRead:
        return DataProvenanceItemRead(
            key=key,
            label=label,
            state=state,
            summary=summary,
            detail=detail,
        )

    def _structural_provenance_item(self) -> DataProvenanceItemRead:
        violations = collect_structural_catalog_violations(self.session)
        if not violations:
            return self._provenance_item(
                key="structural_base",
                label="Base territorial",
                state="real",
                summary="100% real",
                detail=(
                    "Municipios, zonas y corredores viales provienen de un bundle oficial "
                    "con sourceId/sourceRef y trazabilidad de importacion."
                ),
            )
        if self.repository.list_municipalities():
            return self._provenance_item(
                key="structural_base",
                label="Base territorial",
                state="mock",
                summary="Mock/seed",
                detail=violations[0],
            )
        return self._provenance_item(
            key="structural_base",
            label="Base territorial",
            state="unknown",
            summary="Sin verificar",
            detail="No hay un catalogo estructural cargado para clasificar la procedencia.",
        )

    def _source_provenance_item(
        self,
        *,
        source_id: str,
        label: str,
        source_status_by_id: dict[str, SourceStatusRead],
    ) -> DataProvenanceItemRead:
        event = self._latest_sync_event(source_id)
        status = source_status_by_id.get(source_id)
        status_detail = (
            f"Estado visible: {status.status}."
            if status is not None
            else "Sin estado visible en el dashboard."
        )
        note = (status.note or "").strip() if status is not None and status.note else ""

        if event is None:
            if any(token in note.lower() for token in ("semilla", "seed", "synthetic")):
                return self._provenance_item(
                    key=f"source_{source_id.lower()}",
                    label=label,
                    state="mock",
                    summary="Mock/seed",
                    detail=f"{status_detail} {note}".strip(),
                )
            return self._provenance_item(
                key=f"source_{source_id.lower()}",
                label=label,
                state="unknown",
                summary="Sin verificar",
                detail=f"{status_detail} No hay un evento de sincronizacion registrado.",
            )

        transport = (event.transport or "unknown").lower()
        adapter_key = (event.adapter_key or "").lower()
        if transport == "seed" or "seed" in adapter_key:
            return self._provenance_item(
                key=f"source_{source_id.lower()}",
                label=label,
                state="mock",
                summary="Mock/seed",
                detail=(
                    f"{status_detail} Ultima sincronizacion via {event.transport} "
                    f"({event.adapter_key})."
                ),
            )
        if transport in {"http", "file_import"}:
            return self._provenance_item(
                key=f"source_{source_id.lower()}",
                label=label,
                state="real",
                summary="100% real",
                detail=(
                    f"{status_detail} Ultima sincronizacion via {event.transport} "
                    f"({event.adapter_key})."
                ),
            )
        return self._provenance_item(
            key=f"source_{source_id.lower()}",
            label=label,
            state="unknown",
            summary="Sin clasificar",
            detail=(
                f"{status_detail} Ultima sincronizacion via {event.transport} "
                f"({event.adapter_key})."
            ),
        )

    def _analytics_provenance_items(
        self,
        latest_run: PredictionRun,
    ) -> list[DataProvenanceItemRead]:
        model_version = latest_run.model_version or ""
        model_version_lower = model_version.lower()
        run_notes = (latest_run.notes or "").lower()
        if any(token in model_version_lower or token in run_notes for token in ("seed", "semilla", "synthetic")):
            risk_detail = (
                f"El puntaje y la confianza son salidas calculadas, no observaciones directas. "
                f"El modelo activo reporta '{model_version}'."
            )
        elif "operational" in model_version_lower or "real-data" in model_version_lower:
            risk_detail = (
                f"El puntaje y la confianza son salidas calculadas a partir de datos reales. "
                f"El modelo activo reporta '{model_version}'."
            )
        else:
            risk_detail = (
                f"El puntaje y la confianza son salidas calculadas. "
                f"El modelo activo reporta '{model_version}'."
            )

        explanation_modes = sorted(
            {
                prediction.explanation.mode
                for prediction in latest_run.predictions
                if prediction.explanation is not None
            }
        )
        if not explanation_modes:
            explanation_detail = "No hay explicaciones persistidas para clasificar."
        elif explanation_modes == ["template"]:
            explanation_detail = (
                "El texto del panel es generado por plantilla backend a partir de datos y reglas; "
                "no es una observacion cruda."
            )
        else:
            explanation_detail = (
                "El texto del panel es generado por backend a partir de datos y reglas; "
                "no es una observacion cruda."
            )

        return [
            self._provenance_item(
                key="risk_output",
                label="Puntaje y confianza",
                state="derived",
                summary="Derivado",
                detail=risk_detail,
            ),
            self._provenance_item(
                key="zone_explanation",
                label="Resumen y sugerencias",
                state="derived",
                summary="Derivado",
                detail=explanation_detail,
            ),
        ]

    def _data_provenance(
        self,
        *,
        latest_run: PredictionRun,
        source_status: list[SourceStatusRead],
    ) -> DataProvenanceRead:
        source_status_by_id = {item.id: item for item in source_status}
        items = [
            self._structural_provenance_item(),
            self._source_provenance_item(
                source_id="IDEAM",
                label="Lluvia y pronostico IDEAM",
                source_status_by_id=source_status_by_id,
            ),
            self._source_provenance_item(
                source_id="SGC",
                label="Eventos historicos SGC",
                source_status_by_id=source_status_by_id,
            ),
            self._source_provenance_item(
                source_id="UNGRD",
                label="Registros UNGRD",
                source_status_by_id=source_status_by_id,
            ),
            *self._analytics_provenance_items(latest_run),
        ]
        return DataProvenanceRead(
            realDataOnly=self.settings.real_data_only,
            mockDataPresent=any(item.state == "mock" for item in items),
            items=items,
        )

    def _zone_read_from_prediction(self, prediction: ZonePrediction) -> ZoneRead:
        zone = prediction.zone
        return ZoneRead(
            id=zone.id,
            name=zone.name,
            municipality=zone.municipality.name,
            municipalityId=zone.municipality.id,
            type=zone.type,
            centroid=zone.centroid,
            polygon=zone.polygon,
            riskScore=round(prediction.risk_score, 3),
            riskLevel=risk_level_from_score(prediction.risk_score),
            confidence=prediction.confidence,
            drivers=ZoneDriversRead.model_validate(prediction.drivers),
            exposure=ZoneExposureRead.model_validate(zone.exposure),
            assets=ZoneAssetsRead(road_segment_ids=[segment.id for segment in zone.road_segments]),
            lastUpdated=prediction.created_at,
            riskDelta=round(prediction.risk_delta, 3),
            trend=prediction.trend,
        )

    def _run_summary(self, run: PredictionRun, zones: list[ZoneRead], source_status: list[SourceStatusRead]) -> RunSummaryRead:
        high_risk_count = sum(1 for zone in zones if RISK_ORDER[zone.risk_level] >= RISK_ORDER["Naranja"])
        active_sources_count = sum(1 for source in source_status if source.status in {"Fresco", "Retrasado", "Estatico"})
        total_sources_count = len(source_status)
        freshness_weights = {"Fresco": 100, "Retrasado": 65, "Desactualizado": 25, "Estatico": 100}
        freshness_percent = round(
            sum(freshness_weights[source.status] for source in source_status) / max(total_sources_count, 1)
        )
        return RunSummaryRead(
            id=run.id,
            status=run.status,
            modelVersion=run.model_version,
            partialData=run.partial_data,
            startedAt=run.started_at,
            completedAt=run.completed_at,
            zonesMonitored=len(zones),
            highRiskCount=high_risk_count,
            freshnessPercent=freshness_percent,
            activeSourcesCount=active_sources_count,
            totalSourcesCount=total_sources_count,
        )

    def _ensure_runtime_run_allowed(self, run: PredictionRun) -> PredictionRun:
        if not self.settings.real_data_only:
            return run
        note = (run.notes or "").lower()
        if any(token in note for token in ("semilla", "seed", "synthetic")):
            raise ApiError(
                409,
                "legacy_prediction_run_blocked",
                "The persisted prediction run was generated from legacy seed or synthetic data and is unavailable while REAL_DATA_ONLY is enabled.",
            )
        return run

    def _latest_run_or_error(self) -> PredictionRun:
        latest_run = self.repository.get_latest_run()
        if latest_run is None:
            raise ApiError(404, "run_not_found", "No prediction run is available yet.")
        return self._ensure_runtime_run_allowed(latest_run)

    def list_zones(
        self,
        municipality: str | None = None,
        zone_type: str | None = None,
        min_risk_level: str | None = None,
        bounds: BoundingBox | None = None,
    ) -> list[ZoneRead]:
        min_risk_score: float | None = None
        if min_risk_level:
            if min_risk_level not in RISK_ORDER:
                raise ApiError(400, "invalid_risk_level", f"Unsupported risk level: {min_risk_level}")
            min_risk_score = RISK_LEVEL_MIN_SCORE[min_risk_level]
        self._latest_run_or_error()
        predictions = self.repository.list_latest_zone_predictions(
            municipality=municipality,
            zone_type=zone_type,
            min_risk_score=min_risk_score,
            bounds=bounds,
        )
        return [self._zone_read_from_prediction(prediction) for prediction in predictions]

    def get_zone(self, zone_id: str) -> ZoneRead:
        latest_run = self._latest_run_or_error()
        prediction = self.repository.get_prediction_for_zone(zone_id, latest_run.id)
        if prediction is None:
            raise ApiError(404, "zone_not_found", f"Zone '{zone_id}' was not found.")
        return self._zone_read_from_prediction(prediction)

    def list_road_segments(
        self,
        municipality: str | None = None,
        bounds: BoundingBox | None = None,
    ) -> list[RoadSegmentRead]:
        return [
            RoadSegmentRead(
                id=segment.id,
                name=segment.name,
                municipality=segment.municipality.name,
                municipalityId=segment.municipality.id,
                coords=segment.coords,
                riskLevel=segment.risk_level,
                length_km=segment.length_km,
                note=segment.note,
            )
            for segment in self.repository.list_road_segments(municipality=municipality, bounds=bounds)
        ]

    def list_road_segments_for_zone(self, zone_id: str) -> list[RoadSegmentRead]:
        zone = self.repository.get_zone(zone_id)
        if zone is None:
            raise ApiError(404, "zone_not_found", f"Zone '{zone_id}' was not found.")
        return [
            RoadSegmentRead(
                id=segment.id,
                name=segment.name,
                municipality=segment.municipality.name,
                municipalityId=segment.municipality.id,
                coords=segment.coords,
                riskLevel=segment.risk_level,
                length_km=segment.length_km,
                note=segment.note,
            )
            for segment in zone.road_segments
        ]

    def list_historical_events(
        self,
        municipality: str | None = None,
        bounds: BoundingBox | None = None,
    ) -> list[HistoricalEventRead]:
        events = self.repository.list_historical_events(municipality=municipality, bounds=bounds)
        return [
            HistoricalEventRead(
                id=event.id,
                municipality=event.municipality.name,
                date=event.date,
                severity=event.severity,
                type=event.type,
                coords=event.coords,
                source=event.source,
            )
            for event in events
        ]

    def list_zone_events(self, zone_id: str) -> list[HistoricalEventRead]:
        zone = self.repository.get_zone(zone_id)
        if zone is None:
            raise ApiError(404, "zone_not_found", f"Zone '{zone_id}' was not found.")
        events = self.repository.list_historical_events_for_zone(zone_id)
        return [
            HistoricalEventRead(
                id=event.id,
                municipality=event.municipality.name,
                date=event.date,
                severity=event.severity,
                type=event.type,
                coords=event.coords,
                source=event.source,
            )
            for event in events
        ]

    def get_zone_spatial_summary(self, zone_id: str) -> ZoneSpatialSummaryRead:
        zone = self.repository.get_zone(zone_id)
        if zone is None:
            raise ApiError(404, "zone_not_found", f"Zone '{zone_id}' was not found.")

        events = self.repository.list_historical_events_for_zone(zone_id)
        road_segments = self.repository.list_spatial_road_segments_for_zone(zone_id)
        overlays = self.repository.list_rain_overlays_for_zone(zone_id)

        severity_breakdown: dict[str, int] = {}
        for event in events:
            severity_breakdown[event.severity] = severity_breakdown.get(event.severity, 0) + 1

        rain_overlay_intensities = sorted(
            {overlay.intensity for overlay in overlays},
            key=lambda item: RAIN_INTENSITY_ORDER.get(item.lower(), 99),
            reverse=True,
        )

        return ZoneSpatialSummaryRead(
            zoneId=zone.id,
            municipality=zone.municipality.name,
            historicalEventCount=len(events),
            historicalEventIds=[event.id for event in events],
            severityBreakdown=severity_breakdown,
            intersectingRoadSegmentCount=len(road_segments),
            intersectingRoadSegmentIds=[segment.id for segment in road_segments],
            intersectingRoadLengthKm=round(sum(segment.length_km for segment in road_segments), 2),
            rainOverlayCount=len(overlays),
            rainOverlayIntensities=rain_overlay_intensities,
        )

    def list_ungrd_records(self, municipality: str | None = None) -> list[UngrdRecordRead]:
        records = self.repository.list_ungrd_records()
        if municipality:
            records = [record for record in records if record.municipality.name.lower() == municipality.lower()]
        return [
            UngrdRecordRead(
                id=record.id,
                municipality=record.municipality.name,
                date=record.date,
                summary=record.summary,
            )
            for record in records
        ]

    def ungrd_records_by_municipality(self) -> dict[str, list[UngrdRecordRead]]:
        grouped: dict[str, list[UngrdRecordRead]] = {}
        for record in self.list_ungrd_records():
            grouped.setdefault(record.municipality, []).append(record)
        return grouped

    def rain_series_by_municipality(self) -> dict[str, list[RainPointRead]]:
        grouped: dict[str, list[RainPointRead]] = {}
        for point in self.repository.list_rain_points():
            grouped.setdefault(point.municipality.name, []).append(
                RainPointRead(
                    time=point.time_label,
                    observed=point.observed,
                    forecast=point.forecast,
                    forecastLow=point.forecast_low,
                    forecastHigh=point.forecast_high,
                    forecastRange=point.forecast_range,
                )
            )
        return grouped

    def rain_series_for_zone(self, zone_id: str) -> list[RainPointRead]:
        zone = self.repository.get_zone(zone_id)
        if zone is None:
            raise ApiError(404, "zone_not_found", f"Zone '{zone_id}' was not found.")
        return self.rain_series_by_municipality().get(zone.municipality.name, [])

    def rain_overlays_by_municipality(
        self,
        municipality: str | None = None,
        bounds: BoundingBox | None = None,
    ) -> dict[str, list[RainOverlayRead]]:
        grouped: dict[str, list[RainOverlayRead]] = {}
        for overlay in self.repository.list_rain_overlays(municipality=municipality, bounds=bounds):
            grouped.setdefault(overlay.municipality.name, []).append(
                RainOverlayRead(bounds=overlay.bounds, intensity=overlay.intensity)
            )
        return grouped

    def get_zone_explanation(self, zone_id: str) -> ZoneExplanationRead:
        latest_run = self._latest_run_or_error()
        prediction = self.repository.get_prediction_for_zone(zone_id, latest_run.id)
        if prediction is None:
            raise ApiError(404, "zone_not_found", f"Zone '{zone_id}' was not found.")
        if prediction.explanation is None:
            raise ApiError(404, "explanation_not_found", f"No explanation exists for zone '{zone_id}'.")
        return ZoneExplanationRead(
            zoneId=prediction.zone_id,
            runId=prediction.run_id,
            mode=prediction.explanation.mode,
            summary=prediction.explanation.summary,
            driverChips=prediction.explanation.driver_chips,
            suggestions=prediction.explanation.suggestions,
            dataWarnings=prediction.explanation.data_warnings,
            trace=prediction.explanation.trace,
            generatedAt=prediction.explanation.generated_at,
        )

    def get_latest_run_summary(self) -> RunSummaryRead:
        latest_run = self._latest_run_or_error()
        zones = [self._zone_read_from_prediction(prediction) for prediction in latest_run.predictions]
        return self._run_summary(latest_run, zones, self.list_source_status())

    def get_run_detail(self, run_id: int) -> RunDetailRead:
        run = self.repository.get_run(run_id)
        if run is None:
            raise ApiError(404, "run_not_found", f"Run '{run_id}' was not found.")
        self._ensure_runtime_run_allowed(run)
        zones = [self._zone_read_from_prediction(prediction) for prediction in run.predictions]
        return RunDetailRead(
            **self._run_summary(run, zones, self.list_source_status()).model_dump(),
            zones=zones,
        )

    def get_dashboard_summary(self) -> DashboardSummaryRead:
        latest_run = self._latest_run_or_error()
        source_status = self.list_source_status()
        zones = [self._zone_read_from_prediction(prediction) for prediction in latest_run.predictions]
        latest_run_summary = self._run_summary(latest_run, zones, source_status)
        top_risk_zones = sorted(zones, key=lambda item: item.risk_score, reverse=True)[:5]
        return DashboardSummaryRead(
            latestRun=latest_run_summary,
            municipalities=self.list_municipalities(),
            topRiskZones=top_risk_zones,
            sourceStatus=source_status,
            kpis=DashboardKpiRead(
                areasMonitored=len(zones),
                highRiskCount=latest_run_summary.high_risk_count,
                freshnessPercent=latest_run_summary.freshness_percent,
                partialData=latest_run_summary.partial_data,
            ),
        )

    def get_dashboard_bootstrap(self) -> DashboardBootstrapRead:
        latest_run = self._latest_run_or_error()
        source_status = self.list_source_status()
        zones = [self._zone_read_from_prediction(prediction) for prediction in latest_run.predictions]
        return DashboardBootstrapRead(
            municipalities=self.list_municipalities(),
            zones=zones,
            roadSegments=self.list_road_segments(),
            rainSeries=self.rain_series_by_municipality(),
            historicalEvents=self.list_historical_events(),
            ungrdRecords=self.ungrd_records_by_municipality(),
            sourceCatalog=self.list_source_catalog(),
            sourceStatus=source_status,
            rainOverlays=self.rain_overlays_by_municipality(),
            riskColors=RISK_COLORS,
            riskOrder=RISK_ORDER,
            latestRun=self._run_summary(latest_run, zones, source_status),
            dataProvenance=self._data_provenance(
                latest_run=latest_run,
                source_status=source_status,
            ),
        )
