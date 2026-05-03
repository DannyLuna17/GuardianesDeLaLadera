import { useEffect } from 'react'
import {
  MapContainer,
  TileLayer,
  Polygon,
  Polyline,
  CircleMarker,
  Tooltip,
  ZoomControl,
  Rectangle,
  useMap,
} from 'react-leaflet'
import { AlertTriangle, MapPin } from 'lucide-react'
import type {
  HistoricalEvent,
  LatLng,
  Municipality,
  RainOverlay,
  RiskLevel,
  RoadSegment,
  Zone,
} from '../api'
import { formatRelative, RISK_LEVELS } from '../utils'
import { ErrorBoundary } from './ErrorBoundary'

const MapViewUpdater = ({ center, zoom }: { center: LatLng; zoom: number }) => {
  const map = useMap()
  useEffect(() => {
    map.flyTo(center, zoom, { duration: 1.2 })
  }, [center, map, zoom])
  return null
}

interface MapPanelProps {
  activeMunicipality: Municipality
  filteredZones: Zone[]
  selectedZone: Zone | null
  roadSegments: RoadSegment[]
  eventsForMunicipality: HistoricalEvent[]
  rainOverlays: Record<string, RainOverlay[]>
  riskColors: Record<RiskLevel, string>
  showRoadCorridors: boolean
  showSourcesLayer: boolean
  selectedMunicipality: string
  partialData: boolean
  setSelectedZoneId: (id: string) => void
}

export function MapPanel({
  activeMunicipality,
  filteredZones,
  selectedZone,
  roadSegments,
  eventsForMunicipality,
  rainOverlays,
  riskColors,
  showRoadCorridors,
  showSourcesLayer,
  selectedMunicipality,
  partialData,
  setSelectedZoneId,
}: MapPanelProps) {
  return (
    <main className="map-panel" aria-label="Mapa de riesgo">
      <div className="panel-title">
        <MapPin size={16} aria-hidden="true" /> Mapa de riesgo
      </div>
      <ErrorBoundary>
        <div className="map-container">
          <MapContainer
            center={activeMunicipality.center}
            zoom={activeMunicipality.zoom}
            zoomControl={false}
            className="leaflet-map"
          >
            <MapViewUpdater
              center={activeMunicipality.center}
              zoom={activeMunicipality.zoom}
            />
            <ZoomControl position="bottomright" />
            <TileLayer
              attribution="&copy; OpenStreetMap contributors"
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            {showSourcesLayer &&
              (rainOverlays[selectedMunicipality] ?? []).map((overlay, idx) => {
                const color =
                  overlay.intensity === 'alta'
                    ? '#f97316'
                    : overlay.intensity === 'media'
                      ? '#fdba74'
                      : '#fed7aa'
                return (
                  <Rectangle
                    key={`overlay-${selectedMunicipality}-${idx}`}
                    bounds={overlay.bounds}
                    pathOptions={{
                      fillColor: color,
                      fillOpacity: 0.25,
                      color: color,
                      weight: 0.5,
                    }}
                  />
                )
              })}

            {showRoadCorridors &&
              roadSegments
                .filter((segment) => segment.municipality === selectedMunicipality)
                .map((segment) => (
                  <Polyline
                    key={segment.id}
                    pathOptions={{
                      color: riskColors[segment.riskLevel as RiskLevel] ?? '#64748b',
                      weight: 4,
                    }}
                    positions={segment.coords}
                  >
                    <Tooltip sticky>
                      <div className="tooltip-title">{segment.name}</div>
                      <div>Riesgo vial: {segment.riskLevel}</div>
                      <div>{segment.length_km} km</div>
                    </Tooltip>
                  </Polyline>
                ))}

            {filteredZones.map((zone) => {
              const isSelected = zone.id === selectedZone?.id
              return (
                <Polygon
                  key={zone.id}
                  positions={zone.polygon}
                  pathOptions={{
                    color: riskColors[zone.riskLevel],
                    fillColor: riskColors[zone.riskLevel],
                    fillOpacity: isSelected ? 0.55 : 0.35,
                    weight: isSelected ? 2.5 : 1,
                    className: 'risk-polygon',
                  }}
                  eventHandlers={{
                    click: () => setSelectedZoneId(zone.id),
                  }}
                >
                  <Tooltip sticky>
                    <div className="tooltip-title">{zone.name}</div>
                    <div>Riesgo: {zone.riskLevel}</div>
                    <div>Actualizado: {formatRelative(zone.lastUpdated)}</div>
                  </Tooltip>
                </Polygon>
              )
            })}

            {eventsForMunicipality.map((event) => (
              <CircleMarker
                key={event.id}
                center={event.coords}
                radius={event.severity === 'Alta' ? 6 : event.severity === 'Media' ? 5 : 4}
                pathOptions={{
                  color: '#7c2d12',
                  fillColor: '#fca5a5',
                  fillOpacity: 0.9,
                  weight: 1,
                }}
              >
                <Tooltip sticky>
                  <div className="tooltip-title">Evento historico (SGC)</div>
                  <div>{event.date}</div>
                  <div>Severidad: {event.severity}</div>
                </Tooltip>
              </CircleMarker>
            ))}
          </MapContainer>

          <div className="legend">
            <div className="legend-title">Leyenda de riesgo</div>
            {RISK_LEVELS.map((level) => (
              <div key={level} className="legend-item">
                <span
                  className="legend-swatch"
                  style={{ background: riskColors[level] }}
                />
                <span>{level}</span>
              </div>
            ))}
          </div>

          {partialData && (
            <div className="map-banner" role="alert">
              <AlertTriangle size={14} aria-hidden="true" />
              Ejecutando con datos parciales. Algunas fuentes estan retrasadas.
            </div>
          )}
        </div>
      </ErrorBoundary>
    </main>
  )
}
