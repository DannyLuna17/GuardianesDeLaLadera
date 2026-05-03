import { Activity, Filter, Layers } from 'lucide-react'
import type { Municipality, SourceCatalog, Zone } from '../api'
import { RISK_LEVELS, type TimeWindow } from '../utils'

interface SidebarProps {
  selectedMunicipality: string
  setSelectedMunicipality: (value: string) => void
  municipalities: Municipality[]
  availableZoneTypes: string[]
  zoneTypeFilters: Record<string, boolean>
  setZoneTypeFilters: React.Dispatch<React.SetStateAction<Record<string, boolean>>>
  showRoadCorridors: boolean
  setShowRoadCorridors: (value: boolean) => void
  riskThreshold: number
  setRiskThreshold: (value: number) => void
  timeWindow: TimeWindow
  setTimeWindow: (value: TimeWindow) => void
  showSourcesLayer: boolean
  setShowSourcesLayer: (value: boolean) => void
  topRiskZones: Zone[]
  sourceCatalog: SourceCatalog[]
  selectedZone: Zone | null
  setSelectedZoneId: (id: string) => void
}

export function Sidebar({
  selectedMunicipality,
  setSelectedMunicipality,
  municipalities,
  availableZoneTypes,
  zoneTypeFilters,
  setZoneTypeFilters,
  showRoadCorridors,
  setShowRoadCorridors,
  riskThreshold,
  setRiskThreshold,
  timeWindow,
  setTimeWindow,
  showSourcesLayer,
  setShowSourcesLayer,
  topRiskZones,
  sourceCatalog,
  selectedZone,
  setSelectedZoneId,
}: SidebarProps) {
  return (
    <aside className="sidebar" aria-label="Filtros y navegacion">
      <div className="panel-title">
        <Filter size={16} aria-hidden="true" /> Filtros
      </div>

      <div className="field">
        <label htmlFor="municipality-select">Municipio</label>
        <select
          id="municipality-select"
          value={selectedMunicipality}
          onChange={(event) => setSelectedMunicipality(event.target.value)}
        >
          {municipalities.map((item) => (
            <option key={item.id} value={item.name}>
              {item.name}
            </option>
          ))}
        </select>
      </div>

      <fieldset className="field">
        <legend>Tipo de zona</legend>
        <div className="toggle-list" role="group">
          {availableZoneTypes.map((type) => (
            <label key={type} className="toggle">
              <input
                type="checkbox"
                checked={zoneTypeFilters[type] ?? true}
                onChange={(event) =>
                  setZoneTypeFilters((prev) => ({
                    ...prev,
                    [type]: event.target.checked,
                  }))
                }
              />
              <span>{type}</span>
            </label>
          ))}
          <label className="toggle">
            <input
              type="checkbox"
              checked={showRoadCorridors}
              onChange={(event) => setShowRoadCorridors(event.target.checked)}
            />
            <span>Corredores viales</span>
          </label>
        </div>
      </fieldset>

      <div className="field">
        <label htmlFor="risk-threshold">Umbral de riesgo</label>
        <div className="slider-label" aria-live="polite">
          {RISK_LEVELS[riskThreshold]}
        </div>
        <input
          id="risk-threshold"
          type="range"
          min={0}
          max={3}
          step={1}
          value={riskThreshold}
          aria-label={`Umbral de riesgo: ${RISK_LEVELS[riskThreshold]}`}
          onChange={(event) => setRiskThreshold(Number(event.target.value))}
        />
        <div className="slider-scale" aria-hidden="true">
          {RISK_LEVELS.map((level) => (
            <span key={level}>{level}</span>
          ))}
        </div>
      </div>

      <fieldset className="field">
        <legend>Ventana temporal</legend>
        <div className="segmented" role="group">
          {(['6h', '24h', '72h'] as const).map((item) => (
            <button
              key={item}
              className={timeWindow === item ? 'active' : ''}
              onClick={() => setTimeWindow(item)}
              aria-pressed={timeWindow === item}
            >
              Ultimas {item}
            </button>
          ))}
        </div>
      </fieldset>

      <div className="field">
        <label className="toggle">
          <input
            type="checkbox"
            checked={showSourcesLayer}
            onChange={(event) => setShowSourcesLayer(event.target.checked)}
          />
          <span>Mostrar capas IDEAM, SGC, IGAC</span>
        </label>
      </div>

      <div className="panel-title">
        <Activity size={16} aria-hidden="true" /> Zonas criticas
      </div>
      <nav className="zone-list" aria-label="Zonas de mayor riesgo">
        {topRiskZones.map((zone) => {
          return (
            <button
              key={zone.id}
              className={`zone-item ${zone.id === selectedZone?.id ? 'active' : ''}`}
              aria-current={zone.id === selectedZone?.id ? 'true' : undefined}
              onClick={() => {
                setSelectedMunicipality(zone.municipality)
                setSelectedZoneId(zone.id)
              }}
            >
              <span className="zone-name">{zone.name}</span>
              <span className={`badge risk ${zone.riskLevel}`}>
                {zone.riskLevel}
              </span>
              <span className="zone-trend">
                {Math.abs(zone.riskDelta) < 0.01 ? (
                  'Sin cambio'
                ) : (
                  <>
                    {zone.trend === 'subiendo'
                      ? '+'
                      : zone.trend === 'bajando'
                        ? '-'
                        : '='}
                    {Math.abs(zone.riskDelta).toFixed(2)}
                  </>
                )}
              </span>
            </button>
          )
        })}
      </nav>

      <div className="panel-title">
        <Layers size={16} aria-hidden="true" /> Fuentes catalogadas
      </div>
      <div className="source-grid" role="list" aria-label="Fuentes de datos catalogadas">
        {sourceCatalog.map((source) => (
          <span key={source.id} className="chip" role="listitem">
            {source.id}
          </span>
        ))}
      </div>
    </aside>
  )
}
