import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  Cell,
  Line,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Activity, AlertTriangle, CheckCircle2 } from 'lucide-react'
import type {
  HistoricalEvent,
  RainPoint,
  RiskLevel,
  RoadSegment,
  SourceStatus,
  UngrdRecord,
  Zone,
  ZoneExplanation,
} from '../api'
import {
  confidenceTone,
  formatExplanationMode,
  formatNumber,
  formatOptionalMetric,
  formatSourceBadgeMeta,
  formatTime,
  type TimeWindow,
  type TooltipEntry,
} from '../utils'
import { ErrorBoundary } from './ErrorBoundary'

const RainTooltip = ({
  active,
  payload,
  label,
}: {
  active?: boolean
  payload?: TooltipEntry[]
  label?: string
}) => {
  if (!active || !payload || payload.length === 0) return null
  const observed = payload.find((item) => item.dataKey === 'observed')
  const forecast = payload.find((item) => item.dataKey === 'forecast')
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-title">Hora {label}</div>
      {observed?.value !== undefined && (
        <div className="chart-tooltip-row">
          <span>Observado</span>
          <span>{observed.value} mm</span>
        </div>
      )}
      {forecast?.value !== undefined && (
        <div className="chart-tooltip-row">
          <span>Pronostico</span>
          <span>{forecast.value} mm</span>
        </div>
      )}
    </div>
  )
}

const AccumulationTooltip = ({
  active,
  payload,
  label,
}: {
  active?: boolean
  payload?: TooltipEntry[]
  label?: string
}) => {
  if (!active || !payload || payload.length === 0) return null
  const value = payload[0]?.value
  if (value === undefined) return null
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-title">Acumulado {label}</div>
      <div className="chart-tooltip-row">
        <span>Total</span>
        <span>{value} mm</span>
      </div>
    </div>
  )
}

interface AnalystPanelProps {
  selectedZone: Zone | null
  selectedRiskLevel: RiskLevel
  lastRun: Date | null
  activeSourcesCount: number
  sourceStatusCount: number
  partialData: boolean
  zoneExplanation: ZoneExplanation | null
  isLoadingExplanation: boolean
  explanationError: string | null
  explanationWarnings: string[]
  summaryText: string
  driverChips: string[]
  rainData: RainPoint[]
  isRefreshing: boolean
  accumulationData: Array<{ label: '6h' | '24h' | '72h'; mm: number }>
  timeWindow: TimeWindow
  eventsForMunicipality: HistoricalEvent[]
  latestEventDate: string
  roadAssets: RoadSegment[]
  suggestionList: string[]
  ungrdForMunicipality: UngrdRecord[]
  sourceStatus: SourceStatus[]
}

export function AnalystPanel({
  selectedZone,
  selectedRiskLevel,
  lastRun,
  activeSourcesCount,
  sourceStatusCount,
  partialData,
  zoneExplanation,
  isLoadingExplanation,
  explanationError,
  explanationWarnings,
  summaryText,
  driverChips,
  rainData,
  isRefreshing,
  accumulationData,
  timeWindow,
  eventsForMunicipality,
  latestEventDate,
  roadAssets,
  suggestionList,
  ungrdForMunicipality,
  sourceStatus,
}: AnalystPanelProps) {
  return (
    <aside className="agent-panel" id="agent-panel" aria-label="Panel de analisis de zona">
      <div className="panel-title">
        <Activity size={16} aria-hidden="true" /> Salida de agente
      </div>

      {selectedZone ? (
        <>
          <div className="zone-header">
            <div>
              <div className="zone-title">{selectedZone.name}</div>
              <div className="zone-meta">
                {selectedZone.municipality} - {selectedZone.type}
              </div>
            </div>
            <div className="zone-badges">
              <span className={`badge risk ${selectedRiskLevel}`}>
                {selectedRiskLevel}
              </span>
              <span className={`badge ${confidenceTone[selectedZone.confidence]}`}>
                Confianza {selectedZone.confidence}
              </span>
            </div>
          </div>

          <div className="agent-structure">
            <div className="structure-row">
              <span>Ejecucion</span>
              <span>{formatTime(lastRun)} - cada 10 min</span>
            </div>
            <div className="structure-row">
              <span>Fuentes activas</span>
              <span>
                {activeSourcesCount}/{sourceStatusCount} (pondera actualidad)
              </span>
            </div>
            <div className="structure-row">
              <span>Datos</span>
              <span>{partialData ? 'Parciales' : 'Completos'}</span>
            </div>
            <div className="structure-row">
              <span>Explicacion</span>
              <span>
                {formatExplanationMode(
                  zoneExplanation,
                  explanationError,
                  isLoadingExplanation,
                )}
              </span>
            </div>
          </div>

          {explanationError && (
            <div className="inline-alert warning" role="alert">
              <AlertTriangle size={14} aria-hidden="true" />
              <span>{explanationError}</span>
            </div>
          )}

          {explanationWarnings.length > 0 && (
            <div className="inline-alert info" role="status">
              <AlertTriangle size={14} aria-hidden="true" />
              <span>{explanationWarnings.join(' ')}</span>
            </div>
          )}

          <div className="section">
            <div className="section-title">Resumen de riesgo</div>
            <p className="section-text">{summaryText}</p>
          </div>

          <div className="section">
            <div className="section-title">Por que el riesgo esta subiendo</div>
            <div className="chip-row">
              {driverChips.map((chip) => (
                <span key={chip} className="chip">
                  {chip}
                </span>
              ))}
              {driverChips.length === 0 && (
                <div className="muted">
                  Sin factores explicativos disponibles desde el backend.
                </div>
              )}
            </div>
          </div>

          <div className="section">
            <div className="section-title">Evidencia</div>
            <div className="evidence-grid">
              <ErrorBoundary>
                <div className="card span-2">
                  <div className="card-title">Lluvia IDEAM + pronostico</div>
                  {isRefreshing && rainData.length === 0 ? (
                    <div className="skeleton chart" />
                  ) : (
                    <div className="chart-wrapper">
                      <ResponsiveContainer width="100%" height={150}>
                        <AreaChart data={rainData}>
                          <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                          <YAxis tick={{ fontSize: 10 }} />
                          <RechartsTooltip content={<RainTooltip />} />
                          <Area
                            type="monotone"
                            dataKey="forecastLow"
                            stackId="band"
                            stroke="none"
                            fill="transparent"
                          />
                          <Area
                            type="monotone"
                            dataKey="forecastRange"
                            stackId="band"
                            stroke="none"
                            fill="#fed7aa"
                            fillOpacity={0.35}
                          />
                          <Line
                            type="monotone"
                            dataKey="observed"
                            stroke="#ea580c"
                            strokeWidth={2}
                            dot={false}
                          />
                          <Line
                            type="monotone"
                            dataKey="forecast"
                            stroke="#f59e0b"
                            strokeWidth={2}
                            strokeDasharray="4 3"
                            dot={false}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                  <div className="card-foot">
                    Banda de incertidumbre estimada para 6h.
                  </div>
                </div>
              </ErrorBoundary>

              <ErrorBoundary>
                <div className="card">
                  <div className="card-title">Acumulados de lluvia</div>
                  {accumulationData.length > 0 ? (
                    <div className="chart-wrapper small">
                      <ResponsiveContainer width="100%" height={130}>
                        <BarChart data={accumulationData}>
                          <XAxis dataKey="label" tick={{ fontSize: 10 }} />
                          <YAxis tick={{ fontSize: 10 }} />
                          <RechartsTooltip content={<AccumulationTooltip />} />
                          <Bar dataKey="mm" radius={[8, 8, 0, 0]}>
                            {accumulationData.map((item) => (
                              <Cell
                                key={item.label}
                                fill={
                                  item.label === timeWindow ? '#f97316' : '#fdba74'
                                }
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <div className="muted">
                      Sin acumulados oficiales disponibles para esta zona.
                    </div>
                  )}
                  <div className="card-foot">Ventana destacada: {timeWindow}</div>
                </div>
              </ErrorBoundary>

              <div className="card">
                <div className="card-title">Susceptibilidad terreno</div>
                <div className="metric-list">
                  <div>
                    <span>Pendiente</span>
                    <strong>
                      {formatOptionalMetric(selectedZone.drivers.slope_deg, ' deg')}
                    </strong>
                  </div>
                  <div>
                    <span>Geologia</span>
                    <strong>
                      {formatOptionalMetric(selectedZone.drivers.geology_class)}
                    </strong>
                  </div>
                  <div>
                    <span>Suelo</span>
                    <strong>
                      {formatOptionalMetric(selectedZone.drivers.soil_class)}
                    </strong>
                  </div>
                  <div>
                    <span>Cobertura</span>
                    <strong>
                      {selectedZone.drivers.deforestation_proxy !== undefined &&
                      selectedZone.drivers.deforestation_proxy !== null
                        ? `${Math.round(
                            selectedZone.drivers.deforestation_proxy * 100,
                          )}%`
                        : 'No disponible'}
                    </strong>
                  </div>
                </div>
              </div>

              <div className="card">
                <div className="card-title">Eventos cercanos</div>
                <div className="metric-large">
                  {eventsForMunicipality.length}
                </div>
                <div className="card-foot">
                  Ultimo evento: {latestEventDate}
                </div>
              </div>

              <div className="card">
                <div className="card-title">Exposicion (DANE)</div>
                <div className="metric-list">
                  <div>
                    <span>Poblacion</span>
                    <strong>{formatNumber(selectedZone.exposure.population_estimate)}</strong>
                  </div>
                  <div>
                    <span>Hogares</span>
                    <strong>{formatNumber(selectedZone.exposure.households_estimate)}</strong>
                  </div>
                </div>
              </div>

              <div className="card">
                <div className="card-title">Activos viales</div>
                <div className="road-list">
                  {roadAssets.map((segment) => (
                    <div key={segment.id} className="road-item">
                      <span>{segment.name}</span>
                      <span
                        className={
                          ['Verde', 'Amarillo', 'Naranja', 'Rojo'].includes(segment.riskLevel)
                            ? `badge risk ${segment.riskLevel}`
                            : 'badge'
                        }
                      >
                        {segment.riskLevel}
                      </span>
                    </div>
                  ))}
                  {roadAssets.length === 0 && (
                    <div className="muted">Sin segmentos asociados</div>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="section">
            <div className="section-title">Sugerencias a considerar</div>
            <ul className="suggestions">
              {suggestionList.map((item) => (
                <li key={item}>{item}</li>
              ))}
              {suggestionList.length === 0 && (
                <li>Sin sugerencias disponibles desde el backend.</li>
              )}
            </ul>
          </div>

          <div className="section">
            <div className="section-title">Registros UNGRD recientes</div>
            <div className="timeline">
              {ungrdForMunicipality.map((record) => (
                <div key={record.id} className="timeline-item">
                  <div className="timeline-date">{record.date}</div>
                  <div className="timeline-text">{record.summary}</div>
                </div>
              ))}
              {ungrdForMunicipality.length === 0 && (
                <div className="muted">Sin registros recientes.</div>
              )}
            </div>
          </div>

          <div className="section">
            <div className="section-title">Estado de datos</div>
            <div className="status-list">
              {sourceStatus.map((item) => (
                <div key={item.id} className="status-item">
                  <div className="status-left">
                    {item.status === 'Fresco' || item.status === 'Estatico' ? (
                      <CheckCircle2 size={14} aria-hidden="true" />
                    ) : (
                      <AlertTriangle size={14} aria-hidden="true" />
                    )}
                    <span>{item.label}</span>
                  </div>
                  <div className={`badge ${item.status.toLowerCase()}`}>
                    {item.status}
                    {formatSourceBadgeMeta(item)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      ) : (
        <div className="empty">
          No hay zonas visibles con el filtro actual.
        </div>
      )}
    </aside>
  )
}
