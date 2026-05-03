export type RiskLevel = 'Verde' | 'Amarillo' | 'Naranja' | 'Rojo'
export type Confidence = 'Baja' | 'Media' | 'Alta'
export type ZoneType = string
export type LatLng = [number, number]
export type DataProvenanceState = 'real' | 'mock' | 'derived' | 'mixed' | 'unknown'

export type Municipality = {
  id: string
  name: string
  center: LatLng
  zoom: number
}

export type Zone = {
  id: string
  name: string
  municipality: string
  municipalityId: string
  type: ZoneType
  centroid: LatLng
  polygon: LatLng[]
  riskScore: number
  riskLevel: RiskLevel
  confidence: Confidence
  drivers: {
    rain_6h?: number | null
    rain_24h?: number | null
    rain_72h?: number | null
    slope_deg?: number | null
    geology_class?: string | null
    soil_class?: string | null
    deforestation_proxy?: number
  }
  exposure: {
    population_estimate: number | null
    households_estimate: number | null
  }
  assets: {
    road_segment_ids: string[]
  }
  lastUpdated: Date
  riskDelta: number
  trend: 'subiendo' | 'estable' | 'bajando'
}

export type RoadSegment = {
  id: string
  name: string
  municipality: string
  municipalityId: string
  coords: LatLng[]
  riskLevel: string
  length_km: number
  note: string
}

export type RainPoint = {
  time: string
  observed?: number
  forecast?: number
  forecastLow?: number
  forecastHigh?: number
  forecastRange?: number
}

export type HistoricalEvent = {
  id: string
  municipality: string
  date: string
  severity: 'Baja' | 'Media' | 'Alta'
  type: 'Deslizamiento'
  coords: LatLng
  source: 'SGC'
}

export type UngrdRecord = {
  id: string
  municipality: string
  date: string
  summary: string
}

export type SourceCatalog = {
  id: string
  label: string
  category: 'tiempo-real' | 'historico' | 'infraestructura'
}

export type SourceStatus = {
  id: string
  label: string
  category: string
  status: 'Fresco' | 'Retrasado' | 'Desactualizado' | 'Estatico'
  minutes?: number
  updatedAt?: Date
  note?: string
}

export type RainOverlay = {
  bounds: [LatLng, LatLng]
  intensity: 'alta' | 'media' | 'baja'
}

export type RunSummary = {
  id: number
  status: string
  modelVersion: string
  partialData: boolean
  startedAt: Date
  completedAt: Date
  zonesMonitored: number
  highRiskCount: number
  freshnessPercent: number
  activeSourcesCount: number
  totalSourcesCount: number
}

export type ZoneExplanation = {
  zoneId: string
  runId: number
  mode: string
  summary: string
  driverChips: string[]
  suggestions: string[]
  dataWarnings: string[]
  trace: Record<string, unknown>
  generatedAt: Date
}

export type DataProvenanceItem = {
  key: string
  label: string
  state: DataProvenanceState
  summary: string
  detail?: string
}

export type DataProvenance = {
  realDataOnly: boolean
  mockDataPresent: boolean
  items: DataProvenanceItem[]
}

export type DashboardBootstrap = {
  municipalities: Municipality[]
  zones: Zone[]
  roadSegments: RoadSegment[]
  rainSeries: Record<string, RainPoint[]>
  historicalEvents: HistoricalEvent[]
  ungrdRecords: Record<string, UngrdRecord[]>
  sourceCatalog: SourceCatalog[]
  sourceStatus: SourceStatus[]
  rainOverlays: Record<string, RainOverlay[]>
  riskColors: Record<RiskLevel, string>
  riskOrder: Record<RiskLevel, number>
  latestRun: RunSummary
  dataProvenance: DataProvenance
}

type DashboardBootstrapResponse = {
  municipalities: Array<{
    id: string
    name: string
    center: number[]
    zoom: number
  }>
  zones: Array<{
    id: string
    name: string
    municipality: string
    municipalityId: string
    type: string
    centroid: number[]
    polygon: number[][]
    riskScore: number
    riskLevel: RiskLevel
    confidence: Confidence
    drivers: Zone['drivers']
    exposure: Zone['exposure']
    assets: Zone['assets']
    lastUpdated: string
    riskDelta: number
    trend: Zone['trend']
  }>
  roadSegments: Array<{
    id: string
    name: string
    municipality: string
    municipalityId: string
    coords: number[][]
    riskLevel: string
    length_km: number
    note: string
  }>
  rainSeries: Record<string, RainPoint[]>
  historicalEvents: Array<{
    id: string
    municipality: string
    date: string
    severity: HistoricalEvent['severity']
    type: HistoricalEvent['type']
    coords: number[]
    source: HistoricalEvent['source']
  }>
  ungrdRecords: Record<string, UngrdRecord[]>
  sourceCatalog: SourceCatalog[]
  sourceStatus: Array<{
    id: string
    label: string
    category: string
    status: SourceStatus['status']
    minutes?: number
    updatedAt?: string | null
    note?: string | null
  }>
  rainOverlays: Record<
    string,
    Array<{
      bounds: number[][]
      intensity: RainOverlay['intensity']
    }>
  >
  riskColors: Record<RiskLevel, string>
  riskOrder: Record<RiskLevel, number>
  latestRun: {
    id: number
    status: string
    modelVersion: string
    partialData: boolean
    startedAt: string
    completedAt: string
    zonesMonitored: number
    highRiskCount: number
    freshnessPercent: number
    activeSourcesCount: number
    totalSourcesCount: number
  }
  dataProvenance?: {
    realDataOnly: boolean
    mockDataPresent: boolean
    items: Array<{
      key: string
      label: string
      state: DataProvenanceState
      summary: string
      detail?: string | null
    }>
  }
}

type ZoneExplanationResponse = {
  zoneId: string
  runId: number
  mode: string
  summary: string
  driverChips: string[]
  suggestions: string[]
  dataWarnings: string[]
  trace: Record<string, unknown>
  generatedAt: string
}

type ApiErrorPayload = {
  error?: {
    message?: string
  }
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '/api').replace(
  /\/$/,
  '',
)

const toLatLng = (coords: number[]): LatLng => {
  const [lat = 0, lng = 0] = coords
  return [lat, lng]
}

const toDate = (value: string): Date => new Date(value)

const DEFAULT_DATA_PROVENANCE: DataProvenance = {
  realDataOnly: false,
  mockDataPresent: false,
  items: [
    {
      key: 'unknown',
      label: 'Origen de datos',
      state: 'unknown',
      summary: 'Sin verificar',
      detail: 'El backend no expuso metadatos de procedencia para clasificar los datos.',
    },
  ],
}

const adaptRunSummary = (
  latestRun: DashboardBootstrapResponse['latestRun'],
): RunSummary => ({
  ...latestRun,
  startedAt: toDate(latestRun.startedAt),
  completedAt: toDate(latestRun.completedAt),
})

const adaptDashboardBootstrap = (
  payload: DashboardBootstrapResponse,
): DashboardBootstrap => ({
  municipalities: payload.municipalities.map((municipality) => ({
    ...municipality,
    center: toLatLng(municipality.center),
  })),
  zones: payload.zones.map((zone) => ({
    ...zone,
    type: zone.type,
    centroid: toLatLng(zone.centroid),
    polygon: zone.polygon.map(toLatLng),
    lastUpdated: toDate(zone.lastUpdated),
  })),
  roadSegments: payload.roadSegments.map((segment) => ({
    ...segment,
    coords: segment.coords.map(toLatLng),
  })),
  rainSeries: payload.rainSeries,
  historicalEvents: payload.historicalEvents.map((event) => ({
    ...event,
    coords: toLatLng(event.coords),
  })),
  ungrdRecords: payload.ungrdRecords,
  sourceCatalog: payload.sourceCatalog,
  sourceStatus: payload.sourceStatus.map((status) => ({
    ...status,
    updatedAt: status.updatedAt ? toDate(status.updatedAt) : undefined,
    note: status.note ?? undefined,
  })),
  rainOverlays: Object.fromEntries(
    Object.entries(payload.rainOverlays).map(([municipality, overlays]) => [
      municipality,
      overlays.map((overlay) => ({
        ...overlay,
        bounds: [toLatLng(overlay.bounds[0] ?? []), toLatLng(overlay.bounds[1] ?? [])],
      })),
    ]),
  ) as DashboardBootstrap['rainOverlays'],
  riskColors: payload.riskColors,
  riskOrder: payload.riskOrder,
  latestRun: adaptRunSummary(payload.latestRun),
  dataProvenance: payload.dataProvenance
    ? {
        realDataOnly: payload.dataProvenance.realDataOnly,
        mockDataPresent: payload.dataProvenance.mockDataPresent,
        items: payload.dataProvenance.items.map((item) => ({
          key: item.key,
          label: item.label,
          state: item.state,
          summary: item.summary,
          detail: item.detail ?? undefined,
        })),
      }
    : DEFAULT_DATA_PROVENANCE,
})

const adaptZoneExplanation = (
  payload: ZoneExplanationResponse,
): ZoneExplanation => ({
  ...payload,
  generatedAt: toDate(payload.generatedAt),
})

async function requestJson<T>(
  path: string,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      Accept: 'application/json',
    },
    signal,
  })

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`
    try {
      const payload = (await response.json()) as ApiErrorPayload
      if (payload.error?.message) {
        message = payload.error.message
      }
    } catch {
      // Leave the generic message in place when the response is not JSON.
    }
    throw new Error(message)
  }

  return (await response.json()) as T
}

export async function fetchDashboardBootstrap(
  signal?: AbortSignal,
): Promise<DashboardBootstrap> {
  const payload = await requestJson<DashboardBootstrapResponse>(
    '/v1/dashboard/bootstrap',
    signal,
  )
  return adaptDashboardBootstrap(payload)
}

export async function fetchZoneExplanation(
  zoneId: string,
  signal?: AbortSignal,
): Promise<ZoneExplanation> {
  const payload = await requestJson<ZoneExplanationResponse>(
    `/v1/zones/${encodeURIComponent(zoneId)}/explanation`,
    signal,
  )
  return adaptZoneExplanation(payload)
}
