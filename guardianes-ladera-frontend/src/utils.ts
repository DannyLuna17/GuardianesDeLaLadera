import type {
  Confidence,
  DataProvenanceState,
  Municipality,
  RiskLevel,
  SourceStatus,
  ZoneExplanation,
} from './api'

export type TooltipEntry = {
  dataKey?: string
  value?: number | string
}
export type TimeWindow = '6h' | '24h' | '72h'

export const RUN_INTERVAL_SECONDS = 600
export const RISK_LEVELS: RiskLevel[] = ['Verde', 'Amarillo', 'Naranja', 'Rojo']

export const DEFAULT_MUNICIPALITY: Municipality = {
  id: 'default',
  name: 'Mocoa',
  center: [1.147, -76.648],
  zoom: 12,
}

export const DEFAULT_RISK_COLORS: Record<RiskLevel, string> = {
  Verde: '#2e7d32',
  Amarillo: '#f9a825',
  Naranja: '#ef6c00',
  Rojo: '#c62828',
}

export const DEFAULT_RISK_ORDER: Record<RiskLevel, number> = {
  Verde: 0,
  Amarillo: 1,
  Naranja: 2,
  Rojo: 3,
}

export const confidenceTone: Record<Confidence, string> = {
  Alta: 'ok',
  Media: 'warn',
  Baja: 'risk',
}

export const formatTime = (date?: Date | null) =>
  date
    ? date.toLocaleTimeString('es-CO', { hour: '2-digit', minute: '2-digit' })
    : '--:--'

export const formatDateTime = (date?: Date | null) =>
  date
    ? date.toLocaleString('es-CO', {
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
      })
    : ''

export const formatNumber = (value: number | null | undefined) =>
  typeof value === 'number' && Number.isFinite(value)
    ? value.toLocaleString('es-CO')
    : 'No disponible'

export const formatMinutes = (minutes: number) => {
  if (minutes < 60) return `${minutes} min`
  if (minutes < 1440) return `${Math.round(minutes / 60)} h`
  return `${Math.round(minutes / 1440)} dias`
}

export const formatRelative = (date?: Date | null) => {
  if (!date) return 'sin dato'
  const diff = Math.round((Date.now() - date.getTime()) / 60000)
  if (diff <= 1) return 'hace 1 min'
  if (diff < 60) return `hace ${diff} min`
  if (diff < 1440) return `hace ${Math.round(diff / 60)} h`
  return `hace ${Math.round(diff / 1440)} dias`
}

export const formatOptionalMetric = (
  value: number | string | null | undefined,
  suffix = '',
) =>
  value === null || value === undefined || value === ''
    ? 'No disponible'
    : `${value}${suffix}`

export const formatSourceBadgeMeta = (item: SourceStatus) => {
  if (item.minutes !== undefined) return ` - ${formatMinutes(item.minutes)}`
  if (item.updatedAt) return ` - ${formatDateTime(item.updatedAt)}`
  return ''
}

export const formatExplanationMode = (
  explanation: ZoneExplanation | null,
  explanationError: string | null,
  isLoadingExplanation: boolean,
) => {
  if (explanation) {
    return explanation.mode === 'template'
      ? 'Plantilla backend'
      : `Backend ${explanation.mode}`
  }
  if (isLoadingExplanation) return 'Cargando...'
  if (explanationError) return 'No disponible'
  return '-'
}

export const provenanceTone: Record<DataProvenanceState, string> = {
  real: 'provenance-real',
  mock: 'provenance-mock',
  derived: 'provenance-derived',
  mixed: 'provenance-mixed',
  unknown: 'provenance-unknown',
}

export const formatProvenanceState = (state: DataProvenanceState) => {
  switch (state) {
    case 'real':
      return '100% real'
    case 'mock':
      return 'Mock/seed'
    case 'derived':
      return 'Derivado'
    case 'mixed':
      return 'Mixto'
    default:
      return 'Sin verificar'
  }
}
