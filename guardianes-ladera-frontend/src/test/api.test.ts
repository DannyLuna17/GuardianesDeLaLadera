import { describe, it, expect, vi, beforeEach } from 'vitest'
import { fetchDashboardBootstrap, fetchZoneExplanation } from '../api'

const MOCK_BOOTSTRAP_RESPONSE = {
  municipalities: [
    { id: 'mun-1', name: 'Mocoa', center: [1.147, -76.648], zoom: 12 },
  ],
  zones: [
    {
      id: 'zone-1',
      name: 'Test Zone',
      municipality: 'Mocoa',
      municipalityId: 'mun-1',
      type: 'Vereda',
      centroid: [1.15, -76.65],
      polygon: [[1.14, -76.66], [1.15, -76.64], [1.16, -76.65]],
      riskScore: 0.75,
      riskLevel: 'Naranja',
      confidence: 'Alta',
      drivers: { rain_24h: 50 },
      exposure: { population_estimate: 1000, households_estimate: 250 },
      assets: { road_segment_ids: [] },
      lastUpdated: '2026-03-28T10:00:00Z',
      riskDelta: 0.05,
      trend: 'subiendo',
    },
  ],
  roadSegments: [],
  rainSeries: {},
  historicalEvents: [],
  ungrdRecords: {},
  sourceCatalog: [],
  sourceStatus: [],
  rainOverlays: {},
  riskColors: { Verde: '#2e7d32', Amarillo: '#f9a825', Naranja: '#ef6c00', Rojo: '#c62828' },
  riskOrder: { Verde: 0, Amarillo: 1, Naranja: 2, Rojo: 3 },
  latestRun: {
    id: 1,
    status: 'completed',
    modelVersion: 'v1',
    partialData: false,
    startedAt: '2026-03-28T09:50:00Z',
    completedAt: '2026-03-28T10:00:00Z',
    zonesMonitored: 12,
    highRiskCount: 3,
    freshnessPercent: 85,
    activeSourcesCount: 5,
    totalSourcesCount: 8,
  },
  dataProvenance: {
    realDataOnly: true,
    mockDataPresent: false,
    items: [
      {
        key: 'structural_base',
        label: 'Base territorial',
        state: 'real',
        summary: '100% real',
        detail: 'Bundle oficial con trazabilidad.',
      },
      {
        key: 'risk_output',
        label: 'Puntaje y confianza',
        state: 'derived',
        summary: 'Derivado',
        detail: 'Salida calculada por el backend.',
      },
    ],
  },
}

const MOCK_EXPLANATION_RESPONSE = {
  zoneId: 'zone-1',
  runId: 1,
  mode: 'template',
  summary: 'Risk is elevated due to heavy rainfall.',
  driverChips: ['Lluvia 24h: 50mm'],
  suggestions: ['Monitor drainage systems'],
  dataWarnings: [],
  trace: {},
  generatedAt: '2026-03-28T10:00:00Z',
}

beforeEach(() => {
  vi.restoreAllMocks()
})

describe('fetchDashboardBootstrap', () => {
  it('fetches and adapts bootstrap data', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => MOCK_BOOTSTRAP_RESPONSE,
    } as Response)

    const result = await fetchDashboardBootstrap()

    expect(result.municipalities).toHaveLength(1)
    expect(result.municipalities[0].center).toEqual([1.147, -76.648])
    expect(result.zones).toHaveLength(1)
    expect(result.zones[0].lastUpdated).toBeInstanceOf(Date)
    expect(result.zones[0].polygon).toHaveLength(3)
    expect(result.latestRun.startedAt).toBeInstanceOf(Date)
    expect(result.latestRun.completedAt).toBeInstanceOf(Date)
    expect(result.dataProvenance.realDataOnly).toBe(true)
    expect(result.dataProvenance.items[0].state).toBe('real')
  })

  it('throws on HTTP error with parsed message', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({ error: { message: 'Server error' } }),
    } as Response)

    await expect(fetchDashboardBootstrap()).rejects.toThrow('Server error')
  })

  it('throws generic message on non-JSON error', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: false,
      status: 502,
      json: async () => { throw new Error('not json') },
    } as unknown as Response)

    await expect(fetchDashboardBootstrap()).rejects.toThrow('Request failed with status 502')
  })

  it('preserves dynamic contract values and nullable backend fields', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => ({
        ...MOCK_BOOTSTRAP_RESPONSE,
        zones: [
          {
            ...MOCK_BOOTSTRAP_RESPONSE.zones[0],
            type: 'Seccion Urbana Oficial',
            exposure: {
              population_estimate: null,
              households_estimate: null,
            },
          },
        ],
        roadSegments: [
          {
            id: 'road-1',
            name: 'Corredor piloto',
            municipality: 'Mocoa',
            municipalityId: 'mun-1',
            coords: [[1.14, -76.66], [1.15, -76.65]],
            riskLevel: 'Sin clasificar',
            length_km: 1.25,
            note: 'Sin etiqueta estandar',
          },
        ],
        sourceStatus: [
          {
            id: 'SGC',
            label: 'SGC',
            category: 'historico',
            status: 'Retrasado',
            updatedAt: null,
            note: null,
          },
        ],
        latestRun: {
          ...MOCK_BOOTSTRAP_RESPONSE.latestRun,
          partialData: true,
        },
      }),
    } as Response)

    const result = await fetchDashboardBootstrap()

    expect(result.zones[0].type).toBe('Seccion Urbana Oficial')
    expect(result.zones[0].exposure.population_estimate).toBeNull()
    expect(result.zones[0].exposure.households_estimate).toBeNull()
    expect(result.roadSegments[0].riskLevel).toBe('Sin clasificar')
    expect(result.sourceStatus[0].updatedAt).toBeUndefined()
    expect(result.sourceStatus[0].note).toBeUndefined()
    expect(result.latestRun.partialData).toBe(true)
  })

  it('falls back to unknown provenance when the backend omits it', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => {
        const { dataProvenance: _ignored, ...payload } = MOCK_BOOTSTRAP_RESPONSE
        return payload
      },
    } as Response)

    const result = await fetchDashboardBootstrap()

    expect(result.dataProvenance.items).toHaveLength(1)
    expect(result.dataProvenance.items[0].state).toBe('unknown')
  })
})

describe('fetchZoneExplanation', () => {
  it('fetches and adapts explanation data', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => MOCK_EXPLANATION_RESPONSE,
    } as Response)

    const result = await fetchZoneExplanation('zone-1')

    expect(result.zoneId).toBe('zone-1')
    expect(result.mode).toBe('template')
    expect(result.generatedAt).toBeInstanceOf(Date)
    expect(result.driverChips).toHaveLength(1)
  })

  it('encodes zoneId in the URL', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => MOCK_EXPLANATION_RESPONSE,
    } as Response)

    await fetchZoneExplanation('zone/with spaces')

    const calledUrl = fetchSpy.mock.calls[0][0] as string
    expect(calledUrl).toContain('zone%2Fwith%20spaces')
    expect(calledUrl).not.toContain('zone/with spaces')
  })
})
