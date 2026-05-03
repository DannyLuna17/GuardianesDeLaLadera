import { useEffect, useMemo, useState } from 'react'
import type { DashboardBootstrap } from './api'
import {
  DEFAULT_MUNICIPALITY,
  DEFAULT_RISK_COLORS,
  DEFAULT_RISK_ORDER,
  RISK_LEVELS,
  RUN_INTERVAL_SECONDS,
  type TimeWindow,
} from './utils'

export function useDashboardViewModel(dashboard: DashboardBootstrap | null) {
  const [selectedMunicipality, setSelectedMunicipality] = useState('')
  const [selectedZoneId, setSelectedZoneId] = useState<string | null>(null)
  const [zoneTypeFilters, setZoneTypeFilters] = useState<Record<string, boolean>>({})
  const [riskThreshold, setRiskThreshold] = useState(0)
  const [timeWindow, setTimeWindow] = useState<TimeWindow>('24h')
  const [showSourcesLayer, setShowSourcesLayer] = useState(true)
  const [showRoadCorridors, setShowRoadCorridors] = useState(true)
  const [secondsRemaining, setSecondsRemaining] = useState(RUN_INTERVAL_SECONDS)

  const municipalities = dashboard?.municipalities ?? []
  const zonesState = dashboard?.zones ?? []
  const roadSegments = dashboard?.roadSegments ?? []
  const rainOverlays = dashboard?.rainOverlays ?? {}
  const rainSeries = dashboard?.rainSeries ?? {}
  const historicalEvents = dashboard?.historicalEvents ?? []
  const sourceCatalog = dashboard?.sourceCatalog ?? []
  const sourceStatus = dashboard?.sourceStatus ?? []
  const ungrdRecords = dashboard?.ungrdRecords ?? {}
  const riskColors = dashboard?.riskColors ?? DEFAULT_RISK_COLORS
  const riskOrder = dashboard?.riskOrder ?? DEFAULT_RISK_ORDER
  const latestRun = dashboard?.latestRun ?? null
  const lastRun = latestRun?.completedAt ?? null

  useEffect(() => {
    if (municipalities.length === 0) {
      if (selectedMunicipality) setSelectedMunicipality('')
      return
    }
    if (!municipalities.some((item) => item.name === selectedMunicipality)) {
      setSelectedMunicipality(municipalities[0].name)
    }
  }, [municipalities, selectedMunicipality])

  const activeMunicipality = useMemo(
    () =>
      municipalities.find((item) => item.name === selectedMunicipality) ??
      municipalities[0] ??
      DEFAULT_MUNICIPALITY,
    [municipalities, selectedMunicipality],
  )

  const availableZoneTypes = useMemo(
    () =>
      [...new Set(zonesState.map((zone) => zone.type).filter(Boolean))].sort((a, b) =>
        a.localeCompare(b, 'es-CO'),
      ),
    [zonesState],
  )

  useEffect(() => {
    setZoneTypeFilters((prev) => {
      const next = Object.fromEntries(
        availableZoneTypes.map((type) => [type, prev[type] ?? true]),
      )
      const sameKeys =
        Object.keys(prev).length === Object.keys(next).length &&
        Object.keys(next).every((key) => prev[key] === next[key])
      return sameKeys ? prev : next
    })
  }, [availableZoneTypes])

  const filteredZones = useMemo(() => {
    const thresholdLevel = RISK_LEVELS[riskThreshold] ?? 'Verde'
    return zonesState.filter((zone) => {
      if (zone.municipality !== selectedMunicipality) return false
      if (!(zoneTypeFilters[zone.type] ?? true)) return false
      return riskOrder[zone.riskLevel] >= riskOrder[thresholdLevel]
    })
  }, [riskOrder, riskThreshold, selectedMunicipality, zoneTypeFilters, zonesState])

  const selectedZone = useMemo(() => {
    if (!selectedZoneId) return filteredZones[0] ?? null
    return filteredZones.find((zone) => zone.id === selectedZoneId) ?? null
  }, [filteredZones, selectedZoneId])

  useEffect(() => {
    if (filteredZones.length === 0) {
      if (selectedZoneId !== null) setSelectedZoneId(null)
      return
    }
    if (!selectedZone) {
      setSelectedZoneId(filteredZones[0].id)
    }
  }, [filteredZones, selectedZone, selectedZoneId])

  useEffect(() => {
    if (!latestRun) {
      setSecondsRemaining(RUN_INTERVAL_SECONDS)
      return
    }

    const tick = () => {
      const nextRunAt = latestRun.completedAt.getTime() + RUN_INTERVAL_SECONDS * 1000
      setSecondsRemaining(Math.max(0, Math.ceil((nextRunAt - Date.now()) / 1000)))
    }

    tick()
    const interval = window.setInterval(tick, 1000)
    return () => window.clearInterval(interval)
  }, [latestRun])

  const partialData =
    latestRun?.partialData ??
    sourceStatus.some(
      (item) => item.status === 'Retrasado' || item.status === 'Desactualizado',
    )

  const freshnessPercent = latestRun?.freshnessPercent ?? 0
  const activeSourcesCount = latestRun?.activeSourcesCount ?? 0

  const highRiskCount = useMemo(
    () =>
      latestRun?.highRiskCount ??
      zonesState.filter((zone) => riskOrder[zone.riskLevel] >= 2).length,
    [latestRun, riskOrder, zonesState],
  )

  const topRiskZones = useMemo(
    () =>
      [...zonesState]
        .filter((zone) => zone.municipality === selectedMunicipality)
        .sort((a, b) => b.riskScore - a.riskScore)
        .slice(0, 3),
    [zonesState, selectedMunicipality],
  )

  const eventsForMunicipality = useMemo(
    () => historicalEvents.filter((item) => item.municipality === selectedMunicipality),
    [historicalEvents, selectedMunicipality],
  )

  const latestEventDate = useMemo(() => {
    if (eventsForMunicipality.length === 0) return '-'
    return [...eventsForMunicipality].sort((a, b) => (a.date < b.date ? 1 : -1))[0]
      .date
  }, [eventsForMunicipality])

  const ungrdForMunicipality = useMemo(
    () => ungrdRecords[selectedMunicipality] ?? [],
    [selectedMunicipality, ungrdRecords],
  )

  const rainData =
    rainSeries[selectedZone?.municipality ?? selectedMunicipality] ??
    rainSeries[selectedMunicipality] ??
    []

  const selectedRiskLevel = selectedZone?.riskLevel ?? 'Verde'

  const roadAssets = useMemo(() => {
    if (!selectedZone) return []
    return roadSegments.filter((segment) =>
      selectedZone.assets.road_segment_ids.includes(segment.id),
    )
  }, [roadSegments, selectedZone])

  const accumulationData = selectedZone
    ? [
        { label: '6h', mm: selectedZone.drivers.rain_6h },
        { label: '24h', mm: selectedZone.drivers.rain_24h },
        { label: '72h', mm: selectedZone.drivers.rain_72h },
      ].filter(
        (
          item,
        ): item is {
          label: TimeWindow
          mm: number
        } => item.mm !== null && item.mm !== undefined,
      )
    : []

  return {
    activeMunicipality,
    activeSourcesCount,
    accumulationData,
    availableZoneTypes,
    eventsForMunicipality,
    filteredZones,
    freshnessPercent,
    highRiskCount,
    lastRun,
    latestEventDate,
    latestRun,
    municipalities,
    partialData,
    rainData,
    rainOverlays,
    riskColors,
    roadAssets,
    roadSegments,
    riskThreshold,
    secondsRemaining,
    selectedMunicipality,
    selectedRiskLevel,
    selectedZone,
    setRiskThreshold,
    setSelectedMunicipality,
    setSelectedZoneId,
    setShowRoadCorridors,
    setShowSourcesLayer,
    setTimeWindow,
    setZoneTypeFilters,
    showRoadCorridors,
    showSourcesLayer,
    sourceCatalog,
    sourceStatus,
    timeWindow,
    topRiskZones,
    ungrdForMunicipality,
    zoneTypeFilters,
    zonesState,
  }
}
