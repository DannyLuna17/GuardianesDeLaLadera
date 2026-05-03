import { Suspense, lazy, useCallback, useMemo, type ReactNode } from 'react'
import { Activity, AlertTriangle, Download } from 'lucide-react'
// import { DataProvenanceBanner } from './components/DataProvenanceBanner'
import { KpiRow } from './components/KpiRow'
import { Sidebar } from './components/Sidebar'
import { exportDashboardPng, exportZoneReportPdf } from './exporters'
import { useDashboardBootstrap, useZoneExplanation } from './useDashboardApi'
import { useDashboardViewModel } from './useDashboardViewModel'

const MapPanel = lazy(() =>
  import('./components/MapPanel').then((module) => ({ default: module.MapPanel })),
)
const AnalystPanel = lazy(() =>
  import('./components/AnalystPanel').then((module) => ({
    default: module.AnalystPanel,
  })),
)

function PageState({
  title,
  description,
  action,
  tone = 'default',
}: {
  title: string
  description: string
  action?: ReactNode
  tone?: 'default' | 'error'
}) {
  return (
    <div className="app page-shell" role={tone === 'error' ? 'alert' : 'status'} aria-live="polite">
      <header className="app-header">
        <div>
          <h1 className="brand">
            Guardianes de la Ladera <span className="tag">Beta</span>
          </h1>
          <p className="subtitle">
            Sistema de apoyo a decision para gestores de riesgo
          </p>
        </div>
      </header>

      <div className="page-state">
        <div className={`page-state-card ${tone === 'error' ? 'error' : ''}`}>
          {tone === 'error' ? (
            <AlertTriangle size={18} aria-hidden="true" />
          ) : (
            <Activity size={18} aria-hidden="true" />
          )}
          <div className="page-state-title">{title}</div>
          <div className="page-state-text">{description}</div>
          {action}
        </div>
      </div>
    </div>
  )
}

function PanelPlaceholder({ label }: { label: string }) {
  return (
    <section className="panel-placeholder" aria-busy="true" aria-live="polite">
      <div className="panel-title">{label}</div>
      <div className="panel-placeholder-body">Cargando modulo...</div>
    </section>
  )
}

export default function App() {
  const {
    dashboard,
    isLoading: isLoadingDashboard,
    error: dashboardError,
    isRefreshing,
    reload,
  } = useDashboardBootstrap()
  const {
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
    riskThreshold,
    roadAssets,
    roadSegments,
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
  } = useDashboardViewModel(dashboard)

  const {
    explanation: zoneExplanation,
    isLoading: isLoadingExplanation,
    error: explanationError,
  } = useZoneExplanation(selectedZone?.id ?? null, latestRun?.id ?? null)

  const handleRetry = useCallback(() => {
    reload()
  }, [reload])

  const handleExportPng = useCallback(async () => {
    const node = document.getElementById('dashboard-root')
    if (!node) return
    await exportDashboardPng(node)
  }, [])

  const handleExportPdf = useCallback(async () => {
    if (!selectedZone) return
    const node = document.getElementById('agent-panel')
    if (!node) return
    await exportZoneReportPdf(node, {
      zoneId: selectedZone.id,
      zoneName: selectedZone.name,
      confidence: selectedZone.confidence,
      lastRun,
    })
  }, [lastRun, selectedZone])

  const driverChips = zoneExplanation?.driverChips ?? []
  const suggestionList = zoneExplanation?.suggestions ?? []
  const explanationWarnings = useMemo(
    () => zoneExplanation?.dataWarnings ?? [],
    [zoneExplanation],
  )
  const summaryText = useMemo(
    () =>
      selectedZone
        ? zoneExplanation?.summary ??
          'No hay una explicacion disponible desde el backend para esta zona.'
        : 'Seleccione una zona para ver el resumen del riesgo.',
    [selectedZone, zoneExplanation],
  )

  if (isLoadingDashboard && !dashboard) {
    return (
      <PageState
        title="Cargando dashboard"
        description="Consultando el bootstrap del backend para poblar el mapa y el panel operativo."
      />
    )
  }

  if (dashboardError && !dashboard) {
    return (
      <PageState
        title="No fue posible cargar el backend"
        description={dashboardError}
        tone="error"
        action={
          <button className="primary" onClick={handleRetry}>
            Reintentar
          </button>
        }
      />
    )
  }

  return (
    <div className="app" id="dashboard-root">
      <a className="skip-link" href="#main-content">
        Saltar al contenido principal
      </a>
      <header className="app-header">
        <div>
          <h1 className="brand">
            Guardianes de la Ladera <span className="tag">Beta</span>
          </h1>
          <p className="subtitle">
            Sistema de apoyo a decision para gestores de riesgo
          </p>
        </div>
        <div className="header-actions">
          <button
            className="ghost"
            onClick={handleRetry}
            aria-label={isRefreshing ? 'Actualizando datos' : 'Actualizar datos'}
          >
            {isRefreshing ? 'Actualizando...' : 'Actualizar datos'}
          </button>
          <button
            className="ghost"
            onClick={handleExportPng}
            aria-label="Exportar captura del dashboard como PNG"
          >
            <Download size={16} aria-hidden="true" /> Exportar captura (PNG)
          </button>
          <button
            className="primary"
            onClick={handleExportPdf}
            disabled={!selectedZone}
            aria-label="Descargar reporte de zona como PDF"
          >
            <Download size={16} aria-hidden="true" /> Descargar reporte (PDF)
          </button>
        </div>
      </header>

      {dashboardError && (
        <div className="status-banner warning" role="alert">
          <span>{dashboardError}</span>
          <button onClick={handleRetry}>Reintentar</button>
        </div>
      )}

      {/* <DataProvenanceBanner dataProvenance={dashboard!.dataProvenance} /> */}

      <KpiRow
        zonesCount={zonesState.length}
        municipalitiesCount={municipalities.length}
        highRiskCount={highRiskCount}
        lastRun={lastRun}
        latestRunId={latestRun?.id}
        isRefreshing={isRefreshing}
        secondsRemaining={secondsRemaining}
        freshnessPercent={freshnessPercent}
        selectedZone={selectedZone}
      />

      <div className="main-grid" id="main-content">
        <Sidebar
          selectedMunicipality={selectedMunicipality}
          setSelectedMunicipality={setSelectedMunicipality}
          municipalities={municipalities}
          availableZoneTypes={availableZoneTypes}
          zoneTypeFilters={zoneTypeFilters}
          setZoneTypeFilters={setZoneTypeFilters}
          showRoadCorridors={showRoadCorridors}
          setShowRoadCorridors={setShowRoadCorridors}
          riskThreshold={riskThreshold}
          setRiskThreshold={setRiskThreshold}
          timeWindow={timeWindow}
          setTimeWindow={setTimeWindow}
          showSourcesLayer={showSourcesLayer}
          setShowSourcesLayer={setShowSourcesLayer}
          topRiskZones={topRiskZones}
          sourceCatalog={sourceCatalog}
          selectedZone={selectedZone}
          setSelectedZoneId={setSelectedZoneId}
        />

        <Suspense fallback={<PanelPlaceholder label="Cargando mapa de riesgo" />}>
          <MapPanel
            activeMunicipality={activeMunicipality}
            filteredZones={filteredZones}
            selectedZone={selectedZone}
            roadSegments={roadSegments}
            eventsForMunicipality={eventsForMunicipality}
            rainOverlays={rainOverlays}
            riskColors={riskColors}
            showRoadCorridors={showRoadCorridors}
            showSourcesLayer={showSourcesLayer}
            selectedMunicipality={selectedMunicipality}
            partialData={partialData}
            setSelectedZoneId={setSelectedZoneId}
          />
        </Suspense>

        <Suspense fallback={<PanelPlaceholder label="Cargando panel de analisis" />}>
          <AnalystPanel
            selectedZone={selectedZone}
            selectedRiskLevel={selectedRiskLevel}
            lastRun={lastRun}
            activeSourcesCount={activeSourcesCount}
            sourceStatusCount={sourceStatus.length}
            partialData={partialData}
            zoneExplanation={zoneExplanation}
            isLoadingExplanation={isLoadingExplanation}
            explanationError={explanationError}
            explanationWarnings={explanationWarnings}
            summaryText={summaryText}
            driverChips={driverChips}
            rainData={rainData}
            isRefreshing={isRefreshing}
            accumulationData={accumulationData}
            timeWindow={timeWindow}
            eventsForMunicipality={eventsForMunicipality}
            latestEventDate={latestEventDate}
            roadAssets={roadAssets}
            suggestionList={suggestionList}
            ungrdForMunicipality={ungrdForMunicipality}
            sourceStatus={sourceStatus}
          />
        </Suspense>
      </div>
    </div>
  )
}
