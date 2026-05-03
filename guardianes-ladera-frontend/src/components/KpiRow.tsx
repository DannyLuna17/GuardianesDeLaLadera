import { Clock } from 'lucide-react'
import type { Confidence } from '../api'
import { confidenceTone, formatTime } from '../utils'

interface KpiRowProps {
  zonesCount: number
  municipalitiesCount: number
  highRiskCount: number
  lastRun: Date | null
  latestRunId: number | string | null | undefined
  isRefreshing: boolean
  secondsRemaining: number
  freshnessPercent: number
  selectedZone: { confidence: Confidence; name: string } | null
}

export function KpiRow({
  zonesCount,
  municipalitiesCount,
  highRiskCount,
  lastRun,
  latestRunId,
  isRefreshing,
  secondsRemaining,
  freshnessPercent,
  selectedZone,
}: KpiRowProps) {
  return (
    <section className="kpi-row" aria-label="Indicadores clave de rendimiento">
      <div className="kpi-card">
        <div className="kpi-label">Areas monitoreadas</div>
        <div className="kpi-value">{zonesCount}</div>
        <div className="kpi-foot">
          {zonesCount} zonas en {municipalitiesCount} municipios
        </div>
      </div>
      <div className="kpi-card">
        <div className="kpi-label">Zonas alto riesgo</div>
        <div className="kpi-value">{highRiskCount}</div>
        <div className="kpi-foot">Naranja + Rojo</div>
      </div>
      <div className="kpi-card">
        <div className="kpi-label">Ultima corrida</div>
        <div className="kpi-value">{formatTime(lastRun)}</div>
        <div className="kpi-foot">
          {isRefreshing ? 'Sincronizando backend...' : `Run #${latestRunId ?? '-'}`}
        </div>
      </div>
      <div className="kpi-card">
        <div className="kpi-label">Proxima corrida</div>
        <div className="kpi-value">
          {Math.floor(secondsRemaining / 60)
            .toString()
            .padStart(2, '0')}
          :{(secondsRemaining % 60).toString().padStart(2, '0')}
        </div>
        <div className="kpi-foot">
          <Clock size={14} aria-hidden="true" />
          {/* {secondsRemaining > 0 ? 'Ventana estimada' : 'Esperando nueva corrida'} */}
        </div>
      </div>
      <div className="kpi-card">
        <div className="kpi-label">Frescura datos</div>
        <div className="kpi-value">{freshnessPercent}%</div>
        <div className="kpi-foot">
          <div className="progress" role="progressbar" aria-valuenow={freshnessPercent} aria-valuemin={0} aria-valuemax={100} aria-label={`Frescura de datos: ${freshnessPercent}%`}>
            <div
              className="progress-fill"
              style={{ width: `${freshnessPercent}%` }}
            />
          </div>
        </div>
      </div>
      <div className="kpi-card">
        <div className="kpi-label">Confianza zona</div>
        <div className="kpi-value">
          {selectedZone ? selectedZone.confidence : '-'}
        </div>
        <div className="kpi-foot">
          <span
            className={`badge ${confidenceTone[selectedZone?.confidence ?? 'Media']}`}
          >
            {selectedZone ? selectedZone.name : 'Seleccione zona'}
          </span>
        </div>
      </div>
    </section>
  )
}
