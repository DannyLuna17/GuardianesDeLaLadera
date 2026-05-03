import { Activity, AlertTriangle, CheckCircle2 } from 'lucide-react'
import type { DataProvenance } from '../api'
import { formatProvenanceState, provenanceTone } from '../utils'

interface DataProvenanceBannerProps {
  dataProvenance: DataProvenance
}

export function DataProvenanceBanner({
  dataProvenance,
}: DataProvenanceBannerProps) {
  const headline = dataProvenance.mockDataPresent
    ? 'Hay datos mock/seed visibles en este runtime.'
    : dataProvenance.realDataOnly
      ? 'No se detectan datos mock visibles en este runtime.'
      : 'El dashboard separa datos reales, derivados y mock/seed segun la metadata del backend.'

  return (
    <section
      className={`provenance-banner ${dataProvenance.mockDataPresent ? 'warning' : 'ok'}`}
      aria-label="Origen de datos"
    >
      <div className="provenance-banner-head">
        <div className="panel-title">
          {dataProvenance.mockDataPresent ? (
            <AlertTriangle size={16} aria-hidden="true" />
          ) : (
            <CheckCircle2 size={16} aria-hidden="true" />
          )}
          Origen de datos
        </div>
        <span className={`badge ${dataProvenance.realDataOnly ? 'fresco' : 'estatico'}`}>
          {dataProvenance.realDataOnly ? 'REAL_DATA_ONLY activo' : 'Modo mixto permitido'}
        </span>
      </div>

      <p className="provenance-banner-text">{headline}</p>

      <div className="provenance-grid">
        {dataProvenance.items.map((item) => (
          <article key={item.key} className="provenance-card">
            <div className="provenance-card-head">
              <div className="provenance-card-title">
                <Activity size={14} aria-hidden="true" />
                <span>{item.label}</span>
              </div>
              <span className={`badge ${provenanceTone[item.state]}`}>
                {formatProvenanceState(item.state)}
              </span>
            </div>
            <div className="provenance-card-summary">{item.summary}</div>
            {item.detail && <div className="provenance-card-detail">{item.detail}</div>}
          </article>
        ))}
      </div>
    </section>
  )
}
