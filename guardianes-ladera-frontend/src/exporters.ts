import type { Confidence } from './api'
import { formatTime } from './utils'

type ZoneReportContext = {
  zoneId: string
  zoneName: string
  confidence: Confidence
  lastRun: Date | null
}

export async function exportDashboardPng(node: HTMLElement) {
  const { default: html2canvas } = await import('html2canvas')
  const canvas = await html2canvas(node, {
    useCORS: true,
    backgroundColor: '#f6f8fb',
    scale: 2,
  })
  const link = document.createElement('a')
  link.download = `guardianes-ladera-${Date.now()}.png`
  link.href = canvas.toDataURL('image/png')
  link.click()
}

export async function exportZoneReportPdf(
  node: HTMLElement,
  context: ZoneReportContext,
) {
  const [{ default: html2canvas }, { jsPDF }] = await Promise.all([
    import('html2canvas'),
    import('jspdf'),
  ])
  const canvas = await html2canvas(node, {
    useCORS: true,
    backgroundColor: '#ffffff',
    scale: 2,
  })
  const imgData = canvas.toDataURL('image/png')
  const pdf = new jsPDF({ orientation: 'p', unit: 'pt', format: 'a4' })
  const pageWidth = pdf.internal.pageSize.getWidth()
  const margin = 32
  pdf.setFont('helvetica', 'bold')
  pdf.setFontSize(16)
  pdf.text(`Reporte de zona: ${context.zoneName}`, margin, 32)
  pdf.setFont('helvetica', 'normal')
  pdf.setFontSize(11)
  pdf.text(
    `Ultima corrida: ${formatTime(context.lastRun)} | Confianza: ${context.confidence}`,
    margin,
    52,
  )
  const imgWidth = pageWidth - margin * 2
  const imgHeight = (canvas.height * imgWidth) / canvas.width
  pdf.addImage(imgData, 'PNG', margin, 68, imgWidth, imgHeight)
  pdf.save(`reporte-${context.zoneId}.pdf`)
}
