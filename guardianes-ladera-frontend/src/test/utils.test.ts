import { describe, it, expect } from 'vitest'
import {
  formatTime,
  formatNumber,
  formatMinutes,
  formatRelative,
  formatOptionalMetric,
  formatExplanationMode,
  RISK_LEVELS,
  DEFAULT_RISK_ORDER,
} from '../utils'

describe('formatTime', () => {
  it('returns --:-- for null', () => {
    expect(formatTime(null)).toBe('--:--')
  })

  it('returns --:-- for undefined', () => {
    expect(formatTime(undefined)).toBe('--:--')
  })

  it('formats a valid date', () => {
    const date = new Date('2026-03-28T14:30:00Z')
    const result = formatTime(date)
    expect(result).toMatch(/\d{1,2}:\d{2}/)
  })
})

describe('formatNumber', () => {
  it('formats a valid number', () => {
    const result = formatNumber(1234)
    expect(result).toBeTruthy()
    expect(result).not.toBe('No disponible')
  })

  it('returns No disponible for NaN', () => {
    expect(formatNumber(NaN)).toBe('No disponible')
  })

  it('returns No disponible for Infinity', () => {
    expect(formatNumber(Infinity)).toBe('No disponible')
  })

  it('formats zero', () => {
    expect(formatNumber(0)).not.toBe('No disponible')
  })
})

describe('formatMinutes', () => {
  it('shows minutes for values under 60', () => {
    expect(formatMinutes(30)).toBe('30 min')
  })

  it('shows hours for values 60-1439', () => {
    expect(formatMinutes(120)).toBe('2 h')
  })

  it('shows days for values >= 1440', () => {
    expect(formatMinutes(2880)).toBe('2 dias')
  })
})

describe('formatRelative', () => {
  it('returns sin dato for null', () => {
    expect(formatRelative(null)).toBe('sin dato')
  })

  it('returns hace 1 min for recent dates', () => {
    const recent = new Date(Date.now() - 30000)
    expect(formatRelative(recent)).toBe('hace 1 min')
  })
})

describe('formatOptionalMetric', () => {
  it('returns No disponible for null', () => {
    expect(formatOptionalMetric(null)).toBe('No disponible')
  })

  it('returns No disponible for undefined', () => {
    expect(formatOptionalMetric(undefined)).toBe('No disponible')
  })

  it('returns value with suffix', () => {
    expect(formatOptionalMetric(25.5, ' deg')).toBe('25.5 deg')
  })

  it('returns string value as-is', () => {
    expect(formatOptionalMetric('Arcilloso')).toBe('Arcilloso')
  })
})

describe('formatExplanationMode', () => {
  it('returns - when no explanation and not loading', () => {
    expect(formatExplanationMode(null, null, false)).toBe('-')
  })

  it('returns Cargando... when loading', () => {
    expect(formatExplanationMode(null, null, true)).toBe('Cargando...')
  })

  it('returns No disponible on error', () => {
    expect(formatExplanationMode(null, 'some error', false)).toBe('No disponible')
  })
})

describe('constants', () => {
  it('RISK_LEVELS has 4 levels in order', () => {
    expect(RISK_LEVELS).toEqual(['Verde', 'Amarillo', 'Naranja', 'Rojo'])
  })

  it('DEFAULT_RISK_ORDER maps correctly', () => {
    expect(DEFAULT_RISK_ORDER.Verde).toBe(0)
    expect(DEFAULT_RISK_ORDER.Rojo).toBe(3)
  })
})
