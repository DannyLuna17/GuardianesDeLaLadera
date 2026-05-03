import { useCallback, useEffect, useRef, useState } from 'react'
import {
  fetchDashboardBootstrap,
  fetchZoneExplanation,
  type DashboardBootstrap,
  type ZoneExplanation,
} from './api'

const BACKGROUND_REFRESH_INTERVAL_MS = 60_000

const isAbortError = (error: unknown) =>
  error instanceof DOMException && error.name === 'AbortError'

export function useDashboardBootstrap() {
  const [dashboard, setDashboard] = useState<DashboardBootstrap | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const activeRefreshController = useRef<AbortController | null>(null)

  const load = useCallback(
    async (options?: { background?: boolean; signal?: AbortSignal }) => {
      const background = options?.background ?? false

      if (background) {
        // Abort any in-flight background refresh to prevent out-of-order responses
        activeRefreshController.current?.abort()
        const controller = new AbortController()
        activeRefreshController.current = controller
        setIsRefreshing(true)

        try {
          const payload = await fetchDashboardBootstrap(controller.signal)
          setDashboard(payload)
          setError(null)
        } catch (loadError) {
          if (isAbortError(loadError)) return
          setError(
            loadError instanceof Error
              ? loadError.message
              : 'No se pudo cargar el dashboard desde el backend.',
          )
        } finally {
          if (!controller.signal.aborted) {
            setIsRefreshing(false)
            activeRefreshController.current = null
          }
        }
      } else {
        setIsLoading(true)
        try {
          const payload = await fetchDashboardBootstrap(options?.signal)
          setDashboard(payload)
          setError(null)
        } catch (loadError) {
          if (isAbortError(loadError)) return
          setError(
            loadError instanceof Error
              ? loadError.message
              : 'No se pudo cargar el dashboard desde el backend.',
          )
        } finally {
          setIsLoading(false)
        }
      }
    },
    [],
  )

  useEffect(() => {
    const controller = new AbortController()
    void load({ signal: controller.signal })
    return () => controller.abort()
  }, [load])

  useEffect(() => {
    const interval = window.setInterval(() => {
      void load({ background: true })
    }, BACKGROUND_REFRESH_INTERVAL_MS)

    return () => {
      window.clearInterval(interval)
      activeRefreshController.current?.abort()
    }
  }, [load])

  return {
    dashboard,
    isLoading,
    error,
    isRefreshing,
    reload: () => void load(),
  }
}

export function useZoneExplanation(zoneId: string | null, runId: number | null) {
  const [explanation, setExplanation] = useState<ZoneExplanation | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!zoneId) {
      setExplanation(null)
      setError(null)
      setIsLoading(false)
      return
    }

    const controller = new AbortController()
    setExplanation(null)
    setError(null)
    setIsLoading(true)

    void fetchZoneExplanation(zoneId, controller.signal)
      .then((payload) => {
        setExplanation(payload)
      })
      .catch((loadError) => {
        if (isAbortError(loadError)) return
        setError(
          loadError instanceof Error
            ? loadError.message
            : 'No se pudo cargar la explicacion de la zona.',
        )
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setIsLoading(false)
        }
      })

    return () => controller.abort()
  }, [runId, zoneId])

  return { explanation, isLoading, error }
}
