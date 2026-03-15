/**
 * hooks/useHealth.js
 * ===================
 * Polls the health endpoint on mount and every 30s.
 * Updates the store with server/model status.
 */

import { useEffect } from 'react'
import { fetchHealth } from '@/api/client'
import { useStore } from '@/hooks/useStore'

export function useHealth() {
  const setServerStatus = useStore((s) => s.setServerStatus)

  useEffect(() => {
    const check = async () => {
      try {
        const status = await fetchHealth()
        setServerStatus(status)
      } catch {
        setServerStatus({ model_loaded: false, device: 'offline' })
      }
    }

    check()
    const interval = setInterval(check, 30_000)
    return () => clearInterval(interval)
  }, [setServerStatus])
}
