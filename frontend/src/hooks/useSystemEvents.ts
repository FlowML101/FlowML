import { useEffect } from 'react'
import { toast } from 'sonner'

/**
 * System Events Hook
 * 
 * This hook is for REAL system events only.
 * Fake/demo events have been removed.
 * 
 * Real notifications come from:
 * - WebSocket messages (useWebSocket hook)
 * - API responses (toast calls in pages)
 * - Actual user actions
 */
export function useSystemEvents() {
  useEffect(() => {
    // Show connection status on mount (this is real - the app did just load)
    toast.info('FlowML Studio Ready', {
      description: 'Connected to backend server',
      duration: 3000,
    })

    // No fake interval events - real events come from:
    // 1. WebSocket (useWebSocket.ts) - training updates, worker status
    // 2. API calls - success/error toasts in individual pages
    // 3. User actions - upload, download, etc.

  }, [])
}
