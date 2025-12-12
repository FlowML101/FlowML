import { Wifi, WifiOff, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { ConnectionStatus as Status } from '@/hooks/useWebSocket'

interface ConnectionStatusProps {
  status: Status
  className?: string
}

export function ConnectionStatus({ status, className }: ConnectionStatusProps) {
  return (
    <div className={cn("flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm", className)}>
      {status === 'connected' && (
        <>
          <Wifi className="w-4 h-4 text-green-500" />
          <span className="text-green-500 font-medium">Live</span>
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
        </>
      )}
      {status === 'connecting' && (
        <>
          <Loader2 className="w-4 h-4 text-yellow-500 animate-spin" />
          <span className="text-yellow-500">Connecting...</span>
        </>
      )}
      {status === 'disconnected' && (
        <>
          <WifiOff className="w-4 h-4 text-zinc-500" />
          <span className="text-zinc-500">Offline</span>
        </>
      )}
      {status === 'error' && (
        <>
          <WifiOff className="w-4 h-4 text-red-500" />
          <span className="text-red-500">Error</span>
        </>
      )}
    </div>
  )
}
