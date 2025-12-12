import { useEffect, useRef, useState, useCallback } from 'react'
import { toast } from 'sonner'

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error'

export interface WebSocketMessage {
  type: 'job_update' | 'worker_status' | 'log_stream' | 'metric_update' | 'system_event'
  payload: any
  timestamp: string
}

interface UseWebSocketOptions {
  url?: string
  reconnectInterval?: number
  maxReconnectAttempts?: number
  onMessage?: (message: WebSocketMessage) => void
  onConnect?: () => void
  onDisconnect?: () => void
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    url = 'ws://localhost:8000/ws',
    reconnectInterval = 3000,
    maxReconnectAttempts = 5,
    onMessage,
    onConnect,
    onDisconnect,
  } = options

  const [status, setStatus] = useState<ConnectionStatus>('disconnected')
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const reconnectTimeoutRef = useRef<number>()

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    setStatus('connecting')

    try {
      const ws = new WebSocket(url)

      ws.onopen = () => {
        setStatus('connected')
        reconnectAttemptsRef.current = 0
        onConnect?.()
        
        // Send initial handshake
        ws.send(JSON.stringify({
          type: 'handshake',
          payload: { client: 'flowml-studio', version: '1.0.0' }
        }))
      }

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          setLastMessage(message)
          onMessage?.(message)

          // Handle different message types
          switch (message.type) {
            case 'job_update':
              if (message.payload.status === 'completed') {
                toast.success('Training Complete', {
                  description: `${message.payload.jobName} finished with ${message.payload.accuracy}% accuracy`
                })
              }
              break
            case 'worker_status':
              if (message.payload.status === 'online') {
                toast.info('Worker Connected', {
                  description: `${message.payload.workerName} joined the cluster`
                })
              }
              break
            case 'system_event':
              const level = message.payload.level || 'info'
              if (level === 'success') toast.success(message.payload.title, { description: message.payload.description })
              else if (level === 'error') toast.error(message.payload.title, { description: message.payload.description })
              else if (level === 'warning') toast.warning(message.payload.title, { description: message.payload.description })
              else toast.info(message.payload.title, { description: message.payload.description })
              break
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        setStatus('error')
      }

      ws.onclose = () => {
        setStatus('disconnected')
        onDisconnect?.()
        wsRef.current = null

        // Attempt reconnection
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, reconnectInterval)
        }
      }

      wsRef.current = ws
    } catch (error) {
      console.error('Failed to connect WebSocket:', error)
      setStatus('error')
    }
  }, [url, reconnectInterval, maxReconnectAttempts, onConnect, onDisconnect, onMessage])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setStatus('disconnected')
  }, [])

  const sendMessage = useCallback((message: Partial<WebSocketMessage>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        ...message,
        timestamp: new Date().toISOString()
      }))
      return true
    }
    return false
  }, [])

  useEffect(() => {
    connect()
    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  return {
    status,
    lastMessage,
    sendMessage,
    connect,
    disconnect,
    isConnected: status === 'connected'
  }
}
