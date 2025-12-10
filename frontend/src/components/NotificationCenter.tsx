import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Bell, X, CheckCheck } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

export interface Notification {
  id: string
  title: string
  message: string
  type: 'info' | 'success' | 'warning' | 'error'
  timestamp: Date
  read: boolean
}

const mockNotifications: Notification[] = [
  {
    id: '1',
    title: 'Training Complete',
    message: 'CatBoost model finished training with 98.3% accuracy',
    type: 'success',
    timestamp: new Date(Date.now() - 5 * 60000),
    read: false,
  },
  {
    id: '2',
    title: 'Worker Connected',
    message: 'Worker-02 joined the mesh network',
    type: 'info',
    timestamp: new Date(Date.now() - 15 * 60000),
    read: false,
  },
  {
    id: '3',
    title: 'Dataset Uploaded',
    message: 'titanic.csv processed successfully (891 rows)',
    type: 'success',
    timestamp: new Date(Date.now() - 30 * 60000),
    read: true,
  },
  {
    id: '4',
    title: 'Low Memory Warning',
    message: 'Worker-01 memory usage at 85%',
    type: 'warning',
    timestamp: new Date(Date.now() - 60 * 60000),
    read: true,
  },
]

export function NotificationCenter() {
  const [isOpen, setIsOpen] = useState(false)
  const [notifications, setNotifications] = useState<Notification[]>(mockNotifications)

  const unreadCount = notifications.filter(n => !n.read).length

  const markAsRead = (id: string) => {
    setNotifications(prev =>
      prev.map(n => n.id === id ? { ...n, read: true } : n)
    )
  }

  const markAllAsRead = () => {
    setNotifications(prev => prev.map(n => ({ ...n, read: true })))
  }

  const removeNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id))
  }

  const getTypeColor = (type: Notification['type']) => {
    switch (type) {
      case 'success': return 'text-green-400 bg-green-600/10 border-green-600/30'
      case 'error': return 'text-red-400 bg-red-600/10 border-red-600/30'
      case 'warning': return 'text-yellow-400 bg-yellow-600/10 border-yellow-600/30'
      default: return 'text-blue-400 bg-blue-600/10 border-blue-600/30'
    }
  }

  const formatTime = (date: Date) => {
    const now = new Date()
    const diff = Math.floor((now.getTime() - date.getTime()) / 60000)
    if (diff < 1) return 'Just now'
    if (diff < 60) return `${diff}m ago`
    if (diff < 1440) return `${Math.floor(diff / 60)}h ago`
    return `${Math.floor(diff / 1440)}d ago`
  }

  return (
    <div className="relative">
      {/* Bell Button */}
      <Button
        onClick={() => setIsOpen(!isOpen)}
        className="relative"
      >
        <Bell className="w-5 h-5" />
        {unreadCount > 0 && (
          <motion.span
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="absolute -top-1 -right-1 w-5 h-5 bg-red-600 text-white text-xs font-bold rounded-full flex items-center justify-center"
          >
            {unreadCount}
          </motion.span>
        )}
      </Button>

      {/* Dropdown Panel */}
      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsOpen(false)}
              className="fixed inset-0 z-40"
            />

            {/* Panel */}
            <motion.div
              initial={{ opacity: 0, y: -10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              transition={{ duration: 0.2 }}
              className="absolute right-0 top-12 w-96 bg-zinc-900 border border-zinc-700 rounded-lg shadow-2xl z-50 overflow-hidden"
            >
              {/* Header */}
              <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-700 bg-zinc-800/50">
                <div>
                  <h3 className="font-semibold">Notifications</h3>
                  {unreadCount > 0 && (
                    <p className="text-xs text-zinc-500">{unreadCount} unread</p>
                  )}
                </div>
                {unreadCount > 0 && (
                  <Button
                    onClick={markAllAsRead}
                    className="text-xs"
                  >
                    <CheckCheck className="w-4 h-4 mr-1" />
                    Mark all read
                  </Button>
                )}
              </div>

              {/* Notifications List */}
              <div className="max-h-[500px] overflow-y-auto">
                {notifications.length === 0 ? (
                  <div className="px-4 py-12 text-center text-zinc-500">
                    <Bell className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No notifications</p>
                  </div>
                ) : (
                  <div className="divide-y divide-zinc-800">
                    {notifications.map((notif) => (
                      <motion.div
                        key={notif.id}
                        layout
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        onClick={() => !notif.read && markAsRead(notif.id)}
                        className={`px-4 py-3 cursor-pointer transition-colors hover:bg-zinc-800/50 ${
                          !notif.read ? 'bg-purple-600/5' : ''
                        }`}
                      >
                        <div className="flex gap-3">
                          {/* Type Indicator */}
                          <div className={`w-2 h-2 rounded-full mt-2 flex-shrink-0 ${
                            notif.type === 'success' ? 'bg-green-500' :
                            notif.type === 'error' ? 'bg-red-500' :
                            notif.type === 'warning' ? 'bg-yellow-500' :
                            'bg-blue-500'
                          }`} />

                          {/* Content */}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-start justify-between gap-2 mb-1">
                              <h4 className={`font-medium text-sm ${!notif.read ? 'text-white' : 'text-zinc-300'}`}>
                                {notif.title}
                              </h4>
                              <Button
                                onClick={(e) => {
                                  e.stopPropagation()
                                  removeNotification(notif.id)
                                }}
                                className="p-1 h-auto"
                              >
                                <X className="w-3 h-3" />
                              </Button>
                            </div>
                            <p className="text-xs text-zinc-500 mb-1">{notif.message}</p>
                            <div className="flex items-center gap-2">
                              <Badge variant="secondary" className={`text-xs ${getTypeColor(notif.type)}`}>
                                {notif.type}
                              </Badge>
                              <span className="text-xs text-zinc-600">{formatTime(notif.timestamp)}</span>
                            </div>
                          </div>

                          {/* Unread Indicator */}
                          {!notif.read && (
                            <div className="w-2 h-2 bg-purple-500 rounded-full mt-2 flex-shrink-0" />
                          )}
                        </div>
                      </motion.div>
                    ))}
                  </div>
                )}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  )
}
