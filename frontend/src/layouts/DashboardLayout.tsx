import { Link, Outlet, useLocation } from 'react-router-dom'
import { 
  LayoutDashboard, 
  Database, 
  Zap, 
  Activity, 
  Server, 
  Trophy,
  FlaskConical,
  ScrollText,
  ChevronRight,
  Bell,
  Settings,
  User
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { useState } from 'react'
import { cn } from '@/lib/utils'

const navigation = [
  { name: 'Dashboard', href: '/app', icon: LayoutDashboard },
  { name: 'Data Studio', href: '/app/data', icon: Database },
  { name: 'Training', href: '/app/train', icon: Zap },
  { name: 'Live Monitor', href: '/app/running', icon: Activity },
  { name: 'Workers', href: '/app/workers', icon: Server },
  { name: 'Results', href: '/app/results', icon: Trophy },
  { name: 'Model Lab', href: '/app/inference', icon: FlaskConical },
  { name: 'Logs', href: '/app/logs', icon: ScrollText },
]

export function DashboardLayout() {
  const location = useLocation()
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // Generate breadcrumbs
  const pathSegments = location.pathname.split('/').filter(Boolean)
  const breadcrumbs = pathSegments.map((segment, index) => ({
    name: segment === 'app' ? 'Dashboard' : segment.charAt(0).toUpperCase() + segment.slice(1),
    href: '/' + pathSegments.slice(0, index + 1).join('/')
  }))

  return (
    <div className="min-h-screen bg-zinc-950">
      {/* Sidebar */}
      <aside
        className={cn(
          "fixed left-0 top-0 bottom-0 z-40 border-r border-zinc-800 bg-zinc-900/50 backdrop-blur-xl transition-all duration-300",
          sidebarOpen ? "w-64" : "w-16"
        )}
      >
        {/* Logo */}
        <div className="h-16 border-b border-zinc-800 flex items-center px-4 gap-3">
          <div className="w-8 h-8 bg-gradient-to-br from-purple-600 to-blue-600 rounded-lg flex items-center justify-center flex-shrink-0">
            <Activity className="w-5 h-5 text-white" />
          </div>
          {sidebarOpen && (
            <span className="font-bold text-lg bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
              FlowML
            </span>
          )}
        </div>

        {/* Navigation */}
        <nav className="p-3 space-y-1">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href
            return (
              <Link
                key={item.name}
                to={item.href}
                className={cn(
                  "flex items-center gap-3 px-3 py-2 rounded-lg transition-colors group",
                  isActive
                    ? "bg-purple-600/20 text-purple-400"
                    : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-50"
                )}
              >
                <item.icon className={cn("w-5 h-5 flex-shrink-0", isActive && "text-purple-400")} />
                {sidebarOpen && (
                  <span className="text-sm font-medium">{item.name}</span>
                )}
              </Link>
            )
          })}
        </nav>

        {/* Cluster Status */}
        {sidebarOpen && (
          <div className="absolute bottom-4 left-3 right-3 p-3 rounded-lg bg-zinc-800/50 border border-zinc-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-zinc-400">Cluster Status</span>
              <Badge variant="success" className="text-xs">Online</Badge>
            </div>
            <div className="text-xs text-zinc-500">
              <div className="flex justify-between">
                <span>Workers</span>
                <span className="text-zinc-300">2 active</span>
              </div>
            </div>
          </div>
        )}

        {/* Toggle Button */}
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="absolute -right-3 top-20 w-6 h-6 rounded-full bg-zinc-800 border border-zinc-700 flex items-center justify-center hover:bg-zinc-700 transition-colors"
        >
          <ChevronRight className={cn("w-4 h-4 transition-transform", sidebarOpen && "rotate-180")} />
        </button>
      </aside>

      {/* Main Content Area */}
      <div className={cn("transition-all duration-300", sidebarOpen ? "ml-64" : "ml-16")}>
        {/* Top Bar */}
        <header className="h-16 border-b border-zinc-800 bg-zinc-900/30 backdrop-blur-xl flex items-center justify-between px-6 sticky top-0 z-30">
          {/* Breadcrumbs */}
          <div className="flex items-center gap-2 text-sm">
            {breadcrumbs.map((crumb, index) => (
              <div key={crumb.href} className="flex items-center gap-2">
                {index > 0 && <ChevronRight className="w-4 h-4 text-zinc-600" />}
                <Link
                  to={crumb.href}
                  className={cn(
                    "hover:text-purple-400 transition-colors",
                    index === breadcrumbs.length - 1 ? "text-zinc-50" : "text-zinc-500"
                  )}
                >
                  {crumb.name}
                </Link>
              </div>
            ))}
          </div>

          {/* Right Actions */}
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" className="relative">
              <Bell className="w-5 h-5" />
              <span className="absolute top-1 right-1 w-2 h-2 bg-purple-600 rounded-full"></span>
            </Button>
            <Button variant="ghost" size="icon">
              <Settings className="w-5 h-5" />
            </Button>
            <Button variant="ghost" size="icon">
              <User className="w-5 h-5" />
            </Button>
          </div>
        </header>

        {/* Page Content */}
        <main className="p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
