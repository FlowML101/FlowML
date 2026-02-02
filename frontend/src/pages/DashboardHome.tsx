import { useState, useEffect } from 'react'
import { ClusterHealth } from '@/features/dashboard/ClusterHealth'
import { ResourceGauges } from '@/features/dashboard/ResourceGauges'
import { ActiveJobs } from '@/features/dashboard/ActiveJobs'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { TrendingUp, Zap, Database, CheckCircle, LayoutDashboard, Loader2 } from 'lucide-react'
import { statsApi, DashboardStats } from '@/lib/api'

export function DashboardHome() {
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [connected, setConnected] = useState(true)

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const data = await statsApi.getDashboard()
        setStats(data)
        setConnected(true)
      } catch (err) {
        console.error('Failed to fetch stats:', err)
        setConnected(false)
      } finally {
        setLoading(false)
      }
    }

    fetchStats()
    const interval = setInterval(fetchStats, 10000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <LayoutDashboard className="w-8 h-8 text-purple-500" />
            Dashboard Overview
          </h1>
          <p className="text-muted-foreground">Real-time monitoring of your distributed AutoML cluster performance</p>
        </div>
        <Badge variant="outline" className={`text-sm px-4 py-2 flex items-center gap-2 ${connected ? '' : 'border-red-500 text-red-500'}`}>
          <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
          {connected ? 'All Systems Operational' : 'Backend Disconnected'}
        </Badge>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-green-500/10 before:to-transparent before:opacity-50 transition-all duration-300 hover:shadow-md hover:shadow-green-500/15">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
            <CardTitle className="text-sm font-medium">Total Models</CardTitle>
            <CheckCircle className="w-4 h-4 text-green-500" />
          </CardHeader>
          <CardContent className="relative">
            {loading ? (
              <Loader2 className="w-6 h-6 animate-spin text-zinc-500" />
            ) : (
              <>
                <div className="text-2xl font-bold">{stats?.total_models || 0}</div>
                <p className="text-xs text-muted-foreground">+{stats?.models_this_week || 0} this week</p>
              </>
            )}
          </CardContent>
        </Card>

        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-yellow-500/10 before:to-transparent before:opacity-50 transition-all duration-300 hover:shadow-md hover:shadow-yellow-500/15">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
            <CardTitle className="text-sm font-medium">Active Jobs</CardTitle>
            <Zap className="w-4 h-4 text-yellow-500" />
          </CardHeader>
          <CardContent className="relative">
            {loading ? (
              <Loader2 className="w-6 h-6 animate-spin text-zinc-500" />
            ) : (
              <>
                <div className="text-2xl font-bold">{stats?.active_jobs || 0}</div>
                <p className="text-xs text-muted-foreground">{stats?.jobs_running || 0} running, {stats?.jobs_queued || 0} queued</p>
              </>
            )}
          </CardContent>
        </Card>

        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/10 before:to-transparent before:opacity-50 transition-all duration-300 hover:shadow-md hover:shadow-blue-500/15">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
            <CardTitle className="text-sm font-medium">Datasets</CardTitle>
            <Database className="w-4 h-4 text-blue-500" />
          </CardHeader>
          <CardContent className="relative">
            {loading ? (
              <Loader2 className="w-6 h-6 animate-spin text-zinc-500" />
            ) : (
              <>
                <div className="text-2xl font-bold">{stats?.total_datasets || 0}</div>
                <p className="text-xs text-muted-foreground">Ready for training</p>
              </>
            )}
          </CardContent>
        </Card>

        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-purple-500/10 before:to-transparent before:opacity-50 transition-all duration-300 hover:shadow-md hover:shadow-purple-500/15">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
            <CardTitle className="text-sm font-medium">Avg. Accuracy</CardTitle>
            <TrendingUp className="w-4 h-4 text-purple-500" />
          </CardHeader>
          <CardContent className="relative">
            {loading ? (
              <Loader2 className="w-6 h-6 animate-spin text-zinc-500" />
            ) : (
              <>
                <div className="text-2xl font-bold">
                  {stats?.avg_accuracy ? `${(stats.avg_accuracy * 100).toFixed(1)}%` : 'N/A'}
                </div>
                <p className="text-xs text-muted-foreground">Best model accuracy</p>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Resource Gauges */}
      <ResourceGauges />

      {/* Two Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ClusterHealth />
        <ActiveJobs />
      </div>
    </div>
  )
}
