import { ClusterHealth } from '@/features/dashboard/ClusterHealth'
import { ResourceGauges } from '@/features/dashboard/ResourceGauges'
import { ActiveJobs } from '@/features/dashboard/ActiveJobs'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { TrendingUp, Zap, Database, CheckCircle, LayoutDashboard } from 'lucide-react'

export function DashboardHome() {
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
        <Badge variant="outline" className="text-sm px-4 py-2 flex items-center gap-2">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          All Systems Operational
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
            <div className="text-2xl font-bold">47</div>
            <p className="text-xs text-muted-foreground">+3 this week</p>
          </CardContent>
        </Card>

        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-yellow-500/10 before:to-transparent before:opacity-50 transition-all duration-300 hover:shadow-md hover:shadow-yellow-500/15">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
            <CardTitle className="text-sm font-medium">Active Jobs</CardTitle>
            <Zap className="w-4 h-4 text-yellow-500" />
          </CardHeader>
          <CardContent className="relative">
            <div className="text-2xl font-bold">2</div>
            <p className="text-xs text-muted-foreground">1 running, 1 queued</p>
          </CardContent>
        </Card>

        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/10 before:to-transparent before:opacity-50 transition-all duration-300 hover:shadow-md hover:shadow-blue-500/15">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
            <CardTitle className="text-sm font-medium">Datasets</CardTitle>
            <Database className="w-4 h-4 text-blue-500" />
          </CardHeader>
          <CardContent className="relative">
            <div className="text-2xl font-bold">12</div>
            <p className="text-xs text-muted-foreground">856K total rows</p>
          </CardContent>
        </Card>

        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-purple-500/10 before:to-transparent before:opacity-50 transition-all duration-300 hover:shadow-md hover:shadow-purple-500/15">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
            <CardTitle className="text-sm font-medium">Avg. Accuracy</CardTitle>
            <TrendingUp className="w-4 h-4 text-purple-500" />
          </CardHeader>
          <CardContent className="relative">
            <div className="text-2xl font-bold">94.2%</div>
            <p className="text-xs text-muted-foreground">+2.1% from last month</p>
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
