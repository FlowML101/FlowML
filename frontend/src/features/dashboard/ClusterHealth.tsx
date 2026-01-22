import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Activity, Server, Cpu, CheckCircle2, Loader2, AlertCircle } from 'lucide-react'
import { workersApi, statsApi } from '@/lib/api'

interface ClusterData {
  total_nodes: number
  online: number
  offline: number
  busy: number
  master: {
    hostname: string
    status: string
    cpu_percent: number
    ram_percent: number
  }
  workers: Array<{
    id: string
    hostname: string
    status: string
    cpu_percent: number
    ram_percent: number
  }>
}

export function ClusterHealth() {
  const [cluster, setCluster] = useState<ClusterData | null>(null)
  const [masterInfo, setMasterInfo] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchClusterData = async () => {
      try {
        const [clusterData, master] = await Promise.all([
          statsApi.getClusterHealth() as Promise<ClusterData>,
          workersApi.getMaster()
        ])
        setCluster(clusterData)
        setMasterInfo(master)
        setError(null)
      } catch (err) {
        console.error('Failed to fetch cluster data:', err)
        setError('Failed to connect to backend')
      } finally {
        setLoading(false)
      }
    }

    fetchClusterData()
    const interval = setInterval(fetchClusterData, 10000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50">
        <CardContent className="flex items-center justify-center h-64">
          <Loader2 className="w-8 h-8 animate-spin text-zinc-500" />
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50">
        <CardContent className="flex flex-col items-center justify-center h-64 text-zinc-500">
          <AlertCircle className="w-8 h-8 mb-2 text-red-500" />
          <div>{error}</div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-purple-500/10 before:via-transparent before:to-blue-500/10 before:opacity-30 transition-all duration-300 hover:border-purple-500/30">
      <CardHeader className="relative">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-purple-500" />
              Cluster Status
            </CardTitle>
            <CardDescription>Mesh network health and connectivity</CardDescription>
          </div>
          <Badge variant="success" className="flex items-center gap-1">
            <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
            {cluster?.online || 0} Online
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 relative">
        {/* Master Node */}
        <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-purple-600/20 flex items-center justify-center">
              <Server className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <div className="font-medium">{masterInfo?.hostname || 'Master Node'}</div>
              <div className="text-xs text-muted-foreground">
                {masterInfo?.ip || 'localhost'}:8000 • CPU: {cluster?.master?.cpu_percent?.toFixed(0) || 0}%
              </div>
            </div>
          </div>
          <CheckCircle2 className="w-5 h-5 text-green-500" />
        </div>

        {/* Workers */}
        <div className="space-y-2">
          <div className="text-sm font-medium text-zinc-400 flex items-center justify-between">
            <span>Connected Workers</span>
            <span className="text-purple-400">{cluster?.workers?.length || 0} Active</span>
          </div>
          
          {cluster?.workers && cluster.workers.length > 0 ? (
            cluster.workers.map((worker) => (
              <div key={worker.id} className="flex items-center justify-between p-3 rounded-lg bg-muted/30 dark:bg-zinc-800/30">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded bg-blue-600/20 flex items-center justify-center">
                    <Cpu className="w-4 h-4 text-blue-400" />
                  </div>
                  <div>
                    <div className="text-sm font-medium">{worker.hostname}</div>
                    <div className="text-xs text-muted-foreground">
                      CPU: {worker.cpu_percent?.toFixed(0) || 0}% • RAM: {worker.ram_percent?.toFixed(0) || 0}%
                    </div>
                  </div>
                </div>
                <Badge variant={worker.status === 'online' || worker.status === 'busy' ? 'success' : 'secondary'} className="text-xs">
                  {worker.status}
                </Badge>
              </div>
            ))
          ) : (
            <div className="p-4 rounded-lg bg-muted/30 dark:bg-zinc-800/30 text-center text-zinc-500 text-sm">
              No workers connected yet. Start a worker to join the cluster.
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
