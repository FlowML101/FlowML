import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Server, Activity, Pause, RotateCcw, Loader2 } from 'lucide-react'
import { workersApi, statsApi } from '@/lib/api'

interface WorkerData {
  id: string
  hostname: string
  role: string
  status: string
  ip: string
  cpu_count: number
  cpu_percent: number
  ram_total_gb: number
  ram_used_gb: number
  ram_percent: number
  gpu_name?: string
  vram_total_gb?: number
  vram_used_gb?: number
  vram_percent?: number
  uptime: string
}

export function WorkersManager() {
  const [workers, setWorkers] = useState<WorkerData[]>([])
  const [loading, setLoading] = useState(true)
  const [stats, setStats] = useState<any>(null)

  useEffect(() => {
    const fetchWorkers = async () => {
      try {
        const [master, available] = await Promise.all([
          workersApi.getMaster(),
          workersApi.getAvailable()
        ])
        
        // Master is always first
        const allWorkers: WorkerData[] = [master as WorkerData]
        setWorkers(allWorkers)
        setStats(available)
      } catch (err) {
        console.error('Failed to fetch workers:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchWorkers()
    const interval = setInterval(fetchWorkers, 5000)
    return () => clearInterval(interval)
  }, [])

  const onlineWorkers = workers.filter(w => w.status === 'online').length
  const totalWorkers = workers.length

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <Server className="w-8 h-8 text-blue-500" />
            Workers Manager
          </h1>
          <p className="text-muted-foreground">Manage and monitor your distributed compute cluster resources</p>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="text-sm px-4 py-2">
            {onlineWorkers}/{totalWorkers} online
          </Badge>
          <Button size="sm" className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700">
            <Server className="w-4 h-4 mr-2" />
            Add Worker
          </Button>
        </div>
      </div>

      {/* Cluster Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-blue-500/15">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
            <CardTitle className="text-sm font-medium">Total Nodes</CardTitle>
            <Server className="w-4 h-4 text-zinc-400" />
          </CardHeader>
          <CardContent className="relative">
            <div className="text-2xl font-bold">4</div>
            <p className="text-xs text-muted-foreground">3 online, 1 offline</p>
          </CardContent>
        </Card>

        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-blue-500/15">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
            <CardTitle className="text-sm font-medium">Total VRAM</CardTitle>
            <Activity className="w-4 h-4 text-purple-400" />
          </CardHeader>
          <CardContent className="relative">
            <div className="text-2xl font-bold">18 GB</div>
            <p className="text-xs text-muted-foreground">7.1 GB in use</p>
          </CardContent>
        </Card>

        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-cyan-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-cyan-500/15">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
            <CardTitle className="text-sm font-medium">Avg. CPU Load</CardTitle>
            <Activity className="w-4 h-4 text-blue-400" />
          </CardHeader>
          <CardContent className="relative">
            <div className="text-2xl font-bold">49%</div>
            <p className="text-xs text-muted-foreground">Across active nodes</p>
          </CardContent>
        </Card>

        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-green-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-green-500/15">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
            <CardTitle className="text-sm font-medium">Network Latency</CardTitle>
            <Activity className="w-4 h-4 text-green-400" />
          </CardHeader>
          <CardContent className="relative">
            <div className="text-2xl font-bold">12ms</div>
            <p className="text-xs text-muted-foreground">Mesh average</p>
          </CardContent>
        </Card>
      </div>

      {/* Workers Table */}
      <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/10 before:via-purple-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-blue-500/12">
        <CardHeader className="relative">
          <CardTitle>Cluster Nodes</CardTitle>
          <CardDescription>Manage and monitor all workers in your mesh</CardDescription>
        </CardHeader>
        <CardContent className="relative">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 animate-spin text-zinc-500" />
            </div>
          ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Hostname</TableHead>
                <TableHead>Role</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>IP Address</TableHead>
                <TableHead>RAM Usage</TableHead>
                <TableHead>CPU Load</TableHead>
                <TableHead>Uptime</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {workers.map((worker) => (
                <TableRow key={worker.id}>
                  <TableCell className="font-medium">{worker.hostname}</TableCell>
                  <TableCell>
                    <Badge variant={worker.role === 'orchestrator' ? 'default' : 'secondary'}>
                      {worker.role === 'orchestrator' ? 'Master' : 'Worker'}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant={worker.status === 'online' ? 'success' : 'secondary'}
                      className="capitalize"
                    >
                      {worker.status === 'online' && (
                        <span className="w-2 h-2 bg-green-500 rounded-full mr-1 animate-pulse"></span>
                      )}
                      {worker.status}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-muted-foreground font-mono text-xs">
                    {worker.ip}
                  </TableCell>
                  <TableCell>
                    {worker.status === 'online' ? (
                      <div className="space-y-1">
                        <div className="text-sm">{worker.ram_used_gb?.toFixed(1)}/{worker.ram_total_gb?.toFixed(1)} GB</div>
                        <div className="w-24 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-blue-600 transition-all"
                            style={{ width: `${worker.ram_percent || 0}%` }}
                          />
                        </div>
                      </div>
                    ) : (
                      <span className="text-zinc-600">-</span>
                    )}
                  </TableCell>
                  <TableCell className="text-muted-foreground">{worker.cpu_percent?.toFixed(0)}%</TableCell>
                  <TableCell className="text-muted-foreground text-xs">{worker.uptime}</TableCell>
                  <TableCell className="text-right">
                    {worker.status === 'online' && worker.role !== 'orchestrator' && (
                      <div className="flex gap-2 justify-end">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-zinc-400 hover:text-yellow-400"
                        >
                          <Pause className="w-4 h-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-zinc-400 hover:text-blue-400"
                        >
                          <RotateCcw className="w-4 h-4" />
                        </Button>
                      </div>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
