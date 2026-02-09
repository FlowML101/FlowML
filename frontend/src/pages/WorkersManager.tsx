import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Server, Activity, Pause, RotateCcw, Loader2, Copy, Check, Wifi, Terminal } from 'lucide-react'
import { workersApi, clusterApi, TailscalePeer, JoinInstructions } from '@/lib/api'

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
  const [tailscalePeers, setTailscalePeers] = useState<TailscalePeer[]>([])
  const [joinCommand, setJoinCommand] = useState<JoinInstructions | null>(null)
  const [tailscaleEnabled, setTailscaleEnabled] = useState(false)
  const [loading, setLoading] = useState(true)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch all data in parallel
        const [master, remoteWorkers, tailscaleData, joinData] = await Promise.all([
          workersApi.getMaster(),
          workersApi.list().catch(() => [] as WorkerData[]),
          clusterApi.getTailscalePeers().catch(() => ({ enabled: false, peers: [] })),
          clusterApi.getJoinCommand().catch(() => null),
        ])
        
        // Master is always first, then merge any registered remote workers
        const allWorkers: WorkerData[] = [master as WorkerData]
        const registeredHostnames = new Set<string>([(master as any).hostname])
        
        // Remote workers come from the /workers endpoint (registered via CLI)
        for (const rw of remoteWorkers as any[]) {
          if (rw.worker_id === 'master' || rw.hostname === (master as any).hostname) continue
          
          registeredHostnames.add(rw.hostname)
          allWorkers.push({
            id: rw.worker_id || rw.id,
            hostname: rw.hostname,
            role: 'worker',
            status: rw.status || 'offline',
            ip: rw.ip_address || rw.ip || '',
            cpu_count: rw.cpu_count || 0,
            cpu_percent: 0,
            ram_total_gb: rw.total_ram_gb || 0,
            ram_used_gb: (rw.total_ram_gb || 0) - (rw.available_ram_gb || 0),
            ram_percent: rw.total_ram_gb ? Math.round(((rw.total_ram_gb - (rw.available_ram_gb || 0)) / rw.total_ram_gb) * 100) : 0,
            gpu_name: rw.gpu_names ? JSON.parse(rw.gpu_names)[0] : undefined,
            vram_total_gb: rw.total_vram_gb || undefined,
            uptime: rw.status === 'online' ? 'connected' : 'offline',
          })
        }
        
        setWorkers(allWorkers)
        setTailscaleEnabled(tailscaleData.enabled)
        setJoinCommand(joinData)
        
        // Filter Tailscale peers to only show ones not already registered
        if (tailscaleData.enabled && tailscaleData.peers) {
          const unregisteredPeers = tailscaleData.peers.filter(
            (p: TailscalePeer) => !registeredHostnames.has(p.hostname)
          )
          setTailscalePeers(unregisteredPeers)
        } else {
          setTailscalePeers([])
        }
      } catch (err) {
        console.error('Failed to fetch workers:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 5000)
    return () => clearInterval(interval)
  }, [])

  const copyCommand = () => {
    if (joinCommand?.command) {
      navigator.clipboard.writeText(joinCommand.command)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

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
          <Button size="sm" className="bg-gradient-to-r from-purple-600 to-purple-600 hover:from-purple-700 hover:to-violet-700">
            <Server className="w-4 h-4 mr-2" />
            Add Worker
          </Button>
        </div>
      </div>

      {/* Cluster Stats - Computed from real worker data */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-blue-500/15">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
            <CardTitle className="text-sm font-medium">Total Nodes</CardTitle>
            <Server className="w-4 h-4 text-zinc-400" />
          </CardHeader>
          <CardContent className="relative">
            {loading ? (
              <Loader2 className="w-5 h-5 animate-spin text-zinc-500" />
            ) : (
              <>
                <div className="text-2xl font-bold">{totalWorkers}</div>
                <p className="text-xs text-muted-foreground">{onlineWorkers} online, {totalWorkers - onlineWorkers} offline</p>
              </>
            )}
          </CardContent>
        </Card>

        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-blue-500/15">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
            <CardTitle className="text-sm font-medium">Total RAM</CardTitle>
            <Activity className="w-4 h-4 text-purple-400" />
          </CardHeader>
          <CardContent className="relative">
            {loading ? (
              <Loader2 className="w-5 h-5 animate-spin text-zinc-500" />
            ) : (
              <>
                <div className="text-2xl font-bold">
                  {workers.reduce((acc, w) => acc + (w.ram_total_gb || 0), 0).toFixed(1)} GB
                </div>
                <p className="text-xs text-muted-foreground">
                  {workers.reduce((acc, w) => acc + (w.ram_used_gb || 0), 0).toFixed(1)} GB in use
                </p>
              </>
            )}
          </CardContent>
        </Card>

        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-cyan-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-cyan-500/15">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
            <CardTitle className="text-sm font-medium">Avg. CPU Load</CardTitle>
            <Activity className="w-4 h-4 text-blue-400" />
          </CardHeader>
          <CardContent className="relative">
            {loading ? (
              <Loader2 className="w-5 h-5 animate-spin text-zinc-500" />
            ) : (
              <>
                <div className="text-2xl font-bold">
                  {onlineWorkers > 0 
                    ? (workers.filter(w => w.status === 'online').reduce((acc, w) => acc + (w.cpu_percent || 0), 0) / onlineWorkers).toFixed(0)
                    : 0}%
                </div>
                <p className="text-xs text-muted-foreground">Across active nodes</p>
              </>
            )}
          </CardContent>
        </Card>

        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-green-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-green-500/15">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
            <CardTitle className="text-sm font-medium">Total VRAM</CardTitle>
            <Activity className="w-4 h-4 text-green-400" />
          </CardHeader>
          <CardContent className="relative">
            {loading ? (
              <Loader2 className="w-5 h-5 animate-spin text-zinc-500" />
            ) : (
              <>
                <div className="text-2xl font-bold">
                  {workers.reduce((acc, w) => acc + (w.vram_total_gb || 0), 0).toFixed(1)} GB
                </div>
                <p className="text-xs text-muted-foreground">
                  {workers.some(w => w.gpu_name) ? workers.find(w => w.gpu_name)?.gpu_name : 'No GPU detected'}
                </p>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Workers Table */}
      <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-purple-500/10 before:via-purple-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-purple-500/12">
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

      {/* Tailscale Discoverable Devices */}
      {tailscaleEnabled && (
        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-cyan-500/10 before:via-transparent before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-cyan-500/12">
          <CardHeader className="relative">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Wifi className="w-5 h-5 text-cyan-500" />
                  Tailscale Network
                </CardTitle>
                <CardDescription>
                  Devices on your Tailscale mesh that can become workers
                </CardDescription>
              </div>
              <Badge variant="outline" className="text-cyan-400 border-cyan-500/30">
                <Wifi className="w-3 h-3 mr-1" />
                {tailscalePeers.length} discoverable
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4 relative">
            {/* Join Command */}
            {joinCommand && (
              <div className="p-4 rounded-lg bg-zinc-800/50 border border-zinc-700">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2 text-sm font-medium text-zinc-300">
                    <Terminal className="w-4 h-4" />
                    Run this on any Tailscale peer to join as worker:
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={copyCommand}
                    className="text-zinc-400 hover:text-cyan-400"
                  >
                    {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                  </Button>
                </div>
                <code className="block p-3 rounded bg-zinc-900 text-cyan-400 text-xs font-mono overflow-x-auto">
                  {joinCommand.command}
                </code>
              </div>
            )}

            {/* Peer List */}
            {tailscalePeers.length > 0 ? (
              <div className="space-y-2">
                <div className="text-sm font-medium text-zinc-400">Discoverable Devices</div>
                {tailscalePeers.map((peer) => (
                  <div key={peer.id} className="flex items-center justify-between p-3 rounded-lg bg-zinc-800/30">
                    <div className="flex items-center gap-3">
                      <div className={`w-8 h-8 rounded flex items-center justify-center ${peer.online ? 'bg-cyan-600/20' : 'bg-zinc-700/50'}`}>
                        <Server className={`w-4 h-4 ${peer.online ? 'text-cyan-400' : 'text-zinc-500'}`} />
                      </div>
                      <div>
                        <div className="text-sm font-medium">{peer.hostname}</div>
                        <div className="text-xs text-muted-foreground font-mono">
                          {peer.tailscale_ip} • {peer.os || 'unknown'}
                        </div>
                      </div>
                    </div>
                    <Badge variant={peer.online ? 'success' : 'secondary'} className="text-xs">
                      {peer.online ? 'reachable' : 'offline'}
                    </Badge>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-4 rounded-lg bg-zinc-800/30 text-center text-zinc-500 text-sm">
                All Tailscale peers are already registered as workers, or no peers found.
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* No Tailscale - Show instructions */}
      {!tailscaleEnabled && !loading && (
        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-amber-500/10 before:via-transparent before:to-transparent before:opacity-30">
          <CardHeader className="relative">
            <CardTitle className="flex items-center gap-2">
              <Wifi className="w-5 h-5 text-amber-500" />
              Enable Distributed Training
            </CardTitle>
            <CardDescription>
              Connect multiple devices using Tailscale for distributed compute
            </CardDescription>
          </CardHeader>
          <CardContent className="relative">
            <div className="space-y-3 text-sm text-zinc-400">
              <p>1. Install Tailscale on all devices: <a href="https://tailscale.com/download" target="_blank" className="text-cyan-400 hover:underline">tailscale.com/download</a></p>
              <p>2. Run <code className="px-2 py-1 rounded bg-zinc-800 text-cyan-400">tailscale up</code> on each device</p>
              <p>3. All devices on the same tailnet can share compute load automatically</p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
