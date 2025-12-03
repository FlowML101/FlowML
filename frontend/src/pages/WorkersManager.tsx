import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Server, Activity, Pause, RotateCcw } from 'lucide-react'

const mockWorkers = [
  {
    id: '1',
    hostname: 'master-node',
    role: 'Master',
    status: 'online',
    ip: '192.168.1.100',
    vramUsage: '2.1/4.0 GB',
    vramPercent: 52,
    cpuUsage: '45%',
    uptime: '3d 14h',
  },
  {
    id: '2',
    hostname: 'worker-laptop-01',
    role: 'Worker',
    status: 'online',
    ip: '192.168.1.105',
    vramUsage: '3.2/6.0 GB',
    vramPercent: 53,
    cpuUsage: '68%',
    uptime: '2d 8h',
  },
  {
    id: '3',
    hostname: 'worker-desktop-02',
    role: 'Worker',
    status: 'online',
    ip: '192.168.1.142',
    vramUsage: '1.8/8.0 GB',
    vramPercent: 22,
    cpuUsage: '34%',
    uptime: '1d 16h',
  },
  {
    id: '4',
    hostname: 'worker-laptop-03',
    role: 'Worker',
    status: 'offline',
    ip: '192.168.1.156',
    vramUsage: '-',
    vramPercent: 0,
    cpuUsage: '-',
    uptime: '-',
  },
]

export function WorkersManager() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
          <Server className="w-8 h-8 text-blue-500" />
          Workers Manager
        </h1>
        <p className="text-zinc-400">Manage your distributed compute cluster</p>
      </div>

      {/* Cluster Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Nodes</CardTitle>
            <Server className="w-4 h-4 text-zinc-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">4</div>
            <p className="text-xs text-zinc-500">3 online, 1 offline</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total VRAM</CardTitle>
            <Activity className="w-4 h-4 text-purple-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">18 GB</div>
            <p className="text-xs text-zinc-500">7.1 GB in use</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg. CPU Load</CardTitle>
            <Activity className="w-4 h-4 text-blue-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">49%</div>
            <p className="text-xs text-zinc-500">Across active nodes</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Network Latency</CardTitle>
            <Activity className="w-4 h-4 text-green-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12ms</div>
            <p className="text-xs text-zinc-500">Mesh average</p>
          </CardContent>
        </Card>
      </div>

      {/* Workers Table */}
      <Card>
        <CardHeader>
          <CardTitle>Cluster Nodes</CardTitle>
          <CardDescription>Manage and monitor all workers in your mesh</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Hostname</TableHead>
                <TableHead>Role</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>IP Address</TableHead>
                <TableHead>VRAM Usage</TableHead>
                <TableHead>CPU Load</TableHead>
                <TableHead>Uptime</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {mockWorkers.map((worker) => (
                <TableRow key={worker.id}>
                  <TableCell className="font-medium">{worker.hostname}</TableCell>
                  <TableCell>
                    <Badge variant={worker.role === 'Master' ? 'default' : 'secondary'}>
                      {worker.role}
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
                  <TableCell className="text-zinc-400 font-mono text-xs">
                    {worker.ip}
                  </TableCell>
                  <TableCell>
                    {worker.status === 'online' ? (
                      <div className="space-y-1">
                        <div className="text-sm">{worker.vramUsage}</div>
                        <div className="w-24 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-purple-600 transition-all"
                            style={{ width: `${worker.vramPercent}%` }}
                          />
                        </div>
                      </div>
                    ) : (
                      <span className="text-zinc-600">-</span>
                    )}
                  </TableCell>
                  <TableCell className="text-zinc-400">{worker.cpuUsage}</TableCell>
                  <TableCell className="text-zinc-400 text-xs">{worker.uptime}</TableCell>
                  <TableCell className="text-right">
                    {worker.status === 'online' && worker.role !== 'Master' && (
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
        </CardContent>
      </Card>
    </div>
  )
}
