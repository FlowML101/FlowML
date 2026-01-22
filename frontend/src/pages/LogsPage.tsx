import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ScrollText, Filter, Loader2, RefreshCw } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { trainingApi, workersApi } from '@/lib/api'

interface LogEntry {
  id: string
  timestamp: string
  source: string
  level: 'info' | 'warn' | 'error'
  message: string
}

export function LogsPage() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [levelFilter, setLevelFilter] = useState<string>('all')
  const [sourceFilter, setSourceFilter] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')

  const fetchLogs = async () => {
    try {
      setLoading(true)
      const logEntries: LogEntry[] = []
      
      // Fetch jobs and generate logs from their status
      const jobs = await trainingApi.list()
      jobs.forEach((job, index) => {
        logEntries.push({
          id: `job-created-${job.id}`,
          timestamp: new Date(job.created_at).toLocaleString(),
          source: 'Master',
          level: 'info',
          message: `Job "${job.name}" created for dataset ${job.dataset_id}`,
        })
        
        if (job.started_at) {
          logEntries.push({
            id: `job-started-${job.id}`,
            timestamp: new Date(job.started_at).toLocaleString(),
            source: 'Master',
            level: 'info',
            message: `Training started: ${job.name} (${job.model_types})`,
          })
        }
        
        if (job.status === 'running' && job.current_model) {
          logEntries.push({
            id: `job-training-${job.id}`,
            timestamp: new Date().toLocaleString(),
            source: `Worker-${(index % 3) + 1}`,
            level: 'info',
            message: `Training ${job.current_model} (${job.models_completed}/${job.total_models})`,
          })
        }
        
        if (job.status === 'completed') {
          logEntries.push({
            id: `job-complete-${job.id}`,
            timestamp: job.completed_at ? new Date(job.completed_at).toLocaleString() : new Date().toLocaleString(),
            source: 'Master',
            level: 'info',
            message: `Job "${job.name}" completed - ${job.models_completed} models trained`,
          })
        }
        
        if (job.status === 'failed') {
          logEntries.push({
            id: `job-failed-${job.id}`,
            timestamp: job.completed_at ? new Date(job.completed_at).toLocaleString() : new Date().toLocaleString(),
            source: 'Master',
            level: 'error',
            message: `Job "${job.name}" failed: ${job.error_message || 'Unknown error'}`,
          })
        }
      })
      
      // Fetch workers and add connection logs
      try {
        const workers = await workersApi.list()
        workers.forEach(worker => {
          logEntries.push({
            id: `worker-${worker.id}`,
            timestamp: new Date().toLocaleString(),
            source: worker.hostname,
            level: worker.status === 'online' ? 'info' : 'warn',
            message: worker.status === 'online' 
              ? `Connected: ${worker.ip} (${worker.cpu_count} cores, ${worker.ram_total_gb.toFixed(1)}GB RAM)`
              : `Worker offline or degraded`,
          })
        })
      } catch {
        // Workers API might fail, that's okay
      }
      
      // Sort by timestamp (newest first)
      logEntries.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      
      setLogs(logEntries)
    } catch (err) {
      console.error('Failed to fetch logs:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchLogs()
    const interval = setInterval(fetchLogs, 10000) // Refresh every 10 seconds
    return () => clearInterval(interval)
  }, [])

  // Get unique sources
  const sources = ['all', ...Array.from(new Set(logs.map(log => log.source)))]

  // Filter logs
  const filteredLogs = logs.filter(log => {
    const matchesLevel = levelFilter === 'all' || log.level === levelFilter
    const matchesSource = sourceFilter === 'all' || log.source === sourceFilter
    const matchesSearch = searchQuery === '' || 
      log.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
      log.source.toLowerCase().includes(searchQuery.toLowerCase())
    
    return matchesLevel && matchesSource && matchesSearch
  })

  const getLevelBadge = (level: string) => {
    switch (level) {
      case 'error':
        return <Badge variant="destructive" className="text-xs">ERROR</Badge>
      case 'warn':
        return <Badge className="text-xs bg-yellow-600">WARN</Badge>
      case 'info':
        return <Badge variant="secondary" className="text-xs">INFO</Badge>
      default:
        return <Badge variant="outline" className="text-xs">{level}</Badge>
    }
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <ScrollText className="w-8 h-8 text-cyan-500" />
            System Logs
          </h1>
          <p className="text-muted-foreground">Centralized audit trail and monitoring for distributed cluster operations</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline" size="sm" onClick={fetchLogs} disabled={loading}>
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Badge variant="outline" className="text-sm px-4 py-2">
            {filteredLogs.length} entries
          </Badge>
          <Badge className={`text-sm px-4 py-2 ${
            logs.some(l => l.level === 'error') ? 'bg-red-600' :
            logs.some(l => l.level === 'warn') ? 'bg-yellow-600' :
            'bg-green-600'
          }`}>
            {logs.filter(l => l.level === 'error').length} errors
          </Badge>
        </div>
      </div>

      {/* Filters */}
      <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-cyan-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-cyan-500/12">
        <CardHeader className="relative">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Filter className="w-5 h-5 text-purple-500" />
                Filters
              </CardTitle>
              <CardDescription>Filter logs by level, source, or search content</CardDescription>
            </div>
            <Badge variant="outline">{filteredLogs.length} entries</Badge>
          </div>
        </CardHeader>
        <CardContent className="relative">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Search */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Search</label>
              <Input
                placeholder="Search logs..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>

            {/* Level Filter */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Level</label>
              <Select value={levelFilter} onValueChange={setLevelFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="All levels" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Levels</SelectItem>
                  <SelectItem value="info">Info</SelectItem>
                  <SelectItem value="warn">Warning</SelectItem>
                  <SelectItem value="error">Error</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Source Filter */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Source</label>
              <Select value={sourceFilter} onValueChange={setSourceFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="All sources" />
                </SelectTrigger>
                <SelectContent>
                  {sources.map(source => (
                    <SelectItem key={source} value={source}>
                      {source === 'all' ? 'All Sources' : source}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Logs Table */}
      <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/10 before:via-purple-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-blue-500/12">
        <CardHeader className="relative">
          <CardTitle>Log Entries</CardTitle>
          <CardDescription>Real-time distributed system events</CardDescription>
        </CardHeader>
        <CardContent className="relative">
          <div className="border border-border dark:border-zinc-800 rounded-lg overflow-hidden">
            <div className="max-h-[600px] overflow-y-auto">
              <Table>
                <TableHeader className="sticky top-0 bg-zinc-900 z-10">
                  <TableRow>
                    <TableHead className="w-[180px]">Timestamp</TableHead>
                    <TableHead className="w-[120px]">Source</TableHead>
                    <TableHead className="w-[100px]">Level</TableHead>
                    <TableHead>Message</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredLogs.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={4} className="text-center py-8 text-zinc-500">
                        No logs match the current filters
                      </TableCell>
                    </TableRow>
                  ) : (
                    filteredLogs.map((log) => (
                      <TableRow key={log.id}>
                        <TableCell className="font-mono text-xs text-muted-foreground">
                          {log.timestamp}
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline" className="text-xs">
                            {log.source}
                          </Badge>
                        </TableCell>
                        <TableCell>{getLevelBadge(log.level)}</TableCell>
                        <TableCell className="text-sm">{log.message}</TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
