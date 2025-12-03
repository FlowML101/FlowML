import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ScrollText, Filter } from 'lucide-react'
import { Input } from '@/components/ui/input'

interface LogEntry {
  id: string
  timestamp: string
  source: string
  level: 'info' | 'warn' | 'error'
  message: string
}

// Mock distributed system logs
const mockLogs: LogEntry[] = [
  { id: '1', timestamp: '2025-12-03 14:32:15', source: 'Master', level: 'info', message: 'Cluster initialized successfully' },
  { id: '2', timestamp: '2025-12-03 14:32:18', source: 'Worker-01', level: 'info', message: 'Handshake successful with 192.168.1.105' },
  { id: '3', timestamp: '2025-12-03 14:32:19', source: 'Worker-02', level: 'info', message: 'Handshake successful with 192.168.1.142' },
  { id: '4', timestamp: '2025-12-03 14:32:45', source: 'Master', level: 'info', message: 'Dataset uploaded: titanic.csv (891 rows)' },
  { id: '5', timestamp: '2025-12-03 14:33:02', source: 'Master', level: 'info', message: 'Starting AutoML pipeline with 20 models' },
  { id: '6', timestamp: '2025-12-03 14:33:05', source: 'Worker-01', level: 'info', message: 'Training XGBoost model (1/20)' },
  { id: '7', timestamp: '2025-12-03 14:33:18', source: 'Worker-01', level: 'info', message: 'XGBoost training complete - Accuracy: 0.812' },
  { id: '8', timestamp: '2025-12-03 14:33:20', source: 'Worker-02', level: 'info', message: 'Training Random Forest model (2/20)' },
  { id: '9', timestamp: '2025-12-03 14:33:45', source: 'Worker-02', level: 'warn', message: 'High memory usage detected: 87%' },
  { id: '10', timestamp: '2025-12-03 14:33:48', source: 'Worker-02', level: 'info', message: 'Random Forest training complete - Accuracy: 0.798' },
  { id: '11', timestamp: '2025-12-03 14:34:01', source: 'Worker-01', level: 'info', message: 'Training LightGBM model (3/20)' },
  { id: '12', timestamp: '2025-12-03 14:34:22', source: 'Worker-01', level: 'info', message: 'LightGBM training complete - Accuracy: 0.825' },
  { id: '13', timestamp: '2025-12-03 14:34:25', source: 'Worker-02', level: 'info', message: 'Training CatBoost model (4/20)' },
  { id: '14', timestamp: '2025-12-03 14:34:58', source: 'Worker-02', level: 'info', message: 'CatBoost training complete - Accuracy: 0.831' },
  { id: '15', timestamp: '2025-12-03 14:35:02', source: 'Worker-01', level: 'info', message: 'Training Neural Network model (5/20)' },
  { id: '16', timestamp: '2025-12-03 14:35:18', source: 'Worker-01', level: 'error', message: 'OOM Killer triggered: Neural Network training aborted' },
  { id: '17', timestamp: '2025-12-03 14:35:20', source: 'Master', level: 'warn', message: 'Worker-01 recovered from OOM, resuming operations' },
  { id: '18', timestamp: '2025-12-03 14:35:25', source: 'Worker-02', level: 'info', message: 'Training Linear Model (6/20)' },
  { id: '19', timestamp: '2025-12-03 14:35:30', source: 'Worker-02', level: 'info', message: 'Linear Model training complete - Accuracy: 0.752' },
  { id: '20', timestamp: '2025-12-03 14:35:35', source: 'Master', level: 'info', message: 'Workload redistributed across 2 workers' },
  { id: '21', timestamp: '2025-12-03 14:35:40', source: 'Worker-01', level: 'info', message: 'Training Extra Trees model (7/20)' },
  { id: '22', timestamp: '2025-12-03 14:36:05', source: 'Worker-01', level: 'info', message: 'Extra Trees training complete - Accuracy: 0.818' },
  { id: '23', timestamp: '2025-12-03 14:36:10', source: 'Worker-03', level: 'info', message: 'New worker node connected: 192.168.1.156' },
  { id: '24', timestamp: '2025-12-03 14:36:12', source: 'Master', level: 'info', message: 'Cluster expanded to 3 workers' },
  { id: '25', timestamp: '2025-12-03 14:36:15', source: 'Worker-03', level: 'info', message: 'Training AdaBoost model (8/20)' },
  { id: '26', timestamp: '2025-12-03 14:36:42', source: 'Worker-03', level: 'info', message: 'AdaBoost training complete - Accuracy: 0.795' },
  { id: '27', timestamp: '2025-12-03 14:36:50', source: 'Master', level: 'info', message: 'Best model so far: CatBoost (0.831)' },
  { id: '28', timestamp: '2025-12-03 14:37:00', source: 'Worker-02', level: 'warn', message: 'Connection timeout detected, retrying...' },
  { id: '29', timestamp: '2025-12-03 14:37:02', source: 'Worker-02', level: 'info', message: 'Connection re-established successfully' },
  { id: '30', timestamp: '2025-12-03 14:37:10', source: 'Master', level: 'info', message: 'AutoML pipeline 40% complete' },
]

export function LogsPage() {
  const [logs] = useState<LogEntry[]>(mockLogs)
  const [levelFilter, setLevelFilter] = useState<string>('all')
  const [sourceFilter, setSourceFilter] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')

  // Get unique sources
  const sources = ['all', ...Array.from(new Set(mockLogs.map(log => log.source)))]

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
      <div>
        <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
          <ScrollText className="w-8 h-8 text-cyan-500" />
          System Logs
        </h1>
        <p className="text-zinc-400">Centralized audit trail for the distributed cluster</p>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
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
        <CardContent>
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
      <Card>
        <CardHeader>
          <CardTitle>Log Entries</CardTitle>
          <CardDescription>Real-time distributed system events</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="border border-zinc-800 rounded-lg overflow-hidden">
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
                        <TableCell className="font-mono text-xs text-zinc-400">
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
