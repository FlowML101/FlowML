import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { 
  Calendar, Clock, Cpu, Server, CheckCircle, 
  Loader2, Circle, PlayCircle
} from 'lucide-react'
import { motion } from 'framer-motion'

const mockJobs = [
  { id: 1, name: 'XGBoost Training', status: 'completed', start: '08:00', end: '09:30', worker: 'worker-1', date: '2024-12-12' },
  { id: 2, name: 'Data Preprocessing', status: 'completed', start: '09:45', end: '10:15', worker: 'worker-2', date: '2024-12-12' },
  { id: 3, name: 'CatBoost Tuning', status: 'running', start: '10:30', end: null, worker: 'worker-1', date: '2024-12-12' },
  { id: 4, name: 'RF Training', status: 'pending', start: '11:00', end: null, worker: 'worker-3', date: '2024-12-12' },
  { id: 5, name: 'Model Evaluation', status: 'pending', start: '12:00', end: null, worker: 'worker-2', date: '2024-12-12' }
]

const workers = [
  { id: 'worker-1', name: 'GPU Worker 1', type: 'gpu', status: 'busy', utilization: 87 },
  { id: 'worker-2', name: 'CPU Worker 1', type: 'cpu', status: 'idle', utilization: 12 },
  { id: 'worker-3', name: 'GPU Worker 2', type: 'gpu', status: 'idle', utilization: 0 }
]

const timeSlots = ['08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00']

export function ResourceScheduler() {
  const [selectedDate, setSelectedDate] = useState('2024-12-12')
  const [queueFilter, setQueueFilter] = useState<'all' | 'pending' | 'running' | 'completed'>('all')

  const filteredJobs = queueFilter === 'all' 
    ? mockJobs 
    : mockJobs.filter(job => job.status === queueFilter)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <Calendar className="w-8 h-8 text-cyan-500" />
            Resource Scheduling
          </h1>
          <p className="text-muted-foreground">Visualize worker allocation and job timeline</p>
        </div>
        <Button className="bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700">
          <PlayCircle className="w-4 h-4 mr-2" />
          Schedule New Job
        </Button>
      </div>

      {/* Worker Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {workers.map((worker) => (
          <Card 
            key={worker.id}
            className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/10 before:via-transparent before:to-cyan-500/10 before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-blue-500/12"
          >
            <CardHeader className="relative pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">{worker.name}</CardTitle>
                <Badge className={
                  worker.status === 'busy' ? 'bg-yellow-600' : 'bg-green-600'
                }>
                  {worker.status}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="relative">
              <div className="flex items-center gap-3 mb-3">
                <Server className={`w-5 h-5 ${worker.type === 'gpu' ? 'text-purple-400' : 'text-blue-400'}`} />
                <span className="text-sm text-muted-foreground">{worker.type.toUpperCase()}</span>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Utilization</span>
                  <span className="font-medium">{worker.utilization}%</span>
                </div>
                <div className="w-full h-2 bg-zinc-800 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${worker.utilization}%` }}
                    className={`h-full rounded-full ${
                      worker.utilization > 80 ? 'bg-red-500' :
                      worker.utilization > 50 ? 'bg-yellow-500' :
                      'bg-green-500'
                    }`}
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Timeline Calendar */}
        <div className="lg:col-span-2">
          <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <Clock className="w-5 h-5 text-cyan-400" />
                  Job Timeline
                </CardTitle>
                <input 
                  type="date" 
                  value={selectedDate}
                  onChange={(e) => setSelectedDate(e.target.value)}
                  className="px-3 py-1.5 rounded-lg bg-zinc-800 border border-zinc-700 text-sm"
                />
              </div>
              <CardDescription>
                Visual schedule of past and upcoming jobs
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {workers.map((worker) => (
                  <div key={worker.id} className="space-y-2">
                    <div className="flex items-center gap-3">
                      <div className="w-32 text-sm font-medium">{worker.name}</div>
                      <div className="flex-1 relative h-12 bg-zinc-800/50 rounded-lg border border-zinc-700">
                        {/* Time markers */}
                        <div className="absolute inset-0 flex">
                          {timeSlots.map((time, idx) => (
                            <div 
                              key={time}
                              className="flex-1 border-r border-zinc-700/50 text-xs text-muted-foreground flex items-center justify-center"
                            >
                              {idx === 0 && time}
                            </div>
                          ))}
                        </div>

                        {/* Jobs on this worker */}
                        {mockJobs
                          .filter(job => job.worker === worker.id)
                          .map((job) => {
                            const startHour = parseInt(job.start.split(':')[0])
                            const startMin = parseInt(job.start.split(':')[1])
                            const left = ((startHour - 8) * 60 + startMin) / (10 * 60) * 100
                            
                            let width = 15
                            if (job.end) {
                              const endHour = parseInt(job.end.split(':')[0])
                              const endMin = parseInt(job.end.split(':')[1])
                              const duration = (endHour * 60 + endMin) - (startHour * 60 + startMin)
                              width = (duration / (10 * 60)) * 100
                            }

                            return (
                              <motion.div
                                key={job.id}
                                initial={{ opacity: 0, scale: 0.8 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className={`absolute top-1 bottom-1 rounded px-2 flex items-center text-xs font-medium ${
                                  job.status === 'completed' ? 'bg-green-600' :
                                  job.status === 'running' ? 'bg-yellow-600' :
                                  'bg-zinc-600'
                                }`}
                                style={{ left: `${left}%`, width: `${width}%` }}
                              >
                                {width > 12 && <span className="truncate">{job.name}</span>}
                              </motion.div>
                            )
                          })}
                      </div>
                    </div>
                  </div>
                ))}

                {/* Time labels */}
                <div className="flex items-center gap-3">
                  <div className="w-32"></div>
                  <div className="flex-1 flex justify-between px-2">
                    {timeSlots.map((time) => (
                      <span key={time} className="text-xs text-muted-foreground">{time}</span>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Job Queue */}
        <div className="lg:col-span-1">
          <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50">
            <CardHeader>
              <CardTitle className="text-lg">Job Queue</CardTitle>
              <CardDescription>{mockJobs.length} total jobs</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {/* Filter buttons */}
              <div className="flex gap-2">
                <Button 
                  size="sm" 
                  variant={queueFilter === 'all' ? 'default' : 'outline'}
                  onClick={() => setQueueFilter('all')}
                  className="flex-1"
                >
                  All
                </Button>
                <Button 
                  size="sm" 
                  variant={queueFilter === 'pending' ? 'default' : 'outline'}
                  onClick={() => setQueueFilter('pending')}
                  className="flex-1"
                >
                  Pending
                </Button>
                <Button 
                  size="sm" 
                  variant={queueFilter === 'running' ? 'default' : 'outline'}
                  onClick={() => setQueueFilter('running')}
                  className="flex-1"
                >
                  Running
                </Button>
              </div>

              {/* Job List */}
              <div className="space-y-2 max-h-[500px] overflow-y-auto">
                {filteredJobs.map((job) => (
                  <motion.div
                    key={job.id}
                    layout
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-3 rounded-lg border border-zinc-700 bg-zinc-800/30"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium">{job.name}</span>
                      {job.status === 'completed' && <CheckCircle className="w-4 h-4 text-green-500" />}
                      {job.status === 'running' && <Loader2 className="w-4 h-4 text-yellow-500 animate-spin" />}
                      {job.status === 'pending' && <Circle className="w-4 h-4 text-zinc-500" />}
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Clock className="w-3 h-3" />
                      <span>{job.start}</span>
                      {job.end && <span>→ {job.end}</span>}
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground mt-1">
                      <Server className="w-3 h-3" />
                      <span>{job.worker}</span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Heatmap */}
      <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/10 before:via-transparent before:to-cyan-500/10 before:opacity-30">
        <CardHeader className="relative">
          <CardTitle className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-purple-400" />
            Worker Availability Heatmap
          </CardTitle>
          <CardDescription>
            Historical utilization patterns (last 7 days)
          </CardDescription>
        </CardHeader>
        <CardContent className="relative">
          <div className="space-y-3">
            {workers.map((worker) => (
              <div key={worker.id} className="space-y-2">
                <div className="text-sm font-medium">{worker.name}</div>
                <div className="flex gap-1">
                  {Array.from({ length: 24 }).map((_, idx) => {
                    const utilization = Math.random() * 100
                    return (
                      <div 
                        key={idx}
                        className="flex-1 h-8 rounded transition-all hover:scale-110"
                        style={{
                          backgroundColor: 
                            utilization > 80 ? '#ef4444' :
                            utilization > 50 ? '#f59e0b' :
                            utilization > 20 ? '#10b981' :
                            '#27272a'
                        }}
                        title={`${idx}:00 - ${Math.round(utilization)}%`}
                      />
                    )
                  })}
                </div>
              </div>
            ))}
            
            {/* Legend */}
            <div className="flex items-center gap-4 pt-3 border-t border-zinc-700">
              <span className="text-xs text-muted-foreground">Utilization:</span>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded bg-zinc-800"></div>
                <span className="text-xs">Low</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded bg-green-500"></div>
                <span className="text-xs">Medium</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded bg-yellow-500"></div>
                <span className="text-xs">High</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded bg-red-500"></div>
                <span className="text-xs">Critical</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
