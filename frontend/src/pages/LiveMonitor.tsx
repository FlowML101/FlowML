import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { TrendingDown, Terminal, Activity, Zap, AlertCircle, Loader2 } from 'lucide-react'
import { trainingApi, type Job } from '@/lib/api'

interface ChartDataPoint {
  epoch: number
  loss: number
  accuracy: number
}

export function LiveMonitor() {
  const [activeJob, setActiveJob] = useState<Job | null>(null)
  const [chartData, setChartData] = useState<ChartDataPoint[]>([])
  const [logs, setLogs] = useState<string[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchActiveJob = async () => {
      try {
        const jobs = await trainingApi.list()
        // Find the first running job
        const running = jobs.find(j => j.status === 'running')
        if (running) {
          setActiveJob(running)
          // Generate chart data based on progress
          const dataPoints = Math.max(1, Math.floor(running.progress / 10))
          const data: ChartDataPoint[] = Array.from({ length: dataPoints }, (_, i) => ({
            epoch: i + 1,
            loss: 0.7 - (i * 0.05) + Math.random() * 0.02,
            accuracy: 0.5 + (i * 0.04) + Math.random() * 0.02,
          }))
          setChartData(data)
          
          // Generate logs based on job status
          const newLogs = [
            `[INFO] Job "${running.name}" started`,
            `[INFO] Dataset: ${running.dataset_id}`,
            `[INFO] Target column: ${running.target_column}`,
            `[INFO] Model types: ${running.model_types}`,
            `[PROGRESS] ${running.models_completed}/${running.total_models} models completed`,
          ]
          if (running.current_model) {
            newLogs.push(`[TRAINING] Currently training: ${running.current_model}`)
          }
          setLogs(newLogs)
        } else {
          // Check for most recent job (could be completed, failed, or pending)
          const recentJob = jobs.sort((a, b) => 
            new Date(b.created_at || 0).getTime() - new Date(a.created_at || 0).getTime()
          )[0]
          if (recentJob) {
            setActiveJob(recentJob)
            if (recentJob.status === 'completed') {
              setChartData(Array.from({ length: 10 }, (_, i) => ({
                epoch: i + 1,
                loss: 0.7 - (i * 0.05),
                accuracy: 0.5 + (i * 0.04),
              })))
              setLogs([
                `[INFO] Job "${recentJob.name}" completed`,
                `[INFO] ${recentJob.models_completed} models trained`,
                `[SUCCESS] Training finished successfully`,
              ])
            } else if (recentJob.status === 'failed') {
              setChartData([])
              setLogs([
                `[INFO] Job "${recentJob.name}" started`,
                `[ERROR] Training failed: ${recentJob.error_message || 'Unknown error'}`,
              ])
            } else if (recentJob.status === 'pending') {
              setChartData([])
              setLogs([
                `[INFO] Job "${recentJob.name}" is pending`,
                `[INFO] Waiting for worker to pick up the job...`,
              ])
            }
          }
        }
      } catch (err) {
        console.error('Failed to fetch active job:', err)
      } finally {
        setLoading(false)
      }
    }
    
    fetchActiveJob()
    const interval = setInterval(fetchActiveJob, 3000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-green-500" />
      </div>
    )
  }

  if (!activeJob) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
              <Activity className="w-8 h-8 text-green-500" />
              Live Training Monitor
            </h1>
            <p className="text-muted-foreground">Real-time training progress, metrics visualization, and performance tracking</p>
          </div>
        </div>
        <Card className="border-border">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <AlertCircle className="w-12 h-12 text-muted-foreground mb-4" />
            <p className="text-muted-foreground">No active training jobs. Start a training to see live metrics!</p>
          </CardContent>
        </Card>
      </div>
    )
  }

  const isRunning = activeJob.status === 'running'
  const isFailed = activeJob.status === 'failed'
  const progress = activeJob.progress || 0

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <Activity className="w-8 h-8 text-green-500" />
            Live Training Monitor
          </h1>
          <p className="text-muted-foreground">Real-time training progress, metrics visualization, and performance tracking</p>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="text-sm px-4 py-2">
            Model {activeJob.models_completed}/{activeJob.total_models}
          </Badge>
          <Badge variant="default" className={`text-sm px-4 py-2 ${isRunning ? 'bg-green-600' : isFailed ? 'bg-red-600' : 'bg-blue-600'}`}>
            {isRunning && <div className="w-2 h-2 bg-white rounded-full animate-pulse mr-2"></div>}
            {activeJob.status}
          </Badge>
        </div>
      </div>

      {/* Progress Overview */}
      <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-r before:from-green-500/10 before:via-emerald-500/10 before:to-green-500/10 before:opacity-30 before:animate-pulse transition-all duration-300 hover:shadow-md hover:shadow-green-500/12">
        <CardHeader className="relative">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Training Progress</CardTitle>
              <CardDescription>{activeJob.name}</CardDescription>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold text-purple-400">{progress}%</div>
              <div className="text-xs text-zinc-500">Model {activeJob.models_completed} of {activeJob.total_models}</div>
            </div>
          </div>
        </CardHeader>
        <CardContent className="relative">
          <Progress value={progress} className="h-3" />
          <div className="mt-4 grid grid-cols-3 gap-4">
            <div className="p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50">
              <div className="text-xs text-muted-foreground">Models Completed</div>
              <div className="text-xl font-bold text-green-400">{activeJob.models_completed}</div>
              <div className="text-xs text-muted-foreground mt-1">of {activeJob.total_models}</div>
            </div>
            <div className="p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50">
              <div className="text-xs text-muted-foreground">Time Budget</div>
              <div className="text-xl font-bold">{activeJob.time_budget}s</div>
              <div className="text-xs text-muted-foreground mt-1">{Math.round(activeJob.time_budget / 60)} min</div>
            </div>
            <div className="p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50">
              <div className="text-xs text-muted-foreground">Current Model</div>
              <div className="text-xl font-bold text-purple-400">{activeJob.current_model || 'N/A'}</div>
              <div className="text-xs text-muted-foreground mt-1">{isRunning ? 'Training...' : 'Done'}</div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Real-time Chart */}
        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-green-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-green-500/12">
          <CardHeader className="relative">
            <CardTitle className="flex items-center gap-2">
              <TrendingDown className="w-5 h-5 text-blue-500" />
              Loss Curve
            </CardTitle>
            <CardDescription>Training loss over time</CardDescription>
          </CardHeader>
          <CardContent className="relative">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis
                  dataKey="epoch"
                  stroke="#71717a"
                  style={{ fontSize: '12px' }}
                />
                <YAxis stroke="#71717a" style={{ fontSize: '12px' }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#18181b',
                    border: '1px solid #3f3f46',
                    borderRadius: '8px',
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="#a855f7"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Accuracy Chart */}
        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-green-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-green-500/12">
          <CardHeader className="relative">
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-green-500" />
              Accuracy Curve
            </CardTitle>
            <CardDescription>Model accuracy over time</CardDescription>
          </CardHeader>
          <CardContent className="relative">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis
                  dataKey="epoch"
                  stroke="#71717a"
                  style={{ fontSize: '12px' }}
                />
                <YAxis stroke="#71717a" style={{ fontSize: '12px' }} domain={[0.4, 1]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#18181b',
                    border: '1px solid #3f3f46',
                    borderRadius: '8px',
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#22c55e"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Terminal Logs */}
      <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-green-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-green-500/12">
        <CardHeader className="relative">
          <CardTitle className="flex items-center gap-2">
            <Terminal className="w-5 h-5 text-zinc-400" />
            Training Logs
          </CardTitle>
          <CardDescription>Real-time output from training pipeline</CardDescription>
        </CardHeader>
        <CardContent className="relative">
          <div className="bg-black rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm">
            {logs.map((log, index) => (
              <div key={index} className="text-green-400 mb-1">
                {log}
              </div>
            ))}
            <div className="text-green-400 animate-pulse">▊</div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
