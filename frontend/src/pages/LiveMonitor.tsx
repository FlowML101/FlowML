import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { TrendingDown, Terminal, Activity, Zap } from 'lucide-react'
import { useEffect, useState } from 'react'

// Mock training data
const generateMockData = (length: number) => {
  return Array.from({ length }, (_, i) => ({
    epoch: i + 1,
    loss: 0.7 - (i * 0.025) + Math.random() * 0.05,
    accuracy: 0.45 + (i * 0.025) + Math.random() * 0.03,
  }))
}

const mockLogs = [
  '[00:00] Initializing AutoML pipeline...',
  '[00:01] Scanning dataset: 891 rows, 12 features',
  '[00:02] Feature engineering: Created 5 new features',
  '[00:03] Starting model evaluation...',
  '[00:05] Training XGBoost (1/20)...',
  '[00:12] XGBoost complete. Accuracy: 0.812',
  '[00:13] Training Random Forest (2/20)...',
  '[00:22] Random Forest complete. Accuracy: 0.798',
  '[00:23] Training LightGBM (3/20)...',
  '[00:29] LightGBM complete. Accuracy: 0.825',
  '[00:30] Training CatBoost (4/20)...',
  '[00:38] CatBoost complete. Accuracy: 0.831',
  '[00:39] Training Linear Model (5/20)...',
  '[00:41] Linear Model complete. Accuracy: 0.752',
]

export function LiveMonitor() {
  const [chartData, setChartData] = useState(generateMockData(10))
  const [logs, setLogs] = useState(mockLogs.slice(0, 8))
  const [progress, setProgress] = useState(25)

  useEffect(() => {
    const interval = setInterval(() => {
      // Simulate real-time updates
      if (logs.length < mockLogs.length) {
        setLogs((prev) => [...prev, mockLogs[prev.length]])
      }
      if (chartData.length < 25) {
        setChartData((prev) => [
          ...prev,
          {
            epoch: prev.length + 1,
            loss: Math.max(0.15, prev[prev.length - 1].loss - 0.02 + Math.random() * 0.01),
            accuracy: Math.min(0.95, prev[prev.length - 1].accuracy + 0.015 + Math.random() * 0.01),
          },
        ])
      }
      setProgress((prev) => Math.min(100, prev + 3))
    }, 2000)

    return () => clearInterval(interval)
  }, [logs.length, chartData.length])

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <Activity className="w-8 h-8 text-purple-500 animate-pulse" />
            Live Training Monitor
          </h1>
          <p className="text-zinc-400">Real-time model training progress</p>
        </div>
        <Badge variant="default" className="text-base px-4 py-2">
          <span className="w-2 h-2 bg-purple-400 rounded-full animate-pulse mr-2"></span>
          Training in Progress
        </Badge>
      </div>

      {/* Progress Overview */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Training Progress</CardTitle>
              <CardDescription>Customer Churn Model</CardDescription>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold text-purple-400">{progress}%</div>
              <div className="text-xs text-zinc-500">Model 5 of 20</div>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Progress value={progress} className="h-3" />
          <div className="mt-4 grid grid-cols-3 gap-4">
            <div className="p-3 rounded-lg bg-zinc-800/50">
              <div className="text-xs text-zinc-500">Best Accuracy</div>
              <div className="text-xl font-bold text-green-400">83.1%</div>
              <div className="text-xs text-zinc-500 mt-1">CatBoost</div>
            </div>
            <div className="p-3 rounded-lg bg-zinc-800/50">
              <div className="text-xs text-zinc-500">Time Elapsed</div>
              <div className="text-xl font-bold">0:38</div>
              <div className="text-xs text-zinc-500 mt-1">~11 min left</div>
            </div>
            <div className="p-3 rounded-lg bg-zinc-800/50">
              <div className="text-xs text-zinc-500">Current Model</div>
              <div className="text-xl font-bold text-purple-400">Linear</div>
              <div className="text-xs text-zinc-500 mt-1">Training...</div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Real-time Chart */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingDown className="w-5 h-5 text-blue-500" />
              Loss Curve
            </CardTitle>
            <CardDescription>Training loss over time</CardDescription>
          </CardHeader>
          <CardContent>
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
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-green-500" />
              Accuracy Curve
            </CardTitle>
            <CardDescription>Model accuracy over time</CardDescription>
          </CardHeader>
          <CardContent>
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
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Terminal className="w-5 h-5 text-zinc-400" />
            Training Logs
          </CardTitle>
          <CardDescription>Real-time output from training pipeline</CardDescription>
        </CardHeader>
        <CardContent>
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
