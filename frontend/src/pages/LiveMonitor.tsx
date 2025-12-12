import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { TrendingDown, Terminal, Activity, Zap } from 'lucide-react'

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
  // Backend will provide real-time data via WebSocket
  const chartData = generateMockData(10)
  const logs = mockLogs.slice(0, 8)
  const progress = 25

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
            Epoch {chartData.length}/25
          </Badge>
          <Badge variant="default" className="text-sm px-4 py-2 bg-green-600">
            <div className="w-2 h-2 bg-white rounded-full animate-pulse mr-2"></div>
            Training
          </Badge>
        </div>
      </div>

      {/* Progress Overview */}
      <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-r before:from-green-500/10 before:via-emerald-500/10 before:to-green-500/10 before:opacity-30 before:animate-pulse transition-all duration-300 hover:shadow-md hover:shadow-green-500/12">
        <CardHeader className="relative">
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
        <CardContent className="relative">
          <Progress value={progress} className="h-3" />
          <div className="mt-4 grid grid-cols-3 gap-4">
            <div className="p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50">
              <div className="text-xs text-muted-foreground">Best Accuracy</div>
              <div className="text-xl font-bold text-green-400">83.1%</div>
              <div className="text-xs text-muted-foreground mt-1">CatBoost</div>
            </div>
            <div className="p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50">
              <div className="text-xs text-muted-foreground">Time Elapsed</div>
              <div className="text-xl font-bold">0:38</div>
              <div className="text-xs text-muted-foreground mt-1">~11 min left</div>
            </div>
            <div className="p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50">
              <div className="text-xs text-muted-foreground">Current Model</div>
              <div className="text-xl font-bold text-purple-400">Linear</div>
              <div className="text-xs text-muted-foreground mt-1">Training...</div>
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
