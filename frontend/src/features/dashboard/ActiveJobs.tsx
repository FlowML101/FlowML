import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Zap, Clock, TrendingUp } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Link } from 'react-router-dom'

const mockJobs = [
  {
    id: 'job-1',
    name: 'Customer Churn Model',
    status: 'running',
    progress: 65,
    modelsEvaluated: 13,
    totalModels: 20,
    estimatedTime: '8 min',
    currentModel: 'XGBoost',
  },
  {
    id: 'job-2',
    name: 'Fraud Detection',
    status: 'queued',
    progress: 0,
    modelsEvaluated: 0,
    totalModels: 15,
    estimatedTime: '12 min',
    currentModel: '-',
  },
]

export function ActiveJobs() {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-yellow-500" />
              Active Training Jobs
            </CardTitle>
            <CardDescription>Real-time training pipeline status</CardDescription>
          </div>
          <Link to="/app/train">
            <Button size="sm" className="bg-purple-600 hover:bg-purple-700">
              New Job
            </Button>
          </Link>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {mockJobs.length === 0 ? (
          <div className="text-center py-8 text-zinc-500">
            No active jobs. Start training to see progress here.
          </div>
        ) : (
          mockJobs.map((job) => (
            <div key={job.id} className="p-4 rounded-lg bg-zinc-800/50 space-y-3">
              {/* Header */}
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">{job.name}</div>
                  <div className="text-xs text-zinc-500 mt-1">
                    Training: {job.currentModel}
                  </div>
                </div>
                <Badge
                  variant={job.status === 'running' ? 'default' : 'secondary'}
                  className="capitalize"
                >
                  {job.status === 'running' && (
                    <span className="w-2 h-2 bg-purple-400 rounded-full animate-pulse mr-1"></span>
                  )}
                  {job.status}
                </Badge>
              </div>

              {/* Progress */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-zinc-400">
                    Model {job.modelsEvaluated} of {job.totalModels}
                  </span>
                  <span className="text-zinc-400">{job.progress}%</span>
                </div>
                <Progress value={job.progress} className="h-2" />
              </div>

              {/* Footer */}
              <div className="flex items-center justify-between text-xs text-zinc-500">
                <div className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  <span>~{job.estimatedTime} remaining</span>
                </div>
                {job.status === 'running' && (
                  <Link
                    to="/app/running"
                    className="text-purple-400 hover:text-purple-300 flex items-center gap-1"
                  >
                    <TrendingUp className="w-3 h-3" />
                    Monitor
                  </Link>
                )}
              </div>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  )
}
