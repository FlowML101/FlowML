import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Activity, Clock, TrendingUp, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Link } from 'react-router-dom'
import { trainingApi, Job } from '@/lib/api'

export function ActiveJobs() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchJobs = async () => {
      try {
        const allJobs = await trainingApi.list()
        // Filter to show running and pending jobs
        const activeJobs = allJobs.filter(j => j.status === 'running' || j.status === 'pending')
        setJobs(activeJobs)
      } catch (err) {
        console.error('Failed to fetch jobs:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchJobs()
    // Refresh every 3 seconds for active jobs
    const interval = setInterval(fetchJobs, 3000)
    return () => clearInterval(interval)
  }, [])

  return (
    <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-yellow-500/10 before:via-transparent before:to-orange-500/10 before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-yellow-500/12">
      <CardHeader className="relative">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-yellow-500" />
              Active Jobs
            </CardTitle>
            <CardDescription>Real-time training pipeline status</CardDescription>
          </div>
          <Link to="/train">
            <Button size="sm" className="bg-gradient-to-r from-yellow-600 to-orange-600 hover:from-yellow-700 hover:to-orange-700">
              New Job
            </Button>
          </Link>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 relative">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-6 h-6 animate-spin text-zinc-500" />
          </div>
        ) : jobs.length === 0 ? (
          <div className="text-center py-8 text-zinc-500">
            No active jobs. Start training to see progress here.
          </div>
        ) : (
          jobs.map((job) => (
            <div key={job.id} className="p-4 rounded-lg bg-muted/50 dark:bg-zinc-800/50 space-y-3">
              {/* Header */}
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">{job.name}</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Training: {job.current_model || 'Preparing...'}
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
                  <span className="text-muted-foreground">
                    Model {job.models_completed} of {job.total_models || '?'}
                  </span>
                  <span className="text-muted-foreground">{Math.round(job.progress)}%</span>
                </div>
                <Progress value={job.progress} className="h-2" />
              </div>

              {/* Footer */}
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <div className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  <span>Budget: {job.time_budget} min</span>
                </div>
                {job.status === 'running' && (
                  <Link
                    to="/running"
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
