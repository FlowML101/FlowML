import { useState, useEffect, useCallback, useMemo } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Trophy, Download, Radio, Copy, FlaskConical, Loader2, AlertCircle, RefreshCw, Filter } from 'lucide-react'
import { toast } from 'sonner'
import { resultsApi, trainingApi, TrainedModel, Job } from '@/lib/api'

export function Results() {
  const navigate = useNavigate()
  const location = useLocation()
  const [models, setModels] = useState<TrainedModel[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [selectedJobId, setSelectedJobId] = useState<string>('latest')
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchModels = useCallback(async (showRefreshing = false) => {
    try {
      if (showRefreshing) setRefreshing(true)
      else setLoading(true)
      
      // Fetch both models and jobs
      const [modelsData, jobsData] = await Promise.all([
        resultsApi.listAllModels(),
        trainingApi.list()
      ])
      
      // Get completed jobs only
      const completedJobs = jobsData.filter(j => j.status === 'completed')
      setJobs(completedJobs)
      
      // Sort models by created_at (newest first), then by accuracy
      const sorted = modelsData.sort((a: TrainedModel, b: TrainedModel) => {
        // First sort by created_at (newest first)
        const dateA = new Date(a.created_at || 0).getTime()
        const dateB = new Date(b.created_at || 0).getTime()
        if (dateB !== dateA) return dateB - dateA
        // Then by accuracy
        return (b.accuracy || 0) - (a.accuracy || 0)
      })
      setModels(sorted)
      setError(null)
    } catch (err) {
      setError('Failed to load models')
      console.error(err)
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [])

  // Fetch on mount and when navigating to this page
  useEffect(() => {
    fetchModels()
  }, [location.key, fetchModels])

  // Filter models based on selected job
  const filteredModels = useMemo(() => {
    if (selectedJobId === 'all') return models
    if (selectedJobId === 'latest' && models.length > 0) {
      // Get the job_id of the most recent model
      const latestJobId = models[0]?.job_id
      return models.filter(m => m.job_id === latestJobId)
    }
    return models.filter(m => m.job_id === selectedJobId)
  }, [models, selectedJobId])

  // Sort filtered models by rank (accuracy)
  const sortedModels = useMemo(() => {
    return [...filteredModels].sort((a, b) => (b.accuracy || 0) - (a.accuracy || 0))
  }, [filteredModels])

  const bestModel = sortedModels[0]
  
  const formatTime = (seconds?: number) => {
    if (!seconds) return 'N/A'
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}m ${secs}s`
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-yellow-500" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <AlertCircle className="w-12 h-12 text-red-500" />
        <p className="text-muted-foreground">{error}</p>
        <Button onClick={() => window.location.reload()}>Retry</Button>
      </div>
    )
  }

  if (models.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <Trophy className="w-12 h-12 text-muted-foreground" />
        <p className="text-muted-foreground">No trained models yet. Start a training job first!</p>
        <Button onClick={() => navigate('/train')}>Go to Training</Button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Trophy className="w-8 h-8 text-yellow-500" />
            Training Results
          </h1>
        </div>
        <div className="flex items-center gap-3">
          {/* Job Filter */}
          <Select value={selectedJobId} onValueChange={setSelectedJobId}>
            <SelectTrigger className="w-[200px]">
              <Filter className="w-4 h-4 mr-2" />
              <SelectValue placeholder="Filter by job" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="latest">Latest Job</SelectItem>
              <SelectItem value="all">All Jobs</SelectItem>
              {jobs.map(job => (
                <SelectItem key={job.id} value={job.id}>
                  {job.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => fetchModels(true)}
            disabled={refreshing}
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Badge variant="outline" className="text-sm px-4 py-2">
            {sortedModels.length} models
          </Badge>
          <Button className="bg-gradient-to-r from-purple-600 to-violet-600 hover:from-purple-700 hover:to-violet-700">
            <Download className="w-4 h-4 mr-2" />
            Export Report
          </Button>
        </div>
      </div>

      <div className="space-y-6">
        {/* Best Model Card */}
        {bestModel && (
        <Card className=" bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:from-purple-500/20 before:via-purple-500/10  before:opacity-70  shadow-purple-500/20">
          <CardHeader className="relative">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Trophy className="w-5 h-5 text-yellow-500" />
                  Best Model: {bestModel.name}
                </CardTitle>
                <CardDescription>Highest accuracy on validation set</CardDescription>
              </div>
              <div className="flex gap-2">
                <Button 
                  onClick={() => navigate(`/inference?model_id=${bestModel.id}`)}
                  className="bg-gradient-to-r from-purple-600 to-violet-600 hover:from-purple-700 hover:to-violet-700"
                >
                  <FlaskConical className="w-4 h-4 mr-2" />
                  Test in Lab
                </Button>
                <Button className="bg-gradient-to-r from-purple-600 via-purple-600 to-violet-600 hover:from-purple-700 hover:via-violet-700 hover:to-purple-700">
                  <Download className="w-4 h-4 mr-2" />
                  Export Model
                </Button>
              </div>
            </div>
          </CardHeader>
            <CardContent className="relative">
              <div className="grid grid-cols-4 gap-4">
                <div className="p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50">
                  <div className="text-xs text-muted-foreground">Accuracy</div>
                  <div className="text-2xl font-bold text-green-400">
                    {bestModel.accuracy ? `${(bestModel.accuracy * 100).toFixed(1)}%` : 'N/A'}
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50">
                  <div className="text-xs text-muted-foreground">F1 Score</div>
                  <div className="text-2xl font-bold text-blue-400">
                    {bestModel.f1_score?.toFixed(3) || 'N/A'}
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50">
                  <div className="text-xs text-muted-foreground">Precision</div>
                  <div className="text-2xl font-bold text-purple-400">
                    {bestModel.precision?.toFixed(3) || 'N/A'}
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50">
                  <div className="text-xs text-muted-foreground">Recall</div>
                  <div className="text-2xl font-bold text-yellow-400">
                    {bestModel.recall?.toFixed(3) || 'N/A'}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

          {/* All Models Table */}
          <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-yellow-500/10 before:via-transparent before:to-orange-500/10 before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-yellow-500/12">
            <CardHeader className="relative">
              <CardTitle>All Models</CardTitle>
              <CardDescription>Complete ranking of trained models</CardDescription>
            </CardHeader>
            <CardContent className="relative">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-16">Rank</TableHead>
                    <TableHead>Model</TableHead>
                    <TableHead>Accuracy</TableHead>
                    <TableHead>F1 Score</TableHead>
                    <TableHead>Precision</TableHead>
                    <TableHead>Recall</TableHead>
                    <TableHead>Training Time</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sortedModels.map((model, index) => (
                    <TableRow key={model.id}>
                      <TableCell>
                        <Badge
                          variant={index === 0 ? 'default' : 'secondary'}
                          className={index === 0 ? 'bg-yellow-600' : ''}
                        >
                          #{index + 1}
                        </Badge>
                      </TableCell>
                      <TableCell className="font-medium">{model.name}</TableCell>
                      <TableCell>
                        <span className="text-green-400 font-semibold">
                          {model.accuracy ? `${(model.accuracy * 100).toFixed(1)}%` : 'N/A'}
                        </span>
                      </TableCell>
                      <TableCell className="text-foreground">{model.f1_score?.toFixed(3) || 'N/A'}</TableCell>
                      <TableCell className="text-foreground">{model.precision?.toFixed(3) || 'N/A'}</TableCell>
                      <TableCell className="text-foreground">{model.recall?.toFixed(3) || 'N/A'}</TableCell>
                      <TableCell className="text-muted-foreground text-sm">{formatTime(model.training_time)}</TableCell>
                      <TableCell className="text-right">
                        <div className="flex gap-2 justify-end">
                          <Button 
                            onClick={() => navigate(`/inference?model_id=${model.id}`)}
                            className="bg-purple-600 hover:bg-purple-700 text-white"
                          >
                            <FlaskConical className="w-4 h-4 mr-1" />
                            Test in Lab
                          </Button>
                          <Dialog>
                            <DialogTrigger asChild>
                              <Button variant="ghost" size="sm" title="Serve model">
                                <Radio className="w-4 h-4" />
                              </Button>
                            </DialogTrigger>
                            <DialogContent className="sm:max-w-[600px]">
                              <DialogHeader>
                                <DialogTitle className="flex items-center gap-2">
                                  <Radio className="w-5 h-5 text-green-500" />
                                  Local Inference API
                                </DialogTitle>
                                <DialogDescription>
                                  {model.name} model is now serving on your local network
                                </DialogDescription>
                              </DialogHeader>
                              
                              <div className="space-y-4">
                                {/* Status */}
                                <div className="flex items-center justify-between p-3 rounded-lg bg-green-600/10 border border-green-600/30">
                                  <div className="flex items-center gap-2">
                                    <span className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></span>
                                    <span className="font-medium">Endpoint Active</span>
                                  </div>
                                  <Badge variant="success">Port 8000</Badge>
                                </div>

                                {/* CURL Example */}
                                <div className="space-y-2">
                                  <div className="flex items-center justify-between">
                                    <label className="text-sm font-medium">API Request Example</label>
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      onClick={() => {
                                        navigator.clipboard.writeText(`curl -X POST http://localhost:8000/results/models/${model.id}/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": {"age": 25, "sex": "female", "pclass": 1, "fare": 75.0}}'`)
                                        toast.success('Copied to clipboard')
                                      }}
                                    >
                                      <Copy className="w-4 h-4 mr-1" />
                                      Copy
                                    </Button>
                                  </div>
                                  <div className="bg-black rounded-lg p-4 font-mono text-sm overflow-x-auto">
                                    <code className="text-green-400">
                                      curl -X POST http://localhost:8000/results/models/{model.id}/predict \<br />
                                      &nbsp;&nbsp;-H "Content-Type: application/json" \<br />
                                      &nbsp;&nbsp;-d '{`{"features": {"age": 25, "sex": "female", "pclass": 1, "fare": 75.0}}`}'
                                    </code>
                                  </div>
                                </div>

                                {/* Response Example */}
                                <div className="space-y-2">
                                  <label className="text-sm font-medium">Expected Response</label>
                                  <div className="bg-black rounded-lg p-4 font-mono text-sm">
                                    <code className="text-blue-400">
                                      {`{`}<br />
                                      &nbsp;&nbsp;"prediction": 1,<br />
                                      &nbsp;&nbsp;"confidence": 0.87,<br />
                                      &nbsp;&nbsp;"model_id": "{model.id}",<br />
                                      &nbsp;&nbsp;"model_name": "{model.name}"<br />
                                      {`}`}
                                    </code>
                                  </div>
                                </div>

                                {/* Stats */}
                                <div className="grid grid-cols-3 gap-3">
                                  <div className="p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50 text-center">
                                    <div className="text-xs text-muted-foreground">Accuracy</div>
                                    <div className="text-lg font-bold text-green-400">
                                      {model.accuracy ? `${(model.accuracy * 100).toFixed(1)}%` : 'N/A'}
                                    </div>
                                  </div>
                                  <div className="p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50 text-center">
                                    <div className="text-xs text-muted-foreground">Training Time</div>
                                    <div className="text-lg font-bold text-blue-400">{formatTime(model.training_time)}</div>
                                  </div>
                                  <div className="p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50 text-center">
                                    <div className="text-xs text-muted-foreground">Created</div>
                                    <div className="text-lg font-bold text-purple-400">{new Date(model.created_at).toLocaleDateString()}</div>
                                  </div>
                                </div>
                              </div>
                            </DialogContent>
                          </Dialog>
                          <Button variant="ghost" size="sm" title="Download model">
                            <Download className="w-4 h-4" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </div>
    </div>
  )
}
