import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Label } from '@/components/ui/label'
import { Zap, Network, Play, Settings2, Loader2, AlertCircle } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { useDataset } from '@/contexts/DatasetContext'
import { trainingApi, workersApi } from '@/lib/api'
import type { TrainingModelOption } from '@/lib/api'
import { toast } from 'sonner'

type ProblemType = 'classification' | 'regression'

const DEFAULT_MODEL_SELECTION: Record<ProblemType, string[]> = {
  classification: ['xgboost', 'random_forest', 'lightgbm', 'gradient_boosting', 'logistic_regression', 'knn', 'extra_trees'],
  regression: ['xgboost', 'random_forest', 'lightgbm', 'gradient_boosting', 'ridge', 'lasso', 'elastic_net', 'knn'],
}

const CATEGORY_LABELS: Record<string, string> = {
  boosting: 'Boosting',
  tree: 'Tree-based',
  linear: 'Linear',
  instance: 'Instance-based',
  probabilistic: 'Probabilistic',
  other: 'Other',
}

export function TrainingConfig() {
  const navigate = useNavigate()
  const { selectedDataset, previewData, datasets, isLoadingDatasets } = useDataset()
  
  const [targetColumn, setTargetColumn] = useState('')
  const [problemType, setProblemType] = useState<ProblemType>('classification')
  const [timeBudget, setTimeBudget] = useState([15])
  const [selectedModels, setSelectedModels] = useState<string[]>(DEFAULT_MODEL_SELECTION.classification)
  const [modelTypes, setModelTypes] = useState<TrainingModelOption[]>([])
  const [isLoadingModels, setIsLoadingModels] = useState(false)
  const [isStarting, setIsStarting] = useState(false)
  const [workerCount, setWorkerCount] = useState(0)

  // Estimate models that can be trained in the time budget
  const estimatedModels = Math.min(selectedModels.length, Math.max(1, Math.floor(timeBudget[0] / 3)))

  // Get columns from preview data - filter out empty column names
  const columns = (previewData?.columns || []).filter(col => col && col.trim() !== '')

  // Set default target column when data loads
  useEffect(() => {
    if (columns.length > 0 && !targetColumn) {
      // Pick a sensible default: last column that isn't a _duplicated column
      const validColumns = columns.filter(c => !c.startsWith('_duplicated'))
      setTargetColumn(validColumns.length > 0 ? validColumns[validColumns.length - 1] : columns[columns.length - 1])
    }
  }, [previewData?.columns]) // Use previewData.columns directly to avoid recreating dependency

  // Reset target column when dataset changes
  useEffect(() => {
    setTargetColumn('')
  }, [selectedDataset?.id])

  // Fetch worker count
  useEffect(() => {
    workersApi.list().then(workers => {
      setWorkerCount(workers.filter(w => w.status === 'online').length)
    }).catch(() => setWorkerCount(0))
  }, [])

  // Fetch available models for selected problem type
  useEffect(() => {
    const loadModels = async () => {
      setIsLoadingModels(true)
      try {
        const response = await trainingApi.getModels(problemType)
        setModelTypes(response.models)

        const availableIds = new Set(response.models.map(m => m.id))
        const defaults = (DEFAULT_MODEL_SELECTION[problemType] || []).filter(id => availableIds.has(id))
        setSelectedModels(defaults.length > 0 ? defaults : response.models.slice(0, 5).map(m => m.id))
      } catch (err) {
        console.error('Failed to fetch model list:', err)
        toast.error('Failed to load model list for selected problem type')
        setModelTypes([])
        setSelectedModels([])
      } finally {
        setIsLoadingModels(false)
      }
    }

    loadModels()
  }, [problemType])

  const groupedModelTypes = modelTypes.reduce<Record<string, TrainingModelOption[]>>((acc, model) => {
    if (!acc[model.category]) {
      acc[model.category] = []
    }
    acc[model.category].push(model)
    return acc
  }, {})

  const categoryOrder = ['boosting', 'tree', 'linear', 'instance', 'probabilistic', 'other']

  const toggleModel = (modelId: string) => {
    setSelectedModels(prev => 
      prev.includes(modelId) 
        ? prev.filter(m => m !== modelId)
        : [...prev, modelId]
    )
  }

  const handleStartTraining = async () => {
    if (!selectedDataset) {
      toast.error('Please select a dataset first')
      return
    }
    if (!targetColumn) {
      toast.error('Please select a target column')
      return
    }
    if (selectedModels.length === 0) {
      toast.error('Please select at least one model type')
      return
    }

    setIsStarting(true)
    try {
      const job = await trainingApi.start({
        dataset_id: selectedDataset.id,
        target_column: targetColumn,
        time_budget: timeBudget[0],
        model_types: selectedModels,
        problem_type: problemType,
        name: `Training on ${selectedDataset.name}`,
      })
      toast.success('Training job started!')
      navigate('/running', { state: { jobId: job.id } })
    } catch (err: any) {
      console.error('Failed to start training:', err)
      toast.error('Failed to start training: ' + (err.message || 'Unknown error'))
    } finally {
      setIsStarting(false)
    }
  }

  if (isLoadingDatasets) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-purple-500" />
      </div>
    )
  }

  if (datasets.length === 0) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <Zap className="w-8 h-8 text-yellow-500" />
            Training Configuration
          </h1>
          <p className="text-muted-foreground">Configure and launch AutoML training jobs</p>
        </div>
        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50">
          <CardContent className="p-12 text-center">
            <AlertCircle className="w-12 h-12 mx-auto mb-4 text-purple-500" />
            <h3 className="text-lg font-medium mb-2">No Dataset Available</h3>
            <p className="text-muted-foreground mb-4">Upload a dataset in Data Studio to start training</p>
            <Button onClick={() => navigate('/data')}>Go to Data Studio</Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  // If datasets exist but none selected yet, show loading
  if (!selectedDataset) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-purple-500" />
        <span className="ml-3 text-muted-foreground">Loading dataset...</span>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <Zap className="w-8 h-8 text-yellow-500" />
            Training Configuration
          </h1>
          <p className="text-muted-foreground">Configure and launch AutoML training jobs with optimized hyperparameters</p>
        </div>
        <Badge variant="outline" className="text-sm px-4 py-2">
          {workerCount} workers available
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Configuration */}
        <div className="lg:col-span-2 space-y-6">
          {/* Dataset Info */}
          <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-purple-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-violet-500/12">
            <CardHeader className="relative">
              <CardTitle>Dataset</CardTitle>
              <CardDescription>{selectedDataset?.name || 'No dataset selected'}</CardDescription>
            </CardHeader>
            <CardContent className="relative">
              <div className="flex gap-2">
                <Badge variant="outline">{(selectedDataset?.num_rows ?? 0).toLocaleString()} rows</Badge>
                <Badge variant="outline">{selectedDataset?.num_columns ?? 0} columns</Badge>
                <Badge variant="outline" className="capitalize">{problemType}</Badge>
              </div>
            </CardContent>
          </Card>

          {/* Target Column */}
          <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-purple-500/10 before:via-violet-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-yellow-500/12">
            <CardHeader className="relative">
              <CardTitle className="flex items-center gap-2">
                <Settings2 className="w-5 h-5 text-purple-500" />
                Model Configuration
              </CardTitle>
              <CardDescription>Select prediction target and training parameters</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6 relative">
              {/* Target Column */}
              <div className="space-y-2">
                <Label htmlFor="target">Target Column</Label>
                <Select value={targetColumn} onValueChange={setTargetColumn}>
                  <SelectTrigger id="target">
                    <SelectValue placeholder="Select target column" />
                  </SelectTrigger>
                  <SelectContent>
                    {columns.map(col => (
                      <SelectItem key={col} value={col}>{col}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  Column to predict (dependent variable)
                </p>
              </div>

              {/* Problem Type */}
              <div className="space-y-2">
                <Label htmlFor="problem-type">Problem Type</Label>
                <Select value={problemType} onValueChange={(value: ProblemType) => setProblemType(value)}>
                  <SelectTrigger id="problem-type">
                    <SelectValue placeholder="Select problem type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="classification">Classification</SelectItem>
                    <SelectItem value="regression">Regression</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  Choose the ML task to show compatible models only.
                </p>
              </div>

              {/* Time Budget */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label>Time Budget</Label>
                  <Badge variant="secondary">{timeBudget[0]} minutes</Badge>
                </div>
                <Slider
                  value={timeBudget}
                  onValueChange={setTimeBudget}
                  min={1}
                  max={60}
                  step={1}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>1 min (quick)</span>
                  <span>60 min (thorough)</span>
                </div>
                <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
                  <p className="text-xs text-blue-400">
                    ⏱️ <strong>~{estimatedModels} models</strong> will be trained with hyperparameter tuning.
                    {timeBudget[0] < 5 && <span className="text-yellow-400"> Consider 5+ min for better results.</span>}
                  </p>
                </div>
              </div>

              {/* Model Types */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>Model Types</Label>
                  <div className="flex gap-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-xs h-7"
                      onClick={() => setSelectedModels(modelTypes.map(m => m.id))}
                      disabled={isLoadingModels || modelTypes.length === 0}
                    >
                      Select All
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-xs h-7"
                      onClick={() => setSelectedModels([])}
                      disabled={isLoadingModels || modelTypes.length === 0}
                    >
                      Clear
                    </Button>
                  </div>
                </div>
                {isLoadingModels ? (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Loading models for {problemType}...
                  </div>
                ) : (
                  <div className="space-y-3">
                    {categoryOrder
                      .filter(category => groupedModelTypes[category]?.length)
                      .map(category => (
                        <div key={category} className="space-y-2">
                          <div className="text-xs uppercase tracking-wide text-muted-foreground">
                            {CATEGORY_LABELS[category] || category}
                          </div>
                          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-2">
                            {groupedModelTypes[category].map((model) => (
                              <div
                                key={model.id}
                                onClick={() => toggleModel(model.id)}
                                className={`flex items-center space-x-2 p-2.5 rounded-lg cursor-pointer transition-all border ${
                                  selectedModels.includes(model.id)
                                    ? 'bg-purple-500/20 border-purple-500/50 text-purple-300'
                                    : 'bg-zinc-800/30 border-zinc-700 hover:border-zinc-600'
                                }`}
                              >
                                <input
                                  type="checkbox"
                                  checked={selectedModels.includes(model.id)}
                                  onChange={() => {}}
                                  className="w-3.5 h-3.5 rounded border-zinc-600 text-purple-600 focus:ring-purple-500"
                                />
                                <span className="text-xs font-medium">{model.name}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                  </div>
                )}
                <p className="text-xs text-muted-foreground">
                  {selectedModels.length} of {modelTypes.length} models selected
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Compute Configuration */}
        <div className="space-y-6">
          <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-purple-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-green-500/12">
            <CardHeader className="relative">
              <CardTitle>Compute Mode</CardTitle>
              <CardDescription>Automatic distributed training across your mesh</CardDescription>
            </CardHeader>
            <CardContent className="relative">
              <div className="p-4 rounded-lg border-2 border-purple-600 bg-purple-600/10">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Network className="w-5 h-5 text-purple-400" />
                    <span className="font-medium">Auto-Distributed</span>
                  </div>
                  <Badge variant="success" className="text-xs">Active</Badge>
                </div>
                <p className="text-xs text-muted-foreground mb-3">
                  Intelligently distributes workload across your mesh network ({workerCount} workers available)
                </p>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between text-zinc-400">
                    <span>Master Node:</span>
                    <span className="text-zinc-300">localhost</span>
                  </div>
                  <div className="flex justify-between text-zinc-400">
                    <span>Active Workers:</span>
                    <span className="text-green-400">{workerCount} online</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Start Training */}
          <Card className="border-zinc-600/50 bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:opacity-70 transition-all duration-300 hover:shadow-xl hover:shadow-purple-600/15">
            <CardHeader className="relative">
              <CardTitle className="text-lg">Ready to Launch</CardTitle>
            </CardHeader>
            <CardContent className="relative">
              <Button
                onClick={handleStartTraining}
                disabled={isStarting || isLoadingModels || !selectedDataset || !targetColumn || selectedModels.length === 0}
                className="w-full bg-gradient-to-r from-purple-600 to-purple-600 hover:from-purple-700 hover:to-purple-700 text-lg py-6 disabled:opacity-50"
              >
                {isStarting ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5 mr-2" />
                    Start Training
                  </>
                )}
              </Button>
              <p className="text-xs text-muted-foreground mt-3 text-center">
                {timeBudget[0]} min budget • ~{estimatedModels} models trained • Auto-Distributed
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}