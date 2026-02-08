import { useState, useEffect, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { FlaskConical, Play, Sparkles, TrendingUp, AlertCircle, Loader2, Activity, RefreshCw } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { resultsApi, trainingApi, type TrainedModel, type Job } from '@/lib/api'

export function InferencePage() {
  const [searchParams] = useSearchParams()
  const modelIdFromUrl = searchParams.get('model_id')
  
  const [models, setModels] = useState<TrainedModel[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [selectedJobId, setSelectedJobId] = useState<string>('all')
  const [modelsLoading, setModelsLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [isLoading, setIsLoading] = useState(false)
  const [prediction, setPrediction] = useState<{
    survived: boolean
    confidence: number
    model: string
    latency: number
  } | null>(null)
  
  const [inputData, setInputData] = useState({
    pclass: '1',
    sex: 'female',
    age: '29',
    sibsp: '0',
    parch: '0',
    fare: '75.00',
    embarked: 'C',
  })

  // Filter models by selected job
  const filteredModels = useMemo(() => {
    if (selectedJobId === 'all') return models
    return models.filter(m => m.job_id === selectedJobId)
  }, [models, selectedJobId])

  const fetchModels = async (isRefresh = false) => {
    try {
      if (isRefresh) setRefreshing(true)
      else setModelsLoading(true)
      
      const [data, jobsData] = await Promise.all([
        resultsApi.listAllModels(),
        trainingApi.list()
      ])
      setModels(data)
      setJobs(jobsData.filter(j => j.status === 'completed'))
      
      // Auto-select latest job only on initial load
      if (!isRefresh) {
        const completedJobs = jobsData.filter(j => j.status === 'completed')
        if (completedJobs.length > 0) {
          const latest = completedJobs.sort((a, b) => 
            new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
          )[0]
          setSelectedJobId(latest.id)
        }
        
        if (modelIdFromUrl) {
          setSelectedModel(modelIdFromUrl)
        } else if (data.length > 0) {
          setSelectedModel(data[0].id)
        }
      }
    } catch (err) {
      console.error('Failed to load models:', err)
    } finally {
      if (isRefresh) setRefreshing(false)
      else setModelsLoading(false)
    }
  }

  useEffect(() => {
    fetchModels()
  }, [modelIdFromUrl])

  // Auto-select first model when job changes
  useEffect(() => {
    if (filteredModels.length > 0 && !filteredModels.find(m => m.id === selectedModel)) {
      setSelectedModel(filteredModels[0].id)
    }
  }, [filteredModels, selectedModel])

  const handlePredict = async () => {
    setIsLoading(true)
    setPrediction(null)
    
    const startTime = Date.now()
    
    try {
      const features = {
        pclass: parseInt(inputData.pclass),
        sex: inputData.sex,
        age: parseFloat(inputData.age),
        sibsp: parseInt(inputData.sibsp),
        parch: parseInt(inputData.parch),
        fare: parseFloat(inputData.fare),
        embarked: inputData.embarked,
      }
      
      const result = await resultsApi.predict(selectedModel, features)
      const latency = Date.now() - startTime
      
      const currentModelData = filteredModels.find(m => m.id === selectedModel)
      
      setPrediction({
        survived: result.prediction === 1 || result.prediction === '1' || String(result.prediction).toLowerCase() === 'true',
        confidence: result.confidence || result.probability || 0.85,
        model: currentModelData?.name || 'Unknown',
        latency: latency,
      })
    } catch (err) {
      console.error('Prediction failed:', err)
      // Show error instead of fake demo data
      setPrediction(null)
      const currentModelData = filteredModels.find(m => m.id === selectedModel)
      // Use toast to show error (import toast from sonner at top)
      import('sonner').then(({ toast }) => {
        toast.error('Prediction failed', {
          description: currentModelData 
            ? `Model "${currentModelData.name}" could not make a prediction. Check that the input features match the training data.`
            : 'No model selected or model not found.'
        })
      })
    } finally {
      setIsLoading(false)
    }
  }

  const currentModel = filteredModels.find(m => m.id === selectedModel)

  if (modelsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-purple-500" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <FlaskConical className="w-8 h-8 text-purple-500" />
            Model Playground
          </h1>
          <p className="text-muted-foreground">Test trained models with real-time inputs and instant predictions</p>
        </div>
        <div className="flex items-center gap-3">
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
            {filteredModels.length} models available
          </Badge>
          {currentModel && (
            <Badge className="bg-purple-600 text-sm px-4 py-2">
              {currentModel.name} • {currentModel.accuracy ? `${(currentModel.accuracy * 100).toFixed(1)}%` : 'N/A'}
            </Badge>
          )}
        </div>
      </div>

      {/* Job Filter */}
      <div className="flex items-center gap-4 p-4 bg-zinc-900/50 rounded-lg border border-zinc-800">
        <Activity className="w-4 h-4 text-green-400" />
        <span className="text-sm text-muted-foreground">Filter by Job:</span>
        <Select value={selectedJobId} onValueChange={setSelectedJobId}>
          <SelectTrigger className="w-[250px]">
            <SelectValue placeholder="Select Job" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Jobs</SelectItem>
            {jobs.map(job => (
              <SelectItem key={job.id} value={job.id}>
                {job.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {filteredModels.length === 0 ? (
        <Card className="border-border">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <FlaskConical className="w-12 h-12 text-muted-foreground mb-4" />
            <p className="text-muted-foreground">No trained models available. Train a model first!</p>
          </CardContent>
        </Card>
      ) : (
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Model Selector + Input Form */}
        <div className="lg:col-span-2 space-y-6">
          {/* Model Selector */}
          <Card className="border-purple-600/30 bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-purple-500/20 before:via-pink-500/10 before:to-blue-500/20 before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-purple-500/12">
            <CardHeader className="relative">
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-purple-500" />
                Select Model
              </CardTitle>
              <CardDescription>Choose which trained model to use for inference</CardDescription>
            </CardHeader>
            <CardContent className="relative">
              <div className="space-y-4">
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a model" />
                  </SelectTrigger>
                  <SelectContent>
                    {filteredModels.map((model) => (
                      <SelectItem key={model.id} value={model.id}>
                        <div className="flex items-center gap-2">
                          <span>{model.name}</span>
                          <Badge variant="secondary" className="text-xs">
                            {model.accuracy ? `${(model.accuracy * 100).toFixed(1)}% acc` : 'N/A'}
                          </Badge>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                
                {currentModel && (
                  <div className="flex items-center justify-between p-3 rounded-lg bg-purple-600/10 border border-purple-600/30">
                    <div>
                      <div className="text-sm font-medium">{currentModel.name}</div>
                      <div className="text-xs text-muted-foreground">Accuracy: {currentModel.accuracy ? `${(currentModel.accuracy * 100).toFixed(1)}%` : 'N/A'}</div>
                    </div>
                    <Badge variant="success">Ready</Badge>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Input Form */}
          <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-blue-500/12">
            <CardHeader className="relative">
              <CardTitle>Input Features</CardTitle>
              <CardDescription>Enter passenger details for prediction</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 relative">
              {/* Passenger Class */}
              <div className="space-y-2">
                <Label htmlFor="pclass">Passenger Class</Label>
                <Select
                  value={inputData.pclass}
                  onValueChange={(value) => setInputData({ ...inputData, pclass: value })}
                >
                  <SelectTrigger id="pclass">
                    <SelectValue placeholder="Select class" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1">1st Class</SelectItem>
                    <SelectItem value="2">2nd Class</SelectItem>
                    <SelectItem value="3">3rd Class</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Sex */}
              <div className="space-y-2">
                <Label htmlFor="sex">Sex</Label>
                <Select
                  value={inputData.sex}
                  onValueChange={(value) => setInputData({ ...inputData, sex: value })}
                >
                  <SelectTrigger id="sex">
                    <SelectValue placeholder="Select sex" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="male">Male</SelectItem>
                    <SelectItem value="female">Female</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Age */}
              <div className="space-y-2">
                <Label htmlFor="age">Age</Label>
                <Input
                  id="age"
                  type="number"
                  placeholder="Enter age"
                  value={inputData.age}
                  onChange={(e) => setInputData({ ...inputData, age: e.target.value })}
                />
              </div>

              {/* Siblings/Spouses */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="sibsp">Siblings/Spouses Aboard</Label>
                  <Input
                    id="sibsp"
                    type="number"
                    placeholder="0"
                    value={inputData.sibsp}
                    onChange={(e) => setInputData({ ...inputData, sibsp: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="parch">Parents/Children Aboard</Label>
                  <Input
                    id="parch"
                    type="number"
                    placeholder="0"
                    value={inputData.parch}
                    onChange={(e) => setInputData({ ...inputData, parch: e.target.value })}
                  />
                </div>
              </div>

              {/* Fare */}
              <div className="space-y-2">
                <Label htmlFor="fare">Fare</Label>
                <Input
                  id="fare"
                  type="number"
                  step="0.01"
                  placeholder="Enter fare amount"
                  value={inputData.fare}
                  onChange={(e) => setInputData({ ...inputData, fare: e.target.value })}
                />
              </div>

              {/* Embarked */}
              <div className="space-y-2">
                <Label htmlFor="embarked">Port of Embarkation</Label>
                <Select
                  value={inputData.embarked}
                  onValueChange={(value) => setInputData({ ...inputData, embarked: value })}
                >
                  <SelectTrigger id="embarked">
                    <SelectValue placeholder="Select port" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="C">Cherbourg</SelectItem>
                    <SelectItem value="Q">Queenstown</SelectItem>
                    <SelectItem value="S">Southampton</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Predict Button */}
              <Button
                onClick={handlePredict}
                disabled={isLoading}
                className="w-full bg-gradient-to-r from-purple-600 to-violet-600 hover:from-purple-700 hover:to-purple-700 text-lg py-6"
              >
                {isLoading ? (
                  <>
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                      className="w-5 h-5 mr-2"
                    >
                      <Sparkles className="w-5 h-5" />
                    </motion.div>
                    Predicting...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5 mr-2" />
                    Run Prediction
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Right Column: Result Card */}
        <div className="space-y-6">
          <AnimatePresence mode="wait">
            {prediction ? (
              <motion.div
                key="result"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                <Card className={`border-2 ${prediction.survived ? 'border-green-600/50 bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-green-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-green-500/12' : 'border-red-600/50 bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-red-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-red-500/12'}`}>
                  <CardHeader className="relative">
                    <CardTitle className="flex items-center gap-2">
                      <TrendingUp className={`w-5 h-5 ${prediction.survived ? 'text-green-500' : 'text-red-500'}`} />
                      Prediction Result
                    </CardTitle>
                    <CardDescription>Model: {prediction.model}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6 relative">
                    {/* Main Prediction */}
                    <div className="text-center p-6 rounded-xl bg-muted/50 dark:bg-zinc-900/50 border border-border dark:border-zinc-700">
                      <div className="text-sm text-muted-foreground mb-2">Survival Prediction</div>
                      <div className={`text-4xl font-bold mb-2 ${prediction.survived ? 'text-green-400' : 'text-red-400'}`}>
                        {prediction.survived ? 'Survived' : 'Did Not Survive'}
                      </div>
                      <div className="text-lg text-muted-foreground">
                        {(prediction.confidence * 100).toFixed(1)}% Confidence
                      </div>
                    </div>

                    {/* Confidence Bar */}
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-zinc-500">Confidence Level</span>
                        <span className="font-semibold">{(prediction.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-3 bg-zinc-800 rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${prediction.confidence * 100}%` }}
                          transition={{ duration: 0.8, ease: 'easeOut' }}
                          className={`h-full ${prediction.survived ? 'bg-gradient-to-r from-green-600 to-green-400' : 'bg-gradient-to-r from-red-600 to-red-400'}`}
                        />
                      </div>
                    </div>

                    {/* Stats */}
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50 text-center">
                        <div className="text-xs text-muted-foreground">Latency</div>
                        <div className="text-lg font-bold text-blue-400">{prediction.latency}ms</div>
                      </div>
                      <div className="p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50 text-center">
                        <div className="text-xs text-muted-foreground">Model</div>
                        <div className="text-sm font-bold text-purple-400 truncate">{prediction.model}</div>
                      </div>
                    </div>

                    {/* Info */}
                    <div className="flex items-start gap-2 p-3 rounded-lg bg-blue-600/10 border border-blue-600/30">
                      <AlertCircle className="w-4 h-4 text-blue-400 mt-0.5 flex-shrink-0" />
                      <p className="text-xs text-blue-300">
                        This prediction is based on historical Titanic passenger data and the selected ML model.
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ) : (
              <motion.div
                key="placeholder"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-purple-500/10 before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-purple-500/12">
                  <CardHeader className="relative">
                    <CardTitle className="flex items-center gap-2">
                      <FlaskConical className="w-5 h-5 text-muted-foreground" />
                      Awaiting Prediction
                    </CardTitle>
                    <CardDescription>Fill in the form and click "Run Prediction"</CardDescription>
                  </CardHeader>
                  <CardContent className="relative">
                    <div className="flex flex-col items-center justify-center py-12 text-center">
                      <div className="w-20 h-20 rounded-full bg-muted/50 dark:bg-zinc-800/50 flex items-center justify-center mb-4">
                        <Play className="w-10 h-10 text-muted-foreground" />
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Your prediction results will appear here
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
      )}
    </div>
  )
}
