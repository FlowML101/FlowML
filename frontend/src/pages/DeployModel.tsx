import { useState, useEffect, useMemo } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  Rocket, Download, Code2, Copy, Check, 
  Terminal, Server, Loader2, AlertCircle,
  CheckCircle, ExternalLink, FileJson
} from 'lucide-react'
import { motion } from 'framer-motion'
import { resultsApi, trainingApi, type TrainedModel, type Job } from '@/lib/api'
import { toast } from 'sonner'

export function DeployModel() {
  const [models, setModels] = useState<TrainedModel[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [selectedJobId, setSelectedJobId] = useState<string>('')
  const [selectedModelId, setSelectedModelId] = useState<string>('')
  const [isLoading, setIsLoading] = useState(true)
  const [copiedSnippet, setCopiedSnippet] = useState<string | null>(null)
  const [isDownloading, setIsDownloading] = useState(false)

  // Get API base URL from current window location
  const apiBaseUrl = `${window.location.protocol}//${window.location.hostname}:8000`

  // Filter models by selected job
  const filteredModels = useMemo(() => {
    if (!selectedJobId) return models
    return models.filter(m => m.job_id === selectedJobId)
  }, [models, selectedJobId])

  const selectedModel = models.find(m => m.id === selectedModelId)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [modelsData, jobsData] = await Promise.all([
          resultsApi.listAllModels(),
          trainingApi.list()
        ])
        setModels(modelsData)
        const completedJobs = jobsData.filter(j => j.status === 'completed')
        setJobs(completedJobs)

        // Auto-select latest job and its best model
        if (completedJobs.length > 0) {
          const latest = completedJobs.sort((a, b) => 
            new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
          )[0]
          setSelectedJobId(latest.id)
          
          const jobModels = modelsData.filter(m => m.job_id === latest.id)
          if (jobModels.length > 0) {
            // Select best model (rank 1 or first)
            const best = jobModels.find(m => m.rank === 1) || jobModels[0]
            setSelectedModelId(best.id)
          }
        }
      } catch (err) {
        console.error('Failed to load data:', err)
        toast.error('Failed to load models')
      } finally {
        setIsLoading(false)
      }
    }
    fetchData()
  }, [])

  // Update selected model when job changes
  useEffect(() => {
    if (filteredModels.length > 0 && !filteredModels.find(m => m.id === selectedModelId)) {
      const best = filteredModels.find(m => m.rank === 1) || filteredModels[0]
      setSelectedModelId(best.id)
    }
  }, [filteredModels, selectedModelId])

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedSnippet(id)
    toast.success('Copied to clipboard!')
    setTimeout(() => setCopiedSnippet(null), 2000)
  }

  const handleDownload = async () => {
    if (!selectedModelId) return
    setIsDownloading(true)
    try {
      const response = await fetch(`${apiBaseUrl}/api/results/model/${selectedModelId}/download`)
      if (!response.ok) throw new Error('Download failed')
      
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${selectedModel?.name || 'model'}_${selectedModelId}.pkl`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
      toast.success('Model downloaded!')
    } catch (err) {
      console.error('Download failed:', err)
      toast.error('Failed to download model')
    } finally {
      setIsDownloading(false)
    }
  }

  // Generate real, working code snippets
  const pythonSnippet = `import requests

# FlowML Prediction API
url = "${apiBaseUrl}/api/results/predict"

# Your input features (adjust based on your training data)
payload = {
    "model_id": "${selectedModelId}",
    "features": {
        # Add your feature columns here, e.g.:
        # "age": 35,
        # "income": 50000,
        # "category": "A"
    }
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result.get('confidence', 'N/A')}")
print(f"Latency: {result.get('latency_ms', 'N/A')}ms")`

  const curlSnippet = `curl -X POST "${apiBaseUrl}/api/results/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model_id": "${selectedModelId}",
    "features": {
      "feature1": "value1",
      "feature2": 123
    }
  }'`

  const pythonLocalSnippet = `import joblib
import pandas as pd

# Load the downloaded model
model = joblib.load("${selectedModel?.name || 'model'}_${selectedModelId}.pkl")

# Prepare your data (must match training features)
data = pd.DataFrame([{
    # Add your feature columns here
    # "age": 35,
    # "income": 50000,
}])

# Make prediction
prediction = model.predict(data)
print(f"Prediction: {prediction[0]}")

# Get probability (if classification)
if hasattr(model, 'predict_proba'):
    proba = model.predict_proba(data)
    print(f"Probabilities: {proba[0]}")`

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-purple-500" />
      </div>
    )
  }

  if (models.length === 0) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <Rocket className="w-8 h-8 text-orange-500" />
            Deploy Model
          </h1>
          <p className="text-muted-foreground">Export and integrate your trained models</p>
        </div>
        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50">
          <CardContent className="p-12 text-center">
            <AlertCircle className="w-12 h-12 mx-auto mb-4 text-yellow-500" />
            <h3 className="text-lg font-medium mb-2">No Trained Models</h3>
            <p className="text-muted-foreground">Train a model first to deploy it</p>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
          <Rocket className="w-8 h-8 text-orange-500" />
          Deploy Model
        </h1>
        <p className="text-muted-foreground">Export and integrate your trained models into applications</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Model Selection */}
        <div className="space-y-4">
          <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50">
            <CardHeader>
              <CardTitle className="text-lg">Select Model</CardTitle>
              <CardDescription>Choose which model to deploy</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Job Filter */}
              <div className="space-y-2">
                <label className="text-sm text-muted-foreground">Training Job</label>
                <Select value={selectedJobId} onValueChange={setSelectedJobId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select job" />
                  </SelectTrigger>
                  <SelectContent>
                    {jobs.map(job => (
                      <SelectItem key={job.id} value={job.id}>
                        {job.name || `Job ${job.id.slice(0, 8)}`}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Model Selection */}
              <div className="space-y-2">
                <label className="text-sm text-muted-foreground">Model</label>
                <Select value={selectedModelId} onValueChange={setSelectedModelId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select model" />
                  </SelectTrigger>
                  <SelectContent>
                    {filteredModels.map(model => (
                      <SelectItem key={model.id} value={model.id}>
                        <div className="flex items-center gap-2">
                          {model.rank === 1 && <Badge variant="default" className="text-[10px] px-1">Best</Badge>}
                          <span>{model.name}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Model Info */}
          {selectedModel && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg flex items-center gap-2">
                    {selectedModel.name}
                    {selectedModel.rank === 1 && (
                      <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/30">
                        🏆 Best
                      </Badge>
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div className="p-2 rounded bg-zinc-800/50">
                      <p className="text-muted-foreground text-xs">Accuracy</p>
                      <p className="font-medium text-green-400">
                        {((selectedModel.accuracy || 0) * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="p-2 rounded bg-zinc-800/50">
                      <p className="text-muted-foreground text-xs">F1 Score</p>
                      <p className="font-medium text-blue-400">
                        {((selectedModel.f1_score || 0) * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="p-2 rounded bg-zinc-800/50">
                      <p className="text-muted-foreground text-xs">Precision</p>
                      <p className="font-medium text-purple-400">
                        {((selectedModel.precision || 0) * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="p-2 rounded bg-zinc-800/50">
                      <p className="text-muted-foreground text-xs">Recall</p>
                      <p className="font-medium text-orange-400">
                        {((selectedModel.recall || 0) * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>

                  <div className="pt-2 border-t border-zinc-800">
                    <p className="text-xs text-muted-foreground mb-1">Model ID</p>
                    <code className="text-xs bg-zinc-800 px-2 py-1 rounded block truncate">
                      {selectedModel.id}
                    </code>
                  </div>

                  <Button 
                    onClick={handleDownload} 
                    disabled={isDownloading}
                    className="w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700"
                  >
                    {isDownloading ? (
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                      <Download className="w-4 h-4 mr-2" />
                    )}
                    Download Model (.pkl)
                  </Button>
                </CardContent>
              </Card>
            </motion.div>
          )}
        </div>

        {/* Right: Integration Options */}
        <div className="lg:col-span-2">
          <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Code2 className="w-5 h-5 text-blue-500" />
                Integration
              </CardTitle>
              <CardDescription>Use your model via API or download</CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="api" className="w-full">
                <TabsList className="grid grid-cols-3 mb-4">
                  <TabsTrigger value="api" className="flex items-center gap-2">
                    <Server className="w-4 h-4" />
                    REST API
                  </TabsTrigger>
                  <TabsTrigger value="python" className="flex items-center gap-2">
                    <Terminal className="w-4 h-4" />
                    Python
                  </TabsTrigger>
                  <TabsTrigger value="local" className="flex items-center gap-2">
                    <FileJson className="w-4 h-4" />
                    Local File
                  </TabsTrigger>
                </TabsList>

                {/* REST API Tab */}
                <TabsContent value="api" className="space-y-4">
                  <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      <span className="font-medium text-green-400">API Ready</span>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Your model is already deployed and accessible via REST API
                    </p>
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium">Endpoint</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleCopy(`${apiBaseUrl}/api/results/predict`, 'endpoint')}
                      >
                        {copiedSnippet === 'endpoint' ? (
                          <Check className="w-4 h-4 text-green-500" />
                        ) : (
                          <Copy className="w-4 h-4" />
                        )}
                      </Button>
                    </div>
                    <code className="block p-3 rounded bg-zinc-800 text-sm text-blue-400 break-all">
                      POST {apiBaseUrl}/api/results/predict
                    </code>
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium">cURL Example</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleCopy(curlSnippet, 'curl')}
                      >
                        {copiedSnippet === 'curl' ? (
                          <Check className="w-4 h-4 text-green-500" />
                        ) : (
                          <Copy className="w-4 h-4" />
                        )}
                      </Button>
                    </div>
                    <pre className="p-3 rounded bg-zinc-800 text-sm overflow-x-auto">
                      <code className="text-zinc-300">{curlSnippet}</code>
                    </pre>
                  </div>

                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => window.open(`${apiBaseUrl}/docs`, '_blank')}
                    >
                      <ExternalLink className="w-4 h-4 mr-2" />
                      API Docs
                    </Button>
                  </div>
                </TabsContent>

                {/* Python Tab */}
                <TabsContent value="python" className="space-y-4">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium">Python (via API)</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleCopy(pythonSnippet, 'python')}
                      >
                        {copiedSnippet === 'python' ? (
                          <Check className="w-4 h-4 text-green-500" />
                        ) : (
                          <Copy className="w-4 h-4" />
                        )}
                      </Button>
                    </div>
                    <pre className="p-3 rounded bg-zinc-800 text-sm overflow-x-auto max-h-80">
                      <code className="text-zinc-300">{pythonSnippet}</code>
                    </pre>
                  </div>

                  <div className="p-3 rounded bg-blue-500/10 border border-blue-500/20">
                    <p className="text-sm text-blue-400">
                      💡 Replace the features dict with your actual column names and values from your training data.
                    </p>
                  </div>
                </TabsContent>

                {/* Local File Tab */}
                <TabsContent value="local" className="space-y-4">
                  <div className="p-4 rounded-lg bg-purple-500/10 border border-purple-500/20">
                    <p className="text-sm text-purple-400">
                      Download the model file and use it directly in Python with joblib. 
                      No server required - runs completely offline.
                    </p>
                  </div>

                  <Button 
                    onClick={handleDownload} 
                    disabled={isDownloading}
                    className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
                  >
                    {isDownloading ? (
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                      <Download className="w-4 h-4 mr-2" />
                    )}
                    Download Model File
                  </Button>

                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium">Usage Example</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleCopy(pythonLocalSnippet, 'local')}
                      >
                        {copiedSnippet === 'local' ? (
                          <Check className="w-4 h-4 text-green-500" />
                        ) : (
                          <Copy className="w-4 h-4" />
                        )}
                      </Button>
                    </div>
                    <pre className="p-3 rounded bg-zinc-800 text-sm overflow-x-auto max-h-80">
                      <code className="text-zinc-300">{pythonLocalSnippet}</code>
                    </pre>
                  </div>

                  <div className="p-3 rounded bg-yellow-500/10 border border-yellow-500/20">
                    <p className="text-sm text-yellow-400">
                      ⚠️ Requires: <code className="bg-zinc-800 px-1 rounded">pip install joblib pandas scikit-learn</code>
                    </p>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
