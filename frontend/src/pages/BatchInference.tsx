import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { 
  Upload, Download, Code2, Copy, CheckCircle, 
  FileText, Rocket, Terminal, Container, Play,
  Server, Zap
} from 'lucide-react'
import { motion } from 'framer-motion'

const curlSnippet = `curl -X POST https://api.flowml.io/v1/predict \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model_id": "xgboost-v1",
    "features": {
      "age": 29,
      "sex": "male",
      "pclass": 3,
      "sibsp": 0,
      "parch": 0
    }
  }'`

const pythonSnippet = `import requests

url = "https://api.flowml.io/v1/predict"
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "model_id": "xgboost-v1",
    "features": {
        "age": 29,
        "sex": "male",
        "pclass": 3,
        "sibsp": 0,
        "parch": 0
    }
}

response = requests.post(url, json=payload, headers=headers)
prediction = response.json()
print(f"Prediction: {prediction['result']}")`

const dockerSnippet = `# Pull the FlowML Docker image
docker pull flowml/xgboost-v1:latest

# Run inference server on port 8080
docker run -d -p 8080:8080 \\
  --name flowml-inference \\
  -e MODEL_PATH=/models/xgboost-v1.pkl \\
  flowml/xgboost-v1:latest

# Test the endpoint
curl -X POST http://localhost:8080/predict \\
  -H "Content-Type: application/json" \\
  -d '{"age": 29, "sex": "male", "pclass": 3}'`

export function BatchInference() {
  const [uploadedFile, setUploadedFile] = useState<string | null>(null)
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null)

  const handleCopy = (text: string, index: number) => {
    navigator.clipboard.writeText(text)
    setCopiedIndex(index)
    setTimeout(() => setCopiedIndex(null), 2000)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <Zap className="w-8 h-8 text-yellow-500" />
            Batch Inference & API
          </h1>
          <p className="text-muted-foreground">Deploy models and run batch predictions</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Batch Upload */}
        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-purple-500/10 before:via-transparent before:to-blue-500/10 before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-purple-500/12">
          <CardHeader className="relative">
            <CardTitle className="flex items-center gap-2">
              <Upload className="w-5 h-5 text-cyan-400" />
              Batch Predictions
            </CardTitle>
            <CardDescription>
              Upload CSV for batch inference
            </CardDescription>
          </CardHeader>
          <CardContent className="relative space-y-4">
            {/* File Upload Zone */}
            <div className="border-2 border-dashed border-zinc-700 rounded-lg p-8 text-center hover:border-cyan-500 transition-all cursor-pointer">
              <Upload className="w-12 h-12 text-muted-foreground mx-auto mb-3" />
              <p className="text-sm font-medium mb-1">Drop CSV file or click to browse</p>
              <p className="text-xs text-muted-foreground">Max size: 100MB</p>
              <input
                type="file"
                accept=".csv"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files?.[0]) {
                    setUploadedFile(e.target.files[0].name)
                  }
                }}
              />
            </div>

            {uploadedFile && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-4 rounded-lg bg-cyan-500/10 border border-cyan-600"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <FileText className="w-5 h-5 text-cyan-400" />
                    <div>
                      <p className="text-sm font-medium">{uploadedFile}</p>
                      <p className="text-xs text-muted-foreground">Ready for inference</p>
                    </div>
                  </div>
                  <CheckCircle className="w-5 h-5 text-green-500" />
                </div>
              </motion.div>
            )}

            {/* Model Selection */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Select Model</label>
              <select className="w-full p-2 rounded-lg bg-zinc-800 border border-zinc-700 text-sm">
                <option>XGBoost v1 (94.2% accuracy)</option>
                <option>CatBoost v1 (95.1% accuracy)</option>
                <option>Random Forest v2 (92.8% accuracy)</option>
              </select>
            </div>

            {/* Settings */}
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <label className="text-sm font-medium">Batch Size</label>
                <Input type="number" defaultValue="32" className="bg-zinc-800" />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Workers</label>
                <Input type="number" defaultValue="4" className="bg-zinc-800" />
              </div>
            </div>

            <Button className="w-full bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700">
              <Play className="w-4 h-4 mr-2" />
              Run Batch Inference
            </Button>

            {/* Results Preview */}
            {uploadedFile && (
              <div className="space-y-2 pt-4 border-t border-zinc-700">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium">Results</p>
                  <Button variant="outline" size="sm">
                    <Download className="w-3 h-3 mr-2" />
                    Download
                  </Button>
                </div>
                <div className="bg-zinc-950 rounded-lg p-3 border border-zinc-800">
                  <div className="text-xs font-mono space-y-1">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Total Rows:</span>
                      <span className="text-green-400">891</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Processed:</span>
                      <span className="text-green-400">891</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Avg Confidence:</span>
                      <span className="text-blue-400">0.847</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* API Playground */}
        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-purple-500/10 before:via-transparent before:to-blue-500/10 before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-purple-500/12">
          <CardHeader className="relative">
            <CardTitle className="flex items-center gap-2">
              <Terminal className="w-5 h-5 text-blue-400" />
              API Playground
            </CardTitle>
            <CardDescription>
              Test REST API endpoints
            </CardDescription>
          </CardHeader>
          <CardContent className="relative space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">API Endpoint</label>
              <div className="flex gap-2">
                <Badge variant="secondary">POST</Badge>
                <Input 
                  value="https://api.flowml.io/v1/predict" 
                  readOnly
                  className="bg-zinc-800 font-mono text-sm"
                />
              </div>
            </div>

            <Tabs defaultValue="curl">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="curl">cURL</TabsTrigger>
                <TabsTrigger value="python">Python</TabsTrigger>
                <TabsTrigger value="docker">Docker</TabsTrigger>
              </TabsList>

              <TabsContent value="curl" className="space-y-3 mt-4">
                <div className="relative">
                  <pre className="bg-zinc-950 rounded-lg p-4 text-xs font-mono overflow-x-auto border border-zinc-800">
                    <code>{curlSnippet}</code>
                  </pre>
                  <Button
                    size="sm"
                    variant="outline"
                    className="absolute top-2 right-2"
                    onClick={() => handleCopy(curlSnippet, 0)}
                  >
                    {copiedIndex === 0 ? (
                      <CheckCircle className="w-3 h-3 text-green-500" />
                    ) : (
                      <Copy className="w-3 h-3" />
                    )}
                  </Button>
                </div>
              </TabsContent>

              <TabsContent value="python" className="space-y-3 mt-4">
                <div className="relative">
                  <pre className="bg-zinc-950 rounded-lg p-4 text-xs font-mono overflow-x-auto border border-zinc-800">
                    <code>{pythonSnippet}</code>
                  </pre>
                  <Button
                    size="sm"
                    variant="outline"
                    className="absolute top-2 right-2"
                    onClick={() => handleCopy(pythonSnippet, 1)}
                  >
                    {copiedIndex === 1 ? (
                      <CheckCircle className="w-3 h-3 text-green-500" />
                    ) : (
                      <Copy className="w-3 h-3" />
                    )}
                  </Button>
                </div>
              </TabsContent>

              <TabsContent value="docker" className="space-y-3 mt-4">
                <div className="relative">
                  <pre className="bg-zinc-950 rounded-lg p-4 text-xs font-mono overflow-x-auto border border-zinc-800">
                    <code>{dockerSnippet}</code>
                  </pre>
                  <Button
                    size="sm"
                    variant="outline"
                    className="absolute top-2 right-2"
                    onClick={() => handleCopy(dockerSnippet, 2)}
                  >
                    {copiedIndex === 2 ? (
                      <CheckCircle className="w-3 h-3 text-green-500" />
                    ) : (
                      <Copy className="w-3 h-3" />
                    )}
                  </Button>
                </div>
              </TabsContent>
            </Tabs>

            <Button className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700">
              <Play className="w-4 h-4 mr-2" />
              Test API Call
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Deployment Guide */}
      <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-purple-500/10 before:via-transparent before:to-blue-500/10 before:opacity-30">
        <CardHeader className="relative">
          <CardTitle className="flex items-center gap-2">
            <Rocket className="w-5 h-5 text-purple-400" />
            Deployment Guide
          </CardTitle>
          <CardDescription>
            Export and deploy your models
          </CardDescription>
        </CardHeader>
        <CardContent className="relative">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 rounded-lg bg-zinc-800/50 border border-zinc-700 space-y-3">
              <Container className="w-8 h-8 text-blue-400" />
              <h4 className="font-medium">Docker Container</h4>
              <p className="text-sm text-muted-foreground">
                Pre-built containers with optimized inference runtime
              </p>
              <Button variant="outline" size="sm" className="w-full">
                Export Docker Image
              </Button>
            </div>

            <div className="p-4 rounded-lg bg-zinc-800/50 border border-zinc-700 space-y-3">
              <Server className="w-8 h-8 text-green-400" />
              <h4 className="font-medium">REST API</h4>
              <p className="text-sm text-muted-foreground">
                Production-ready FastAPI server with authentication
              </p>
              <Button variant="outline" size="sm" className="w-full">
                Generate API Server
              </Button>
            </div>

            <div className="p-4 rounded-lg bg-zinc-800/50 border border-zinc-700 space-y-3">
              <Code2 className="w-8 h-8 text-purple-400" />
              <h4 className="font-medium">Python Package</h4>
              <p className="text-sm text-muted-foreground">
                Standalone .pkl or .joblib for custom integration
              </p>
              <Button variant="outline" size="sm" className="w-full">
                Download Model File
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
