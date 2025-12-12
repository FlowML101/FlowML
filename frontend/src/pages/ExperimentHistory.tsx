import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  History, Search, 
  Download, Play, Code2, Database, Cpu,
  CheckCircle, GitCommit, Info
} from 'lucide-react'
import { motion } from 'framer-motion'

const mockExperiments = [
  {
    id: 'exp-2024-001',
    name: 'Titanic Survival - XGBoost Tuning',
    status: 'completed',
    timestamp: '2024-12-12 14:23:15',
    duration: '2m 14s',
    accuracy: 94.2,
    model: 'XGBoost',
    dataset: 'titanic_v2.csv',
    params: {
      max_depth: 6,
      learning_rate: 0.1,
      n_estimators: 200,
      subsample: 0.8
    },
    environment: {
      python: '3.11.5',
      sklearn: '1.3.0',
      xgboost: '2.0.1'
    },
    seed: 42,
    gpuUsed: true
  },
  {
    id: 'exp-2024-002',
    name: 'Feature Engineering Experiment',
    status: 'completed',
    timestamp: '2024-12-12 15:47:32',
    duration: '1m 52s',
    accuracy: 92.8,
    model: 'Random Forest',
    dataset: 'titanic_v3_engineered.csv',
    params: {
      n_estimators: 150,
      max_depth: 10,
      min_samples_split: 4
    },
    environment: {
      python: '3.11.5',
      sklearn: '1.3.0'
    },
    seed: 42,
    gpuUsed: false
  },
  {
    id: 'exp-2024-003',
    name: 'CatBoost Deep Tuning',
    status: 'running',
    timestamp: '2024-12-12 16:12:05',
    duration: '0m 45s',
    accuracy: null,
    model: 'CatBoost',
    dataset: 'titanic_v2.csv',
    params: {
      iterations: 500,
      learning_rate: 0.05,
      depth: 7
    },
    environment: {
      python: '3.11.5',
      catboost: '1.2.0'
    },
    seed: 42,
    gpuUsed: true
  }
]

export function ExperimentHistory() {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedExperiment, setSelectedExperiment] = useState(mockExperiments[0])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <History className="w-8 h-8 text-purple-500" />
            Experiment History
          </h1>
          <p className="text-muted-foreground">Track, version, and reproduce all training runs</p>
        </div>
        <Button className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700">
          <Download className="w-4 h-4 mr-2" />
          Export History
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Experiments List */}
        <div className="lg:col-span-1 space-y-4">
          <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50">
            <CardHeader>
              <CardTitle className="text-lg">All Experiments</CardTitle>
              <CardDescription>
                {mockExperiments.length} total runs
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="relative">
                <Search className="absolute left-3 top-2.5 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search experiments..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9 bg-zinc-800/50"
                />
              </div>

              <div className="space-y-2 max-h-[600px] overflow-y-auto">
                {mockExperiments.map((exp) => (
                  <motion.button
                    key={exp.id}
                    whileHover={{ scale: 1.02 }}
                    onClick={() => setSelectedExperiment(exp)}
                    className={`w-full p-3 rounded-lg border text-left transition-all ${
                      selectedExperiment.id === exp.id
                        ? 'border-purple-500 bg-purple-500/10'
                        : 'border-zinc-700 hover:border-zinc-600'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium truncate">{exp.name}</span>
                      {exp.status === 'completed' ? (
                        <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" />
                      ) : (
                        <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse flex-shrink-0" />
                      )}
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Badge variant="outline" className="text-xs">{exp.model}</Badge>
                      {exp.accuracy && (
                        <span className="text-green-400 font-medium">{exp.accuracy}%</span>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground mt-2">{exp.timestamp}</p>
                  </motion.button>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Experiment Details */}
        <div className="lg:col-span-2 space-y-4">
          <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-green-500/10 before:via-emerald-500/10 before:to-transparent before:opacity-30">
            <CardHeader className="relative">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-xl">{selectedExperiment.name}</CardTitle>
                  <CardDescription className="flex items-center gap-2 mt-2">
                    <GitCommit className="w-3 h-3" />
                    {selectedExperiment.id}
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Badge className={selectedExperiment.status === 'completed' ? 'bg-green-600' : 'bg-yellow-600'}>
                    {selectedExperiment.status}
                  </Badge>
                  {selectedExperiment.status === 'completed' && (
                    <Button size="sm" className="bg-gradient-to-r from-purple-600 to-blue-600">
                      <Play className="w-3 h-3 mr-1.5" />
                      Reproduce
                    </Button>
                  )}
                </div>
              </div>
            </CardHeader>
            <CardContent className="relative space-y-6">
              {/* Metrics */}
              {selectedExperiment.accuracy && (
                <div className="grid grid-cols-4 gap-4">
                  <div className="p-4 rounded-lg bg-zinc-800/50 border border-zinc-700">
                    <div className="text-2xl font-bold text-green-400">{selectedExperiment.accuracy}%</div>
                    <p className="text-xs text-muted-foreground mt-1">Accuracy</p>
                  </div>
                  <div className="p-4 rounded-lg bg-zinc-800/50 border border-zinc-700">
                    <div className="text-2xl font-bold">{selectedExperiment.duration}</div>
                    <p className="text-xs text-muted-foreground mt-1">Duration</p>
                  </div>
                  <div className="p-4 rounded-lg bg-zinc-800/50 border border-zinc-700">
                    <div className="text-2xl font-bold">{selectedExperiment.seed}</div>
                    <p className="text-xs text-muted-foreground mt-1">Random Seed</p>
                  </div>
                  <div className="p-4 rounded-lg bg-zinc-800/50 border border-zinc-700">
                    <div className="flex items-center gap-2">
                      <Cpu className={`w-5 h-5 ${selectedExperiment.gpuUsed ? 'text-green-500' : 'text-zinc-500'}`} />
                      <span className="text-lg font-bold">{selectedExperiment.gpuUsed ? 'GPU' : 'CPU'}</span>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">Compute</p>
                  </div>
                </div>
              )}

              <Tabs defaultValue="params">
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="params">Parameters</TabsTrigger>
                  <TabsTrigger value="diff">Parameter Diff</TabsTrigger>
                  <TabsTrigger value="environment">Environment</TabsTrigger>
                  <TabsTrigger value="dataset">Dataset</TabsTrigger>
                </TabsList>

                <TabsContent value="params" className="space-y-3 mt-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Code2 className="w-4 h-4 text-purple-400" />
                    <h4 className="font-medium">Model Configuration</h4>
                  </div>
                  <div className="bg-zinc-950 rounded-lg p-4 border border-zinc-800">
                    <pre className="text-sm font-mono">
                      <code>{JSON.stringify(selectedExperiment.params, null, 2)}</code>
                    </pre>
                  </div>
                  <Button variant="outline" size="sm" className="w-full">
                    <Download className="w-3 h-3 mr-2" />
                    Export Configuration
                  </Button>
                </TabsContent>

                <TabsContent value="diff" className="space-y-3 mt-4">
                  <div className="flex items-center gap-2 mb-3">
                    <GitCommit className="w-4 h-4 text-cyan-400" />
                    <h4 className="font-medium">Git-like Parameter Diff</h4>
                  </div>
                  <div className="bg-zinc-950 rounded-lg p-4 border border-zinc-800 font-mono text-sm space-y-1">
                    <div className="text-muted-foreground mb-2">
                      <span className="text-red-400">--- exp-2024-001</span> (previous experiment)<br/>
                      <span className="text-green-400">+++ {selectedExperiment.id}</span> (current)
                    </div>
                    <div className="border-t border-zinc-700 pt-2 mt-2">
                      <div className="text-cyan-400 mb-1">@@ Model Parameters @@</div>
                      {selectedExperiment.id === 'exp-2024-001' ? (
                        <>
                          <div className="bg-green-500/10 text-green-400 pl-2">+ max_depth: 6</div>
                          <div className="bg-green-500/10 text-green-400 pl-2">+ learning_rate: 0.1</div>
                          <div className="bg-green-500/10 text-green-400 pl-2">+ n_estimators: 200</div>
                          <div className="bg-green-500/10 text-green-400 pl-2">+ subsample: 0.8</div>
                        </>
                      ) : selectedExperiment.id === 'exp-2024-002' ? (
                        <>
                          <div className="bg-red-500/10 text-red-400 pl-2">- max_depth: 6</div>
                          <div className="bg-green-500/10 text-green-400 pl-2">+ max_depth: 10</div>
                          <div className="bg-red-500/10 text-red-400 pl-2">- learning_rate: 0.1</div>
                          <div className="pl-2 text-zinc-500">  (removed - using defaults)</div>
                          <div className="bg-red-500/10 text-red-400 pl-2">- n_estimators: 200</div>
                          <div className="bg-green-500/10 text-green-400 pl-2">+ n_estimators: 150</div>
                          <div className="bg-red-500/10 text-red-400 pl-2">- subsample: 0.8</div>
                          <div className="pl-2 text-zinc-500">  (removed)</div>
                          <div className="bg-green-500/10 text-green-400 pl-2">+ min_samples_split: 4</div>
                        </>
                      ) : (
                        <>
                          <div className="bg-red-500/10 text-red-400 pl-2">- n_estimators: 150</div>
                          <div className="bg-green-500/10 text-green-400 pl-2">+ iterations: 500</div>
                          <div className="bg-red-500/10 text-red-400 pl-2">- max_depth: 10</div>
                          <div className="bg-green-500/10 text-green-400 pl-2">+ depth: 7</div>
                          <div className="bg-green-500/10 text-green-400 pl-2">+ learning_rate: 0.05</div>
                        </>
                      )}
                    </div>
                    <div className="border-t border-zinc-700 pt-2 mt-2">
                      <div className="text-cyan-400 mb-1">@@ Metadata @@</div>
                      <div className="pl-2 text-zinc-400">  seed: 42 (unchanged)</div>
                      {selectedExperiment.id === 'exp-2024-002' ? (
                        <>
                          <div className="bg-red-500/10 text-red-400 pl-2">- dataset: titanic_v2.csv</div>
                          <div className="bg-green-500/10 text-green-400 pl-2">+ dataset: titanic_v3_engineered.csv</div>
                        </>
                      ) : (
                        <div className="pl-2 text-zinc-400">  dataset: titanic_v2.csv (unchanged)</div>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2 p-3 rounded-lg bg-blue-500/10 border border-blue-600 text-sm">
                    <Info className="w-4 h-4 text-blue-400 flex-shrink-0" />
                    <span>Changes shown relative to previous experiment in timeline</span>
                  </div>
                </TabsContent>

                <TabsContent value="environment" className="space-y-3 mt-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Cpu className="w-4 h-4 text-blue-400" />
                    <h4 className="font-medium">Runtime Environment</h4>
                  </div>
                  <div className="space-y-2">
                    {Object.entries(selectedExperiment.environment).map(([key, value]) => (
                      <div key={key} className="flex items-center justify-between p-3 rounded-lg bg-zinc-800/50 border border-zinc-700">
                        <span className="text-sm font-medium capitalize">{key}</span>
                        <Badge variant="secondary">{value}</Badge>
                      </div>
                    ))}
                  </div>
                </TabsContent>

                <TabsContent value="dataset" className="space-y-3 mt-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Database className="w-4 h-4 text-cyan-400" />
                    <h4 className="font-medium">Dataset Information</h4>
                  </div>
                  <div className="p-4 rounded-lg bg-zinc-800/50 border border-zinc-700 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Filename</span>
                      <span className="text-sm font-medium">{selectedExperiment.dataset}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Rows</span>
                      <span className="text-sm font-medium">891</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Features</span>
                      <span className="text-sm font-medium">12</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Version Hash</span>
                      <Badge variant="outline" className="font-mono text-xs">a3b7c9d1</Badge>
                    </div>
                  </div>
                  <Button variant="outline" size="sm" className="w-full">
                    <Download className="w-3 h-3 mr-2" />
                    Download Dataset Snapshot
                  </Button>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          {/* Reproducibility */}
          <Card className="border-border border-green-600/50 bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-green-500/20 before:to-transparent before:opacity-70">
            <CardHeader className="relative">
              <CardTitle className="text-lg flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-green-500" />
                Reproducibility Dashboard
              </CardTitle>
            </CardHeader>
            <CardContent className="relative">
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center">
                  <CheckCircle className="w-8 h-8 text-green-500 mx-auto mb-2" />
                  <p className="text-sm font-medium">Seed Locked</p>
                  <p className="text-xs text-muted-foreground">Random state: {selectedExperiment.seed}</p>
                </div>
                <div className="text-center">
                  <CheckCircle className="w-8 h-8 text-green-500 mx-auto mb-2" />
                  <p className="text-sm font-medium">Environment Captured</p>
                  <p className="text-xs text-muted-foreground">All dependencies versioned</p>
                </div>
                <div className="text-center">
                  <CheckCircle className="w-8 h-8 text-green-500 mx-auto mb-2" />
                  <p className="text-sm font-medium">Dataset Versioned</p>
                  <p className="text-xs text-muted-foreground">Hash: a3b7c9d1</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
