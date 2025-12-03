import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Trophy, Download, Play, TrendingUp, Radio, Copy } from 'lucide-react'
import { toast } from 'sonner'

const mockModels = [
  { rank: 1, name: 'CatBoost', accuracy: 0.983, f1: 0.981, precision: 0.985, recall: 0.978, time: '2m 15s' },
  { rank: 2, name: 'XGBoost', accuracy: 0.978, f1: 0.976, precision: 0.980, recall: 0.972, time: '1m 52s' },
  { rank: 3, name: 'LightGBM', accuracy: 0.975, f1: 0.973, precision: 0.977, recall: 0.969, time: '1m 38s' },
  { rank: 4, name: 'Random Forest', accuracy: 0.968, f1: 0.965, precision: 0.971, recall: 0.960, time: '3m 05s' },
  { rank: 5, name: 'Extra Trees', accuracy: 0.962, f1: 0.959, precision: 0.964, recall: 0.954, time: '2m 42s' },
]

export function Results() {
  const [inferenceData, setInferenceData] = useState({
    pclass: '1',
    sex: 'female',
    age: '29',
    sibsp: '0',
    parch: '0',
    fare: '75.00',
  })
  const [prediction, setPrediction] = useState<{ survived: boolean; confidence: number } | null>(null)

  const handlePredict = () => {
    // Mock prediction
    setPrediction({
      survived: true,
      confidence: 0.87,
    })
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
          <Trophy className="w-8 h-8 text-yellow-500" />
          Results & Inference
        </h1>
        <p className="text-zinc-400">Model leaderboard and prediction interface</p>
      </div>

      <Tabs defaultValue="leaderboard" className="w-full">
        <TabsList className="grid w-full grid-cols-2 max-w-md">
          <TabsTrigger value="leaderboard">Leaderboard</TabsTrigger>
          <TabsTrigger value="inference">Inference Lab</TabsTrigger>
        </TabsList>

        {/* Leaderboard Tab */}
        <TabsContent value="leaderboard" className="space-y-6 mt-6">
          {/* Best Model Card */}
          <Card className="border-yellow-600/50 bg-yellow-600/5">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Trophy className="w-5 h-5 text-yellow-500" />
                    Best Model: CatBoost
                  </CardTitle>
                  <CardDescription>Highest accuracy on validation set</CardDescription>
                </div>
                <Button className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700">
                  <Download className="w-4 h-4 mr-2" />
                  Export Model
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-4 gap-4">
                <div className="p-3 rounded-lg bg-zinc-800/50">
                  <div className="text-xs text-zinc-500">Accuracy</div>
                  <div className="text-2xl font-bold text-green-400">98.3%</div>
                </div>
                <div className="p-3 rounded-lg bg-zinc-800/50">
                  <div className="text-xs text-zinc-500">F1 Score</div>
                  <div className="text-2xl font-bold text-blue-400">0.981</div>
                </div>
                <div className="p-3 rounded-lg bg-zinc-800/50">
                  <div className="text-xs text-zinc-500">Precision</div>
                  <div className="text-2xl font-bold text-purple-400">0.985</div>
                </div>
                <div className="p-3 rounded-lg bg-zinc-800/50">
                  <div className="text-xs text-zinc-500">Recall</div>
                  <div className="text-2xl font-bold text-yellow-400">0.978</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* All Models Table */}
          <Card>
            <CardHeader>
              <CardTitle>All Models</CardTitle>
              <CardDescription>Complete ranking of trained models</CardDescription>
            </CardHeader>
            <CardContent>
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
                  {mockModels.map((model) => (
                    <TableRow key={model.rank}>
                      <TableCell>
                        <Badge
                          variant={model.rank === 1 ? 'default' : 'secondary'}
                          className={model.rank === 1 ? 'bg-yellow-600' : ''}
                        >
                          #{model.rank}
                        </Badge>
                      </TableCell>
                      <TableCell className="font-medium">{model.name}</TableCell>
                      <TableCell>
                        <span className="text-green-400 font-semibold">
                          {(model.accuracy * 100).toFixed(1)}%
                        </span>
                      </TableCell>
                      <TableCell className="text-zinc-300">{model.f1.toFixed(3)}</TableCell>
                      <TableCell className="text-zinc-300">{model.precision.toFixed(3)}</TableCell>
                      <TableCell className="text-zinc-300">{model.recall.toFixed(3)}</TableCell>
                      <TableCell className="text-zinc-400 text-sm">{model.time}</TableCell>
                      <TableCell className="text-right">
                        <div className="flex gap-2 justify-end">
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
                                        navigator.clipboard.writeText(`curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"age": 25, "sex": "female", "pclass": 1, "fare": 75.0}'`)
                                        toast.success('Copied to clipboard')
                                      }}
                                    >
                                      <Copy className="w-4 h-4 mr-1" />
                                      Copy
                                    </Button>
                                  </div>
                                  <div className="bg-black rounded-lg p-4 font-mono text-sm overflow-x-auto">
                                    <code className="text-green-400">
                                      curl -X POST http://localhost:8000/predict \<br />
                                      &nbsp;&nbsp;-H "Content-Type: application/json" \<br />
                                      &nbsp;&nbsp;-d '{`{"age": 25, "sex": "female", "pclass": 1, "fare": 75.0}`}'
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
                                      &nbsp;&nbsp;"model": "${model.name}",<br />
                                      &nbsp;&nbsp;"latency_ms": 12<br />
                                      {`}`}
                                    </code>
                                  </div>
                                </div>

                                {/* Stats */}
                                <div className="grid grid-cols-3 gap-3">
                                  <div className="p-3 rounded-lg bg-zinc-800/50 text-center">
                                    <div className="text-xs text-zinc-500">Avg Latency</div>
                                    <div className="text-lg font-bold text-green-400">12ms</div>
                                  </div>
                                  <div className="p-3 rounded-lg bg-zinc-800/50 text-center">
                                    <div className="text-xs text-zinc-500">Requests</div>
                                    <div className="text-lg font-bold text-blue-400">1,247</div>
                                  </div>
                                  <div className="p-3 rounded-lg bg-zinc-800/50 text-center">
                                    <div className="text-xs text-zinc-500">Uptime</div>
                                    <div className="text-lg font-bold text-purple-400">99.9%</div>
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
        </TabsContent>

        {/* Inference Lab Tab */}
        <TabsContent value="inference" className="space-y-6 mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Input Form */}
            <Card>
              <CardHeader>
                <CardTitle>Input Features</CardTitle>
                <CardDescription>Enter passenger details for prediction</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Passenger Class</label>
                  <Input
                    type="number"
                    value={inferenceData.pclass}
                    onChange={(e) =>
                      setInferenceData({ ...inferenceData, pclass: e.target.value })
                    }
                    placeholder="1, 2, or 3"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Sex</label>
                  <Input
                    value={inferenceData.sex}
                    onChange={(e) =>
                      setInferenceData({ ...inferenceData, sex: e.target.value })
                    }
                    placeholder="male or female"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Age</label>
                  <Input
                    type="number"
                    value={inferenceData.age}
                    onChange={(e) =>
                      setInferenceData({ ...inferenceData, age: e.target.value })
                    }
                    placeholder="Age in years"
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Siblings/Spouses</label>
                    <Input
                      type="number"
                      value={inferenceData.sibsp}
                      onChange={(e) =>
                        setInferenceData({ ...inferenceData, sibsp: e.target.value })
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Parents/Children</label>
                    <Input
                      type="number"
                      value={inferenceData.parch}
                      onChange={(e) =>
                        setInferenceData({ ...inferenceData, parch: e.target.value })
                      }
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Fare ($)</label>
                  <Input
                    type="number"
                    value={inferenceData.fare}
                    onChange={(e) =>
                      setInferenceData({ ...inferenceData, fare: e.target.value })
                    }
                    placeholder="Ticket fare"
                  />
                </div>
                <Button onClick={handlePredict} className="w-full bg-purple-600 hover:bg-purple-700">
                  <Play className="w-4 h-4 mr-2" />
                  Predict Survival
                </Button>
              </CardContent>
            </Card>

            {/* Prediction Result */}
            <Card>
              <CardHeader>
                <CardTitle>Prediction Result</CardTitle>
                <CardDescription>Model inference output</CardDescription>
              </CardHeader>
              <CardContent>
                {prediction ? (
                  <div className="space-y-6">
                    <div
                      className={`p-8 rounded-lg text-center ${
                        prediction.survived
                          ? 'bg-green-600/20 border-2 border-green-600'
                          : 'bg-red-600/20 border-2 border-red-600'
                      }`}
                    >
                      <div className="text-6xl mb-4">
                        {prediction.survived ? '✅' : '❌'}
                      </div>
                      <div className="text-2xl font-bold mb-2">
                        {prediction.survived ? 'SURVIVED' : 'DID NOT SURVIVE'}
                      </div>
                      <div className="text-zinc-400">
                        Confidence: {(prediction.confidence * 100).toFixed(1)}%
                      </div>
                    </div>

                    <div className="p-4 rounded-lg bg-zinc-800/50">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">Confidence Score</span>
                        <span className="text-sm text-purple-400">
                          {(prediction.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full h-2 bg-zinc-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-purple-600 transition-all"
                          style={{ width: `${prediction.confidence * 100}%` }}
                        />
                      </div>
                    </div>

                    <div className="space-y-2 text-sm text-zinc-400">
                      <div className="flex justify-between">
                        <span>Model Used:</span>
                        <span className="text-zinc-100">CatBoost (Best)</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Inference Time:</span>
                        <span className="text-zinc-100">23ms</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12 text-zinc-500">
                    <TrendingUp className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>Enter passenger details and click "Predict Survival" to see results</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
