import { useNavigate } from 'react-router-dom'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Trophy, Download, Radio, Copy, FlaskConical } from 'lucide-react'
import { toast } from 'sonner'

const mockModels = [
  { rank: 1, id: 'catboost-v1', name: 'CatBoost', accuracy: 0.983, f1: 0.981, precision: 0.985, recall: 0.978, time: '2m 15s' },
  { rank: 2, id: 'xgboost-v1', name: 'XGBoost', accuracy: 0.978, f1: 0.976, precision: 0.980, recall: 0.972, time: '1m 52s' },
  { rank: 3, id: 'lightgbm-v2', name: 'LightGBM', accuracy: 0.975, f1: 0.973, precision: 0.977, recall: 0.969, time: '1m 38s' },
  { rank: 4, id: 'rf-v1', name: 'Random Forest', accuracy: 0.968, f1: 0.965, precision: 0.971, recall: 0.960, time: '3m 05s' },
  { rank: 5, id: 'et-v1', name: 'Extra Trees', accuracy: 0.962, f1: 0.959, precision: 0.964, recall: 0.954, time: '2m 42s' },
]

export function Results() {
  const navigate = useNavigate()
  const bestModel = mockModels[0]

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <Trophy className="w-8 h-8 text-yellow-500" />
            Training Results
          </h1>
          <p className="text-muted-foreground">Performance leaderboard and model evaluation metrics</p>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="text-sm px-4 py-2">
            {mockModels.length} models trained
          </Badge>
          <Button className="bg-gradient-to-r from-yellow-600 to-orange-600 hover:from-yellow-700 hover:to-orange-700">
            <Download className="w-4 h-4 mr-2" />
            Export Report
          </Button>
        </div>
      </div>

      <div className="space-y-6">
        {/* Best Model Card */}
        <Card className="border-yellow-600/50 bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-yellow-500/20 before:via-amber-500/10 before:to-orange-500/20 before:opacity-70 shadow-lg shadow-yellow-500/20">
          <CardHeader className="relative">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Trophy className="w-5 h-5 text-yellow-500" />
                  Best Model: CatBoost
                </CardTitle>
                <CardDescription>Highest accuracy on validation set</CardDescription>
              </div>
              <div className="flex gap-2">
                <Button 
                  onClick={() => navigate('/app/inference?model_id=catboost-v1')}
                  className="bg-gradient-to-r from-yellow-600 to-orange-600 hover:from-yellow-700 hover:to-orange-700"
                >
                  <FlaskConical className="w-4 h-4 mr-2" />
                  Test in Lab
                </Button>
                <Button className="bg-gradient-to-r from-yellow-600 via-amber-600 to-orange-600 hover:from-yellow-700 hover:via-amber-700 hover:to-orange-700">
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
                  <div className="text-2xl font-bold text-green-400">98.3%</div>
                </div>
                <div className="p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50">
                  <div className="text-xs text-muted-foreground">F1 Score</div>
                  <div className="text-2xl font-bold text-blue-400">0.981</div>
                </div>
                <div className="p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50">
                  <div className="text-xs text-muted-foreground">Precision</div>
                  <div className="text-2xl font-bold text-purple-400">0.985</div>
                </div>
                <div className="p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50">
                  <div className="text-xs text-muted-foreground">Recall</div>
                  <div className="text-2xl font-bold text-yellow-400">0.978</div>
                </div>
              </div>
            </CardContent>
          </Card>

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
                      <TableCell className="text-foreground">{model.f1.toFixed(3)}</TableCell>
                      <TableCell className="text-foreground">{model.precision.toFixed(3)}</TableCell>
                      <TableCell className="text-foreground">{model.recall.toFixed(3)}</TableCell>
                      <TableCell className="text-muted-foreground text-sm">{model.time}</TableCell>
                      <TableCell className="text-right">
                        <div className="flex gap-2 justify-end">
                          <Button 
                            onClick={() => navigate(`/app/inference?model_id=${model.id}`)}
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
                                  <div className="p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50 text-center">
                                    <div className="text-xs text-muted-foreground">Avg Latency</div>
                                    <div className="text-lg font-bold text-green-400">12ms</div>
                                  </div>
                                  <div className="p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50 text-center">
                                    <div className="text-xs text-muted-foreground">Requests</div>
                                    <div className="text-lg font-bold text-blue-400">1,247</div>
                                  </div>
                                  <div className="p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50 text-center">
                                    <div className="text-xs text-muted-foreground">Uptime</div>
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
        </div>
    </div>
  )
}
