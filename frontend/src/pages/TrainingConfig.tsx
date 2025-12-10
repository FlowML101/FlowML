import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Zap, Network, Play, Settings2 } from 'lucide-react'
import { Badge } from '@/components/ui/badge'

const Label2 = ({ children, ...props }: any) => (
  <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70" {...props}>
    {children}
  </label>
)

export function TrainingConfig() {
  const navigate = useNavigate()
  const [targetColumn, setTargetColumn] = useState('Survived')
  const [timeBudget, setTimeBudget] = useState([15])

  const handleStartTraining = () => {
    navigate('/app/running')
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
          <Zap className="w-8 h-8 text-yellow-500" />
          Training Configuration
        </h1>
        <p className="text-zinc-400">Configure AutoML pipeline for your dataset</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Configuration */}
        <div className="lg:col-span-2 space-y-6">
          {/* Dataset Info */}
          <Card>
            <CardHeader>
              <CardTitle>Dataset</CardTitle>
              <CardDescription>titanic.csv</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <Badge variant="outline">891 rows</Badge>
                <Badge variant="outline">12 columns</Badge>
                <Badge variant="outline">Binary Classification</Badge>
              </div>
            </CardContent>
          </Card>

          {/* Target Column */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings2 className="w-5 h-5 text-purple-500" />
                Model Configuration
              </CardTitle>
              <CardDescription>Select prediction target and training parameters</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Target Column */}
              <div className="space-y-2">
                <Label2 htmlFor="target">Target Column</Label2>
                <Select value={targetColumn} onValueChange={setTargetColumn}>
                  <SelectTrigger id="target">
                    <SelectValue placeholder="Select target column" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Survived">Survived</SelectItem>
                    <SelectItem value="Pclass">Pclass</SelectItem>
                    <SelectItem value="Age">Age</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-zinc-500">
                  Column to predict (dependent variable)
                </p>
              </div>

              {/* Time Budget */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label2>Time Budget</Label2>
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
                <div className="flex justify-between text-xs text-zinc-500">
                  <span>1 min</span>
                  <span>60 min</span>
                </div>
                <p className="text-xs text-zinc-500">
                  Maximum time to search for best model
                </p>
              </div>

              {/* Model Types */}
              <div className="space-y-2">
                <Label2>Model Types</Label2>
                <div className="grid grid-cols-2 gap-3">
                  {['XGBoost', 'Random Forest', 'LightGBM', 'CatBoost', 'Linear', 'Neural Net'].map(
                    (model) => (
                      <div
                        key={model}
                        className="flex items-center space-x-2 p-3 rounded-lg bg-zinc-800/30 border border-zinc-700"
                      >
                        <input
                          type="checkbox"
                          defaultChecked
                          className="w-4 h-4 rounded border-zinc-600 text-purple-600 focus:ring-purple-500"
                        />
                        <span className="text-sm">{model}</span>
                      </div>
                    )
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Compute Configuration */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Compute Mode</CardTitle>
              <CardDescription>Automatic distributed training across your mesh</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="p-4 rounded-lg border-2 border-purple-600 bg-purple-600/10">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Network className="w-5 h-5 text-purple-400" />
                    <span className="font-medium">Auto-Distributed</span>
                  </div>
                  <Badge variant="success" className="text-xs">Active</Badge>
                </div>
                <p className="text-xs text-zinc-500 mb-3">
                  Intelligently distributes workload across your mesh network (2 workers available)
                </p>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between text-zinc-400">
                    <span>Master Node:</span>
                    <span className="text-zinc-300">localhost</span>
                  </div>
                  <div className="flex justify-between text-zinc-400">
                    <span>Active Workers:</span>
                    <span className="text-green-400">2 online</span>
                  </div>
                  <div className="flex justify-between text-zinc-400">
                    <span>Total VRAM:</span>
                    <span className="text-zinc-300">18 GB</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Start Training */}
          <Card className="border-purple-600/50">
            <CardHeader>
              <CardTitle className="text-lg">Ready to Launch</CardTitle>
            </CardHeader>
            <CardContent>
              <Button
                onClick={handleStartTraining}
                className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-lg py-6"
              >
                <Play className="w-5 h-5 mr-2" />
                Start Training
              </Button>
              <p className="text-xs text-zinc-500 mt-3 text-center">
                Estimated: {timeBudget[0]} min • ~20 models • Auto-Distributed
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}