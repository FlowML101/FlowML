import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  TrendingUp, BarChart3, GitCompare, Trophy, 
  Target, Clock, Check
} from 'lucide-react'
import { motion } from 'framer-motion'

const mockModels = [
  {
    id: 'xgboost-v1',
    name: 'XGBoost',
    version: 'v1.0',
    accuracy: 94.2,
    precision: 93.8,
    recall: 94.5,
    f1: 94.1,
    trainTime: '2.2m',
    status: 'production' as const
  },
  {
    id: 'catboost-v1',
    name: 'CatBoost',
    version: 'v1.0',
    accuracy: 95.1,
    precision: 95.4,
    recall: 94.8,
    f1: 95.1,
    trainTime: '3.5m',
    status: 'production' as const
  },
  {
    id: 'randomforest-v2',
    name: 'Random Forest',
    version: 'v2.0',
    accuracy: 92.8,
    precision: 92.3,
    recall: 93.1,
    f1: 92.7,
    trainTime: '1.8m',
    status: 'staging' as const
  }
]

const confusionMatrixData = {
  'xgboost-v1': [[85, 15], [8, 92]],
  'catboost-v1': [[88, 12], [6, 94]],
  'randomforest-v2': [[82, 18], [10, 90]]
}

const featureImportance = {
  'xgboost-v1': [
    { feature: 'Sex', importance: 0.287 },
    { feature: 'Fare', importance: 0.243 },
    { feature: 'Age', importance: 0.198 },
    { feature: 'Pclass', importance: 0.145 },
    { feature: 'Embarked', importance: 0.062 }
  ],
  'catboost-v1': [
    { feature: 'Sex', importance: 0.312 },
    { feature: 'Fare', importance: 0.228 },
    { feature: 'Age', importance: 0.185 },
    { feature: 'Pclass', importance: 0.167 },
    { feature: 'Embarked', importance: 0.058 }
  ],
  'randomforest-v2': [
    { feature: 'Sex', importance: 0.245 },
    { feature: 'Fare', importance: 0.267 },
    { feature: 'Age', importance: 0.203 },
    { feature: 'Pclass', importance: 0.132 },
    { feature: 'Embarked', importance: 0.073 }
  ]
}

export function ModelComparison() {
  const [selectedModels, setSelectedModels] = useState<string[]>(['xgboost-v1', 'catboost-v1'])

  const toggleModel = (modelId: string) => {
    if (selectedModels.includes(modelId)) {
      if (selectedModels.length > 1) {
        setSelectedModels(selectedModels.filter(id => id !== modelId))
      }
    } else if (selectedModels.length < 3) {
      setSelectedModels([...selectedModels, modelId])
    }
  }

  // Calculate radar chart points
  const getRadarPoints = (model: typeof mockModels[0]) => {
    const metrics = [
      model.accuracy,
      model.precision,
      model.recall,
      model.f1,
      100 - (parseFloat(model.trainTime.replace('m', '')) * 8) // Speed score (inverse)
    ]
    const angles = [0, 72, 144, 216, 288] // Pentagon: 360/5 = 72°
    const centerX = 150
    const centerY = 150
    const maxRadius = 110

    return metrics.map((value, i) => {
      const angle = (angles[i] - 90) * (Math.PI / 180) // Start from top
      const radius = (value / 100) * maxRadius
      return {
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle)
      }
    })
  }

  const modelColors = {
    'xgboost-v1': '#3b82f6',
    'catboost-v1': '#10b981',
    'randomforest-v2': '#f59e0b'
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <GitCompare className="w-8 h-8 text-cyan-500" />
            Model Comparison
          </h1>
          <p className="text-muted-foreground">Professional model performance analysis</p>
        </div>
        <Badge variant="outline" className="text-sm px-4 py-2">
          {selectedModels.length} of 3 models
        </Badge>
      </div>

      {/* Model Selector */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {mockModels.map((model) => {
          const isSelected = selectedModels.includes(model.id)
          return (
            <motion.div
              key={model.id}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <button
                onClick={() => toggleModel(model.id)}
                className={`w-full p-5 rounded-xl border-2 transition-all text-left ${
                  isSelected
                    ? 'border-cyan-500 bg-cyan-500/10 shadow-lg shadow-cyan-500/20'
                    : 'border-zinc-700 hover:border-zinc-600 bg-zinc-900/50'
                }`}
              >
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h3 className="font-bold text-lg">{model.name}</h3>
                    <p className="text-xs text-muted-foreground">{model.version}</p>
                  </div>
                  {isSelected && (
                    <div className="w-6 h-6 rounded-full bg-cyan-500 flex items-center justify-center">
                      <Check className="w-4 h-4 text-white" />
                    </div>
                  )}
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Accuracy</span>
                    <span className="font-semibold text-green-400">{model.accuracy}%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">F1 Score</span>
                    <span className="font-semibold">{model.f1}%</span>
                  </div>
                  <Badge className={model.status === 'production' ? 'bg-green-600' : 'bg-yellow-600'}>
                    {model.status}
                  </Badge>
                </div>
              </button>
            </motion.div>
          )
        })}
      </div>

      {/* Performance Radar Chart */}
      <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-cyan-500/10 before:via-purple-500/10 before:to-transparent before:opacity-30">
        <CardHeader className="relative">
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-cyan-400" />
            Performance Radar
          </CardTitle>
          <CardDescription>
            Multi-dimensional performance comparison across key metrics
          </CardDescription>
        </CardHeader>
        <CardContent className="relative">
          <div className="flex flex-col items-center">
            <svg width="300" height="300" className="mx-auto">
              {/* Background grid circles */}
              {[20, 40, 60, 80, 100].map((percent) => (
                <circle
                  key={percent}
                  cx="150"
                  cy="150"
                  r={(percent / 100) * 110}
                  fill="none"
                  stroke="#27272a"
                  strokeWidth="1"
                  opacity="0.5"
                />
              ))}

              {/* Axis lines */}
              {[0, 72, 144, 216, 288].map((angle, i) => {
                const rad = (angle - 90) * (Math.PI / 180)
                const x = 150 + 110 * Math.cos(rad)
                const y = 150 + 110 * Math.sin(rad)
                return (
                  <line
                    key={i}
                    x1="150"
                    y1="150"
                    x2={x}
                    y2={y}
                    stroke="#3f3f46"
                    strokeWidth="1"
                  />
                )
              })}

              {/* Model data polygons */}
              {selectedModels.map((modelId) => {
                const model = mockModels.find(m => m.id === modelId)!
                const points = getRadarPoints(model)
                const pathData = points.map((p, i) => 
                  `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`
                ).join(' ') + ' Z'
                
                return (
                  <g key={modelId}>
                    <motion.path
                      initial={{ opacity: 0, scale: 0 }}
                      animate={{ opacity: 0.3, scale: 1 }}
                      transition={{ duration: 0.5 }}
                      d={pathData}
                      fill={modelColors[modelId as keyof typeof modelColors]}
                      stroke={modelColors[modelId as keyof typeof modelColors]}
                      strokeWidth="2"
                    />
                    {points.map((point, i) => (
                      <motion.circle
                        key={i}
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: i * 0.1 }}
                        cx={point.x}
                        cy={point.y}
                        r="5"
                        fill={modelColors[modelId as keyof typeof modelColors]}
                      />
                    ))}
                  </g>
                )
              })}

              {/* Metric labels */}
              {['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Speed'].map((label, i) => {
                const angle = ([0, 72, 144, 216, 288][i] - 90) * (Math.PI / 180)
                const x = 150 + 130 * Math.cos(angle)
                const y = 150 + 130 * Math.sin(angle)
                return (
                  <text
                    key={label}
                    x={x}
                    y={y}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className="fill-muted-foreground text-xs font-medium"
                  >
                    {label}
                  </text>
                )
              })}
            </svg>

            {/* Legend */}
            <div className="flex items-center gap-6 mt-6">
              {selectedModels.map((modelId) => {
                const model = mockModels.find(m => m.id === modelId)!
                return (
                  <div key={modelId} className="flex items-center gap-2">
                    <div 
                      className="w-4 h-4 rounded-full" 
                      style={{ backgroundColor: modelColors[modelId as keyof typeof modelColors] }}
                    />
                    <span className="text-sm font-medium">{model.name}</span>
                  </div>
                )
              })}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Metrics Table */}
      <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="w-5 h-5 text-purple-400" />
            Detailed Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-zinc-700">
                  <th className="py-3 px-4 text-left text-sm font-medium text-muted-foreground">Metric</th>
                  {selectedModels.map((modelId) => {
                    const model = mockModels.find(m => m.id === modelId)!
                    return (
                      <th key={modelId} className="py-3 px-4 text-center text-sm font-medium">
                        {model.name}
                      </th>
                    )
                  })}
                </tr>
              </thead>
              <tbody>
                {['accuracy', 'precision', 'recall', 'f1'].map((metric) => {
                  const values = selectedModels.map(id => {
                    const model = mockModels.find(m => m.id === id)!
                    return model[metric as keyof typeof model] as number
                  })
                  const maxValue = Math.max(...values)

                  return (
                    <tr key={metric} className="border-b border-zinc-800">
                      <td className="py-4 px-4 text-sm font-medium capitalize">{metric}</td>
                      {selectedModels.map((modelId) => {
                        const model = mockModels.find(m => m.id === modelId)!
                        const value = model[metric as keyof typeof model] as number
                        const isMax = value === maxValue

                        return (
                          <td key={modelId} className="py-4 px-4 text-center">
                            <div className="flex items-center justify-center gap-2">
                              <span className={`text-sm font-semibold ${isMax ? 'text-green-400' : ''}`}>
                                {value}%
                              </span>
                              {isMax && <Trophy className="w-4 h-4 text-yellow-500" />}
                            </div>
                          </td>
                        )
                      })}
                    </tr>
                  )
                })}
                <tr className="border-b border-zinc-800">
                  <td className="py-4 px-4 text-sm font-medium">Training Time</td>
                  {selectedModels.map((modelId) => {
                    const model = mockModels.find(m => m.id === modelId)!
                    return (
                      <td key={modelId} className="py-4 px-4 text-center">
                        <div className="flex items-center justify-center gap-2">
                          <Clock className="w-4 h-4 text-blue-400" />
                          <span className="text-sm">{model.trainTime}</span>
                        </div>
                      </td>
                    )
                  })}
                </tr>
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ROC Curves */}
        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-green-500/10 before:to-transparent before:opacity-30">
          <CardHeader className="relative">
            <CardTitle className="flex items-center gap-2 text-lg">
              <TrendingUp className="w-5 h-5 text-green-400" />
              ROC Curves
            </CardTitle>
            <CardDescription>
              Model discrimination performance
            </CardDescription>
          </CardHeader>
          <CardContent className="relative">
            <div className="relative h-80 bg-zinc-950 rounded-lg p-4 border border-zinc-800">
              {/* Grid */}
              <div className="absolute inset-4 grid grid-cols-10 grid-rows-10">
                {Array.from({ length: 100 }).map((_, i) => (
                  <div key={i} className="border-r border-b border-zinc-800/30"></div>
                ))}
              </div>

              <svg className="absolute inset-4 w-[calc(100%-2rem)] h-[calc(100%-2rem)]">
                {/* Diagonal reference */}
                <line x1="0" y1="100%" x2="100%" y2="0" stroke="#52525b" strokeWidth="2" strokeDasharray="5,5" />
                
                {/* Model ROC curves */}
                {selectedModels.map((modelId, idx) => {
                  const paths = {
                    'xgboost-v1': 'M 0 100 L 2 85 L 5 70 L 10 55 L 18 40 L 30 28 L 45 18 L 65 10 L 85 4 L 95 2 L 100 0',
                    'catboost-v1': 'M 0 100 L 1 80 L 3 60 L 6 45 L 12 32 L 22 22 L 38 14 L 60 7 L 82 3 L 94 1 L 100 0',
                    'randomforest-v2': 'M 0 100 L 3 88 L 8 75 L 15 60 L 25 45 L 38 33 L 52 22 L 70 13 L 88 6 L 96 3 L 100 0'
                  }
                  
                  return (
                    <motion.path
                      key={modelId}
                      initial={{ pathLength: 0 }}
                      animate={{ pathLength: 1 }}
                      transition={{ duration: 1.5, delay: idx * 0.2 }}
                      d={paths[modelId as keyof typeof paths]}
                      stroke={modelColors[modelId as keyof typeof modelColors]}
                      strokeWidth="3"
                      fill="none"
                      vectorEffect="non-scaling-stroke"
                    />
                  )
                })}
              </svg>

              {/* Axes labels */}
              <div className="absolute bottom-0 left-1/2 -translate-x-1/2 text-xs text-muted-foreground">
                False Positive Rate
              </div>
              <div className="absolute left-0 top-1/2 -translate-y-1/2 -rotate-90 text-xs text-muted-foreground">
                True Positive Rate
              </div>
            </div>

            {/* AUC Scores */}
            <div className="space-y-2 mt-4">
              {selectedModels.map((modelId) => {
                const model = mockModels.find(m => m.id === modelId)!
                const auc = modelId === 'xgboost-v1' ? 0.94 : modelId === 'catboost-v1' ? 0.96 : 0.92
                return (
                  <div key={modelId} className="flex items-center justify-between p-3 rounded-lg bg-zinc-800/50">
                    <div className="flex items-center gap-2">
                      <div 
                        className="w-3 h-3 rounded-full" 
                        style={{ backgroundColor: modelColors[modelId as keyof typeof modelColors] }}
                      />
                      <span className="text-sm font-medium">{model.name}</span>
                    </div>
                    <span className="text-sm text-muted-foreground">AUC: {auc.toFixed(3)}</span>
                  </div>
                )
              })}
            </div>
          </CardContent>
        </Card>

        {/* Confusion Matrices */}
        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/10 before:via-green-500/10 before:to-orange-500/10 before:opacity-30">
          <CardHeader className="relative">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Target className="w-5 h-5 text-purple-400" />
              Confusion Matrices
            </CardTitle>
            <CardDescription>
              Classification performance breakdown
            </CardDescription>
          </CardHeader>
          <CardContent className="relative space-y-6">
            {selectedModels.map((modelId) => {
              const model = mockModels.find(m => m.id === modelId)!
              const matrix = confusionMatrixData[modelId as keyof typeof confusionMatrixData]

              return (
                <div key={modelId} className="space-y-2">
                  <div className="flex items-center gap-2 mb-3">
                    <div 
                      className="w-3 h-3 rounded-full" 
                      style={{ backgroundColor: modelColors[modelId as keyof typeof modelColors] }}
                    />
                    <h4 className="font-medium text-sm">{model.name}</h4>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {matrix.map((row, i) => 
                      row.map((value, j) => (
                        <div
                          key={`${i}-${j}`}
                          className={`p-4 rounded-lg text-center transition-all hover:scale-105 ${
                            i === j 
                              ? 'bg-green-500/20 border-2 border-green-500/40' 
                              : 'bg-red-500/20 border-2 border-red-500/40'
                          }`}
                        >
                          <div className="text-2xl font-bold">{value}</div>
                          <div className="text-xs text-muted-foreground mt-1">
                            {i === 0 && j === 0 && 'True Neg'}
                            {i === 0 && j === 1 && 'False Pos'}
                            {i === 1 && j === 0 && 'False Neg'}
                            {i === 1 && j === 1 && 'True Pos'}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )
            })}
          </CardContent>
        </Card>
      </div>

      {/* Feature Importance */}
      <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-cyan-500/10 before:to-transparent before:opacity-30">
        <CardHeader className="relative">
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-cyan-400" />
            Feature Importance Comparison
          </CardTitle>
          <CardDescription>
            Top contributing features across models
          </CardDescription>
        </CardHeader>
        <CardContent className="relative">
          <Tabs defaultValue={selectedModels[0]}>
            <TabsList className="grid w-full mb-6" style={{ gridTemplateColumns: `repeat(${selectedModels.length}, 1fr)` }}>
              {selectedModels.map((modelId) => {
                const model = mockModels.find(m => m.id === modelId)!
                return (
                  <TabsTrigger key={modelId} value={modelId}>
                    {model.name}
                  </TabsTrigger>
                )
              })}
            </TabsList>
            {selectedModels.map((modelId) => {
              const features = featureImportance[modelId as keyof typeof featureImportance]
              return (
                <TabsContent key={modelId} value={modelId} className="space-y-4">
                  {features.map((item, index) => (
                    <div key={item.feature} className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-cyan-400">#{index + 1}</span>
                          <span className="font-medium">{item.feature}</span>
                        </div>
                        <span className="text-muted-foreground">{(item.importance * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-3 bg-zinc-800 rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${item.importance * 100}%` }}
                          transition={{ duration: 0.5, delay: index * 0.1 }}
                          className="h-full rounded-full"
                          style={{ 
                            background: `linear-gradient(to right, ${modelColors[modelId as keyof typeof modelColors]}, ${modelColors[modelId as keyof typeof modelColors]}dd)` 
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </TabsContent>
              )
            })}
          </Tabs>
        </CardContent>
      </Card>

      {/* Winner Summary */}
      <Card className="border-2 border-green-600/50 bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-green-500/20 before:via-yellow-500/10 before:to-transparent before:opacity-70 shadow-xl shadow-green-500/20">
        <CardContent className="relative pt-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-yellow-500 to-orange-500 flex items-center justify-center shadow-lg">
                <Trophy className="w-8 h-8 text-white" />
              </div>
              <div>
                <h3 className="text-2xl font-bold flex items-center gap-3">
                  Recommended Model
                  <Badge className="bg-green-600">Best Overall</Badge>
                </h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Highest accuracy with balanced precision-recall trade-off
                </p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-4xl font-bold text-green-400">CatBoost</div>
              <p className="text-sm text-muted-foreground">95.1% accuracy • v1.0</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
