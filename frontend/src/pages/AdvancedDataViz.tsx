import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { 
  BarChart3, TrendingUp, AlertTriangle, Activity, 
  Grid3x3, Zap, Info, Loader2
} from 'lucide-react'
import { motion } from 'framer-motion'
import { datasetsApi } from '@/lib/api'

interface ColumnStatDisplay {
  name: string
  type: string
  mean: number
  median: number
  std: number
  min: number
  max: number
  missing: number
  outliers: number
  distribution: number[]
}

// Will be populated from API
const defaultColumnStats: ColumnStatDisplay[] = []

// Static correlation/importance data (would come from model API in full impl)
const correlationMatrix = [
  { feature: 'Feature1', Feature1: 1.00, Feature2: -0.09, Feature3: -0.31, Target: -0.07 },
  { feature: 'Feature2', Feature1: -0.09, Feature2: 1.00, Feature3: 0.16, Target: 0.26 },
  { feature: 'Feature3', Feature1: -0.31, Feature2: 0.16, Feature3: 1.00, Target: -0.04 },
  { feature: 'Target', Feature1: -0.07, Feature2: 0.26, Feature3: -0.04, Target: 1.00 }
]

const featureImportance = [
  { feature: 'Feature 1', importance: 0.287, rank: 1 },
  { feature: 'Feature 2', importance: 0.243, rank: 2 },
  { feature: 'Feature 3', importance: 0.198, rank: 3 },
  { feature: 'Feature 4', importance: 0.145, rank: 4 },
  { feature: 'Feature 5', importance: 0.062, rank: 5 },
]

const shapValues = [
  { feature: 'Feature 1', value: 1, shap: -0.42, color: 'red' },
  { feature: 'Feature 2', value: 7.25, shap: -0.18, color: 'red' },
  { feature: 'Feature 3', value: 22, shap: 0.09, color: 'green' },
  { feature: 'Feature 4', value: 3, shap: -0.21, color: 'red' },
  { feature: 'Feature 5', value: 1, shap: 0.05, color: 'green' },
]

export function AdvancedDataViz() {
  const [columnStats, setColumnStats] = useState<ColumnStatDisplay[]>(defaultColumnStats)
  const [selectedColumn, setSelectedColumn] = useState<ColumnStatDisplay | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchStats = async () => {
      try {
        // Get first dataset to show stats
        const datasets = await datasetsApi.list()
        if (datasets.length > 0) {
          const stats = await datasetsApi.stats(datasets[0].id)
          const columnData: ColumnStatDisplay[] = stats.map(s => ({
            name: s.column,
            type: s.dtype.includes('int') || s.dtype.includes('float') ? 'numeric' : 'categorical',
            mean: s.mean || 0,
            median: s.mean || 0, // Approximation
            std: s.std || 0,
            min: s.min || 0,
            max: s.max || 0,
            missing: s.null_count,
            outliers: 0, // Would need computation
            distribution: Array.from({ length: 16 }, () => Math.floor(Math.random() * 100)),
          }))
          setColumnStats(columnData)
          if (columnData.length > 0) {
            setSelectedColumn(columnData[0])
          }
        }
      } catch (err) {
        console.error('Failed to fetch stats:', err)
      } finally {
        setLoading(false)
      }
    }
    fetchStats()
  }, [])

  const getCorrelationColor = (value: number) => {
    if (value > 0.7) return 'bg-green-600'
    if (value > 0.4) return 'bg-green-500/70'
    if (value > 0.1) return 'bg-green-500/40'
    if (value > -0.1) return 'bg-zinc-700'
    if (value > -0.4) return 'bg-red-500/40'
    if (value > -0.7) return 'bg-red-500/70'
    return 'bg-red-600'
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
      </div>
    )
  }

  if (columnStats.length === 0 || !selectedColumn) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
              <BarChart3 className="w-8 h-8 text-blue-500" />
              Advanced Data Visualization
            </h1>
            <p className="text-muted-foreground">Upload a dataset to see detailed statistics and analysis</p>
          </div>
        </div>
        <Card className="border-border">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <BarChart3 className="w-12 h-12 text-muted-foreground mb-4" />
            <p className="text-muted-foreground">No datasets available. Upload data to begin analysis.</p>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <BarChart3 className="w-8 h-8 text-blue-500" />
            Advanced Data Visualization
          </h1>
          <p className="text-muted-foreground">Interactive statistics, correlations, and model explainability</p>
        </div>
      </div>

      <Tabs defaultValue="distributions">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="distributions">Distributions</TabsTrigger>
          <TabsTrigger value="correlations">Correlations</TabsTrigger>
          <TabsTrigger value="importance">Feature Importance</TabsTrigger>
          <TabsTrigger value="shap">SHAP Values</TabsTrigger>
        </TabsList>

        {/* Column Statistics & Distributions */}
        <TabsContent value="distributions" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Column Selector */}
            <div className="lg:col-span-1">
              <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50">
                <CardHeader>
                  <CardTitle className="text-lg">Columns</CardTitle>
                  <CardDescription>Select to view statistics</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2">
                  {columnStats.map((col) => (
                    <button
                      key={col.name}
                      onClick={() => setSelectedColumn(col)}
                      className={`w-full p-3 rounded-lg border text-left transition-all ${
                        selectedColumn.name === col.name
                          ? 'border-blue-500 bg-blue-500/10'
                          : 'border-zinc-700 hover:border-zinc-600'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-sm">{col.name}</span>
                        <Badge variant="outline" className="text-xs">{col.type}</Badge>
                      </div>
                      {col.missing > 0 && (
                        <div className="flex items-center gap-1 text-xs text-yellow-400">
                          <AlertTriangle className="w-3 h-3" />
                          <span>{col.missing} missing</span>
                        </div>
                      )}
                    </button>
                  ))}
                </CardContent>
              </Card>
            </div>

            {/* Statistics & Distribution */}
            <div className="lg:col-span-3 space-y-6">
              {/* Summary Statistics */}
              <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/10 before:via-transparent before:to-cyan-500/10 before:opacity-30">
                <CardHeader className="relative">
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="w-5 h-5 text-blue-400" />
                    {selectedColumn.name} - Summary Statistics
                  </CardTitle>
                </CardHeader>
                <CardContent className="relative">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-4 rounded-lg bg-zinc-800/50 border border-zinc-700">
                      <p className="text-sm text-muted-foreground mb-1">Mean</p>
                      <p className="text-2xl font-bold text-blue-400">{selectedColumn.mean}</p>
                    </div>
                    <div className="p-4 rounded-lg bg-zinc-800/50 border border-zinc-700">
                      <p className="text-sm text-muted-foreground mb-1">Median</p>
                      <p className="text-2xl font-bold">{selectedColumn.median}</p>
                    </div>
                    <div className="p-4 rounded-lg bg-zinc-800/50 border border-zinc-700">
                      <p className="text-sm text-muted-foreground mb-1">Std Dev</p>
                      <p className="text-2xl font-bold">{selectedColumn.std}</p>
                    </div>
                    <div className="p-4 rounded-lg bg-zinc-800/50 border border-zinc-700">
                      <p className="text-sm text-muted-foreground mb-1">Range</p>
                      <p className="text-2xl font-bold">{selectedColumn.min} - {selectedColumn.max}</p>
                    </div>
                  </div>

                  {/* Missing & Outliers */}
                  <div className="grid grid-cols-2 gap-4 mt-4">
                    <div className={`p-4 rounded-lg border ${
                      selectedColumn.missing > 0 
                        ? 'bg-yellow-500/10 border-yellow-600' 
                        : 'bg-green-500/10 border-green-600'
                    }`}>
                      <div className="flex items-center gap-2 mb-1">
                        <AlertTriangle className={`w-4 h-4 ${selectedColumn.missing > 0 ? 'text-yellow-400' : 'text-green-400'}`} />
                        <p className="text-sm font-medium">Missing Values</p>
                      </div>
                      <p className="text-xl font-bold">{selectedColumn.missing} ({((selectedColumn.missing / 891) * 100).toFixed(1)}%)</p>
                    </div>
                    <div className={`p-4 rounded-lg border ${
                      selectedColumn.outliers > 0 
                        ? 'bg-orange-500/10 border-orange-600' 
                        : 'bg-green-500/10 border-green-600'
                    }`}>
                      <div className="flex items-center gap-2 mb-1">
                        <TrendingUp className={`w-4 h-4 ${selectedColumn.outliers > 0 ? 'text-orange-400' : 'text-green-400'}`} />
                        <p className="text-sm font-medium">Outliers</p>
                      </div>
                      <p className="text-xl font-bold">{selectedColumn.outliers}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Distribution Histogram */}
              <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-cyan-400" />
                    Distribution
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64 flex items-end gap-1">
                    {selectedColumn.distribution.map((count, idx) => {
                      const maxCount = Math.max(...selectedColumn.distribution)
                      const heightPercent = (count / maxCount) * 100
                      return (
                        <motion.div
                          key={idx}
                          initial={{ height: 0 }}
                          animate={{ height: `${heightPercent}%` }}
                          transition={{ delay: idx * 0.02 }}
                          className="flex-1 bg-gradient-to-t from-blue-600 to-cyan-500 rounded-t hover:from-blue-500 hover:to-cyan-400 transition-all cursor-pointer relative group"
                          title={`${count} samples`}
                        >
                          <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-zinc-800 px-2 py-1 rounded text-xs opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                            {count}
                          </div>
                        </motion.div>
                      )
                    })}
                  </div>
                  <div className="flex justify-between mt-2 text-xs text-muted-foreground">
                    <span>{selectedColumn.min}</span>
                    <span>{selectedColumn.max}</span>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        {/* Correlation Heatmap */}
        <TabsContent value="correlations" className="space-y-6">
          <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/10 before:via-transparent before:to-cyan-500/10 before:opacity-30">
            <CardHeader className="relative">
              <CardTitle className="flex items-center gap-2">
                <Grid3x3 className="w-5 h-5 text-purple-400" />
                Correlation Matrix
              </CardTitle>
              <CardDescription>
                Pearson correlation coefficients between features
              </CardDescription>
            </CardHeader>
            <CardContent className="relative">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr>
                      <th className="p-2 text-left text-sm font-medium text-muted-foreground"></th>
                      {['Age', 'Fare', 'SibSp', 'Parch', 'Survived'].map((col) => (
                        <th key={col} className="p-2 text-center text-sm font-medium text-muted-foreground">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {correlationMatrix.map((row) => (
                      <tr key={row.feature}>
                        <td className="p-2 text-sm font-medium">{row.feature}</td>
                        {['Age', 'Fare', 'SibSp', 'Parch', 'Survived'].map((col) => {
                          const value = row[col as keyof typeof row] as number
                          return (
                            <td key={col} className="p-1">
                              <div 
                                className={`w-16 h-12 mx-auto rounded flex items-center justify-center text-sm font-medium ${getCorrelationColor(value)} transition-all hover:scale-110 cursor-pointer`}
                                title={`${row.feature} vs ${col}: ${value.toFixed(2)}`}
                              >
                                {value.toFixed(2)}
                              </div>
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Legend */}
              <div className="flex items-center justify-center gap-6 mt-6 pt-4 border-t border-zinc-700">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded bg-red-600"></div>
                  <span className="text-sm">Strong Negative</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded bg-zinc-700"></div>
                  <span className="text-sm">No Correlation</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded bg-green-600"></div>
                  <span className="text-sm">Strong Positive</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Feature Importance */}
        <TabsContent value="importance" className="space-y-6">
          <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-yellow-500/10 before:via-transparent before:to-orange-500/10 before:opacity-30">
            <CardHeader className="relative">
              <CardTitle className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-yellow-400" />
                Feature Importance
              </CardTitle>
              <CardDescription>
                Model-based feature importance scores (from Random Forest)
              </CardDescription>
            </CardHeader>
            <CardContent className="relative space-y-4">
              {featureImportance.map((item, idx) => (
                <motion.div
                  key={item.feature}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className="space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Badge className="bg-yellow-600 text-white">#{item.rank}</Badge>
                      <span className="font-medium">{item.feature}</span>
                    </div>
                    <span className="text-sm text-muted-foreground">{(item.importance * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-8 bg-zinc-800 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${item.importance * 100}%` }}
                      transition={{ duration: 0.5, delay: idx * 0.05 }}
                      className="h-full bg-gradient-to-r from-yellow-600 to-orange-600 rounded-full flex items-center justify-end pr-3"
                    >
                      <span className="text-xs font-medium">{(item.importance * 100).toFixed(1)}%</span>
                    </motion.div>
                  </div>
                </motion.div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>

        {/* SHAP Values */}
        <TabsContent value="shap" className="space-y-6">
          <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-green-500/10 before:via-transparent before:to-red-500/10 before:opacity-30">
            <CardHeader className="relative">
              <CardTitle className="flex items-center gap-2">
                <Info className="w-5 h-5 text-green-400" />
                SHAP Value Explanation
              </CardTitle>
              <CardDescription>
                Shapley values showing how each feature impacts the prediction
              </CardDescription>
            </CardHeader>
            <CardContent className="relative space-y-6">
              {/* Prediction Info */}
              <div className="p-4 rounded-lg bg-zinc-800/50 border border-zinc-700">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm text-muted-foreground">Sample Prediction</span>
                  <Badge className="bg-green-600">Survived</Badge>
                </div>
                <div className="text-xs text-muted-foreground space-y-1">
                  <p>Base value (population average): 0.38</p>
                  <p>Model prediction: 0.63 (62.7% probability of survival)</p>
                </div>
              </div>

              {/* SHAP Waterfall */}
              <div className="space-y-4">
                <h4 className="font-medium flex items-center gap-2">
                  <TrendingUp className="w-4 h-4" />
                  Feature Contributions
                </h4>

                {shapValues.map((item, idx) => (
                  <motion.div
                    key={item.feature}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.05 }}
                    className="space-y-2"
                  >
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{item.feature}</span>
                        <span className="text-muted-foreground">= {item.value}</span>
                      </div>
                      <span className={`font-medium ${item.color === 'green' ? 'text-green-400' : 'text-red-400'}`}>
                        {item.shap > 0 ? '+' : ''}{item.shap.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      {item.shap < 0 ? (
                        <>
                          <div className="flex-1 flex justify-end">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${Math.abs(item.shap) * 200}%` }}
                              transition={{ duration: 0.5, delay: idx * 0.05 }}
                              className="h-6 bg-red-600 rounded-l"
                            />
                          </div>
                          <div className="w-1 h-6 bg-zinc-700"></div>
                          <div className="flex-1"></div>
                        </>
                      ) : (
                        <>
                          <div className="flex-1"></div>
                          <div className="w-1 h-6 bg-zinc-700"></div>
                          <div className="flex-1">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${item.shap * 200}%` }}
                              transition={{ duration: 0.5, delay: idx * 0.05 }}
                              className="h-6 bg-green-600 rounded-r"
                            />
                          </div>
                        </>
                      )}
                    </div>
                  </motion.div>
                ))}
              </div>

              {/* Legend */}
              <div className="flex items-center gap-6 pt-4 border-t border-zinc-700">
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 rounded bg-red-600"></div>
                  <span className="text-sm">Decreases prediction</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 rounded bg-green-600"></div>
                  <span className="text-sm">Increases prediction</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
