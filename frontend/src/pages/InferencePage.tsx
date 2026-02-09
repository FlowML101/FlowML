import { useState, useEffect, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { FlaskConical, Play, Sparkles, TrendingUp, AlertCircle, Loader2, Activity, RefreshCw, Wand2, X, ChevronDown, ChevronRight } from 'lucide-react'
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
  const [modelMetadata, setModelMetadata] = useState<any>(null)
  const [metadataLoading, setMetadataLoading] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [groupsCollapsed, setGroupsCollapsed] = useState<Record<string, boolean>>({})
  const [prediction, setPrediction] = useState<{
    survived: boolean
    confidence: number
    model: string
    latency: number
  } | null>(null)
  
  const [inputData, setInputData] = useState<Record<string, any>>({})

  // Debug: Log when inputData changes
  useEffect(() => {
    console.log('inputData state updated:', inputData)
  }, [inputData])

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

  // Fetch metadata when model is selected
  useEffect(() => {
    const fetchMetadata = async () => {
      if (!selectedModel) {
        setModelMetadata(null)
        setInputData({})
        return
      }

      setMetadataLoading(true)
      try {
        const metadata = await resultsApi.getMetadata(selectedModel)
        console.log('📊 LOADED METADATA:', metadata)
        console.log('Feature names:', metadata.feature_names)
        console.log('Numeric features:', metadata.numeric_features)
        console.log('Categorical features:', metadata.categorical_features)
        console.log('Categories:', metadata.onehot_categories)
        console.log('Numeric stats:', metadata.numeric_stats)
        setModelMetadata(metadata)
        
        // Initialize input data with default values (but keep empty for manual entry)
        const defaults: Record<string, any> = {}
        for (const feature of metadata.feature_names || []) {
          defaults[feature] = ''
        }
        setInputData(defaults)
        
        // Initialize all groups as expanded by default
        setGroupsCollapsed({})
      } catch (err) {
        console.error('Failed to load model metadata:', err)
        // Fallback for old models without metadata - show basic titanic fields
        const fallbackMetadata = {
          feature_names: ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked'],
          numeric_features: ['age', 'sibsp', 'parch', 'fare', 'pclass'],
          categorical_features: ['sex', 'embarked'],
          onehot_categories: {
            sex: ['male', 'female'],
            embarked: ['C', 'Q', 'S']
          },
          categorical_modes: {
            sex: 'male',
            embarked: 'S'
          },
          numeric_medians: {
            age: 28,
            sibsp: 0,
            parch: 0,
            fare: 14.5,
            pclass: 3
          },
          numeric_stats: {
            age: { min: 0.42, max: 80, mean: 29.7, median: 28, std: 14 },
            sibsp: { min: 0, max: 8, mean: 0.5, median: 0, std: 1 },
            parch: { min: 0, max: 6, mean: 0.4, median: 0, std: 0.8 },
            fare: { min: 0, max: 512, mean: 32, median: 14.5, std: 50 },
            pclass: { min: 1, max: 3, mean: 2.3, median: 3, std: 0.8 }
          }
        }
        console.log('Using fallback metadata for old model:', fallbackMetadata)
        setModelMetadata(fallbackMetadata)
        setInputData({
          pclass: '',
          sex: '',
          age: '',
          sibsp: '',
          parch: '',
          fare: '',
          embarked: ''
        })
      } finally {
        setMetadataLoading(false)
      }
    }

    fetchMetadata()
  }, [selectedModel])

  const handleAutoFill = () => {
    if (!modelMetadata) {
      console.log('No metadata available for autofill')
      return
    }
    
    console.log('Starting autofill with metadata:', modelMetadata)
    
    // Example values for common titanic features (fallback)
    const exampleDefaults: Record<string, any> = {
      pclass: '3',
      sex: 'male',
      age: '20-30',
      sibsp: '0',
      parch: '0',
      fare: '7.25',
      embarked: 'S'
    }
    
    const exampleData: Record<string, any> = {}
    for (const feature of modelMetadata.feature_names || []) {
      if (modelMetadata.binary_features?.includes(feature)) {
        // Binary features: use 0 as default
        exampleData[feature] = '0'
        console.log(`${feature} (binary): 0`)
      } else if (modelMetadata.numeric_features?.includes(feature)) {
        const stats = modelMetadata.numeric_stats?.[feature]
        const numericOptions = getNumericOptions(feature, stats)
        
        if (numericOptions && numericOptions.length > 0) {
          // Use middle option from dropdown
          const middleIndex = Math.floor(numericOptions.length / 2)
          exampleData[feature] = numericOptions[middleIndex]
          console.log(`${feature} (numeric dropdown): ${numericOptions[middleIndex]}`)
        } else {
          // Use median/mean as before
          let numValue = stats?.median || stats?.mean || modelMetadata.numeric_medians?.[feature]
          
          if (numValue !== undefined && numValue !== null) {
            if (Number.isInteger(numValue) || feature.toLowerCase().includes('count')) {
              exampleData[feature] = String(Math.round(numValue))
            } else {
              exampleData[feature] = String(Number(numValue).toFixed(2))
            }
          } else {
            exampleData[feature] = exampleDefaults[feature] || '0'
          }
          console.log(`${feature} (numeric input): ${exampleData[feature]}`)
        }
      } else if (modelMetadata.categorical_features?.includes(feature)) {
        exampleData[feature] = modelMetadata.categorical_modes?.[feature] || 
                              modelMetadata.onehot_categories?.[feature]?.[0] ||
                              exampleDefaults[feature] || ''
        console.log(`${feature} (categorical): ${exampleData[feature]}`)
      } else {
        exampleData[feature] = exampleDefaults[feature] || ''
        console.log(`${feature} (other): ${exampleData[feature]}`)
      }
    }
    
    console.log('Final autofill data:', exampleData)
    console.log('Current inputData before update:', inputData)
    setInputData(exampleData)
    console.log('setInputData called with:', exampleData)
  }

  // Helper to generate dropdown options for numeric features
  const getNumericOptions = (feature: string, stats: any) => {
    console.log(`🔍 getNumericOptions called for ${feature}:`, stats)
    
    const featureLower = feature.toLowerCase()
    
    // Check for common feature patterns and provide smart options
    if (featureLower.includes('class') || featureLower === 'pclass') {
      console.log(`  → Pattern match: class → ['1','2','3']`)
      return ['1', '2', '3']
    }
    
    if (featureLower.includes('age')) {
      console.log(`  → Pattern match: age → range bins`)
      return ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+']
    }
    
    if (featureLower.includes('sibsp') || featureLower.includes('parch') || featureLower.includes('siblings') || featureLower.includes('parents')) {
      console.log(`  → Pattern match: siblings/parents → ['0'-'5+']`)
      return ['0', '1', '2', '3', '4', '5+']
    }
    
    if (stats && stats.min !== undefined && stats.max !== undefined) {
      const range = stats.max - stats.min
      const min = stats.min
      const max = stats.max
      
      console.log(`  → Stats available: min=${min}, max=${max}, range=${range}`)
      
      // If small integer range (like 0-10), show each value
      if (range <= 15 && Number.isInteger(min) && Number.isInteger(max)) {
        const options = []
        for (let i = Math.floor(min); i <= Math.ceil(max); i++) {
          options.push(String(i))
        }
        console.log(`  → Small integer range → ${options.length} options`)
        return options
      }
      
      // For larger ranges, create bins
      if (range > 0) {
        const bins = 8
        const step = range / bins
        const options = []
        for (let i = 0; i < bins; i++) {
          const rangeStart = (min + i * step).toFixed(1)
          const rangeEnd = (min + (i + 1) * step).toFixed(1)
          options.push(`${rangeStart}-${rangeEnd}`)
        }
        console.log(`  → Large range → ${bins} bins`)
        return options
      }
    }
    
    console.log(`  → No match, returning null → will use text input`)
    return null // Return null to use text input
  }

  // Helper to parse range values back to numbers
  const parseRangeValue = (value: string): string => {
    if (value.includes('-')) {
      // Return midpoint of range
      const parts = value.split('-')
      const start = parseFloat(parts[0])
      const end = parseFloat(parts[1])
      return String((start + end) / 2)
    }
    if (value.includes('+')) {
      // For "5+" return the number
      return value.replace('+', '')
    }
    return value
  }

  const handleClear = () => {
    if (!modelMetadata) return
    
    const emptyData: Record<string, any> = {}
    for (const feature of modelMetadata.feature_names || []) {
      emptyData[feature] = ''
    }
    setInputData(emptyData)
  }

  const handlePredict = async () => {
    setIsLoading(true)
    setPrediction(null)
    
    const startTime = Date.now()
    
    try {
      // Convert string inputs to appropriate types
      const features: Record<string, any> = {}
      
      if (modelMetadata) {
        for (const [key, value] of Object.entries(inputData)) {
          if (modelMetadata.numeric_features?.includes(key)) {
            // Parse range values (e.g., "20-30" becomes 25)
            const parsedValue = parseRangeValue(String(value))
            // Convert to number
            features[key] = parsedValue === '' ? null : parseFloat(parsedValue)
          } else {
            // Keep as string for categorical
            features[key] = value
          }
        }
      } else {
        // Fallback: try to parse numbers
        for (const [key, value] of Object.entries(inputData)) {
          const parsed = parseFloat(value as string)
          features[key] = isNaN(parsed) ? value : parsed
        }
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
      setPrediction(null)
      const currentModelData = filteredModels.find(m => m.id === selectedModel)
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
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Input Features</CardTitle>
                  <CardDescription>
                    {modelMetadata 
                      ? `Enter values for ${modelMetadata.feature_names?.length || 0} features`
                      : 'Select a model to see input fields'}
                  </CardDescription>
                </div>
                {modelMetadata && (
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleAutoFill}
                      className="gap-2 bg-purple-600/20"
                    >
                      <Wand2 className="w-4 h-4" />
                      Auto-Fill v2
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleClear}
                      className="gap-2"
                    >
                      <X className="w-4 h-4" />
                      Clear
                    </Button>
                  </div>
                )}
              </div>
            </CardHeader>
            <CardContent className="space-y-4 relative">
              {metadataLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-8 h-8 animate-spin text-purple-500" />
                </div>
              ) : modelMetadata && modelMetadata.feature_names ? (
                <>
                  {(() => {
                    const features = modelMetadata.feature_names
                    const numericFeatures = features.filter((f: string) => modelMetadata.numeric_features?.includes(f))
                    const binaryFeatures = features.filter((f: string) => modelMetadata.binary_features?.includes(f))
                    const categoricalFeatures = features.filter((f: string) => modelMetadata.categorical_features?.includes(f))
                    const shouldCollapse = features.length > 12  // Only collapse if more than 12 features
                    
                    const renderFeature = (feature: string) => {
                      const isNumeric = modelMetadata.numeric_features?.includes(feature)
                      const isBinary = modelMetadata.binary_features?.includes(feature)
                      const isCategorical = modelMetadata.categorical_features?.includes(feature)
                      const categories = modelMetadata.onehot_categories?.[feature]
                      const stats = modelMetadata.numeric_stats?.[feature]
                      const numericOptions = isNumeric ? getNumericOptions(feature, stats) : null
                      
                      console.log(`🎨 Rendering ${feature}:`, { 
                        isNumeric, 
                        isBinary,
                        isCategorical, 
                        hasCategories: categories?.length > 0, 
                        hasStats: !!stats,
                        numericOptions 
                      })
                      
                      return (
                        <div key={feature} className="space-y-2">
                          <div className="flex items-center justify-between">
                            <Label htmlFor={feature} className="capitalize">
                              {feature.replace(/_/g, ' ')}
                            </Label>
                            {isNumeric && stats && !numericOptions && (
                              <span className="text-xs text-muted-foreground">
                                {stats.min?.toFixed(0)}-{stats.max?.toFixed(0)} (avg: {stats.mean?.toFixed(1)})
                              </span>
                            )}
                          </div>
                          
                          {(isBinary || isCategorical) && categories && categories.length > 0 ? (
                            // Binary or Categorical dropdown
                            <Select
                              value={String(inputData[feature] || '')}
                              onValueChange={(value) => setInputData({ ...inputData, [feature]: value })}
                            >
                              <SelectTrigger id={feature}>
                                <SelectValue placeholder={`Select ${feature}`} />
                              </SelectTrigger>
                              <SelectContent>
                                {categories.map((cat: string) => (
                                  <SelectItem key={cat} value={cat}>
                                    {cat}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          ) : numericOptions ? (
                            // Numeric dropdown with preset options
                            <Select
                              value={String(inputData[feature] || '')}
                              onValueChange={(value) => {
                                const parsedValue = parseRangeValue(value)
                                setInputData({ ...inputData, [feature]: parsedValue })
                              }}
                            >
                              <SelectTrigger id={feature}>
                                <SelectValue placeholder={`Select ${feature}`} />
                              </SelectTrigger>
                              <SelectContent>
                                {numericOptions.map((option: string) => (
                                  <SelectItem key={option} value={option}>
                                    {option}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          ) : (
                            // Text/number input for other fields
                            <Input
                              id={feature}
                              type={isNumeric ? "number" : "text"}
                              step={isNumeric ? "any" : undefined}
                              placeholder={isNumeric && stats ? `e.g., ${stats.median?.toFixed(1)}` : `Enter ${feature}`}
                              value={inputData[feature] || ''}
                              onChange={(e) => setInputData({ ...inputData, [feature]: e.target.value })}
                            />
                          )}
                        </div>
                      )
                    }
                    
                    if (!shouldCollapse) {
                      // Show all features without grouping (8 or fewer)
                      return <div className="space-y-4">{features.map(renderFeature)}</div>
                    }
                    
                    // Show collapsible groups for many features (9+)
                    return (
                      <div className="space-y-4">
                        {numericFeatures.length > 0 && (
                          <div className="space-y-2">
                            <button
                              type="button"
                              onClick={() => setGroupsCollapsed({ ...groupsCollapsed, numeric: !groupsCollapsed.numeric })}
                              className="flex items-center gap-2 text-sm font-medium hover:text-purple-400 transition-colors w-full text-left"
                            >
                              {groupsCollapsed.numeric ? <ChevronRight className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                              🔢 Numeric Features ({numericFeatures.length})
                            </button>
                            {!groupsCollapsed.numeric && (
                              <div className="space-y-4 pl-6 border-l-2 border-purple-600/20">
                                {numericFeatures.map(renderFeature)}
                              </div>
                            )}
                          </div>
                        )}
                        
                        {binaryFeatures.length > 0 && (
                          <div className="space-y-2">
                            <button
                              type="button"
                              onClick={() => setGroupsCollapsed({ ...groupsCollapsed, binary: !groupsCollapsed.binary })}
                              className="flex items-center gap-2 text-sm font-medium hover:text-green-400 transition-colors w-full text-left"
                            >
                              {groupsCollapsed.binary ? <ChevronRight className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                              ✓ Binary Features ({binaryFeatures.length})
                            </button>
                            {!groupsCollapsed.binary && (
                              <div className="space-y-4 pl-6 border-l-2 border-green-600/20">
                                {binaryFeatures.map(renderFeature)}
                              </div>
                            )}
                          </div>
                        )}
                        
                        {categoricalFeatures.length > 0 && (
                          <div className="space-y-2">
                            <button
                              type="button"
                              onClick={() => setGroupsCollapsed({ ...groupsCollapsed, categorical: !groupsCollapsed.categorical })}
                              className="flex items-center gap-2 text-sm font-medium hover:text-blue-400 transition-colors w-full text-left"
                            >
                              {groupsCollapsed.categorical ? <ChevronRight className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                              📝 Categorical Features ({categoricalFeatures.length})
                            </button>
                            {!groupsCollapsed.categorical && (
                              <div className="space-y-4 pl-6 border-l-2 border-blue-600/20">
                                {categoricalFeatures.map(renderFeature)}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )
                  })()}
                </>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  No model metadata available. Select a trained model first.
                </div>
              )}

              {/* Predict Button */}
              <Button
                onClick={handlePredict}
                disabled={isLoading || !modelMetadata}
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
