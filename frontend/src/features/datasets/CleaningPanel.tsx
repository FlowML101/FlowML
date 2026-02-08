import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { 
  Sparkles, Eye, Check, X, AlertCircle, TrendingUp, 
  Trash2, Play, ArrowRight, Wand2
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'

interface CleaningOperation {
  operation: string
  column?: string
  columns?: string[]
  description: string
  confidence: number
  parameters: Record<string, any>
  isEditing?: boolean
  enabled?: boolean
}

interface CleaningPanelProps {
  datasetId: string
  datasetName: string
  onDatasetUpdated?: (newDatasetId: string) => void
}

export function CleaningPanel({ datasetId, datasetName, onDatasetUpdated }: CleaningPanelProps) {
  const [suggestions, setSuggestions] = useState<CleaningOperation[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isPreviewing, setIsPreviewing] = useState(false)
  const [isApplying, setIsApplying] = useState(false)
  const [previewData, setPreviewData] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)

  const analyzeDataset = async () => {
    setIsAnalyzing(true)
    setError(null)
    setSuggestions([])
    
    try {
      const response = await fetch(`${API_BASE}/llm/suggest-cleaning`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: datasetId }),
      })
      
      if (!response.ok) throw new Error('Failed to analyze dataset')
      
      const data = await response.json()
      
      if (!data.llm_available) {
        setError('AI assistant is offline. Please start Ollama to get cleaning suggestions.')
        return
      }
      
      if (data.suggestions.length === 0) {
        setSuccessMessage('✨ Your dataset looks clean! No issues detected.')
        return
      }
      
      // Enable all suggestions by default
      setSuggestions(data.suggestions.map((s: CleaningOperation) => ({ ...s, enabled: true })))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze dataset')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleParameterChange = (index: number, paramName: string, value: any) => {
    setSuggestions(prev => prev.map((sug, i) => 
      i === index 
        ? { ...sug, parameters: { ...sug.parameters, [paramName]: value } }
        : sug
    ))
  }

  const toggleSuggestion = (index: number) => {
    setSuggestions(prev => prev.map((sug, i) => 
      i === index ? { ...sug, enabled: !sug.enabled } : sug
    ))
  }

  const removeSuggestion = (index: number) => {
    setSuggestions(prev => prev.filter((_, i) => i !== index))
  }

  const previewOperations = async () => {
    setIsPreviewing(true)
    setError(null)
    setPreviewData(null)
    
    try {
      const enabledOps = suggestions.filter(s => s.enabled)
      
      if (enabledOps.length === 0) {
        setError('Please enable at least one operation to preview')
        return
      }
      
      const response = await fetch(`${API_BASE}/llm/apply-operations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: datasetId,
          operations: enabledOps,
          preview_only: true,
        }),
      })
      
      if (!response.ok) throw new Error('Failed to preview operations')
      
      const data = await response.json()
      
      if (!data.success) {
        setError(data.error || 'Preview failed')
        return
      }
      
      setPreviewData(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to preview operations')
    } finally {
      setIsPreviewing(false)
    }
  }

  const applyOperations = async () => {
    if (!confirm(`Create a new cleaned version of "${datasetName}"?\n\nOriginal dataset will be preserved.`)) {
      return
    }
    
    setIsApplying(true)
    setError(null)
    
    try {
      const enabledOps = suggestions.filter(s => s.enabled)
      
      const response = await fetch(`${API_BASE}/llm/apply-operations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: datasetId,
          operations: enabledOps,
          preview_only: false,
          new_dataset_name: `${datasetName}_cleaned`,
        }),
      })
      
      if (!response.ok) throw new Error('Failed to apply operations')
      
      const data = await response.json()
      
      if (!data.success) {
        setError(data.error || 'Failed to apply operations')
        return
      }
      
      setSuccessMessage(
        `✅ Created "${data.new_dataset_name}"! ${data.operations_applied.length} operations applied. ` +
        `(${data.rows_before} → ${data.rows_after} rows, ${data.columns_before} → ${data.columns_after} columns)`
      )
      
      // Notify parent component
      if (onDatasetUpdated && data.new_dataset_id) {
        onDatasetUpdated(data.new_dataset_id)
      }
      
      // Reset state
      setSuggestions([])
      setPreviewData(null)
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to apply operations')
    } finally {
      setIsApplying(false)
    }
  }

  const getOperationIcon = (opType: string) => {
    const icons: Record<string, any> = {
      fill_missing: '🩹',
      drop_missing: '🗑️',
      drop_duplicates: '🔄',
      cast_type: '🔢',
      remove_outliers: '📊',
      standardize: '📏',
      normalize: '📉',
      string_clean: '🧹',
      rename_column: '✏️',
      drop_column: '❌',
      encode_categorical: '🏷️',
    }
    return icons[opType] || '⚙️'
  }

  const renderParameterEditor = (sug: CleaningOperation, index: number) => {
    const { operation, parameters } = sug

    if (operation === 'fill_missing') {
      return (
        <div className="space-y-2">
          <Label className="text-xs">Strategy</Label>
          <Select
            value={parameters.strategy || 'mean'}
            onValueChange={(val) => handleParameterChange(index, 'strategy', val)}
          >
            <SelectTrigger className="h-8">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="mean">Mean (Average)</SelectItem>
              <SelectItem value="median">Median (Middle value)</SelectItem>
              <SelectItem value="mode">Mode (Most common)</SelectItem>
              <SelectItem value="forward">Forward fill</SelectItem>
              <SelectItem value="backward">Backward fill</SelectItem>
              <SelectItem value="constant">Constant value</SelectItem>
            </SelectContent>
          </Select>
          {parameters.strategy === 'constant' && (
            <Input
              type="text"
              placeholder="Fill value"
              value={parameters.fill_value || ''}
              onChange={(e) => handleParameterChange(index, 'fill_value', e.target.value)}
              className="h-8"
            />
          )}
        </div>
      )
    }

    if (operation === 'remove_outliers') {
      return (
        <div className="space-y-2">
          <Label className="text-xs">Method</Label>
          <Select
            value={parameters.method || 'iqr'}
            onValueChange={(val) => handleParameterChange(index, 'method', val)}
          >
            <SelectTrigger className="h-8">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="iqr">IQR (Interquartile Range)</SelectItem>
              <SelectItem value="zscore">Z-Score</SelectItem>
              <SelectItem value="percentile">Percentile</SelectItem>
            </SelectContent>
          </Select>
          <Label className="text-xs">Threshold</Label>
          <Input
            type="number"
            step="0.1"
            value={parameters.threshold || 1.5}
            onChange={(e) => handleParameterChange(index, 'threshold', parseFloat(e.target.value))}
            className="h-8"
          />
        </div>
      )
    }

    if (operation === 'cast_type') {
      return (
        <div className="space-y-2">
          <Label className="text-xs">Target Type</Label>
          <Select
            value={parameters.target_type || 'string'}
            onValueChange={(val) => handleParameterChange(index, 'target_type', val)}
          >
            <SelectTrigger className="h-8">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="int">Integer</SelectItem>
              <SelectItem value="float">Decimal</SelectItem>
              <SelectItem value="string">Text</SelectItem>
              <SelectItem value="datetime">Date/Time</SelectItem>
              <SelectItem value="boolean">True/False</SelectItem>
            </SelectContent>
          </Select>
        </div>
      )
    }

    if (operation === 'string_clean') {
      const operations = parameters.operations || []
      const availableOps = ['lowercase', 'uppercase', 'strip', 'remove_special', 'remove_numbers']
      
      return (
        <div className="space-y-2">
          <Label className="text-xs">String Operations</Label>
          <div className="flex flex-wrap gap-1">
            {availableOps.map(op => (
              <Badge
                key={op}
                variant={operations.includes(op) ? 'default' : 'outline'}
                className="cursor-pointer text-xs"
                onClick={() => {
                  const newOps = operations.includes(op)
                    ? operations.filter((o: string) => o !== op)
                    : [...operations, op]
                  handleParameterChange(index, 'operations', newOps)
                }}
              >
                {op.replace('_', ' ')}
              </Badge>
            ))}
          </div>
        </div>
      )
    }

    return null
  }

  return (
    <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50">
      <CardHeader className="border-b border-border">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Wand2 className="w-5 h-5 text-purple-500" />
            Data Cleaning Assistant
          </CardTitle>
          <Button
            onClick={analyzeDataset}
            disabled={isAnalyzing}
            variant="outline"
            size="sm"
          >
            {isAnalyzing ? (
              <>
                <Sparkles className="w-4 h-4 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4 mr-2" />
                Analyze Dataset
              </>
            )}
          </Button>
        </div>
      </CardHeader>

      <CardContent className="p-6 space-y-4">
        {error && (
          <div className="flex items-center gap-2 p-4 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400">
            <AlertCircle className="w-4 h-4" />
            <p className="text-sm">{error}</p>
          </div>
        )}

        {successMessage && (
          <div className="flex items-center gap-2 p-4 rounded-lg bg-green-500/10 border border-green-500/30 text-green-400">
            <Check className="w-4 h-4" />
            <p className="text-sm">{successMessage}</p>
          </div>
        )}

        {suggestions.length === 0 && !isAnalyzing && !successMessage && (
          <div className="text-center py-12 text-muted-foreground">
            <Wand2 className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>Click "Analyze Dataset" to get AI-powered cleaning suggestions</p>
          </div>
        )}

        <AnimatePresence>
          {suggestions.map((sug, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, x: -20 }}
            >
              <Card className={`border ${sug.enabled ? 'border-purple-500/30' : 'border-zinc-700 opacity-60'}`}>
                <CardHeader className="pb-3">
                  <div className="flex items-start gap-3">
                    <div className="flex items-center gap-2 flex-1">
                      <span className="text-2xl">{getOperationIcon(sug.operation)}</span>
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <h4 className="font-semibold text-sm">
                            {sug.operation.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </h4>
                          {sug.column && (
                            <Badge variant="outline" className="text-xs">
                              {sug.column}
                            </Badge>
                          )}
                          <Badge 
                            variant={sug.confidence > 0.8 ? 'default' : 'secondary'}
                            className="text-xs"
                          >
                            {Math.round(sug.confidence * 100)}% confident
                          </Badge>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">{sug.description}</p>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-1">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => toggleSuggestion(index)}
                        className="h-7 w-7 p-0"
                      >
                        {sug.enabled ? (
                          <Check className="w-4 h-4 text-green-500" />
                        ) : (
                          <X className="w-4 h-4" />
                        )}
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => removeSuggestion(index)}
                        className="h-7 w-7 p-0 text-red-500"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                
                {sug.enabled && Object.keys(sug.parameters).length > 0 && (
                  <CardContent className="pt-0 pb-3">
                    {renderParameterEditor(sug, index)}
                  </CardContent>
                )}
              </Card>
            </motion.div>
          ))}
        </AnimatePresence>

        {suggestions.length > 0 && (
          <div className="flex gap-2 pt-4 border-t border-border">
            <Button
              onClick={previewOperations}
              disabled={isPreviewing || suggestions.filter(s => s.enabled).length === 0}
              variant="outline"
              className="flex-1"
            >
              {isPreviewing ? (
                <>
                  <Sparkles className="w-4 h-4 mr-2 animate-spin" />
                  Previewing...
                </>
              ) : (
                <>
                  <Eye className="w-4 h-4 mr-2" />
                  Preview Changes
                </>
              )}
            </Button>
            <Button
              onClick={applyOperations}
              disabled={isApplying || suggestions.filter(s => s.enabled).length === 0}
              className="flex-1 bg-gradient-to-r from-purple-600 to-blue-600"
            >
              {isApplying ? (
                <>
                  <Sparkles className="w-4 h-4 mr-2 animate-spin" />
                  Applying...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Apply & Create New Dataset
                </>
              )}
            </Button>
          </div>
        )}

        {previewData && (
          <Card className="border-green-500/30 bg-green-500/5">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-green-500" />
                Preview Results
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">Rows</p>
                  <p className="font-semibold">
                    {previewData.rows_before} <ArrowRight className="w-3 h-3 inline" /> {previewData.rows_after}
                  </p>
                </div>
                <div>
                  <p className="text-muted-foreground">Columns</p>
                  <p className="font-semibold">
                    {previewData.columns_before} <ArrowRight className="w-3 h-3 inline" /> {previewData.columns_after}
                  </p>
                </div>
              </div>
              <div className="text-xs text-muted-foreground">
                <p>Applied operations: {previewData.operations_applied?.length || 0}</p>
              </div>
              {previewData.preview_rows && previewData.preview_rows.length > 0 && (
                <div className="overflow-x-auto">
                  <p className="text-xs text-muted-foreground mb-2">First few rows after cleaning:</p>
                  <div className="bg-zinc-950 rounded p-2 max-h-48 overflow-y-auto">
                    <pre className="text-xs">
                      {JSON.stringify(previewData.preview_rows.slice(0, 3), null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </CardContent>
    </Card>
  )
}
