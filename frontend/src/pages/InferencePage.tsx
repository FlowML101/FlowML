import { useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { FlaskConical, Play, Sparkles, TrendingUp, AlertCircle } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

const Label2 = ({ children, ...props }: any) => (
  <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70" {...props}>
    {children}
  </label>
)

const mockModels = [
  { id: 'catboost-v1', name: 'CatBoost v1', accuracy: 0.983, status: 'ready' },
  { id: 'xgboost-v1', name: 'XGBoost v1', accuracy: 0.978, status: 'ready' },
  { id: 'lightgbm-v2', name: 'LightGBM v2', accuracy: 0.975, status: 'ready' },
  { id: 'rf-v1', name: 'Random Forest v1', accuracy: 0.968, status: 'ready' },
]

export function InferencePage() {
  const [searchParams] = useSearchParams()
  const modelIdFromUrl = searchParams.get('model_id')
  
  const [selectedModel, setSelectedModel] = useState(modelIdFromUrl || mockModels[0].id)
  const [isLoading, setIsLoading] = useState(false)
  const [prediction, setPrediction] = useState<{
    survived: boolean
    confidence: number
    model: string
    latency: number
  } | null>(null)
  
  const [inputData, setInputData] = useState({
    pclass: '1',
    sex: 'female',
    age: '29',
    sibsp: '0',
    parch: '0',
    fare: '75.00',
    embarked: 'C',
  })

  const handlePredict = async () => {
    setIsLoading(true)
    setPrediction(null)
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1200))
    
    // Mock prediction based on inputs
    const ageNum = parseInt(inputData.age) || 0
    const fareNum = parseFloat(inputData.fare) || 0
    const isFemale = inputData.sex === 'female'
    const isFirstClass = inputData.pclass === '1'
    
    // Simple heuristic for demo
    let survivalChance = 0.5
    if (isFemale) survivalChance += 0.3
    if (isFirstClass) survivalChance += 0.2
    if (ageNum < 18) survivalChance += 0.1
    if (fareNum > 50) survivalChance += 0.1
    
    survivalChance = Math.min(Math.max(survivalChance, 0.1), 0.95)
    
    const selectedModelData = mockModels.find(m => m.id === selectedModel)
    
    setPrediction({
      survived: survivalChance > 0.5,
      confidence: survivalChance,
      model: selectedModelData?.name || 'CatBoost v1',
      latency: Math.floor(Math.random() * 20) + 8,
    })
    
    setIsLoading(false)
  }

  const currentModel = mockModels.find(m => m.id === selectedModel)

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
          <FlaskConical className="w-8 h-8 text-purple-500" />
          Model Playground
        </h1>
        <p className="text-zinc-400">Test your trained models with real-time inputs</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Model Selector + Input Form */}
        <div className="lg:col-span-2 space-y-6">
          {/* Model Selector */}
          <Card className="border-purple-600/30">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-purple-500" />
                Select Model
              </CardTitle>
              <CardDescription>Choose which trained model to use for inference</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a model" />
                  </SelectTrigger>
                  <SelectContent>
                    {mockModels.map((model) => (
                      <SelectItem key={model.id} value={model.id}>
                        <div className="flex items-center gap-2">
                          <span>{model.name}</span>
                          <Badge variant="secondary" className="text-xs">
                            {(model.accuracy * 100).toFixed(1)}% acc
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
                      <div className="text-xs text-zinc-500">Accuracy: {(currentModel.accuracy * 100).toFixed(1)}%</div>
                    </div>
                    <Badge variant="success">Ready</Badge>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Input Form */}
          <Card>
            <CardHeader>
              <CardTitle>Input Features</CardTitle>
              <CardDescription>Enter passenger details for prediction</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Passenger Class */}
              <div className="space-y-2">
                <Label2 htmlFor="pclass">Passenger Class</Label2>
                <Select
                  value={inputData.pclass}
                  onValueChange={(value) => setInputData({ ...inputData, pclass: value })}
                >
                  <SelectTrigger id="pclass">
                    <SelectValue placeholder="Select class" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1">1st Class</SelectItem>
                    <SelectItem value="2">2nd Class</SelectItem>
                    <SelectItem value="3">3rd Class</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Sex */}
              <div className="space-y-2">
                <Label2 htmlFor="sex">Sex</Label2>
                <Select
                  value={inputData.sex}
                  onValueChange={(value) => setInputData({ ...inputData, sex: value })}
                >
                  <SelectTrigger id="sex">
                    <SelectValue placeholder="Select sex" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="male">Male</SelectItem>
                    <SelectItem value="female">Female</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Age */}
              <div className="space-y-2">
                <Label2 htmlFor="age">Age</Label2>
                <Input
                  id="age"
                  type="number"
                  placeholder="Enter age"
                  value={inputData.age}
                  onChange={(e) => setInputData({ ...inputData, age: e.target.value })}
                />
              </div>

              {/* Siblings/Spouses */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label2 htmlFor="sibsp">Siblings/Spouses Aboard</Label2>
                  <Input
                    id="sibsp"
                    type="number"
                    placeholder="0"
                    value={inputData.sibsp}
                    onChange={(e) => setInputData({ ...inputData, sibsp: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <Label2 htmlFor="parch">Parents/Children Aboard</Label2>
                  <Input
                    id="parch"
                    type="number"
                    placeholder="0"
                    value={inputData.parch}
                    onChange={(e) => setInputData({ ...inputData, parch: e.target.value })}
                  />
                </div>
              </div>

              {/* Fare */}
              <div className="space-y-2">
                <Label2 htmlFor="fare">Fare</Label2>
                <Input
                  id="fare"
                  type="number"
                  step="0.01"
                  placeholder="Enter fare amount"
                  value={inputData.fare}
                  onChange={(e) => setInputData({ ...inputData, fare: e.target.value })}
                />
              </div>

              {/* Embarked */}
              <div className="space-y-2">
                <Label2 htmlFor="embarked">Port of Embarkation</Label2>
                <Select
                  value={inputData.embarked}
                  onValueChange={(value) => setInputData({ ...inputData, embarked: value })}
                >
                  <SelectTrigger id="embarked">
                    <SelectValue placeholder="Select port" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="C">Cherbourg</SelectItem>
                    <SelectItem value="Q">Queenstown</SelectItem>
                    <SelectItem value="S">Southampton</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Predict Button */}
              <Button
                onClick={handlePredict}
                disabled={isLoading}
                className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-lg py-6"
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
                <Card className={`border-2 ${prediction.survived ? 'border-green-600/50' : 'border-red-600/50 bg-red-600/5'}`}>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <TrendingUp className={`w-5 h-5 ${prediction.survived ? 'text-green-500' : 'text-red-500'}`} />
                      Prediction Result
                    </CardTitle>
                    <CardDescription>Model: {prediction.model}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {/* Main Prediction */}
                    <div className="text-center p-6 rounded-xl bg-zinc-900/50 border border-zinc-700">
                      <div className="text-sm text-zinc-500 mb-2">Survival Prediction</div>
                      <div className={`text-4xl font-bold mb-2 ${prediction.survived ? 'text-green-400' : 'text-red-400'}`}>
                        {prediction.survived ? 'Survived' : 'Did Not Survive'}
                      </div>
                      <div className="text-lg text-zinc-400">
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
                      <div className="p-3 rounded-lg bg-zinc-800/50 text-center">
                        <div className="text-xs text-zinc-500">Latency</div>
                        <div className="text-lg font-bold text-blue-400">{prediction.latency}ms</div>
                      </div>
                      <div className="p-3 rounded-lg bg-zinc-800/50 text-center">
                        <div className="text-xs text-zinc-500">Model</div>
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
                <Card className="border-zinc-700">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <FlaskConical className="w-5 h-5 text-zinc-500" />
                      Awaiting Prediction
                    </CardTitle>
                    <CardDescription>Fill in the form and click "Run Prediction"</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-col items-center justify-center py-12 text-center">
                      <div className="w-20 h-20 rounded-full bg-zinc-800/50 flex items-center justify-center mb-4">
                        <Play className="w-10 h-10 text-zinc-600" />
                      </div>
                      <p className="text-sm text-zinc-500">
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
    </div>
  )
}
