import { useState, useRef, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { 
  Bot, Send, Sparkles, Code2, Copy, Check, Zap, 
  TrendingUp, AlertCircle, Lightbulb, WifiOff, RefreshCw, Database
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { useDataset } from '@/contexts/DatasetContext'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'

interface Message {
  role: 'user' | 'assistant'
  content: string
  code?: string
  codeLanguage?: string
  suggestions?: string[]
  timestamp: Date
}

interface LLMStatus {
  available: boolean
  url: string
  default_model: string
  models: string[]
}

export function EnhancedCopilot() {
  const { selectedDataset, previewData } = useDataset()
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [copiedCode, setCopiedCode] = useState(false)
  const [llmStatus, setLlmStatus] = useState<LLMStatus | null>(null)
  const [isCheckingStatus, setIsCheckingStatus] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(scrollToBottom, [messages])

  // Check LLM status on mount
  useEffect(() => {
    checkLLMStatus()
  }, [])

  const checkLLMStatus = async () => {
    setIsCheckingStatus(true)
    try {
      const response = await fetch(`${API_BASE}/llm/status`)
      if (response.ok) {
        const status: LLMStatus = await response.json()
        setLlmStatus(status)
        
        if (status.available) {
          // Add welcome message when LLM is available
          setMessages([{
            role: 'assistant',
            content: `👋 I'm your AI Data Assistant powered by ${status.default_model}. I can help you clean data, engineer features, and explain model results. Select a dataset to get started!`,
            suggestions: ['Suggest data cleaning', 'Explain results', 'Feature engineering tips'],
            timestamp: new Date()
          }])
        }
      } else {
        setLlmStatus({ available: false, url: '', default_model: '', models: [] })
      }
    } catch {
      setLlmStatus({ available: false, url: '', default_model: '', models: [] })
    } finally {
      setIsCheckingStatus(false)
    }
  }

  const handleSend = async (text?: string) => {
    const messageText = text || input
    if (!messageText.trim() || !llmStatus?.available) return

    const userMessage: Message = {
      role: 'user',
      content: messageText,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsTyping(true)

    try {
      // Build context from dataset if available
      let context = ''
      if (selectedDataset && previewData) {
        context = `Dataset: ${selectedDataset.name}\n`
        context += `Rows: ${selectedDataset.num_rows}, Columns: ${selectedDataset.num_columns}\n`
        context += `Columns: ${previewData.columns.join(', ')}`
      }
      
      // Call real chat API
      const response = await fetch(`${API_BASE}/llm/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: messageText,
          context: context || undefined,
        }),
      })

      if (!response.ok) {
        throw new Error('Chat request failed')
      }

      const data = await response.json()
      
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.response || 'No response from AI',
        timestamp: new Date()
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage: Message = {
        role: 'assistant',
        content: selectedDataset
          ? '❌ Sorry, I encountered an error. Please try again.'
          : '⚠️ Please select a dataset first to get context-aware analysis.',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsTyping(false)
    }
  }

  const copyCode = (code: string) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(true)
    setTimeout(() => setCopiedCode(false), 2000)
  }

  // Loading state
  if (isCheckingStatus) {
    return (
      <Card className="flex flex-col border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 h-full">
        <CardHeader className="border-b border-border">
          <CardTitle className="flex items-center gap-2 text-lg">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-purple-600 via-blue-600 to-pink-600 flex items-center justify-center shadow-lg animate-pulse">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <span>Connecting to AI...</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-1 flex items-center justify-center">
          <RefreshCw className="w-8 h-8 text-muted-foreground animate-spin" />
        </CardContent>
      </Card>
    )
  }

  // LLM not available state
  if (!llmStatus?.available) {
    return (
      <Card className="flex flex-col border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 h-full">
        <CardHeader className="border-b border-border">
          <CardTitle className="flex items-center gap-2 text-lg">
            <div className="w-9 h-9 rounded-xl bg-zinc-700 flex items-center justify-center shadow-lg">
              <WifiOff className="w-5 h-5 text-zinc-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                AI Copilot
                <Badge variant="outline" className="text-xs text-yellow-500 border-yellow-500/30">
                  Offline
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground font-normal">LLM service not connected</p>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col items-center justify-center gap-4 p-8">
          <div className="w-16 h-16 rounded-2xl bg-zinc-800 flex items-center justify-center">
            <Bot className="w-8 h-8 text-zinc-500" />
          </div>
          <div className="text-center space-y-2">
            <h3 className="font-semibold">Ollama Not Connected</h3>
            <p className="text-sm text-muted-foreground max-w-xs">
              To enable AI-powered data analysis, start Ollama with any model you prefer.
            </p>
          </div>
          <div className="bg-zinc-800/50 rounded-lg p-4 w-full max-w-sm">
            <p className="text-xs text-muted-foreground mb-2">Example:</p>
            <code className="text-xs text-green-400 font-mono">
              ollama run llama3
            </code>
            <p className="text-xs text-muted-foreground mt-2">or any other model</p>
          </div>
          <Button 
            onClick={checkLLMStatus}
            variant="outline"
            className="mt-2"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Retry Connection
          </Button>
        </CardContent>
      </Card>
    )
  }

  // LLM available - show chat interface
  return (
    <Card className="flex flex-col border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/10 before:via-cyan-500/10 before:to-blue-500/10 before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-blue-500/12 h-full">
      <CardHeader className="border-b border-border relative pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-lg">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-purple-600 via-blue-600 to-pink-600 flex items-center justify-center shadow-lg">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                AI Copilot
                <Badge variant="secondary" className="text-xs">
                  <Sparkles className="w-3 h-3 mr-1" />
                  {llmStatus.default_model}
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground font-normal">Connected to Ollama</p>
            </div>
          </CardTitle>

          {/* Connection status */}
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span className="text-xs text-muted-foreground">Online</span>
          </div>
        </div>
        
        {/* Dataset indicator */}
        {selectedDataset && (
          <Alert className="mt-3 bg-blue-500/10 border-blue-500/30">
            <Database className="w-4 h-4" />
            <AlertDescription className="text-xs">
              Analyzing: <span className="font-semibold">{selectedDataset.name}</span> 
              <span className="text-muted-foreground ml-2">
                ({selectedDataset.num_rows} rows, {selectedDataset.num_columns} columns)
              </span>
            </AlertDescription>
          </Alert>
        )}
        
        {!selectedDataset && (
          <Alert className="mt-3 bg-yellow-500/10 border-yellow-500/30">
            <AlertCircle className="w-4 h-4" />
            <AlertDescription className="text-xs">
              Select a dataset to get context-aware analysis
            </AlertDescription>
          </Alert>
        )}
      </CardHeader>

      <CardContent className="flex-1 overflow-y-auto p-4 space-y-4 relative">
        <AnimatePresence>
          {messages.map((message, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-[85%] ${message.role === 'user' ? '' : 'space-y-3'}`}>
                {message.role === 'assistant' && (
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-purple-600 to-blue-600 flex items-center justify-center">
                      <Bot className="w-3.5 h-3.5 text-white" />
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                )}

                <div
                  className={`rounded-xl p-4 ${
                    message.role === 'user'
                      ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white shadow-lg'
                      : 'bg-zinc-800/50 border border-zinc-700/50 text-zinc-100'
                  }`}
                >
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
                </div>

                {message.code && (
                  <div className="relative group">
                    <div className="flex items-center justify-between px-3 py-2 bg-zinc-950 border border-zinc-700 rounded-t-lg">
                      <div className="flex items-center gap-2">
                        <Code2 className="w-4 h-4 text-purple-400" />
                        <span className="text-xs text-zinc-400 font-mono">{message.codeLanguage}</span>
                      </div>
                      <Button
                        onClick={() => copyCode(message.code!)}
                        size="sm"
                        variant="ghost"
                        className="h-6 px-2 hover:bg-zinc-800"
                      >
                        {copiedCode ? (
                          <Check className="w-3 h-3 text-green-500" />
                        ) : (
                          <Copy className="w-3 h-3" />
                        )}
                      </Button>
                    </div>
                    <pre className="bg-zinc-950 border border-t-0 border-zinc-700 rounded-b-lg p-3 overflow-x-auto">
                      <code className="text-xs text-zinc-300 font-mono">{message.code}</code>
                    </pre>
                  </div>
                )}

                {message.suggestions && message.suggestions.length > 0 && (
                  <div className="flex flex-wrap gap-2">
                    {message.suggestions.map((suggestion, i) => (
                      <Button
                        key={i}
                        onClick={() => handleSend(suggestion)}
                        size="sm"
                        variant="outline"
                        className="text-xs border-zinc-700 hover:bg-zinc-800 hover:border-purple-500/50 transition-colors"
                      >
                        <Lightbulb className="w-3 h-3 mr-1.5" />
                        {suggestion}
                      </Button>
                    ))}
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {isTyping && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center gap-2"
          >
            <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-purple-600 to-blue-600 flex items-center justify-center">
              <Bot className="w-3.5 h-3.5 text-white" />
            </div>
            <div className="bg-zinc-800/50 border border-zinc-700/50 rounded-xl px-4 py-3">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 bg-pink-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </CardContent>

      <div className="p-4 border-t border-border space-y-3">
        {/* Quick Actions */}
        <div className="flex gap-2 overflow-x-auto pb-2">
          <Button
            onClick={() => handleSend("Analyze missing values in my dataset")}
            size="sm"
            variant="outline"
            className="text-xs whitespace-nowrap border-zinc-700 hover:bg-zinc-800"
          >
            <AlertCircle className="w-3 h-3 mr-1.5" />
            Missing Data
          </Button>
          <Button
            onClick={() => handleSend("Suggest feature engineering ideas")}
            size="sm"
            variant="outline"
            className="text-xs whitespace-nowrap border-zinc-700 hover:bg-zinc-800"
          >
            <Zap className="w-3 h-3 mr-1.5" />
            Feature Ideas
          </Button>
          <Button
            onClick={() => handleSend("Explain correlation patterns")}
            size="sm"
            variant="outline"
            className="text-xs whitespace-nowrap border-zinc-700 hover:bg-zinc-800"
          >
            <TrendingUp className="w-3 h-3 mr-1.5" />
            Correlations
          </Button>
        </div>

        {/* Input */}
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleSend()
              }
            }}
            placeholder="Ask about your data..."
            className="flex-1 bg-zinc-800/50 border-zinc-700 focus:border-purple-500/50"
          />
          <Button 
            onClick={() => handleSend()} 
            disabled={!input.trim() || isTyping}
            className="bg-gradient-to-r from-purple-600 via-blue-600 to-pink-600 hover:from-purple-700 hover:via-blue-700 hover:to-pink-700 shadow-lg disabled:opacity-50"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>

        <p className="text-xs text-muted-foreground text-center">
          Press Enter to send • Connected to {llmStatus.url || 'Ollama'}
        </p>
      </div>
    </Card>
  )
}
