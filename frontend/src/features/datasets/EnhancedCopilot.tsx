import { useState, useRef, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { 
  Bot, Send, Sparkles, Code2, Copy, Check, Zap, 
  TrendingUp, AlertCircle, Lightbulb 
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

interface Message {
  role: 'user' | 'assistant'
  content: string
  code?: string
  codeLanguage?: string
  suggestions?: string[]
  timestamp: Date
}

const mockSuggestions = [
  "Handle missing values in Age column",
  "Create feature: FamilySize = SibSp + Parch",
  "Encode categorical variables",
  "Normalize numeric features"
]

const mockCodeSamples = {
  fillna: `# Fill missing Age values with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Verify changes
print(f"Missing values: {df['Age'].isnull().sum()}")`,
  
  feature: `# Create new feature: FamilySize
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Create IsAlone binary feature  
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)`,

  encode: `# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Label encode Pclass
df['Pclass'] = df['Pclass'].map({1: 0, 2: 1, 3: 2})`,
}

export function EnhancedCopilot() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: "👋 I'm your AI Data Assistant. I can help you clean data, engineer features, and explain insights. What would you like to know?",
      suggestions: mockSuggestions,
      timestamp: new Date()
    },
  ])
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [copiedCode, setCopiedCode] = useState(false)
  const [tokenUsage, setTokenUsage] = useState({ used: 1420, total: 100000 })
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(scrollToBottom, [messages])

  const handleSend = (text?: string) => {
    const messageText = text || input
    if (!messageText.trim()) return

    const userMessage: Message = {
      role: 'user',
      content: messageText,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsTyping(true)

    // Simulate AI response
    setTimeout(() => {
      const responses = [
        {
          content: "I analyzed the Age column. There are 177 missing values (19.9% of data). The distribution is right-skewed with median=28 and mean=29.7 years.",
          code: mockCodeSamples.fillna,
          codeLanguage: 'python'
        },
        {
          content: "Creating a FamilySize feature is a great idea! This combines SibSp and Parch to capture family dynamics, which historically correlated with survival rates.",
          code: mockCodeSamples.feature,
          codeLanguage: 'python'
        },
        {
          content: "I'll help you encode the categorical variables. Sex and Embarked can be one-hot encoded, while Pclass works well as ordinal.",
          code: mockCodeSamples.encode,
          codeLanguage: 'python'
        }
      ]

      const response = responses[Math.floor(Math.random() * responses.length)]
      
      setMessages(prev => [...prev, {
        ...response,
        role: 'assistant',
        timestamp: new Date(),
        suggestions: ['Apply this code', 'Explain more', 'Show visualization', 'Alternative approach']
      }])
      setIsTyping(false)
      setTokenUsage(prev => ({ ...prev, used: prev.used + Math.floor(Math.random() * 200) + 100 }))
    }, 1500)
  }

  const copyCode = (code: string) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(true)
    setTimeout(() => setCopiedCode(false), 2000)
  }

  const tokenPercentage = (tokenUsage.used / tokenUsage.total) * 100

  return (
    <Card className="flex flex-col border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/10 before:via-cyan-500/10 before:to-blue-500/10 before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-blue-500/12">
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
                  GPT-4
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground font-normal">Powered by FlowML Intelligence</p>
            </div>
          </CardTitle>

          {/* Token Usage */}
          <div className="flex flex-col items-end gap-1">
            <span className="text-xs text-muted-foreground">
              {tokenUsage.used.toLocaleString()} / {tokenUsage.total.toLocaleString()} tokens
            </span>
            <div className="w-32 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-purple-600 to-blue-600 transition-all duration-300"
                style={{ width: `${tokenPercentage}%` }}
              />
            </div>
          </div>
        </div>
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
                  <p className="text-sm leading-relaxed">{message.content}</p>
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
            onClick={() => handleSend("Explain the missing values")}
            size="sm"
            variant="outline"
            className="text-xs whitespace-nowrap border-zinc-700 hover:bg-zinc-800"
          >
            <AlertCircle className="w-3 h-3 mr-1.5" />
            Missing Data
          </Button>
          <Button
            onClick={() => handleSend("Suggest feature engineering")}
            size="sm"
            variant="outline"
            className="text-xs whitespace-nowrap border-zinc-700 hover:bg-zinc-800"
          >
            <Zap className="w-3 h-3 mr-1.5" />
            Feature Ideas
          </Button>
          <Button
            onClick={() => handleSend("Show correlation insights")}
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
            placeholder="Ask anything about your data, or request code..."
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
          Press Enter to send • Shift+Enter for new line
        </p>
      </div>
    </Card>
  )
}
