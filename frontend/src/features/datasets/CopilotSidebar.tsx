import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Bot, Send, Sparkles } from 'lucide-react'
import { Badge } from '@/components/ui/badge'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

const mockResponses = [
  "The dataset contains 891 passengers with 12 features. The survival rate is approximately 38%.",
  "Age has missing values (177 nulls). The median age is 28 years.",
  "Passengers in 1st class had a survival rate of 63%, compared to 24% for 3rd class.",
]

export function CopilotSidebar() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: "Hi! I'm your Data Copilot. Ask me anything about the Titanic dataset.",
    },
  ])
  const [input, setInput] = useState('')
  const [responseIndex, setResponseIndex] = useState(0)

  const handleSend = () => {
    if (!input.trim()) return

    // Add user message
    const newMessages: Message[] = [
      ...messages,
      { role: 'user', content: input },
      {
        role: 'assistant',
        content: mockResponses[responseIndex % mockResponses.length],
      },
    ]

    setMessages(newMessages)
    setInput('')
    setResponseIndex(responseIndex + 1)
  }

  return (
    <Card className="h-[600px] flex flex-col">
      <CardHeader className="border-b border-zinc-800">
        <CardTitle className="flex items-center gap-2 text-lg">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-600 to-blue-600 flex items-center justify-center">
            <Bot className="w-5 h-5 text-white" />
          </div>
          Data Copilot
          <Badge variant="secondary" className="ml-auto text-xs">
            <Sparkles className="w-3 h-3 mr-1" />
            Beta
          </Badge>
        </CardTitle>
      </CardHeader>

      <CardContent className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-lg p-3 ${
                message.role === 'user'
                  ? 'bg-purple-600 text-white'
                  : 'bg-zinc-800/50 text-zinc-100'
              }`}
            >
              <p className="text-sm">{message.content}</p>
            </div>
          </div>
        ))}
      </CardContent>

      <div className="p-4 border-t border-zinc-800">
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSend()
            }}
            placeholder="Ask about your data..."
            className="flex-1"
          />
          <Button onClick={handleSend} size="icon" className="bg-purple-600 hover:bg-purple-700">
            <Send className="w-4 h-4" />
          </Button>
        </div>
        <p className="text-xs text-zinc-500 mt-2">
          Try: "What's the survival rate?" or "Show missing values"
        </p>
      </div>
    </Card>
  )
}
