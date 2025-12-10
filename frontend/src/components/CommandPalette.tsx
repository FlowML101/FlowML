import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Dialog, DialogContent } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { 
  LayoutDashboard, 
  Database, 
  Zap, 
  Activity, 
  Server, 
  Trophy,
  FlaskConical,
  ScrollText,
  Search,
  ArrowRight
} from 'lucide-react'

const pages = [
  { name: 'Dashboard', path: '/app', icon: LayoutDashboard, keywords: ['home', 'overview'] },
  { name: 'Data Studio', path: '/app/data', icon: Database, keywords: ['upload', 'dataset', 'csv'] },
  { name: 'Training', path: '/app/train', icon: Zap, keywords: ['config', 'automl', 'train'] },
  { name: 'Live Monitor', path: '/app/running', icon: Activity, keywords: ['progress', 'running', 'live'] },
  { name: 'Workers', path: '/app/workers', icon: Server, keywords: ['nodes', 'cluster', 'mesh'] },
  { name: 'Results', path: '/app/results', icon: Trophy, keywords: ['leaderboard', 'models', 'metrics'] },
  { name: 'Model Lab', path: '/app/inference', icon: FlaskConical, keywords: ['inference', 'predict', 'test'] },
  { name: 'Logs', path: '/app/logs', icon: ScrollText, keywords: ['history', 'audit', 'debug'] },
]

export function CommandPalette() {
  const [isOpen, setIsOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [selectedIndex, setSelectedIndex] = useState(0)
  const navigate = useNavigate()

  const filteredPages = pages.filter(page => {
    const searchTerm = query.toLowerCase()
    return (
      page.name.toLowerCase().includes(searchTerm) ||
      page.keywords.some(keyword => keyword.includes(searchTerm))
    )
  })

  const handleSelect = useCallback((path: string) => {
    navigate(path)
    setIsOpen(false)
    setQuery('')
    setSelectedIndex(0)
  }, [navigate])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+K or Cmd+K to open
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault()
        setIsOpen(true)
      }

      if (!isOpen) return

      // Escape to close
      if (e.key === 'Escape') {
        setIsOpen(false)
        setQuery('')
        setSelectedIndex(0)
      }

      // Arrow navigation
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setSelectedIndex(prev => (prev + 1) % filteredPages.length)
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault()
        setSelectedIndex(prev => (prev - 1 + filteredPages.length) % filteredPages.length)
      }

      // Enter to select
      if (e.key === 'Enter' && filteredPages[selectedIndex]) {
        e.preventDefault()
        handleSelect(filteredPages[selectedIndex].path)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, filteredPages, selectedIndex, handleSelect])

  useEffect(() => {
    setSelectedIndex(0)
  }, [query])

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogContent className="sm:max-w-[600px] p-0 gap-0 overflow-hidden">
        {/* Search Input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-zinc-700">
          <Search className="w-5 h-5 text-zinc-500" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search pages..."
            className="border-0 focus-visible:ring-0 focus-visible:ring-offset-0 text-base"
            autoFocus
          />
          <Badge variant="secondary" className="text-xs">⌘K</Badge>
        </div>

        {/* Results */}
        <div className="max-h-[400px] overflow-y-auto">
          {filteredPages.length === 0 ? (
            <div className="px-4 py-12 text-center text-zinc-500">
              <Search className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No pages found</p>
            </div>
          ) : (
            <div className="py-2">
              {filteredPages.map((page, index) => {
                const Icon = page.icon
                const isSelected = index === selectedIndex
                return (
                  <button
                    key={page.path}
                    onClick={() => handleSelect(page.path)}
                    onMouseEnter={() => setSelectedIndex(index)}
                    className={`w-full flex items-center gap-3 px-4 py-3 transition-colors ${
                      isSelected
                        ? 'bg-purple-600/20 text-purple-400'
                        : 'hover:bg-zinc-800/50 text-zinc-300'
                    }`}
                  >
                    <Icon className={`w-5 h-5 ${isSelected ? 'text-purple-400' : 'text-zinc-500'}`} />
                    <span className="flex-1 text-left font-medium">{page.name}</span>
                    {isSelected && <ArrowRight className="w-4 h-4 text-purple-400" />}
                  </button>
                )
              })}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-2 border-t border-zinc-700 bg-zinc-900/50">
          <div className="flex items-center justify-between text-xs text-zinc-500">
            <div className="flex items-center gap-4">
              <span><Badge variant="secondary" className="text-xs mr-1">↑↓</Badge> Navigate</span>
              <span><Badge variant="secondary" className="text-xs mr-1">↵</Badge> Select</span>
              <span><Badge variant="secondary" className="text-xs mr-1">Esc</Badge> Close</span>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
