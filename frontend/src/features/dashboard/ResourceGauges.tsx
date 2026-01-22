import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Cpu, HardDrive, Activity, Loader2 } from 'lucide-react'
import { statsApi, ResourceStats } from '@/lib/api'

interface ResourceGaugeProps {
  label: string
  used: number
  total: number
  unit: string
  icon: React.ElementType
  color: string
}

function ResourceGauge({ label, used, total, unit, icon: Icon, color }: ResourceGaugeProps) {
  const percentage = total > 0 ? (used / total) * 100 : 0
  const circumference = 2 * Math.PI * 45
  const strokeDashoffset = circumference - (percentage / 100) * circumference

  const gradientClass = label.includes('VRAM') ? 'before:from-purple-500/10' : 
                        label.includes('RAM') ? 'before:from-blue-500/10' : 
                        'before:from-cyan-500/10'

  const shadowClass = label.includes('VRAM') ? 'hover:shadow-purple-500/15' : 
                      label.includes('RAM') ? 'hover:shadow-blue-500/15' : 
                      'hover:shadow-cyan-500/15'

  return (
    <Card className={`border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br ${gradientClass} before:to-transparent before:opacity-30 transition-all duration-300 hover:shadow-md ${shadowClass}`}>
      <CardHeader className="pb-3 relative">
        <CardTitle className="text-base flex items-center gap-2">
          <Icon className={`w-4 h-4 ${color}`} />
          {label}
        </CardTitle>
      </CardHeader>
      <CardContent className="relative">
        <div className="flex items-center justify-center">
          <div className="relative w-32 h-32">
            {/* Background circle */}
            <svg className="w-full h-full transform -rotate-90">
              <circle
                cx="64"
                cy="64"
                r="45"
                stroke="currentColor"
                strokeWidth="8"
                fill="none"
                className="text-muted dark:text-zinc-800"
              />
              {/* Progress circle */}
              <circle
                cx="64"
                cy="64"
                r="45"
                stroke="currentColor"
                strokeWidth="8"
                fill="none"
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
                className={color}
                strokeLinecap="round"
              />
            </svg>
            {/* Center text */}
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <div className="text-2xl font-bold">{percentage.toFixed(0)}%</div>
              <div className="text-xs text-muted-foreground">{used.toFixed(1)}/{total.toFixed(1)} {unit}</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export function ResourceGauges() {
  const [resources, setResources] = useState<ResourceStats | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchResources = async () => {
      try {
        const data = await statsApi.getResources()
        setResources(data)
      } catch (err) {
        console.error('Failed to fetch resources:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchResources()
    // Refresh every 5 seconds
    const interval = setInterval(fetchResources, 5000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[1, 2, 3].map((i) => (
          <Card key={i} className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50">
            <CardContent className="flex items-center justify-center h-48">
              <Loader2 className="w-8 h-8 animate-spin text-zinc-500" />
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {resources?.vram_total_gb ? (
        <ResourceGauge
          label="VRAM Usage"
          used={resources.vram_used_gb || 0}
          total={resources.vram_total_gb}
          unit="GB"
          icon={Cpu}
          color="text-purple-500"
        />
      ) : (
        <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-purple-500/10 before:to-transparent before:opacity-30">
          <CardHeader className="pb-3 relative">
            <CardTitle className="text-base flex items-center gap-2">
              <Cpu className="w-4 h-4 text-purple-500" />
              VRAM Usage
            </CardTitle>
          </CardHeader>
          <CardContent className="relative flex items-center justify-center h-32">
            <div className="text-center text-zinc-500">
              <div className="text-sm">No GPU detected</div>
            </div>
          </CardContent>
        </Card>
      )}
      <ResourceGauge
        label="RAM Usage"
        used={resources?.ram_used_gb || 0}
        total={resources?.ram_total_gb || 16}
        unit="GB"
        icon={HardDrive}
        color="text-blue-500"
      />
      <ResourceGauge
        label="CPU Load"
        used={resources?.cpu_percent || 0}
        total={100}
        unit="%"
        icon={Activity}
        color="text-cyan-500"
      />
    </div>
  )
}
