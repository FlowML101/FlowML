import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Cpu, HardDrive, Activity } from 'lucide-react'

interface ResourceGaugeProps {
  label: string
  used: number
  total: number
  unit: string
  icon: React.ElementType
  color: string
}

function ResourceGauge({ label, used, total, unit, icon: Icon, color }: ResourceGaugeProps) {
  const percentage = (used / total) * 100
  const circumference = 2 * Math.PI * 45
  const strokeDashoffset = circumference - (percentage / 100) * circumference

  const gradientClass = label.includes('VRAM') ? 'before:from-purple-500/10' : 
                        label.includes('RAM') ? 'before:from-blue-500/10' : 
                        'before:from-cyan-500/10'

  const shadowClass = label.includes('VRAM') ? 'hover:shadow-blue-500/15' : 
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
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <ResourceGauge
        label="VRAM Usage"
        used={3.2}
        total={4.0}
        unit="GB"
        icon={Cpu}
        color="text-purple-500"
      />
      <ResourceGauge
        label="RAM Usage"
        used={12.8}
        total={16.0}
        unit="GB"
        icon={HardDrive}
        color="text-blue-500"
      />
      <ResourceGauge
        label="CPU Load"
        used={45}
        total={100}
        unit="%"
        icon={Activity}
        color="text-cyan-500"
      />
    </div>
  )
}
