import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Activity, Server, Cpu, CheckCircle2 } from 'lucide-react'

export function ClusterHealth() {
  return (
    <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-purple-500/10 before:via-transparent before:to-blue-500/10 before:opacity-30 transition-all duration-300 hover:border-purple-500/30">
      <CardHeader className="relative">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-purple-500" />
              Cluster Status
            </CardTitle>
            <CardDescription>Mesh network health and connectivity</CardDescription>
          </div>
          <Badge variant="success" className="flex items-center gap-1">
            <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
            Online
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 relative">
        {/* Master Node */}
        <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50 dark:bg-zinc-800/50">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-purple-600/20 flex items-center justify-center">
              <Server className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <div className="font-medium">Master Node</div>
              <div className="text-xs text-muted-foreground">localhost:8080</div>
            </div>
          </div>
          <CheckCircle2 className="w-5 h-5 text-green-500" />
        </div>

        {/* Workers */}
        <div className="space-y-2">
          <div className="text-sm font-medium text-zinc-400 flex items-center justify-between">
            <span>Connected Workers</span>
            <span className="text-purple-400">2 Active</span>
          </div>
          
          <div className="flex items-center justify-between p-3 rounded-lg bg-muted/30 dark:bg-zinc-800/30">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded bg-blue-600/20 flex items-center justify-center">
                <Cpu className="w-4 h-4 text-blue-400" />
              </div>
              <div>
                <div className="text-sm font-medium">worker-laptop-01</div>
                <div className="text-xs text-muted-foreground">192.168.1.105</div>
              </div>
            </div>
            <Badge variant="success" className="text-xs">Active</Badge>
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg bg-muted/30 dark:bg-zinc-800/30">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded bg-blue-600/20 flex items-center justify-center">
                <Cpu className="w-4 h-4 text-blue-400" />
              </div>
              <div>
                <div className="text-sm font-medium">worker-desktop-02</div>
                <div className="text-xs text-muted-foreground">192.168.1.142</div>
              </div>
            </div>
            <Badge variant="success" className="text-xs">Active</Badge>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
