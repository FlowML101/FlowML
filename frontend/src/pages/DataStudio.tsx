import { UploadZone } from '@/features/datasets/UploadZone'
import { DataPreview } from '@/features/datasets/DataPreview'
import { EnhancedCopilot } from '@/features/datasets/EnhancedCopilot'
import { Button } from '@/components/ui/button'
import { Sparkles } from 'lucide-react'
import { Link } from 'react-router-dom'

export function DataStudio() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <Sparkles className="w-8 h-8 text-blue-500" />
            Data Studio
          </h1>
          <p className="text-muted-foreground">Upload datasets, explore patterns, and leverage AI-powered data insights</p>
        </div>
        <div className="flex items-center gap-3">
          <Link to="/app/train">
            <Button className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700">
              <Sparkles className="w-4 h-4 mr-2" />
              Start Training
            </Button>
          </Link>
        </div>
      </div>

      {/* Upload Zone */}
      <UploadZone />

      {/* Main Content: Preview + Copilot */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <DataPreview />
        </div>
        <div>
          <EnhancedCopilot />
        </div>
      </div>
    </div>
  )
}
