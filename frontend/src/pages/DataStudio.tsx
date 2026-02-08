import { UploadZone } from '@/features/datasets/UploadZone'
import { DataPreview } from '@/features/datasets/DataPreview'
import { EnhancedCopilot } from '@/features/datasets/EnhancedCopilot'
import { CleaningPanel } from '@/features/datasets/CleaningPanel'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Sparkles, Database } from 'lucide-react'
import { Link } from 'react-router-dom'
import { useDataset } from '@/contexts/DatasetContext'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

export function DataStudio() {
  const { selectedDataset, datasets, setSelectedDataset, refreshDatasets } = useDataset()

  const handleDatasetUpdated = async (newDatasetId: string) => {
    // Refresh the dataset list and select the new dataset
    await refreshDatasets()
    const newDataset = datasets.find(d => d.id === newDatasetId)
    if (newDataset) {
      setSelectedDataset(newDataset)
    }
  }

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
          <Link to="/train">
            <Button className="bg-gradient-to-r from-purple-600 to-purple-600 hover:from-purple-700 hover:to-purple-700">
              <Sparkles className="w-4 h-4 mr-2" />
              Start Training
            </Button>
          </Link>
        </div>
      </div>

      {/* Dataset Selector */}
      {datasets.length > 0 && (
        <div className="flex items-center gap-3 p-4 bg-zinc-900/50 border border-border rounded-lg">
          <Database className="w-5 h-5 text-blue-500" />
          <span className="text-sm font-medium">Active Dataset:</span>
          <Select
            value={selectedDataset?.id || ''}
            onValueChange={(id) => {
              const dataset = datasets.find(d => d.id === id)
              if (dataset) setSelectedDataset(dataset)
            }}
          >
            <SelectTrigger className="w-[300px]">
              <SelectValue placeholder="Select a dataset" />
            </SelectTrigger>
            <SelectContent>
              {datasets.map((dataset) => (
                <SelectItem key={dataset.id} value={dataset.id}>
                  {dataset.name} ({dataset.num_rows} rows, {dataset.num_columns} cols)
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      {/* Upload Zone */}
      <UploadZone />

      {/* Main Content: Tabs for different functionality */}
      {selectedDataset && (
        <Tabs defaultValue="preview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3 lg:w-[600px]">
            <TabsTrigger value="preview">Data Preview</TabsTrigger>
            <TabsTrigger value="cleaning">Data Cleaning</TabsTrigger>
            <TabsTrigger value="copilot">AI Copilot</TabsTrigger>
          </TabsList>

          <TabsContent value="preview">
            <DataPreview />
          </TabsContent>

          <TabsContent value="cleaning">
            <CleaningPanel 
              datasetId={selectedDataset.id}
              datasetName={selectedDataset.name}
              onDatasetUpdated={handleDatasetUpdated}
            />
          </TabsContent>

          <TabsContent value="copilot">
            <EnhancedCopilot />
          </TabsContent>
        </Tabs>
      )}

      {/* Show only preview if no dataset selected */}
      {!selectedDataset && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <DataPreview />
          </div>
          <div>
            <EnhancedCopilot />
          </div>
        </div>
      )}
    </div>
  )
}
