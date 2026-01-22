import { useState, useRef } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Upload, FileText, CheckCircle, Loader2, X, AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { datasetsApi, Dataset } from '@/lib/api'
import { toast } from 'sonner'

interface UploadZoneProps {
  onUploadComplete?: (dataset: Dataset) => void
}

export function UploadZone({ onUploadComplete }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [uploadedDataset, setUploadedDataset] = useState<Dataset | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFile = async (file: File) => {
    if (!file.name.endsWith('.csv') && !file.name.endsWith('.parquet')) {
      setError('Only CSV and Parquet files are supported')
      return
    }

    setIsUploading(true)
    setError(null)

    try {
      const dataset = await datasetsApi.upload(file)
      setUploadedDataset(dataset)
      toast.success(`Dataset "${dataset.name}" uploaded successfully!`)
      onUploadComplete?.(dataset)
    } catch (err: any) {
      console.error('Upload failed:', err)
      setError(err.message || 'Upload failed')
      toast.error('Upload failed: ' + (err.message || 'Unknown error'))
    } finally {
      setIsUploading(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    
    const file = e.dataTransfer.files[0]
    if (file) {
      handleFile(file)
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFile(file)
    }
  }

  const clearUpload = () => {
    setUploadedDataset(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/10 before:via-transparent before:to-cyan-500/10 before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-blue-500/12">
      <CardHeader className="relative">
        <CardTitle className="flex items-center gap-2">
          <Upload className="w-5 h-5 text-purple-500" />
          Dataset Upload
        </CardTitle>
        <CardDescription>Drop CSV or Parquet files to upload</CardDescription>
      </CardHeader>
      <CardContent className="relative">
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.parquet"
          onChange={handleFileSelect}
          className="hidden"
        />
        
        {isUploading ? (
          <div className="border-2 border-dashed rounded-lg p-12 text-center border-purple-500 bg-purple-500/10">
            <Loader2 className="w-12 h-12 mx-auto mb-4 text-purple-500 animate-spin" />
            <p className="text-lg font-medium mb-2">Uploading...</p>
            <p className="text-sm text-muted-foreground">Analyzing dataset structure</p>
          </div>
        ) : error ? (
          <div className="border-2 border-dashed rounded-lg p-12 text-center border-red-500 bg-red-500/10">
            <AlertCircle className="w-12 h-12 mx-auto mb-4 text-red-500" />
            <p className="text-lg font-medium mb-2 text-red-400">{error}</p>
            <Button variant="outline" onClick={clearUpload} className="mt-4">
              Try Again
            </Button>
          </div>
        ) : !uploadedDataset ? (
          <div
            className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors cursor-pointer ${
              isDragging
                ? 'border-purple-500 bg-purple-500/10'
                : 'border-border dark:border-zinc-700 hover:border-zinc-600'
            }`}
            onDragOver={(e) => {
              e.preventDefault()
              setIsDragging(true)
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload className="w-12 h-12 mx-auto mb-4 text-zinc-600" />
            <p className="text-lg font-medium mb-2">Drop your CSV file here</p>
            <p className="text-sm text-muted-foreground mb-4">or click to browse</p>
            <Button variant="outline" onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click() }}>
              Select File
            </Button>
          </div>
        ) : (
          <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50 dark:bg-zinc-800/50 border border-border dark:border-zinc-700">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-green-600/20 flex items-center justify-center">
                <FileText className="w-5 h-5 text-green-400" />
              </div>
              <div>
                <div className="font-medium">{uploadedDataset.name}</div>
                <div className="text-xs text-muted-foreground">
                  {uploadedDataset.num_rows.toLocaleString()} rows × {uploadedDataset.num_columns} columns
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-500" />
              <Button variant="ghost" size="sm" onClick={clearUpload}>
                <X className="w-4 h-4" />
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
