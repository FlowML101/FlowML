import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Upload, FileText, CheckCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'

export function UploadZone() {
  const [isDragging, setIsDragging] = useState(false)
  const [uploadedFile, setUploadedFile] = useState<string | null>('titanic.csv')

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Upload className="w-5 h-5 text-purple-500" />
          Dataset Upload
        </CardTitle>
        <CardDescription>Drop CSV files or click to browse</CardDescription>
      </CardHeader>
      <CardContent>
        {!uploadedFile ? (
          <div
            className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
              isDragging
                ? 'border-purple-500 bg-purple-500/10'
                : 'border-zinc-700 hover:border-zinc-600'
            }`}
            onDragOver={(e) => {
              e.preventDefault()
              setIsDragging(true)
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => {
              e.preventDefault()
              setIsDragging(false)
              // Mock file upload
              setUploadedFile('titanic.csv')
            }}
          >
            <Upload className="w-12 h-12 mx-auto mb-4 text-zinc-600" />
            <p className="text-lg font-medium mb-2">Drop your CSV file here</p>
            <p className="text-sm text-zinc-500 mb-4">or click to browse</p>
            <Button variant="outline">Select File</Button>
          </div>
        ) : (
          <div className="flex items-center justify-between p-4 rounded-lg bg-zinc-800/50 border border-zinc-700">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-green-600/20 flex items-center justify-center">
                <FileText className="w-5 h-5 text-green-400" />
              </div>
              <div>
                <div className="font-medium">{uploadedFile}</div>
                <div className="text-xs text-zinc-500">891 rows × 12 columns</div>
              </div>
            </div>
            <CheckCircle className="w-5 h-5 text-green-500" />
          </div>
        )}
      </CardContent>
    </Card>
  )
}
