import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Database, Loader2 } from 'lucide-react'
import { useDataset } from '@/contexts/DatasetContext'

export function DataPreview() {
  const { selectedDataset, previewData, isLoadingPreview } = useDataset()

  if (!selectedDataset) {
    return (
      <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/10 before:via-transparent before:to-cyan-500/10 before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-blue-500/12">
        <CardHeader className="relative">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Database className="w-5 h-5 text-blue-500" />
                Dataset Preview
              </CardTitle>
              <CardDescription>Upload a dataset to see preview</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="relative">
          <div className="border border-dashed border-border rounded-lg p-12 text-center">
            <Database className="w-12 h-12 mx-auto mb-4 text-zinc-600" />
            <p className="text-muted-foreground">No dataset selected</p>
            <p className="text-sm text-muted-foreground mt-2">Upload a CSV file to get started</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (isLoadingPreview) {
    return (
      <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/10 before:via-transparent before:to-cyan-500/10 before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-blue-500/12">
        <CardHeader className="relative">
          <CardTitle className="flex items-center gap-2">
            <Database className="w-5 h-5 text-blue-500" />
            Dataset Preview
          </CardTitle>
          <CardDescription>Loading {selectedDataset.name}...</CardDescription>
        </CardHeader>
        <CardContent className="relative">
          <div className="flex items-center justify-center p-12">
            <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
          </div>
        </CardContent>
      </Card>
    )
  }

  const columns = previewData?.columns || []
  const rows = previewData?.preview_rows || []

  return (
    <Card className="border-border bg-gradient-to-br from-zinc-900 to-zinc-900/50 relative overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/10 before:via-transparent before:to-cyan-500/10 before:opacity-30 transition-all duration-300 hover:shadow-md hover:shadow-blue-500/12">
      <CardHeader className="relative">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Database className="w-5 h-5 text-blue-500" />
              Dataset Preview
            </CardTitle>
            <CardDescription>First {rows.length} rows of {selectedDataset.name}</CardDescription>
          </div>
          <div className="flex gap-2">
            <Badge variant="outline">{selectedDataset.num_rows.toLocaleString()} rows</Badge>
            <Badge variant="outline">{selectedDataset.num_columns} columns</Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="relative">
        <div className="border border-border dark:border-zinc-800 rounded-lg overflow-hidden">
          <div className="overflow-x-auto max-h-[400px]">
            <Table>
              <TableHeader>
                <TableRow>
                  {columns.map((col) => (
                    <TableHead key={col} className="whitespace-nowrap sticky top-0 bg-zinc-900">
                      {col}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((row, idx) => (
                  <TableRow key={idx}>
                    {columns.map((col) => {
                      const value = row[col]
                      const displayValue = value === null || value === undefined 
                        ? 'N/A' 
                        : typeof value === 'number'
                          ? Number.isInteger(value) ? value : value.toFixed(2)
                          : String(value)
                      
                      return (
                        <TableCell key={col} className="font-mono text-xs max-w-[200px] truncate">
                          {displayValue === 'N/A' ? (
                            <span className="text-muted-foreground">{displayValue}</span>
                          ) : (
                            displayValue
                          )}
                        </TableCell>
                      )
                    })}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
        
        <div className="mt-4 text-xs text-muted-foreground text-center">
          Showing {rows.length} of {selectedDataset.num_rows.toLocaleString()} rows
        </div>
      </CardContent>
    </Card>
  )
}
