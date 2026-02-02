import { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { Dataset, DatasetPreview, datasetsApi } from '@/lib/api'

interface DatasetContextType {
  // Current selected dataset
  selectedDataset: Dataset | null
  setSelectedDataset: (dataset: Dataset | null) => void
  
  // Dataset preview data
  previewData: DatasetPreview | null
  isLoadingPreview: boolean
  
  // All datasets
  datasets: Dataset[]
  isLoadingDatasets: boolean
  refreshDatasets: () => Promise<void>
}

const DatasetContext = createContext<DatasetContextType | undefined>(undefined)

export function DatasetProvider({ children }: { children: ReactNode }) {
  const [selectedDataset, setSelectedDatasetState] = useState<Dataset | null>(null)
  const [previewData, setPreviewData] = useState<DatasetPreview | null>(null)
  const [isLoadingPreview, setIsLoadingPreview] = useState(false)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [isLoadingDatasets, setIsLoadingDatasets] = useState(true)

  // Load all datasets on mount
  const refreshDatasets = async () => {
    setIsLoadingDatasets(true)
    try {
      const data = await datasetsApi.list()
      setDatasets(data)
      // If no selected dataset and we have datasets, select the most recent
      if (!selectedDataset && data.length > 0) {
        setSelectedDatasetState(data[0])
      }
    } catch (err) {
      console.error('Failed to load datasets:', err)
    } finally {
      setIsLoadingDatasets(false)
    }
  }

  useEffect(() => {
    refreshDatasets()
  }, [])

  // Load preview when selected dataset changes
  useEffect(() => {
    if (!selectedDataset) {
      setPreviewData(null)
      return
    }

    const loadPreview = async () => {
      setIsLoadingPreview(true)
      try {
        const preview = await datasetsApi.preview(selectedDataset.id, 10)
        setPreviewData(preview)
      } catch (err) {
        console.error('Failed to load preview:', err)
        setPreviewData(null)
      } finally {
        setIsLoadingPreview(false)
      }
    }

    loadPreview()
  }, [selectedDataset?.id])

  const setSelectedDataset = (dataset: Dataset | null) => {
    setSelectedDatasetState(dataset)
  }

  return (
    <DatasetContext.Provider value={{
      selectedDataset,
      setSelectedDataset,
      previewData,
      isLoadingPreview,
      datasets,
      isLoadingDatasets,
      refreshDatasets,
    }}>
      {children}
    </DatasetContext.Provider>
  )
}

export function useDataset() {
  const context = useContext(DatasetContext)
  if (!context) {
    throw new Error('useDataset must be used within DatasetProvider')
  }
  return context
}
