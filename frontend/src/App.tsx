import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'sonner'
import { ThemeProvider } from '@/contexts/ThemeContext'
import { DatasetProvider } from '@/contexts/DatasetContext'
import { CommandPalette } from '@/components/CommandPalette'
import { DashboardLayout } from '@/layouts/DashboardLayout'
import { DashboardHome } from '@/pages/DashboardHome'
import { DataStudio } from '@/pages/DataStudio'
import { TrainingConfig } from '@/pages/TrainingConfig'
import { LiveMonitor } from '@/pages/LiveMonitor'
import { WorkersManager } from '@/pages/WorkersManager'
import { Results } from '@/pages/Results'
import { InferencePage } from '@/pages/InferencePage'
import { ModelComparison } from '@/pages/ModelComparison'
import { AdvancedDataViz } from '@/pages/AdvancedDataViz'
import { DeployModel } from '@/pages/DeployModel'
import { LogsPage } from '@/pages/LogsPage'
import { useSystemEvents } from '@/hooks/useSystemEvents'

function App() {
  useSystemEvents()

  return (
    <ThemeProvider>
      <DatasetProvider>
      <BrowserRouter>
        <Toaster position="top-right" richColors />
        <CommandPalette />
        <Routes>
        {/* App Zone - No landing page, straight to dashboard */}
        <Route path="/" element={<DashboardLayout />}>
          <Route index element={<DashboardHome />} />
          <Route path="data" element={<DataStudio />} />
          <Route path="train" element={<TrainingConfig />} />
          <Route path="running" element={<LiveMonitor />} />
          <Route path="workers" element={<WorkersManager />} />
          <Route path="results" element={<Results />} />
          <Route path="inference" element={<InferencePage />} />
          <Route path="compare" element={<ModelComparison />} />
          <Route path="deploy" element={<DeployModel />} />
          <Route path="visualizations" element={<AdvancedDataViz />} />
          <Route path="logs" element={<LogsPage />} />
        </Route>

        {/* Legacy /app routes redirect to root */}
        <Route path="/app/*" element={<Navigate to="/" replace />} />
        
        {/* Catch all - redirect to home */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      </BrowserRouter>
      </DatasetProvider>
    </ThemeProvider>
  )
}

export default App
