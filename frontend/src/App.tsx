import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'sonner'
import { ThemeProvider } from '@/contexts/ThemeContext'
import { CommandPalette } from '@/components/CommandPalette'
import { MarketingLayout } from '@/layouts/MarketingLayout'
import { DashboardLayout } from '@/layouts/DashboardLayout'
import { LandingPage } from '@/pages/LandingPage'
import { DashboardHome } from '@/pages/DashboardHome'
import { DataStudio } from '@/pages/DataStudio'
import { TrainingConfig } from '@/pages/TrainingConfig'
import { LiveMonitor } from '@/pages/LiveMonitor'
import { WorkersManager } from '@/pages/WorkersManager'
import { Results } from '@/pages/Results'
import { InferencePage } from '@/pages/InferencePage'
import { LogsPage } from '@/pages/LogsPage'
import { useSystemEvents } from '@/hooks/useSystemEvents'

function App() {
  useSystemEvents()

  return (
    <ThemeProvider>
      <BrowserRouter>
        <Toaster position="top-right" richColors />
        <CommandPalette />
        <Routes>
        {/* Marketing Zone */}
        <Route element={<MarketingLayout />}>
          <Route path="/" element={<LandingPage />} />
        </Route>

        {/* Dashboard/App Zone */}
        <Route path="/app" element={<DashboardLayout />}>
          <Route index element={<DashboardHome />} />
          <Route path="data" element={<DataStudio />} />
          <Route path="train" element={<TrainingConfig />} />
          <Route path="running" element={<LiveMonitor />} />
          <Route path="workers" element={<WorkersManager />} />
          <Route path="results" element={<Results />} />
          <Route path="inference" element={<InferencePage />} />
          <Route path="logs" element={<LogsPage />} />
        </Route>

        {/* Catch all - redirect to home */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      </BrowserRouter>
    </ThemeProvider>
  )
}

export default App
