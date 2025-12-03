import { Routes, Route, Navigate } from 'react-router-dom'
import { AppShell } from './components/layout/AppShell'
import Dashboard from './pages/Dashboard'
import Datasets from './pages/Datasets'
import Jobs from './pages/Jobs'
import Workers from './pages/Workers'
import Runs from './pages/Runs';
import Monitor from './pages/Monitor';
import ModelLab from './pages/ModelLab';
import WorkerLogs from './pages/WorkerLogs';
import Copilot from './pages/Copilot';
import Settings from './pages/Settings';

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/datasets" element={<Datasets />} />
        <Route path="/jobs" element={<Jobs />} />
        <Route path="/workers" element={<Workers />} />
          <Route path="/runs" element={<Runs />} />
          <Route path="/monitor" element={<Monitor />} />
          <Route path="/model-lab" element={<ModelLab />} />
          <Route path="/worker-logs" element={<WorkerLogs />} />
          <Route path="/copilot" element={<Copilot />} />
          <Route path="/settings" element={<Settings />} />
      </Routes>
    </AppShell>
  )
}
