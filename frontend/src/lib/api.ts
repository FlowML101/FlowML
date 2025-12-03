const base = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${base}${path}`, { headers: { 'Content-Type': 'application/json' }, ...init })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

// Stubs to wire later
export const API = {
  listDatasets: () => fetchJSON('/api/datasets'),
  uploadDataset: (form: FormData) => fetch(`${base}/api/datasets/upload`, { method: 'POST', body: form }),
  listJobs: () => fetchJSON('/api/jobs'),
  createJob: (payload: unknown) => fetchJSON('/api/jobs', { method: 'POST', body: JSON.stringify(payload) }),
  listWorkers: () => fetchJSON('/api/workers'),
  listRuns: () => fetchJSON('/api/runs')
}
