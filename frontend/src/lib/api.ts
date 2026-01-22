/**
 * FlowML API Client
 * Centralized API calls for the frontend
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'

interface ApiError {
  detail: string
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error: ApiError = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }
  return response.json()
}

// ============ Datasets ============

export interface Dataset {
  id: string
  name: string
  filename: string
  file_size: number
  num_rows: number
  num_columns: number
  columns: string
  dtypes: string
  description?: string
  created_at: string
}

export interface DatasetPreview {
  id: string
  name: string
  columns: string[]
  dtypes: Record<string, string>
  preview_rows: Record<string, unknown>[]
  num_rows: number
  num_columns: number
}

export interface ColumnStats {
  column: string
  dtype: string
  non_null_count: number
  null_count: number
  unique_count: number
  mean?: number
  std?: number
  min?: number
  max?: number
  top_values?: { value: string; count: number }[]
}

export const datasetsApi = {
  upload: async (file: File, name?: string, description?: string): Promise<Dataset> => {
    const formData = new FormData()
    formData.append('file', file)
    if (name) formData.append('name', name)
    if (description) formData.append('description', description)

    const response = await fetch(`${API_BASE}/datasets/upload`, {
      method: 'POST',
      body: formData,
    })
    return handleResponse<Dataset>(response)
  },

  list: async (): Promise<Dataset[]> => {
    const response = await fetch(`${API_BASE}/datasets`)
    return handleResponse<Dataset[]>(response)
  },

  get: async (id: string): Promise<Dataset> => {
    const response = await fetch(`${API_BASE}/datasets/${id}`)
    return handleResponse<Dataset>(response)
  },

  preview: async (id: string, rows = 100): Promise<DatasetPreview> => {
    const response = await fetch(`${API_BASE}/datasets/${id}/preview?rows=${rows}`)
    return handleResponse<DatasetPreview>(response)
  },

  stats: async (id: string): Promise<ColumnStats[]> => {
    const response = await fetch(`${API_BASE}/datasets/${id}/stats`)
    return handleResponse<ColumnStats[]>(response)
  },

  delete: async (id: string): Promise<void> => {
    const response = await fetch(`${API_BASE}/datasets/${id}`, { method: 'DELETE' })
    return handleResponse<void>(response)
  },
}

// ============ Training ============

export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface Job {
  id: string
  name: string
  dataset_id: string
  target_column: string
  time_budget: number
  model_types: string
  problem_type?: string
  status: JobStatus
  progress: number
  current_model?: string
  models_completed: number
  total_models: number
  error_message?: string
  created_at: string
  started_at?: string
  completed_at?: string
  // Additional fields for UI compatibility
  algorithm?: string
  config?: Record<string, unknown>
  worker_id?: string
}

export interface JobCreate {
  dataset_id: string
  target_column: string
  time_budget?: number
  model_types?: string[] | string
  problem_type?: string
  name?: string
}

export const trainingApi = {
  start: async (config: JobCreate): Promise<Job> => {
    const response = await fetch(`${API_BASE}/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    })
    return handleResponse<Job>(response)
  },

  list: async (status?: JobStatus): Promise<Job[]> => {
    const url = status ? `${API_BASE}/training?status=${status}` : `${API_BASE}/training`
    const response = await fetch(url)
    return handleResponse<Job[]>(response)
  },

  get: async (id: string): Promise<Job> => {
    const response = await fetch(`${API_BASE}/training/${id}`)
    return handleResponse<Job>(response)
  },

  cancel: async (id: string): Promise<void> => {
    const response = await fetch(`${API_BASE}/training/${id}/cancel`, { method: 'POST' })
    return handleResponse<void>(response)
  },
}

// ============ Results ============

export interface TrainedModel {
  id: string
  job_id: string
  algorithm: string
  metrics: {
    accuracy?: number
    f1_score?: number
    precision?: number
    recall?: number
    auc?: number
    rmse?: number
    mae?: number
    r2?: number
    mse?: number
    training_time?: number
  }
  file_path: string
  created_at: string
}

export interface PredictionRequest {
  model_id: string
  features: Record<string, unknown>
}

export interface PredictionResponse {
  prediction: string | number
  probability?: number
  confidence?: number
  model_name: string
  latency_ms: number
}

export const resultsApi = {
  getJobResults: async (jobId: string): Promise<TrainedModel[]> => {
    const response = await fetch(`${API_BASE}/results/job/${jobId}`)
    return handleResponse<TrainedModel[]>(response)
  },

  getLeaderboard: async (jobId: string): Promise<TrainedModel[]> => {
    const response = await fetch(`${API_BASE}/results/job/${jobId}/leaderboard`)
    return handleResponse<TrainedModel[]>(response)
  },

  getModel: async (id: string): Promise<TrainedModel> => {
    const response = await fetch(`${API_BASE}/results/model/${id}`)
    return handleResponse<TrainedModel>(response)
  },

  downloadModel: (id: string): string => {
    return `${API_BASE}/results/model/${id}/download`
  },

  predict: async (modelId: string, features: Record<string, unknown>): Promise<PredictionResponse> => {
    const response = await fetch(`${API_BASE}/results/models/${modelId}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features }),
    })
    return handleResponse<PredictionResponse>(response)
  },

  listAllModels: async (): Promise<TrainedModel[]> => {
    const response = await fetch(`${API_BASE}/results/models`)
    return handleResponse<TrainedModel[]>(response)
  },
}

// ============ Workers ============

export interface Worker {
  id: string
  hostname: string
  role: string
  status: string
  ip: string
  tailscale_ip?: string
  cpu_count: number
  cpu_percent: number
  ram_total_gb: number
  ram_used_gb: number
  ram_percent: number
  gpu_name?: string
  vram_total_gb?: number
  vram_used_gb?: number
  vram_percent?: number
  uptime: string
  // Additional fields for UI compatibility
  capabilities?: {
    gpu?: boolean
    cpu?: boolean
  }
  current_task?: string
}

export const workersApi = {
  list: async (): Promise<Worker[]> => {
    const response = await fetch(`${API_BASE}/workers`)
    return handleResponse<Worker[]>(response)
  },

  getMaster: async (): Promise<Worker> => {
    const response = await fetch(`${API_BASE}/workers/master`)
    return handleResponse<Worker>(response)
  },

  getAvailable: async (): Promise<{ total: number; available: number; busy: number }> => {
    const response = await fetch(`${API_BASE}/workers/available`)
    return handleResponse(response)
  },
}

// ============ Stats ============

export interface DashboardStats {
  total_models: number
  active_jobs: number
  total_datasets: number
  avg_accuracy?: number
  models_this_week: number
  jobs_running: number
  jobs_queued: number
}

export interface ResourceStats {
  cpu_percent: number
  ram_percent: number
  ram_used_gb: number
  ram_total_gb: number
  vram_percent?: number
  vram_used_gb?: number
  vram_total_gb?: number
}

export const statsApi = {
  getDashboard: async (): Promise<DashboardStats> => {
    const response = await fetch(`${API_BASE}/stats`)
    return handleResponse<DashboardStats>(response)
  },

  getResources: async (): Promise<ResourceStats> => {
    const response = await fetch(`${API_BASE}/stats/resources`)
    return handleResponse<ResourceStats>(response)
  },

  getClusterHealth: async (): Promise<unknown> => {
    const response = await fetch(`${API_BASE}/stats/cluster`)
    return handleResponse(response)
  },
}

// ============ Health ============

export const healthApi = {
  check: async (): Promise<{ status: string; app: string; version: string }> => {
    const response = await fetch(`${API_BASE.replace('/api', '')}/health`)
    return handleResponse(response)
  },
}
