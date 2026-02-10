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

  // New visualization endpoints
  correlation: async (id: string): Promise<{ matrix: Record<string, number | string>[]; columns: string[] }> => {
    const response = await fetch(`${API_BASE}/datasets/${id}/correlation`)
    return handleResponse(response)
  },

  distribution: async (id: string, column: string, bins = 20): Promise<{
    bins?: number[];
    counts: number[];
    categories?: string[];
    type: 'numeric' | 'categorical'
  }> => {
    const response = await fetch(`${API_BASE}/datasets/${id}/distribution/${encodeURIComponent(column)}?bins=${bins}`)
    return handleResponse(response)
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
  dataset_id: string
  name: string  // Model algorithm name (e.g., "random_forest", "xgboost")
  // Metrics are at top level, not nested
  accuracy?: number
  f1_score?: number
  precision?: number
  recall?: number
  auc?: number
  rmse?: number
  mae?: number
  r2?: number
  training_time: number
  rank: number
  model_path?: string
  model_size?: number
  created_at: string
  hyperparameters?: string
  feature_importance?: string
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

  getMetadata: async (modelId: string): Promise<{
    model_id: string;
    algorithm: string;
    problem_type: string;
    feature_names: string[];
    numeric_features: string[];
    categorical_features: string[];
    low_cardinality_features: string[];
    high_cardinality_features: string[];
    categorical_modes: Record<string, string>;
    numeric_medians: Record<string, number>;
    numeric_stats?: Record<string, { min: number; max: number; mean: number; median: number; std: number }>;
    onehot_categories: Record<string, string[]>;
  }> => {
    const response = await fetch(`${API_BASE}/results/model/${modelId}/metadata`)
    return handleResponse(response)
  },

  // New analysis endpoints
  featureImportance: async (modelId: string): Promise<{
    model_id: string;
    model_name: string;
    features: { feature: string; importance: number; rank: number }[];
  }> => {
    const response = await fetch(`${API_BASE}/results/model/${modelId}/feature-importance`)
    return handleResponse(response)
  },

  confusionMatrix: async (modelId: string): Promise<{
    model_id: string;
    model_name: string;
    matrix: number[][] | null;
    labels?: string[];
  }> => {
    const response = await fetch(`${API_BASE}/results/model/${modelId}/confusion-matrix`)
    return handleResponse(response)
  },

  compareModels: async (modelIds: string[]): Promise<{
    models: Array<{
      id: string;
      name: string;
      metrics: Record<string, number | null>;
      training_time: number;
      rank: number;
    }>;
    metrics_ranking: Record<string, { best_model_id: string; best_value: number }>;
    recommendation: string | null;
  }> => {
    const response = await fetch(`${API_BASE}/results/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(modelIds),
    })
    return handleResponse(response)
  },

  hyperparameters: async (modelId: string): Promise<{
    model_id: string;
    model_name: string;
    hyperparameters: Record<string, unknown>;
  }> => {
    const response = await fetch(`${API_BASE}/results/model/${modelId}/hyperparameters`)
    return handleResponse(response)
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

  getAll: async (): Promise<Worker[]> => {
    const response = await fetch(`${API_BASE}/workers/all`)
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

// ============ Cluster ============

export interface TailscalePeer {
  id: string
  hostname: string
  dns_name?: string
  tailscale_ip: string
  online: boolean
  os?: string
  last_seen?: string
}

export interface ClusterInfo {
  mode: string
  role: string
  redis_host: string
  redis_url: string
  tailscale: {
    enabled: boolean
    installed: boolean
    tailnet?: string
    self_ip?: string
    self_hostname?: string
    peer_count: number
    online_peers: number
  }
  local: {
    hostname: string
    local_ip: string
  }
}

export interface JoinInstructions {
  command: string
  prerequisites: string[]
  mode: string
  master_address: string
}

export const clusterApi = {
  getInfo: async (): Promise<ClusterInfo> => {
    const response = await fetch(`${API_BASE}/cluster/info`)
    return handleResponse<ClusterInfo>(response)
  },

  getTailscalePeers: async (): Promise<{
    enabled: boolean
    tailnet?: string
    self?: { hostname: string; ip: string }
    peers: TailscalePeer[]
    message?: string
  }> => {
    const response = await fetch(`${API_BASE}/cluster/tailscale/peers`)
    return handleResponse(response)
  },

  getJoinCommand: async (): Promise<JoinInstructions> => {
    const response = await fetch(`${API_BASE}/cluster/join`)
    return handleResponse<JoinInstructions>(response)
  },

  pingPeer: async (hostname: string): Promise<{ success: boolean; latency_ms?: number }> => {
    const response = await fetch(`${API_BASE}/cluster/tailscale/ping/${hostname}`, { method: 'POST' })
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
