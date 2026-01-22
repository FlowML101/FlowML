-- FlowML Database Initialization
-- This runs automatically when Postgres container starts for the first time

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ===================
-- Datasets Table
-- ===================
CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size BIGINT NOT NULL DEFAULT 0,
    num_rows INTEGER NOT NULL DEFAULT 0,
    num_columns INTEGER NOT NULL DEFAULT 0,
    columns TEXT,  -- JSON array of column names
    dtypes TEXT,   -- JSON object of column -> dtype
    description TEXT,
    checksum VARCHAR(64),  -- SHA256 hash for dedup
    storage_uri VARCHAR(500),  -- S3 URI if using MinIO
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_datasets_name ON datasets(name);
CREATE INDEX idx_datasets_checksum ON datasets(checksum);
CREATE INDEX idx_datasets_created_at ON datasets(created_at DESC);

-- ===================
-- Jobs Table
-- ===================
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255),
    dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    target_column VARCHAR(255) NOT NULL,
    problem_type VARCHAR(50),  -- classification, regression
    model_types TEXT,  -- JSON array of model types to try
    time_budget INTEGER DEFAULT 300,  -- seconds
    
    -- Status tracking
    status VARCHAR(50) DEFAULT 'pending',  -- pending, queued, running, completed, failed, cancelled
    progress INTEGER DEFAULT 0,  -- 0-100
    current_model VARCHAR(255),
    models_completed INTEGER DEFAULT 0,
    total_models INTEGER DEFAULT 0,
    
    -- Execution details
    worker_id UUID,
    celery_task_id VARCHAR(255),
    error_message TEXT,
    
    -- Config
    config TEXT,  -- Full JSON config
    seed INTEGER DEFAULT 42,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_dataset_id ON jobs(dataset_id);
CREATE INDEX idx_jobs_worker_id ON jobs(worker_id);
CREATE INDEX idx_jobs_created_at ON jobs(created_at DESC);

-- ===================
-- Trained Models Table
-- ===================
CREATE TABLE IF NOT EXISTS trained_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    
    -- Model info
    algorithm VARCHAR(255) NOT NULL,
    framework VARCHAR(100),  -- sklearn, xgboost, lightgbm, pytorch
    
    -- Metrics (JSON for flexibility)
    metrics TEXT,  -- JSON: {accuracy, f1, precision, recall, auc, rmse, mae, r2, etc.}
    
    -- Storage
    file_path VARCHAR(500),
    storage_uri VARCHAR(500),  -- S3 URI
    file_size BIGINT,
    
    -- Reproducibility
    params TEXT,  -- JSON: hyperparameters
    feature_importance TEXT,  -- JSON: feature -> importance
    training_time_seconds FLOAT,
    
    -- Metadata
    is_best BOOLEAN DEFAULT FALSE,
    rank INTEGER,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_trained_models_job_id ON trained_models(job_id);
CREATE INDEX idx_trained_models_algorithm ON trained_models(algorithm);
CREATE INDEX idx_trained_models_is_best ON trained_models(is_best);

-- ===================
-- Workers Table
-- ===================
CREATE TABLE IF NOT EXISTS workers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hostname VARCHAR(255) NOT NULL,
    
    -- Network
    ip VARCHAR(45),
    tailscale_ip VARCHAR(45),
    
    -- Hardware capabilities
    cpu_count INTEGER,
    cpu_model VARCHAR(255),
    ram_total_gb FLOAT,
    gpu_name VARCHAR(255),
    gpu_count INTEGER DEFAULT 0,
    vram_total_gb FLOAT,
    cuda_version VARCHAR(50),
    
    -- Software capabilities
    python_version VARCHAR(50),
    torch_version VARCHAR(50),
    sklearn_version VARCHAR(50),
    xgboost_version VARCHAR(50),
    lightgbm_version VARCHAR(50),
    
    -- Runtime
    role VARCHAR(50) DEFAULT 'worker',  -- master, worker
    status VARCHAR(50) DEFAULT 'offline',  -- online, busy, offline, draining
    current_task VARCHAR(255),
    max_concurrency INTEGER DEFAULT 4,
    current_slots INTEGER DEFAULT 0,
    tags TEXT,  -- JSON array: ["gpu", "cpu", "llm"]
    queues TEXT,  -- JSON array of subscribed queues
    
    -- Health
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    uptime_seconds BIGINT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_workers_status ON workers(status);
CREATE INDEX idx_workers_hostname ON workers(hostname);
CREATE INDEX idx_workers_last_heartbeat ON workers(last_heartbeat);

-- ===================
-- Runs Table (Experiment Tracking)
-- ===================
CREATE TABLE IF NOT EXISTS runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    
    -- Run info
    run_name VARCHAR(255),
    status VARCHAR(50) DEFAULT 'running',
    
    -- Artifacts location
    artifacts_uri VARCHAR(500),
    
    -- Metrics and params (JSON)
    metrics TEXT,
    params TEXT,
    tags TEXT,
    
    -- Environment capture
    environment TEXT,  -- JSON: pip freeze output
    git_commit VARCHAR(40),
    
    -- Timestamps
    start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_time TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_runs_job_id ON runs(job_id);
CREATE INDEX idx_runs_status ON runs(status);

-- ===================
-- Events Table (Audit Log)
-- ===================
CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,  -- job.created, job.started, worker.registered, etc.
    
    -- References
    job_id UUID REFERENCES jobs(id) ON DELETE SET NULL,
    worker_id UUID REFERENCES workers(id) ON DELETE SET NULL,
    dataset_id UUID REFERENCES datasets(id) ON DELETE SET NULL,
    
    -- Event data
    payload TEXT,  -- JSON
    severity VARCHAR(20) DEFAULT 'info',  -- debug, info, warning, error
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_events_type ON events(event_type);
CREATE INDEX idx_events_job_id ON events(job_id);
CREATE INDEX idx_events_created_at ON events(created_at DESC);

-- ===================
-- Update timestamp trigger
-- ===================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_datasets_updated_at BEFORE UPDATE ON datasets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_jobs_updated_at BEFORE UPDATE ON jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workers_updated_at BEFORE UPDATE ON workers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ===================
-- Grant permissions
-- ===================
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO flowml;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO flowml;

-- Done!
SELECT 'FlowML database initialized successfully!' as message;
