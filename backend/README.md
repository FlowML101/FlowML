# FlowML Backend

**Production-grade AutoML backend with distributed task execution**

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   FastAPI   │  │  WebSocket  │  │   SQLite/   │              │
│  │   REST API  │  │   Manager   │  │  PostgreSQL │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
│         └────────────────┼────────────────┘                      │
│                          │                                       │
│              ┌───────────▼───────────┐                          │
│              │  Scheduler Interface  │                          │
│              │  (Celery / Ray)       │                          │
│              └───────────┬───────────┘                          │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │    Redis    │
                    │   (Broker)  │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼───────┐  ┌───────▼───────┐  ┌───────▼───────┐
│   Worker 1    │  │   Worker 2    │  │   Worker N    │
│  ┌─────────┐  │  │  ┌─────────┐  │  │  ┌─────────┐  │
│  │  CPU    │  │  │  │  GPU    │  │  │  │  LLM    │  │
│  │  Queue  │  │  │  │  Queue  │  │  │  │  Queue  │  │
│  └─────────┘  │  │  └─────────┘  │  │  └─────────┘  │
│               │  │               │  │               │
│  Capabilities:│  │  Capabilities:│  │  Capabilities:│
│  - 8 CPUs     │  │  - 4 CPUs     │  │  - 8 CPUs     │
│  - 16GB RAM   │  │  - 32GB RAM   │  │  - 32GB RAM   │
│  - No GPU     │  │  - RTX 3080   │  │  - RTX 4090   │
│               │  │  - 10GB VRAM  │  │  - 24GB VRAM  │
└───────────────┘  └───────────────┘  └───────────────┘
```

## Features

- **Capability-based task routing**: Jobs automatically routed to workers with required resources
- **Multiple queue types**: `cpu`, `gpu`, `gpu.vram6/8/12/24`, `llm`, `priority`
- **Worker auto-discovery**: Workers probe their hardware and register capabilities
- **Real-time updates**: WebSocket broadcasts for job progress
- **Graceful degradation**: Falls back to local execution if no workers available

## Quick Start

### 1. Install Dependencies

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Start Redis (required for Celery)

```bash
# Docker
docker run -d -p 6379:6379 redis:alpine

# Or install locally
# Windows: https://github.com/microsoftarchive/redis/releases
# Mac: brew install redis && brew services start redis
```

### 3. Start the Orchestrator

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Start a Worker

```bash
cd backend
python -m worker.cli --register http://localhost:8000
```

Or manually with Celery:

```bash
celery -A worker.celery_app worker --loglevel=INFO -Q cpu
```

## API Endpoints

### Datasets
- `POST /api/datasets/upload` - Upload CSV/Parquet
- `GET /api/datasets` - List datasets
- `GET /api/datasets/{id}/preview` - Preview rows
- `GET /api/datasets/{id}/stats` - Column statistics
- `DELETE /api/datasets/{id}` - Delete dataset

### Training
- `POST /api/training/start` - Start AutoML job
- `GET /api/training` - List jobs
- `GET /api/training/{id}` - Job details
- `GET /api/training/{id}/progress` - Real-time progress
- `POST /api/training/{id}/cancel` - Cancel job

### Results
- `GET /api/results/job/{job_id}/leaderboard` - Model leaderboard
- `GET /api/results/model/{model_id}` - Model details
- `POST /api/results/model/{model_id}/predict` - Run prediction
- `GET /api/results/model/{model_id}/download` - Download model

### Workers
- `POST /api/workers/register` - Register worker
- `POST /api/workers/heartbeat` - Worker heartbeat
- `GET /api/workers` - List workers
- `GET /api/workers/stats` - Aggregate stats
- `GET /api/workers/queues` - Queue information
- `GET /api/workers/local/info` - Orchestrator info

### Stats
- `GET /api/stats/overview` - System overview
- `GET /api/stats/jobs` - Job statistics
- `GET /api/stats/models` - Model statistics

### Health
- `GET /health` - Health check
- `WebSocket /ws` - Real-time updates

## Configuration

Environment variables (`.env`):

```env
# Server
DEBUG=true
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_URL=sqlite+aiosqlite:///./flowml.db
# DATABASE_URL=postgresql+asyncpg://user:pass@localhost/flowml

# Redis / Celery
REDIS_URL=redis://localhost:6379/0

# Scheduler: "celery" or "ray"
SCHEDULER_BACKEND=celery

# Limits
MAX_CONCURRENT_JOBS=3
MAX_TIME_BUDGET=60
MAX_UPLOAD_SIZE_MB=500

# CORS
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

## Worker CLI

```bash
# Probe capabilities only
python -m worker.cli --probe-only

# Start with auto-detected queues
python -m worker.cli

# Start with specific queues
python -m worker.cli --queues cpu,gpu

# Start with custom concurrency
python -m worker.cli --concurrency 4

# Register with orchestrator
python -m worker.cli --register http://orchestrator:8000
```

## Task Queues

| Queue | Requirements | Tasks |
|-------|--------------|-------|
| `cpu` | Any worker | CPU training, preprocessing |
| `gpu` | Any GPU | GPU-accelerated training |
| `gpu.vram6` | ≥6GB VRAM | Medium models |
| `gpu.vram8` | ≥8GB VRAM | Large models |
| `gpu.vram12` | ≥12GB VRAM | Very large models |
| `gpu.vram24` | ≥24GB VRAM | Huge models |
| `llm` | GPU + Ollama | LLM tasks |
| `priority` | Any | High-priority tasks |

## Development

### Run Tests

```bash
pytest tests/
```

### Code Style

```bash
ruff check .
ruff format .
```

### Type Checking

```bash
mypy backend/
```

## Phase Roadmap

- **Phase 0 (Current)**: Celery + Redis, SQLite, local storage, PyCaret AutoML
- **Phase 1**: PostgreSQL, MinIO S3, Optuna HPO, MLflow tracking, Ollama LLM
- **Phase 2**: Ray scheduler option, multi-tenant, advanced HPO, distributed DL
