# FlowML Studio

<div align="center">

**Production-Ready Distributed AutoML Platform with GPU Acceleration**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![React 18](https://img.shields.io/badge/react-18.2-blue.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/typescript-5.2-blue.svg)](https://www.typescriptlang.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)

*Privacy-first, self-hosted AutoML platform for distributed model training across multiple machines*

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Documentation](#-documentation)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Core Components](#-core-components)
- [Package Reference](#-package-reference)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Development](#-development)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## 🌟 Overview

**FlowML Studio** is a self-hosted, production-grade distributed AutoML platform that enables you to train machine learning models across multiple machines (local, LAN, or via Tailscale VPN) with automatic GPU acceleration. Built for privacy, performance, and scalability.

### Key Highlights

- **🔒 Privacy-First**: All data stays on your infrastructure - no cloud dependencies
- **⚡ GPU Accelerated**: Automatic NVIDIA GPU detection and utilization for XGBoost, LightGBM, CatBoost
- **🌐 Distributed Training**: Scale across multiple machines via Celery + Redis task queue
- **🤖 AutoML Pipeline**: Automated model selection, hyperparameter optimization (Optuna), and evaluation
- **📊 Real-Time Monitoring**: WebSocket-based live training metrics and logs
- **🔧 Capability-Based Routing**: Jobs automatically routed to workers with appropriate resources
- **🎯 Production Ready**: Full async backend, type-safe TypeScript frontend, containerized deployment

---

## ✨ Features

### Machine Learning

- **Automated Model Selection**: Tests 15+ algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, SVM, etc.)
- **Hyperparameter Optimization**: Optuna-powered Bayesian optimization (TPE sampler)
- **Problem Type Detection**: Automatic classification vs regression detection
- **Cross-Validation**: K-fold CV for robust model evaluation
- **Feature Engineering**: Automatic preprocessing, scaling, imputation, encoding
- **Model Export**: ONNX format for cross-platform deployment
- **Inference Engine**: REST API for batch and real-time predictions

### Distributed Computing

- **Multi-Machine Training**: Celery worker pool across LAN or Tailscale VPN
- **Dynamic Queue Routing**: `cpu`, `gpu`, `gpu.vram6/8/12/24`, `llm`, `priority` queues
- **Worker Auto-Discovery**: Hardware capability probing (CPU, RAM, GPU, VRAM)
- **Graceful Degradation**: Falls back to local execution if workers unavailable
- **Load Balancing**: Round-robin task distribution with capability matching

### User Experience

- **Modern UI**: React 18 + TypeScript + Tailwind CSS dark mode interface
- **Real-Time Updates**: WebSocket dashboard for job progress and metrics
- **Dataset Management**: Support for CSV, Parquet, Excel, JSON, YAML, SQLite
- **Interactive Training**: Live loss/accuracy charts, scrolling logs, cancellation
- **Model Comparison**: Sortable leaderboard with metrics visualization
- **Inference Playground**: Test models with custom input data

### Infrastructure

- **Flexible Deployment**: SQLite (dev) or PostgreSQL (production)
- **Storage Options**: Local filesystem or MinIO S3-compatible storage
- **Containerized**: Docker Compose for easy deployment
- **Monitoring**: Structured logging with Loguru, health checks
- **Secure**: CORS configuration, environment-based secrets

---

## 🏗️ Architecture

FlowML uses a **master-worker** distributed architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR (Master)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   FastAPI    │  │  WebSocket   │  │   Database   │               │
│  │   REST API   │  │   Manager    │  │ SQLite/PG    │               │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
│         │                 │                 │                        │
│         └─────────────────┼─────────────────┘                        │
│                           │                                          │
│               ┌───────────▼────────────┐                            │
│               │  Celery Scheduler      │                            │
│               │  (Task Queue Router)   │                            │
│               └───────────┬────────────┘                            │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
                     ┌──────▼──────┐
                     │    Redis    │
                     │   (Broker)  │
                     └──────┬──────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
  ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐
  │  Worker 1   │    │  Worker 2   │    │  Worker N   │
  │  ┌───────┐  │    │  ┌───────┐  │    │  ┌───────┐  │
  │  │  CPU  │  │    │  │  GPU  │  │    │  │  LLM  │  │
  │  │ Queue │  │    │  │ Queue │  │    │  │ Queue │  │
  │  └───────┘  │    │  └───────┘  │    │  └───────┘  │
  │  8 CPUs     │    │  RTX 3080   │    │  RTX 4090   │
  │  16GB RAM   │    │  10GB VRAM  │    │  24GB VRAM  │
  └─────────────┘    └─────────────┘    └─────────────┘
```

### Data Flow

1. **User uploads dataset** → FastAPI stores in database and filesystem/S3
2. **Training job submitted** → Master creates Celery task with config
3. **Redis distributes task** → Routes to worker queue based on requirements
4. **Worker executes training** → Runs Optuna HPO trials, trains models
5. **Progress updates** → Worker sends WebSocket messages via Redis pub/sub
6. **Results saved** → Models stored, metrics written to database
7. **Frontend displays** → Real-time charts, logs, leaderboard

### Networking Options

- **Local**: Single machine (master = worker)
- **LAN**: Multiple machines on same network
- **Tailscale**: Secure mesh VPN for distributed teams

---

## 🛠️ Technology Stack

### Backend (Python 3.11+)

| **Category** | **Package** | **Version** | **Purpose** |
|-------------|------------|------------|-------------|
| **Web Framework** | FastAPI | 0.109+ | Async REST API with OpenAPI docs |
| | Uvicorn | 0.27+ | ASGI server with hot reload |
| | python-multipart | 0.0.6+ | File upload handling |
| **Database** | SQLModel | 0.0.14+ | SQLAlchemy + Pydantic ORM |
| | aiosqlite | 0.19+ | Async SQLite driver (dev) |
| | asyncpg | 0.29+ | Async PostgreSQL driver (prod) |
| **Task Queue** | Celery | 5.3+ | Distributed task execution |
| | Redis | 5.0+ | Message broker & result backend |
| | Kombu | 5.3+ | Messaging library |
| **Data Processing** | Polars | 0.20+ | Fast DataFrame library (Rust backend) |
| | Pandas | 2.1+ | Data manipulation |
| | NumPy | 1.26+ | Numerical computing |
| | PyArrow | 14.0+ | Columnar data format |
| | OpenPyXL | 3.1+ | Excel file support |
| | FastExcel | 0.9+ | Fast Excel reading |
| **Machine Learning** | scikit-learn | 1.4+ | Classical ML algorithms |
| | XGBoost | 2.0+ | Gradient boosting (GPU support) |
| | LightGBM | 4.2+ | Fast gradient boosting (GPU support) |
| | CatBoost | 1.2+ | Categorical feature handling |
| **Hyperparameter Opt** | Optuna | 3.5+ | Bayesian optimization (TPE) |
| **Model Export** | ONNX | 1.15+ | Model interoperability format |
| | skl2onnx | 1.16+ | scikit-learn to ONNX conversion |
| **Hardware Detection** | psutil | 5.9+ | CPU, RAM, disk monitoring |
| | nvidia-ml-py | 12.0+ | NVIDIA GPU detection (NVML) |
| **Storage** | boto3 | 1.34+ | S3/MinIO client |
| **Real-Time** | websockets | 12.0+ | WebSocket connections |
| **Utilities** | Loguru | 0.7+ | Structured logging |
| | pydantic-settings | 2.1+ | Config validation |
| | python-dotenv | 1.0+ | Environment variable loading |

### Frontend (React 18 + TypeScript)

| **Category** | **Package** | **Version** | **Purpose** |
|-------------|------------|------------|-------------|
| **Framework** | React | 18.2 | UI library |
| | TypeScript | 5.2 | Type-safe JavaScript |
| | Vite | 5.0 | Fast build tool & dev server |
| **UI Components** | Radix UI | Various | Headless accessible components |
| | ShadCN/UI | Latest | Pre-built component library |
| | Tailwind CSS | 3.3 | Utility-first styling |
| | Lucide React | 0.294 | Icon library |
| **Data Visualization** | Recharts | 2.10 | Chart library for React |
| **Animation** | Framer Motion | 10.16 | Motion library for animations |
| **Routing** | React Router | 6.20 | Client-side routing |
| **State Management** | React Hooks | Built-in | useState, useEffect, useContext |
| **Utilities** | class-variance-authority | 0.7 | CSS class composition |
| | clsx | 2.0 | Conditional classnames |
| | tailwind-merge | 2.1 | Merge Tailwind classes |

### Infrastructure

| **Component** | **Technology** | **Purpose** |
|--------------|---------------|------------|
| **Message Broker** | Redis 7 Alpine | Celery task queue broker |
| **Database** | PostgreSQL 15 | Production database |
| | SQLite 3 | Development database |
| **Storage** | MinIO | S3-compatible object storage |
| **Networking** | Tailscale | Secure mesh VPN (optional) |
| **Container** | Docker + Compose | Service orchestration |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ ([Download](https://www.python.org/downloads/))
- Node.js 18+ ([Download](https://nodejs.org/))
- Docker Desktop ([Download](https://www.docker.com/products/docker-desktop/))

### Setup & Run

```batch
setup-and-run.bat
```

This script will automatically:
- Create Python virtual environment
- Install all dependencies (backend + frontend)
- Start Docker services (PostgreSQL, Redis, MinIO)
- Launch all services
- Open browser to http://localhost:5173

**First run:** 5-10 minutes  
**Subsequent runs:** Use `quick-start.bat` (30 seconds)

### Manual Setup (Alternative)

**Backend:**
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**Docker Services:**
```bash
cd deploy
docker-compose up -d postgres redis
```

### Access Points

| **Service** | **URL** | **Description** |
|------------|---------|----------------|
| Frontend | http://localhost:5173 | React web interface |
| Backend API | http://localhost:8000 | FastAPI REST endpoints |
| API Docs | http://localhost:8000/docs | Swagger UI documentation |
| ReDoc | http://localhost:8000/redoc | Alternative API docs |
| Health Check | http://localhost:8000/health | System health status |

---

## 📁 Project Structure

```
FlowML/
├── backend/                    # Python FastAPI backend
│   ├── config.py              # Configuration management
│   ├── database.py            # SQLModel database setup
│   ├── main.py               # FastAPI app entry point
│   ├── exceptions.py         # Custom exception handlers
│   ├── requirements.txt      # Python dependencies
│   │
│   ├── models/               # SQLModel database models
│   │   ├── dataset.py        # Dataset table
│   │   ├── job.py           # Training job table
│   │   ├── trained_model.py  # Model metadata table
│   │   └── worker.py        # Worker node table
│   │
│   ├── routers/              # FastAPI route handlers
│   │   ├── datasets.py       # Dataset CRUD operations
│   │   ├── training.py       # Training job management
│   │   ├── results.py        # Model results & inference
│   │   ├── workers.py        # Worker management
│   │   ├── cluster.py        # Cluster info & health
│   │   ├── stats.py          # Statistics aggregation
│   │   ├── llm.py           # LLM integration (Ollama)
│   │   └── logs.py          # Log streaming
│   │
│   ├── services/             # Business logic services
│   │   ├── optuna_automl.py  # Core AutoML engine
│   │   ├── storage.py        # File & S3 storage abstraction
│   │   ├── data_formats.py   # Multi-format data loading
│   │   ├── llm_service.py    # Ollama LLM client
│   │   ├── cluster.py        # Cluster management
│   │   └── websocket_manager.py  # WebSocket connections
│   │
│   └── worker/               # Celery worker components
│       ├── celery_app.py     # Celery configuration
│       ├── cli.py           # Worker CLI tool
│       ├── capabilities.py   # Hardware detection
│       └── tasks/           # Celery task definitions
│           ├── training.py   # Training tasks
│           ├── preprocessing.py  # Data preprocessing
│           ├── inference.py  # Model inference
│           ├── llm.py       # LLM tasks
│           └── system.py    # System tasks
│
├── frontend/                  # React TypeScript frontend
│   ├── src/
│   │   ├── main.tsx          # React entry point
│   │   ├── App.tsx          # Router configuration
│   │   │
│   │   ├── layouts/          # Layout components
│   │   │   └── DashboardLayout.tsx  # Sidebar + topbar
│   │   │
│   │   ├── pages/            # Page components
│   │   │   ├── DashboardHome.tsx    # Dashboard overview
│   │   │   ├── DataStudio.tsx       # Dataset management
│   │   │   ├── TrainingConfig.tsx   # Training setup
│   │   │   ├── LiveMonitor.tsx      # Real-time training
│   │   │   ├── Results.tsx          # Model leaderboard
│   │   │   ├── InferencePage.tsx    # Model testing
│   │   │   ├── WorkersManager.tsx   # Worker nodes
│   │   │   └── LogsPage.tsx         # System logs
│   │   │
│   │   ├── features/         # Feature modules
│   │   │   ├── dashboard/    # Dashboard widgets
│   │   │   └── datasets/     # Dataset components
│   │   │
│   │   ├── components/ui/    # Reusable UI components
│   │   ├── contexts/         # React contexts
│   │   ├── hooks/           # Custom React hooks
│   │   └── lib/             # Utilities
│   │
│   ├── package.json          # Node dependencies
│   ├── vite.config.ts        # Vite configuration
│   ├── tailwind.config.js    # Tailwind CSS config
│   └── tsconfig.json         # TypeScript config
│
├── deploy/                    # Deployment configuration
│   ├── docker-compose.yml     # Multi-service orchestration
│   ├── SETUP.md              # Deployment guide
│   └── init-scripts/         # Database initialization
│       └── postgres/
│           └── 01-init.sql   # PostgreSQL schema
│
├── quick-start.bat           # Windows quick-start
├── start-dev.bat            # Windows dev launcher
├── README.md                # This file
├── BACKEND_INTEGRATION_READY.md  # API integration docs
└── CLUSTER_SETUP.md         # Distributed setup guide
```

---

## 🔧 Core Components

### 1. AutoML Engine (`services/optuna_automl.py`)

**Purpose**: Core machine learning pipeline with automated model selection and hyperparameter optimization.

**Key Functions**:

- `train()`: Main training orchestrator
  - Loads and preprocesses dataset
  - Detects problem type (classification/regression)
  - Runs Optuna HPO trials for each algorithm
  - Performs K-fold cross-validation
  - Extracts feature importance
  - Saves best models

- `_create_objective()`: Creates Optuna objective function
  - Samples hyperparameters using TPE sampler
  - Trains model with sampled params
  - Evaluates on validation set
  - Returns metric to optimize

- `_detect_problem_type()`: Auto-detects classification vs regression
  - Checks target column dtype
  - Counts unique values
  - Returns ProblemType enum

- `_preprocess_data()`: Feature engineering pipeline
  - Handles missing values (SimpleImputer)
  - Scales numerical features (StandardScaler)
  - Encodes categorical features (LabelEncoder)
  - Splits train/test sets

**Supported Algorithms**:

**Classification**:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier (GPU)
- LightGBM Classifier (GPU)
- CatBoost Classifier
- Extra Trees Classifier
- AdaBoost Classifier
- K-Nearest Neighbors
- Support Vector Machine
- Decision Tree
- Naive Bayes

**Regression**:
- Ridge Regression
- Lasso Regression
- Elastic Net
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor (GPU)
- LightGBM Regressor (GPU)
- CatBoost Regressor
- Extra Trees Regressor
- AdaBoost Regressor
- K-Nearest Neighbors
- Support Vector Regressor
- Decision Tree

**GPU Acceleration**:
```python
# XGBoost with CUDA
device = "cuda" if use_gpu else "cpu"
tree_method = "hist"  # GPU-compatible
params = {"device": device, "tree_method": tree_method}

# LightGBM with GPU
device = "gpu" if use_gpu else "cpu"
params = {"device": device}
```

### 2. Task Queue System (`worker/celery_app.py`)

**Purpose**: Distributed task execution across multiple worker machines.

**Queue Types**:
- `cpu`: General CPU tasks (all workers)
- `gpu`: GPU-accelerated tasks (GPU workers)
- `gpu.vram6/8/12/24`: VRAM-specific queues
- `llm`: Large language model tasks (Ollama)
- `priority`: High-priority tasks

**Configuration**:
```python
task_acks_late=True  # Acknowledge after completion
task_reject_on_worker_lost=True  # Requeue on failure
worker_prefetch_multiplier=1  # One task at a time
task_time_limit=3600 * 4  # 4-hour hard limit
```

**Routing Logic**:
```python
def get_queues(self) -> List[str]:
    queues = ["cpu"]  # All workers can do CPU
    
    if self.has_gpu and self.total_vram_gb >= 4:
        queues.append("gpu")
        
        if self.total_vram_gb >= 6:
            queues.append("gpu.vram6")
        # ... more VRAM tiers
    
    return queues
```

### 3. Capability Detection (`worker/capabilities.py`)

**Purpose**: Automatically probe hardware and software capabilities of worker machines.

**Detected Information**:

**Hardware**:
- CPU count (physical + logical cores)
- CPU frequency (MHz)
- Total & available RAM
- GPU presence & count (via nvidia-ml-py)
- GPU names & VRAM (total + free)
- CUDA version & driver version

**Software**:
- Python version
- Operating system & version
- PyTorch version & CUDA support
- ML library versions (scikit-learn, XGBoost, etc.)

**Network**:
- Local IP address
- Tailscale status & IP (if installed)
- Hostname

**Key Functions**:

- `probe_capabilities()`: Main entry point, returns `WorkerCapabilities` dataclass
- `_get_gpu_info()`: NVML-based GPU detection
- `_get_torch_info()`: PyTorch CUDA availability check
- `_get_tailscale_info()`: Tailscale VPN status

### 4. Real-Time Updates (`services/websocket_manager.py`)

**Purpose**: Broadcast training progress to all connected frontend clients.

**WebSocket Events**:
- `training_started`: New job started
- `training_progress`: Progress update (0-100%)
- `training_completed`: Job finished successfully
- `training_failed`: Job encountered error
- `training_cancelled`: User cancelled job

**Usage**:
```python
await manager.broadcast({
    "type": "training_progress",
    "job_id": job_id,
    "progress": 65,
    "stage": "optimizing",
    "message": "Training XGBoost (trial 8/10)"
})
```

### 5. Data Format Support (`services/data_formats.py`)

**Purpose**: Universal data loading across 15+ file formats.

**Supported Formats**:

| Format | Extensions | Library | Notes |
|--------|-----------|---------|-------|
| CSV | .csv, .tsv, .txt | Polars | Fast, parallel parsing |
| Excel | .xlsx, .xls | FastExcel | Calamine backend |
| Parquet | .parquet, .pq | Polars | Columnar format |
| Feather | .feather, .arrow | Polars | Apache Arrow |
| JSON | .json, .jsonl | Polars | Streaming for large files |
| YAML | .yaml, .yml | pyyaml | Config files |
| TOML | .toml | tomli | Config files |
| SQLite | .db, .sqlite | Polars | Direct SQL query |

**Key Functions**:
```python
load_data(path: Path) -> pl.DataFrame
detect_target_column(df: pl.DataFrame) -> str
infer_column_types(df: pl.DataFrame) -> Dict[str, str]
```

### 6. Storage Abstraction (`services/storage.py`)

**Purpose**: Unified API for local filesystem and S3/MinIO storage.

**Storage Modes**:
- `local`: Files stored in `./uploads/` and `./trained_models/`
- `s3`: Files stored in MinIO with bucket isolation

**Key Functions**:
```python
save_dataset(file: UploadFile, dataset_id: str) -> str
get_dataset_path(dataset_id: str) -> Path
save_model(job_id: str, model_name: str, data: bytes) -> str
get_model_path(job_id: str, model_name: str) -> Path
```

**S3 Configuration**:
```python
S3_ENDPOINT = "http://localhost:9000"
S3_ACCESS_KEY = "flowml-admin"
S3_SECRET_KEY = "flowml-secret"
S3_BUCKET_PREFIX = "flowml"
```

---

## 📦 Package Reference

### Core Python Packages Explained

**FastAPI** (`fastapi>=0.109.0`):
- Modern async web framework for building REST APIs
- Automatic OpenAPI/Swagger documentation generation
- Built-in request validation via Pydantic
- WebSocket support for real-time updates

**Uvicorn** (`uvicorn[standard]>=0.27.0`):
- Lightning-fast ASGI server
- Hot reload for development (`--reload` flag)
- HTTP/1.1 and HTTP/2 support
- WebSocket protocol support

**SQLModel** (`sqlmodel>=0.0.14`):
- Combines SQLAlchemy ORM with Pydantic validation
- Type-safe database models
- Async database operations with `AsyncSession`

**Celery** (`celery[redis]>=5.3.0`):
- Distributed task queue for worker orchestration
- Supports multiple brokers (Redis, RabbitMQ)
- Task routing, retries, time limits
- Beat scheduler for periodic tasks

**Optuna** (`optuna>=3.5.0`):
- Hyperparameter optimization framework
- TPE (Tree-structured Parzen Estimator) sampler
- Pruning strategies for early stopping
- Multi-objective optimization support

**Polars** (`polars>=0.20.0`):
- Blazingly fast DataFrame library (Rust-based)
- Lazy evaluation for query optimization
- Better memory efficiency than Pandas
- Native Arrow format support

**scikit-learn** (`scikit-learn>=1.4.0`):
- Classical machine learning algorithms
- Preprocessing utilities (scaling, imputation)
- Model evaluation metrics
- Pipeline API for workflow composition

**XGBoost** (`xgboost>=2.0.0`):
- Gradient boosting library with GPU support
- `device="cuda"` enables NVIDIA GPU training
- `tree_method="hist"` for faster training
- Handles missing values natively

**LightGBM** (`lightgbm>=4.2.0`):
- Fast gradient boosting framework
- `device="gpu"` for GPU training
- Leaf-wise tree growth (vs level-wise)
- Low memory usage

**CatBoost** (`catboost>=1.2.0`):
- Gradient boosting with native categorical feature support
- No need for label encoding
- Handles missing values automatically
- Built-in overfitting detection

### Frontend Package Details

**React** (`react@18.2`):
- Declarative UI library with hooks API
- Virtual DOM for efficient updates
- Component-based architecture

**TypeScript** (`typescript@5.2`):
- Type-safe JavaScript superset
- Compile-time error detection
- Better IDE autocomplete & refactoring

**Vite** (`vite@5.0`):
- Next-generation build tool
- Lightning-fast HMR (Hot Module Replacement)
- Optimized production builds with Rollup

**Tailwind CSS** (`tailwindcss@3.3`):
- Utility-first CSS framework
- JIT (Just-In-Time) compiler
- Dark mode support
- Responsive design utilities

**Radix UI**:
- Headless UI component primitives
- WAI-ARIA compliant accessibility
- Unstyled for full design control
- Components: Dialog, Select, Slider, Tabs, etc.

**Recharts** (`recharts@2.10`):
- React charting library built on D3
- Responsive charts
- Line, bar, pie, area, scatter plots
- Real-time data updates

**Framer Motion** (`framer-motion@10.16`):
- Production-ready animation library
- Declarative animations with `animate` prop
- Page transitions with `AnimatePresence`
- Gesture recognition (drag, tap, hover)

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# ==================
# Application
# ==================
APP_NAME=FlowML Studio
APP_VERSION=1.0.0
DEBUG=true
ENVIRONMENT=development  # development | staging | production

# ==================
# Server
# ==================
HOST=0.0.0.0
PORT=8000

# ==================
# Database
# ==================
# Development (SQLite)
DATABASE_URL=sqlite+aiosqlite:///./flowml.db

# Production (PostgreSQL)
# DATABASE_URL=postgresql+asyncpg://flowml:flowml@localhost:5432/flowml

DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10

# ==================
# Storage
# ==================
STORAGE_MODE=local  # local | s3

# Local mode settings
UPLOAD_DIR=./uploads
MODELS_DIR=./trained_models
LOGS_DIR=./logs

# S3/MinIO settings
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=flowml-admin
S3_SECRET_KEY=flowml-secret
S3_REGION=us-east-1
S3_BUCKET_PREFIX=flowml

# ==================
# AutoML
# ==================
DEFAULT_TIME_BUDGET=5  # minutes
MAX_TIME_BUDGET=60     # minutes
MAX_CONCURRENT_JOBS=3
MAX_UPLOAD_SIZE_MB=500

# ==================
# Redis / Celery
# ==================
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
SCHEDULER_BACKEND=celery  # celery | ray

WORKER_TTL_SECONDS=90

# ==================
# LLM (Ollama)
# ==================
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
LLM_MAX_TOKENS=2048
LLM_TIMEOUT=120.0

# ==================
# Cluster
# ==================
CLUSTER_MODE=local  # local | lan | tailscale
FLOWML_ROLE=auto    # master | worker | auto

TAILSCALE_ENABLED=false
MASTER_ADDRESS=localhost

# ==================
# Security
# ==================
SECRET_KEY=change-me-in-production-flowml-secret-key

# ==================
# CORS
# ==================
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

### GPU Configuration (NVIDIA)

**Forcing NVIDIA GPU Usage** (Windows dual-GPU systems):

The backend automatically sets these environment variables at startup:

```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first NVIDIA GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # PCI ordering
```

**Windows Graphics Settings**:
1. Settings → Display → Graphics
2. Add `backend\.venv\Scripts\python.exe`
3. Set to **High performance** (NVIDIA GPU)

**Verify GPU Usage**:
```bash
# Check backend logs for:
# "🚀 GPU Training ENABLED - RTX 3080 (10GB VRAM)"
# "⚡ XGBoost GPU acceleration enabled"
# "⚡ LightGBM GPU acceleration enabled"
```

---

## 📚 API Documentation

### REST Endpoints

#### Datasets

**Upload Dataset**
```http
POST /api/datasets/upload
Content-Type: multipart/form-data

file: <binary>
name: "Titanic Dataset"
```

**List Datasets**
```http
GET /api/datasets
Response: Dataset[]
```

**Get Dataset Preview**
```http
GET /api/datasets/{id}/preview?rows=10
Response: {columns: string[], rows: any[][]}
```

**Get Column Statistics**
```http
GET /api/datasets/{id}/stats
Response: {column: string, type: string, missing: number, unique: number}[]
```

#### Training

**Start Training Job**
```http
POST /api/training/start
Content-Type: application/json

{
  "dataset_id": "uuid",
  "target_column": "Survived",
  "problem_type": "classification",  // or "regression" or "auto"
  "time_budget": 5,  // minutes
  "model_types": ["xgboost", "random_forest"],  // optional
  "n_trials_per_model": 10
}

Response: {job_id: string, status: "pending"}
```

**List Jobs**
```http
GET /api/training
Response: Job[]
```

**Get Job Details**
```http
GET /api/training/{id}
Response: Job
```

**Cancel Job**
```http
POST /api/training/{id}/cancel
Response: {status: "cancelled"}
```

#### Results

**Get Model Leaderboard**
```http
GET /api/results/job/{job_id}/leaderboard
Response: {
  models: [{
    algorithm: string,
    metrics: {accuracy: number, f1: number},
    training_time: number
  }]
}
```

**Run Prediction**
```http
POST /api/results/model/{model_id}/predict
Content-Type: application/json

{
  "data": {
    "Age": 25,
    "Pclass": 3,
    "Sex": "male"
  }
}

Response: {
  prediction: number,
  probability: number[],
  execution_time: number
}
```

**Download Model**
```http
GET /api/results/model/{model_id}/download
Response: <binary> (joblib pickle or ONNX)
```

#### Workers

**Register Worker**
```http
POST /api/workers/register
Content-Type: application/json

{
  "worker_id": "worker-001",
  "hostname": "laptop",
  "capabilities": {
    "cpu_count": 8,
    "total_ram_gb": 16,
    "has_gpu": true,
    "gpu_names": ["RTX 3080"],
    "total_vram_gb": 10
  }
}
```

**Worker Heartbeat**
```http
POST /api/workers/heartbeat
Content-Type: application/json

{
  "worker_id": "worker-001",
  "status": "online",
  "current_jobs": 1
}
```

**List Workers**
```http
GET /api/workers
Response: Worker[]
```

### WebSocket API

**Connect to Live Updates**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch(data.type) {
    case 'training_progress':
      console.log(`Job ${data.job_id}: ${data.progress}%`);
      break;
    case 'training_completed':
      console.log(`Job ${data.job_id} finished`);
      break;
  }
};
```

**Event Types**:
- `training_started`
- `training_progress`
- `training_completed`
- `training_failed`
- `training_cancelled`
- `worker_connected`
- `worker_disconnected`

---

## 👨‍💻 Development

### Running Tests

```bash
cd backend
pytest tests/ -v
```

### Code Quality

**Linting**:
```bash
ruff check .
ruff format .
```

**Type Checking**:
```bash
mypy backend/ --strict
```

### Frontend Development

**Run dev server**:
```bash
cd frontend
npm run dev
```

**Build for production**:
```bash
npm run build
npm run preview
```

**Type checking**:
```bash
npx tsc --noEmit
```

### Adding New ML Algorithms

1. **Define model in `optuna_automl.py`**:
```python
self.CLASSIFICATION_MODELS["new_model"] = {
    "class": NewClassifier,
    "params": lambda trial: {
        "param1": trial.suggest_int("param1", 1, 100),
        "param2": trial.suggest_float("param2", 0.1, 1.0)
    }
}
```

2. **Import algorithm**:
```python
from sklearn.some_module import NewClassifier
```

3. **Test with HPO**:
```python
# Optuna will automatically optimize hyperparameters
```

---

## 🚀 Deployment

### Docker Compose (Recommended)

```bash
cd deploy
docker-compose up -d
```

Services started:
- PostgreSQL (port 5432)
- Redis (port 6379)
- MinIO (port 9000, console 9001)
- Backend (port 8000)

### Manual Deployment

**Backend**:
```bash
cd backend
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**Frontend**:
```bash
cd frontend
npm run build
# Serve dist/ folder with Nginx or any static server
```

### Distributed Setup (Tailscale)

See [CLUSTER_SETUP.md](CLUSTER_SETUP.md) for full guide.

**Quick setup**:

1. **Install Tailscale on all machines**
```bash
# Windows
winget install tailscale.tailscale

# Mac
brew install tailscale

# Linux
curl -fsSL https://tailscale.com/install.sh | sh
```

2. **Connect machines**
```bash
tailscale up
```

3. **Start master node**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

4. **Start workers**
```bash
cd backend
celery -A worker.celery_app worker \
  --broker=redis://master-hostname.ts.net:6379/0 \
  -Q cpu,gpu --hostname=%h
```

---

## 🐛 Troubleshooting

### GPU Not Detected

**Check NVIDIA drivers**:
```bash
nvidia-smi
```

**Verify CUDA in Python**:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

**Check XGBoost GPU support**:
```python
import xgboost as xgb
print(xgb.build_info())  # Should show USE_CUDA: True
```

### Redis Connection Failed

**Check Redis is running**:
```bash
docker ps | grep redis
```

**Test connection**:
```bash
redis-cli ping
# Should return: PONG
```

### Database Migration Issues

**Reset database**:
```bash
rm flowml.db
python -c "from database import init_db; import asyncio; asyncio.run(init_db())"
```

### Frontend Build Errors

**Clear node_modules**:
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Worker Not Connecting

**Check worker logs**:
```bash
celery -A worker.celery_app worker --loglevel=DEBUG
```

**Verify Redis URL**:
```bash
# In .env
REDIS_URL=redis://localhost:6379/0
```

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit changes** (`git commit -m 'Add amazing feature'`)
4. **Push to branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Code Standards

- **Python**: PEP 8, type hints, docstrings
- **TypeScript**: ESLint rules, strict mode
- **Commits**: Conventional Commits format
- **Tests**: Maintain >80% coverage

---

## 📄 License

This project is proprietary software. All rights reserved.

For licensing inquiries, contact: [your-email@example.com](mailto:your-email@example.com)

---

## 👥 Authors

- **Your Name** - *Initial work* - [@yourusername](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- **FastAPI** community for the amazing framework
- **Optuna** team for hyperparameter optimization
- **XGBoost**, **LightGBM**, **CatBoost** developers
- **Radix UI** for accessible components
- **Tailwind CSS** for utility-first styling

---

## 📞 Support

- **Documentation**: [Full Docs](https://docs.flowml.com) *(coming soon)*
- **Issues**: [GitHub Issues](https://github.com/yourusername/flowml-studio/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/flowml-studio/discussions)
- **Email**: support@flowml.com

---

<div align="center">

**Built with ❤️ for the ML community**

*Privacy-first • Self-hosted • Production-ready*

</div>
