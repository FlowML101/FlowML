# FlowML Infrastructure Setup Guide

This guide explains how to set up the FlowML infrastructure services.

## Prerequisites

### 1. Install Docker Desktop

Download and install Docker Desktop for Windows:
- https://www.docker.com/products/docker-desktop/

After installation:
1. Start Docker Desktop
2. Wait for it to fully start (the whale icon in the system tray should stop animating)
3. Open a new terminal and verify: `docker --version`

### 2. Install Node.js (for frontend)

If not already installed:
- https://nodejs.org/ (LTS version recommended)

### 3. Install Python 3.11+

If not already installed:
- https://www.python.org/downloads/

---

## Quick Start

Once Docker is installed:

```bash
# Navigate to FlowML directory
cd C:\Users\apurv\Documents\FlowML

# Start all infrastructure services
docker-compose -f deploy/docker-compose.yml up -d

# Wait for services to be ready (~30 seconds)
# Check status
docker-compose -f deploy/docker-compose.yml ps
```

---

## Services

The Docker Compose stack includes:

| Service | Port | Description |
|---------|------|-------------|
| **PostgreSQL** | 5432 | Metadata database |
| **Redis** | 6379 | Queue broker + cache |
| **MinIO** | 9000 (API), 9001 (Console) | S3-compatible storage |
| **MLflow** (optional) | 5000 | Experiment tracking |

### Access Points

- **MinIO Console**: http://localhost:9001
  - Username: `flowml-admin`
  - Password: `flowml-secret`

- **PostgreSQL**: `localhost:5432`
  - Database: `flowml`
  - User: `flowml`
  - Password: `flowml`

- **Redis**: `localhost:6379`
  - No password required (dev mode)

---

## Configuration Modes

### Mode 1: Development (SQLite + Local Files)
Default mode - no Docker needed for basic development.

```env
# backend/.env
DATABASE_URL=sqlite+aiosqlite:///./flowml.db
STORAGE_MODE=local
```

### Mode 2: Production-like (Postgres + MinIO)
Requires Docker containers running.

```env
# backend/.env
DATABASE_URL=postgresql+asyncpg://flowml:flowml@localhost:5432/flowml
STORAGE_MODE=s3
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=flowml-admin
S3_SECRET_KEY=flowml-secret
```

---

## Full Stack Startup

### Option A: Use the batch script (Windows)

```bash
# Start everything
start-dev.bat all

# Or start individually
start-dev.bat infra    # Docker services only
start-dev.bat backend  # FastAPI server
start-dev.bat frontend # Vite dev server
```

### Option B: Manual startup

```bash
# Terminal 1: Infrastructure
docker-compose -f deploy/docker-compose.yml up -d

# Terminal 2: Backend
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 3: Frontend
cd frontend
npm install
npm run dev
```

---

## Useful Commands

```bash
# View logs
docker-compose -f deploy/docker-compose.yml logs -f

# View specific service logs
docker-compose -f deploy/docker-compose.yml logs -f postgres

# Stop all services
docker-compose -f deploy/docker-compose.yml down

# Stop and remove volumes (reset data)
docker-compose -f deploy/docker-compose.yml down -v

# Rebuild containers
docker-compose -f deploy/docker-compose.yml up -d --build
```

---

## Storage Buckets

MinIO is automatically configured with these buckets:

- `flowml-datasets` - Uploaded dataset files
- `flowml-artifacts` - Trained model files
- `flowml-runs` - Experiment run data
- `flowml-logs` - System logs

---

## Troubleshooting

### Docker not starting
1. Make sure Docker Desktop is running
2. On Windows, check if WSL2 is properly configured
3. Try restarting Docker Desktop

### PostgreSQL connection refused
1. Wait for the container to fully start
2. Check container health: `docker-compose ps`
3. View logs: `docker-compose logs postgres`

### MinIO bucket creation failed
The `minio-setup` container creates buckets on first run.
If it failed, manually create buckets via the console at http://localhost:9001

### Backend can't connect to services
1. Ensure containers are running: `docker-compose ps`
2. Verify `.env` has correct URLs
3. Check that ports aren't blocked by firewall

---

## Next Steps

Once infrastructure is running:

1. **Start the backend**: Uses the database and storage services
2. **Start the frontend**: Connects to backend API
3. **Upload a dataset**: Test the full pipeline
4. **Start a training job**: Verify worker queue works

The system will fall back to SQLite + local storage if Docker services are unavailable.
