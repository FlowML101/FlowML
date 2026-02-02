# FlowML Distributed Cluster Setup

## Overview

FlowML supports distributed training across multiple machines using **Tailscale** for secure mesh networking. This allows you to:

- Add workers from different locations (home, office, friend's PC)
- No port forwarding or complex network setup required
- WireGuard encryption for all traffic
- Automatic peer discovery via MagicDNS

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TAILSCALE MESH (100.x.x.x)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   MASTER    │     │  WORKER 1   │     │  WORKER 2   │       │
│  │  (Your PC)  │◄───►│  (Laptop)   │◄───►│  (Friend's) │       │
│  │             │     │             │     │             │       │
│  │ • FastAPI   │     │ • Celery    │     │ • Celery    │       │
│  │ • Redis     │     │   Worker    │     │   Worker    │       │
│  │ • Postgres  │     │ • GPU/CPU   │     │ • GPU/CPU   │       │
│  │ • Scheduler │     │             │     │             │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│   100.64.0.1          100.64.0.2          100.64.0.3           │
│   flowml-master       laptop1             desktop2              │
└─────────────────────────────────────────────────────────────────┘
```

## Setup

### Step 1: Install Tailscale (All Machines)

**Windows:**
```powershell
winget install tailscale.tailscale
```

**macOS:**
```bash
brew install tailscale
```

**Linux:**
```bash
curl -fsSL https://tailscale.com/install.sh | sh
```

### Step 2: Connect to Tailscale (All Machines)

```bash
tailscale up
```

This opens a browser to authenticate. All machines should join the same Tailscale account/network.

### Step 3: Start Master Node (Your Main PC)

```bash
cd backend

# Start Redis (or use Docker)
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Start FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Step 4: Start Worker Nodes (Other Machines)

Get your master's Tailscale hostname:
```bash
tailscale status
# Example output: flowml-master.your-tailnet.ts.net
```

On worker machines:
```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Start Celery worker connecting to master via Tailscale
celery -A worker.celery_app worker \
  --broker=redis://flowml-master.your-tailnet.ts.net:6379/0 \
  -Q cpu,gpu \
  --hostname=%h-%n
```

## API Endpoints

### Cluster Info
```
GET /api/cluster/info
```
Returns cluster mode, role, and Tailscale status.

### Join Command
```
GET /api/cluster/join
```
Returns the exact command workers should run.

### Tailscale Status
```
GET /api/cluster/tailscale/status
```
Full Tailscale status including peer list.

### Cluster Topology
```
GET /api/cluster/topology
```
Network topology data for visualization.

### Health Check
```
GET /api/cluster/health
```
Checks Redis connectivity and Tailscale status.

## Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `local` | Single machine | Development |
| `lan` | Same local network | Home/office cluster |
| `tailscale` | Mesh VPN | Distributed workers |

## Capabilities

Workers automatically probe and report:
- CPU cores and frequency
- RAM total/available
- GPU count and VRAM
- Tailscale IP and hostname
- Installed ML libraries

Workers subscribe to queues based on capabilities:
- `cpu` - All workers
- `gpu` - Workers with 4+ GB VRAM
- `gpu.vram6` - Workers with 6+ GB VRAM
- `gpu.vram8` - Workers with 8+ GB VRAM
- `llm` - Workers with GPU for LLM inference

## Security Notes

1. **Tailscale encrypts all traffic** - No additional VPN needed
2. **ACLs available** - Control which devices can access Redis
3. **No ports exposed** - Workers don't need public IPs
4. **Works through NAT** - Home networks, mobile hotspots, etc.

## Troubleshooting

### Worker can't connect to Redis
```bash
# Test connectivity
tailscale ping flowml-master

# Check Redis is accessible
redis-cli -h flowml-master.your-tailnet.ts.net ping
```

### Tailscale not detecting
```bash
# Check status
tailscale status

# Restart if needed
tailscale down
tailscale up
```

### Worker not appearing in dashboard
1. Check worker logs for registration errors
2. Verify Redis URL is correct
3. Ensure heartbeat task is running
