"""
Stats routes - dashboard overview statistics
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, func
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta

from database import get_session
from models.dataset import Dataset
from models.job import Job, JobStatus
from models.trained_model import TrainedModel

router = APIRouter(prefix="/stats", tags=["stats"])


class DashboardStats(BaseModel):
    """Dashboard overview statistics"""
    total_models: int
    active_jobs: int
    total_datasets: int
    avg_accuracy: Optional[float]
    
    # Trends (for UI badges)
    models_this_week: int
    jobs_running: int
    jobs_queued: int


class ResourceStats(BaseModel):
    """Resource usage stats"""
    cpu_percent: float
    ram_percent: float
    ram_used_gb: float
    ram_total_gb: float
    vram_percent: Optional[float]
    vram_used_gb: Optional[float]
    vram_total_gb: Optional[float]


@router.get("", response_model=DashboardStats)
async def get_dashboard_stats(
    session: AsyncSession = Depends(get_session)
):
    """Get dashboard overview statistics"""
    
    # Total models
    models_result = await session.execute(select(func.count(TrainedModel.id)))
    total_models = models_result.scalar() or 0
    
    # Active jobs (running + pending)
    active_result = await session.execute(
        select(func.count(Job.id)).where(
            Job.status.in_([JobStatus.RUNNING, JobStatus.PENDING])
        )
    )
    active_jobs = active_result.scalar() or 0
    
    # Running vs queued
    running_result = await session.execute(
        select(func.count(Job.id)).where(Job.status == JobStatus.RUNNING)
    )
    jobs_running = running_result.scalar() or 0
    jobs_queued = active_jobs - jobs_running
    
    # Total datasets
    datasets_result = await session.execute(select(func.count(Dataset.id)))
    total_datasets = datasets_result.scalar() or 0
    
    # Average accuracy (of best models per job)
    # Get rank=1 models
    best_models = await session.execute(
        select(TrainedModel.accuracy).where(
            TrainedModel.rank == 1,
            TrainedModel.accuracy.isnot(None)
        )
    )
    accuracies = [m for m in best_models.scalars().all() if m is not None]
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else None
    
    # Models this week (simplified - just count recent)
    from datetime import datetime, timedelta
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_result = await session.execute(
        select(func.count(TrainedModel.id)).where(
            TrainedModel.created_at >= week_ago
        )
    )
    models_this_week = recent_result.scalar() or 0
    
    return DashboardStats(
        total_models=total_models,
        active_jobs=active_jobs,
        total_datasets=total_datasets,
        avg_accuracy=round(avg_accuracy, 4) if avg_accuracy else None,
        models_this_week=models_this_week,
        jobs_running=jobs_running,
        jobs_queued=jobs_queued
    )


@router.get("/resources", response_model=ResourceStats)
async def get_resource_stats():
    """Get current resource usage"""
    import psutil
    
    cpu_percent = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory()
    
    # GPU detection
    vram_percent = None
    vram_used = None
    vram_total = None
    
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 2:
                vram_total = float(parts[0]) / 1024  # MB to GB
                vram_used = float(parts[1]) / 1024
                vram_percent = (vram_used / vram_total) * 100 if vram_total > 0 else 0
    except:
        pass
    
    return ResourceStats(
        cpu_percent=cpu_percent,
        ram_percent=ram.percent,
        ram_used_gb=round(ram.used / (1024**3), 1),
        ram_total_gb=round(ram.total / (1024**3), 1),
        vram_percent=round(vram_percent, 1) if vram_percent else None,
        vram_used_gb=round(vram_used, 1) if vram_used else None,
        vram_total_gb=round(vram_total, 1) if vram_total else None
    )


@router.get("/cluster")
async def get_cluster_health(
    session: AsyncSession = Depends(get_session)
):
    """Get cluster health overview"""
    from routers.workers import get_local_machine_info, WORKER_TTL_SECONDS
    from models.worker import Worker
    from datetime import timedelta
    import psutil
    
    # Get master info
    master = get_local_machine_info()
    
    # Get online workers from DB
    cutoff = datetime.utcnow() - timedelta(seconds=WORKER_TTL_SECONDS)
    result = await session.execute(
        select(Worker).where(Worker.last_heartbeat >= cutoff)
    )
    db_workers = result.scalars().all()
    
    # Convert to worker-like objects
    workers_list = []
    for w in db_workers:
        workers_list.append({
            "id": w.worker_id,
            "hostname": w.hostname,
            "status": w.status,
            "cpu_percent": psutil.cpu_percent(interval=0),  # Approximate since we don't store it
            "ram_percent": round((1 - w.available_ram_gb / w.total_ram_gb) * 100, 1) if w.total_ram_gb > 0 else 0,
        })
    
    online = len([w for w in workers_list if w["status"] in ("online", "busy")]) + 1  # +1 for master
    offline = len([w for w in workers_list if w["status"] == "offline"])
    busy = len([w for w in workers_list if w["status"] == "busy" or (w.get("current_tasks", 0) > 0)])
    
    return {
        "total_nodes": len(workers_list) + 1,  # +1 for master
        "online": online,
        "offline": offline,
        "busy": busy,
        "master": {
            "hostname": master.hostname,
            "status": master.status,
            "cpu_percent": master.cpu_percent,
            "ram_percent": master.ram_percent
        },
        "workers": workers_list
    }
