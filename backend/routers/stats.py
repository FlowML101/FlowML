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


class GPUStatus(BaseModel):
    """GPU status for ML training"""
    available: bool
    count: int
    name: str
    vram_total_gb: float
    vram_used_gb: float
    vram_free_gb: float
    utilization_percent: float
    
    # ML Framework GPU support
    xgboost_gpu: bool
    lightgbm_gpu: bool
    catboost_gpu: bool
    pytorch_cuda: bool
    
    # Training status
    training_accelerated: bool
    acceleration_libs: list[str]


@router.get("/gpu", response_model=GPUStatus)
async def get_gpu_status():
    """Get GPU status and ML acceleration capabilities"""
    from services.optuna_automl import (
        GPU_AVAILABLE, GPU_COUNT, GPU_VRAM_GB, GPU_NAME,
        XGBOOST_GPU_AVAILABLE, LIGHTGBM_GPU_AVAILABLE, CATBOOST_AVAILABLE
    )
    
    vram_used = 0.0
    vram_free = GPU_VRAM_GB
    utilization = 0.0
    
    # Get real-time GPU stats
    if GPU_AVAILABLE:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_used = mem_info.used / (1024**3)
            vram_free = mem_info.free / (1024**3)
            
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utilization = util.gpu
            pynvml.nvmlShutdown()
        except Exception:
            pass
    
    # Check PyTorch CUDA
    pytorch_cuda = False
    try:
        import torch
        pytorch_cuda = torch.cuda.is_available()
    except ImportError:
        pass
    
    # Build acceleration libs list
    accel_libs = []
    if XGBOOST_GPU_AVAILABLE:
        accel_libs.append("XGBoost")
    if LIGHTGBM_GPU_AVAILABLE:
        accel_libs.append("LightGBM")
    if CATBOOST_AVAILABLE and GPU_AVAILABLE:
        accel_libs.append("CatBoost")
    if pytorch_cuda:
        accel_libs.append("PyTorch")
    
    return GPUStatus(
        available=GPU_AVAILABLE,
        count=GPU_COUNT,
        name=GPU_NAME,
        vram_total_gb=round(GPU_VRAM_GB, 2),
        vram_used_gb=round(vram_used, 2),
        vram_free_gb=round(vram_free, 2),
        utilization_percent=utilization,
        xgboost_gpu=XGBOOST_GPU_AVAILABLE,
        lightgbm_gpu=LIGHTGBM_GPU_AVAILABLE,
        catboost_gpu=CATBOOST_AVAILABLE and GPU_AVAILABLE,
        pytorch_cuda=pytorch_cuda,
        training_accelerated=len(accel_libs) > 0,
        acceleration_libs=accel_libs
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
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_total = mem_info.total / (1024**3)
        vram_used = mem_info.used / (1024**3)
        vram_percent = (vram_used / vram_total) * 100 if vram_total > 0 else 0
        pynvml.nvmlShutdown()
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
    """Get cluster health overview - includes Celery workers"""
    from routers.workers import get_local_machine_info, WORKER_TTL_SECONDS
    from models.worker import Worker
    from datetime import timedelta
    import psutil
    from loguru import logger
    
    # Get master info
    master = get_local_machine_info()
    
    # Track seen hostnames to avoid duplicates
    seen_hostnames = {master.hostname.lower()}
    workers_list = []
    
    # First, get Celery workers directly from broker (most reliable)
    try:
        from worker.celery_app import celery_app
        inspector = celery_app.control.inspect(timeout=2.0)
        active_queues = inspector.active_queues() or {}
        
        for worker_name, queues in active_queues.items():
            hostname = worker_name.split('@')[-1] if '@' in worker_name else worker_name
            
            if hostname.lower() in seen_hostnames:
                continue
            seen_hostnames.add(hostname.lower())
            
            queue_names = [q.get('name', 'unknown') for q in queues]
            
            workers_list.append({
                "id": worker_name,
                "hostname": hostname,
                "status": "online",
                "cpu_percent": 0,  # Not available from Celery inspect
                "ram_percent": 0,
                "queues": queue_names,
            })
        
        logger.debug(f"Found {len(workers_list)} Celery workers")
    except Exception as e:
        logger.warning(f"Failed to inspect Celery workers: {e}")
    
    # Also get DB-registered workers (in case any are registered but not running Celery)
    cutoff = datetime.utcnow() - timedelta(seconds=WORKER_TTL_SECONDS)
    result = await session.execute(
        select(Worker).where(Worker.last_heartbeat >= cutoff)
    )
    db_workers = result.scalars().all()
    
    for w in db_workers:
        if w.hostname.lower() in seen_hostnames:
            continue
        seen_hostnames.add(w.hostname.lower())
        
        workers_list.append({
            "id": w.worker_id,
            "hostname": w.hostname,
            "status": w.status,
            "cpu_percent": 0,
            "ram_percent": round((1 - w.available_ram_gb / w.total_ram_gb) * 100, 1) if w.total_ram_gb > 0 else 0,
        })
    
    online = len([w for w in workers_list if w["status"] in ("online", "busy")]) + 1  # +1 for master
    offline = len([w for w in workers_list if w["status"] == "offline"])
    busy = len([w for w in workers_list if w["status"] == "busy"])
    
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
