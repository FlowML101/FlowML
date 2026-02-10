"""
Workers routes - Manage distributed Celery workers
Supports registration, heartbeats, and capability-based routing
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import json
from loguru import logger

from database import get_session
from models.worker import Worker, WorkerCreate, WorkerRead, WorkerHeartbeat
from exceptions import not_found, bad_request

router = APIRouter(prefix="/workers", tags=["workers"])


# Worker TTL - consider offline after this many seconds without heartbeat
WORKER_TTL_SECONDS = 90


class WorkerStats(BaseModel):
    """Aggregated worker statistics"""
    total_workers: int
    online_workers: int
    offline_workers: int
    total_cpus: int
    total_ram_gb: float
    total_gpus: int
    total_vram_gb: float
    queues: Dict[str, int]  # queue -> worker count


class QueueInfo(BaseModel):
    """Queue information"""
    name: str
    worker_count: int
    pending_tasks: int


@router.post("/register", response_model=WorkerRead)
async def register_worker(
    worker_data: WorkerCreate,
    session: AsyncSession = Depends(get_session)
):
    """
    Register a new worker or update existing registration.
    Called by workers on startup with their capabilities.
    """
    # Check if worker already exists
    result = await session.execute(
        select(Worker).where(Worker.worker_id == worker_data.worker_id)
    )
    existing = result.scalar_one_or_none()
    
    if existing:
        # Update existing worker
        for field, value in worker_data.model_dump().items():
            setattr(existing, field, value)
        existing.last_heartbeat = datetime.utcnow()
        existing.status = "online"
        worker = existing
        logger.info(f"Worker {worker_data.worker_id} re-registered")
    else:
        # Create new worker
        worker = Worker(**worker_data.model_dump())
        session.add(worker)
        logger.info(f"New worker registered: {worker_data.worker_id}")
    
    await session.commit()
    await session.refresh(worker)
    
    return worker


@router.post("/heartbeat")
async def worker_heartbeat(
    heartbeat: WorkerHeartbeat,
    session: AsyncSession = Depends(get_session)
):
    """
    Worker heartbeat - update status and resources.
    Workers should call this every 30 seconds.
    """
    result = await session.execute(
        select(Worker).where(Worker.worker_id == heartbeat.worker_id)
    )
    worker = result.scalar_one_or_none()
    
    if not worker:
        raise not_found("Worker", heartbeat.worker_id)
    
    worker.available_ram_gb = heartbeat.available_ram_gb
    worker.current_tasks = heartbeat.current_tasks
    worker.status = heartbeat.status
    worker.last_heartbeat = datetime.utcnow()
    
    await session.commit()
    
    return {"status": "ok", "worker_id": heartbeat.worker_id}


@router.get("", response_model=List[WorkerRead])
async def list_workers(
    session: AsyncSession = Depends(get_session),
    include_offline: bool = False
):
    """
    List all registered workers.
    By default, only shows workers with recent heartbeats.
    """
    query = select(Worker)
    
    if not include_offline:
        cutoff = datetime.utcnow() - timedelta(seconds=WORKER_TTL_SECONDS)
        query = query.where(Worker.last_heartbeat >= cutoff)
    
    result = await session.execute(query.order_by(Worker.hostname))
    workers = result.scalars().all()
    
    # Update status based on heartbeat
    for worker in workers:
        if (datetime.utcnow() - worker.last_heartbeat).total_seconds() > WORKER_TTL_SECONDS:
            worker.status = "offline"
    
    return workers


@router.get("/stats", response_model=WorkerStats)
async def get_worker_stats(
    session: AsyncSession = Depends(get_session)
):
    """Get aggregated statistics across all workers"""
    cutoff = datetime.utcnow() - timedelta(seconds=WORKER_TTL_SECONDS)
    
    result = await session.execute(select(Worker))
    all_workers = result.scalars().all()
    
    online = [w for w in all_workers if w.last_heartbeat >= cutoff]
    offline = [w for w in all_workers if w.last_heartbeat < cutoff]
    
    # Aggregate queue coverage
    queues: Dict[str, int] = {}
    for worker in online:
        try:
            tags = json.loads(worker.tags)
            for tag in tags:
                queues[tag] = queues.get(tag, 0) + 1
        except:
            pass
    
    return WorkerStats(
        total_workers=len(all_workers),
        online_workers=len(online),
        offline_workers=len(offline),
        total_cpus=sum(w.cpu_count for w in online),
        total_ram_gb=sum(w.total_ram_gb for w in online),
        total_gpus=sum(w.gpu_count for w in online),
        total_vram_gb=sum(w.total_vram_gb for w in online),
        queues=queues,
    )


@router.get("/queues", response_model=List[QueueInfo])
async def get_queue_info(
    session: AsyncSession = Depends(get_session)
):
    """Get information about task queues and worker coverage"""
    from scheduler import get_default_scheduler
    
    # Get queue stats from scheduler
    try:
        scheduler = get_default_scheduler()
        stats = await scheduler.get_queue_stats()
    except Exception as e:
        logger.warning(f"Failed to get queue stats: {e}")
        stats = {"queues": {}}
    
    # Get worker coverage
    cutoff = datetime.utcnow() - timedelta(seconds=WORKER_TTL_SECONDS)
    result = await session.execute(
        select(Worker).where(Worker.last_heartbeat >= cutoff)
    )
    online_workers = result.scalars().all()
    
    queue_workers: Dict[str, int] = {}
    for worker in online_workers:
        try:
            tags = json.loads(worker.tags)
            for tag in tags:
                queue_workers[tag] = queue_workers.get(tag, 0) + 1
        except:
            pass
    
    # Build queue info
    all_queues = set(stats.get("queues", {}).keys()) | set(queue_workers.keys())
    queue_info = []
    
    for queue in sorted(all_queues):
        queue_info.append(QueueInfo(
            name=queue,
            worker_count=queue_workers.get(queue, 0),
            pending_tasks=stats.get("queues", {}).get(queue, {}).get("length", 0),
        ))
    
    return queue_info


# --- Frontend-compatible endpoints (MUST be before /{worker_id}) ---

class WorkerFrontend(BaseModel):
    """Worker model matching frontend interface"""
    id: str
    hostname: str
    role: str
    status: str
    ip: str
    tailscale_ip: Optional[str] = None
    cpu_count: int
    cpu_percent: float
    ram_total_gb: float
    ram_used_gb: float
    ram_percent: float
    gpu_name: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_used_gb: Optional[float] = None
    vram_percent: Optional[float] = None
    uptime: str


def get_local_machine_info() -> WorkerFrontend:
    """Get info about the local machine (orchestrator) in frontend-compatible format"""
    import socket
    import psutil
    
    cpu_percent = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory()
    
    # GPU detection
    gpu_name = None
    vram_total = None
    vram_used = None
    vram_percent = None
    
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            gpu_name = name
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_total = round(mem_info.total / (1024**3), 1)
            vram_used = round(mem_info.used / (1024**3), 1)
            vram_percent = round(mem_info.used / mem_info.total * 100, 1)
        pynvml.nvmlShutdown()
    except:
        pass
    
    # Uptime
    boot_time = datetime.fromtimestamp(psutil.boot_time())
    uptime = datetime.now() - boot_time
    hours, remainder = divmod(int(uptime.total_seconds()), 3600)
    minutes, _ = divmod(remainder, 60)
    
    # Get IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except:
        ip = "127.0.0.1"
    
    return WorkerFrontend(
        id="master",
        hostname=socket.gethostname(),
        role="orchestrator",
        status="online",
        ip=ip,
        cpu_count=psutil.cpu_count(),
        cpu_percent=cpu_percent,
        ram_total_gb=round(ram.total / (1024**3), 1),
        ram_used_gb=round(ram.used / (1024**3), 1),
        ram_percent=ram.percent,
        gpu_name=gpu_name,
        vram_total_gb=vram_total,
        vram_used_gb=vram_used,
        vram_percent=vram_percent,
        uptime=f"{hours}h {minutes}m",
    )


@router.get("/master", response_model=WorkerFrontend)
async def get_master():
    """Get master node info (frontend compatible)"""
    return get_local_machine_info()


@router.get("/celery-workers")
async def get_celery_workers():
    """
    Get active Celery workers directly from the broker.
    This shows workers connected to Redis even if not registered via CLI.
    """
    try:
        from worker.celery_app import celery_app
        
        inspector = celery_app.control.inspect(timeout=2.0)
        active_queues = inspector.active_queues() or {}
        stats = inspector.stats() or {}
        
        workers = []
        for worker_name, queues in active_queues.items():
            worker_stats = stats.get(worker_name, {})
            
            # Parse hostname from worker name (format: celery@HOSTNAME)
            hostname = worker_name.split('@')[-1] if '@' in worker_name else worker_name
            
            # Get queue names this worker listens to
            queue_names = [q.get('name', 'unknown') for q in queues]
            
            workers.append({
                "worker_id": worker_name,
                "hostname": hostname,
                "status": "online",
                "queues": queue_names,
                "pool": worker_stats.get("pool", {}).get("implementation", "unknown"),
                "concurrency": worker_stats.get("pool", {}).get("max-concurrency", 1),
                "pid": worker_stats.get("pid"),
            })
        
        return {"workers": workers, "count": len(workers)}
    except Exception as e:
        logger.warning(f"Failed to inspect Celery workers: {e}")
        return {"workers": [], "count": 0, "error": str(e)}


@router.get("/all")
async def get_all_workers(
    session: AsyncSession = Depends(get_session)
):
    """
    Get all workers - both DB-registered and Celery-connected.
    This is the comprehensive view for the frontend.
    """
    # Get master info
    master = get_local_machine_info()
    all_workers = [master.model_dump()]
    seen_hostnames = {master.hostname.lower()}
    
    # Get Celery workers from broker
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
            has_gpu = 'gpu' in queue_names
            
            all_workers.append({
                "id": worker_name,
                "hostname": hostname,
                "role": "celery-worker",
                "status": "online",
                "ip": "",  # Not available from Celery inspect
                "cpu_count": 0,
                "cpu_percent": 0,
                "ram_total_gb": 0,
                "ram_used_gb": 0,
                "ram_percent": 0,
                "gpu_name": "GPU available" if has_gpu else None,
                "vram_total_gb": None,
                "vram_used_gb": None,
                "vram_percent": None,
                "uptime": "connected",
                "queues": queue_names,
            })
    except Exception as e:
        logger.warning(f"Failed to inspect Celery workers: {e}")
    
    # Also get DB-registered workers
    cutoff = datetime.utcnow() - timedelta(seconds=WORKER_TTL_SECONDS)
    result = await session.execute(select(Worker).where(Worker.last_heartbeat >= cutoff))
    db_workers = result.scalars().all()
    
    for w in db_workers:
        if w.hostname.lower() in seen_hostnames:
            continue
        seen_hostnames.add(w.hostname.lower())
        
        all_workers.append({
            "id": w.worker_id,
            "hostname": w.hostname,
            "role": "worker",
            "status": w.status,
            "ip": w.ip_address,
            "cpu_count": w.cpu_count,
            "cpu_percent": 0,
            "ram_total_gb": w.total_ram_gb,
            "ram_used_gb": w.total_ram_gb - w.available_ram_gb,
            "ram_percent": round(((w.total_ram_gb - w.available_ram_gb) / w.total_ram_gb) * 100) if w.total_ram_gb else 0,
            "gpu_name": json.loads(w.gpu_names)[0] if w.gpu_names else None,
            "vram_total_gb": w.total_vram_gb or None,
            "uptime": "connected",
        })
    
    return all_workers


@router.get("/available")
async def get_available_workers(
    session: AsyncSession = Depends(get_session)
):
    """Get worker availability counts"""
    cutoff = datetime.utcnow() - timedelta(seconds=WORKER_TTL_SECONDS)
    
    result = await session.execute(select(Worker))
    all_workers = result.scalars().all()
    
    online = [w for w in all_workers if w.last_heartbeat >= cutoff]
    busy = [w for w in online if w.status == "busy" or w.current_tasks > 0]
    
    return {
        "total": len(all_workers) + 1,  # +1 for master
        "available": len(online) - len(busy) + 1,  # +1 for master (always available)
        "busy": len(busy),
    }


@router.get("/local/info")
async def get_local_info():
    """Get info about the local machine (orchestrator) - legacy endpoint"""
    return get_local_machine_info()


# --- Parameterized routes (MUST be after specific routes) ---

@router.get("/{worker_id}", response_model=WorkerRead)
async def get_worker(
    worker_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Get a specific worker by ID"""
    result = await session.execute(
        select(Worker).where(Worker.worker_id == worker_id)
    )
    worker = result.scalar_one_or_none()
    
    if not worker:
        raise not_found("Worker", worker_id)
    
    # Update status based on heartbeat
    if (datetime.utcnow() - worker.last_heartbeat).total_seconds() > WORKER_TTL_SECONDS:
        worker.status = "offline"
    
    return worker


@router.post("/{worker_id}/drain")
async def drain_worker(
    worker_id: str,
    session: AsyncSession = Depends(get_session)
):
    """
    Put worker in draining mode - no new tasks, wait for current to complete.
    Used for graceful shutdown/updates.
    """
    result = await session.execute(
        select(Worker).where(Worker.worker_id == worker_id)
    )
    worker = result.scalar_one_or_none()
    
    if not worker:
        raise not_found("Worker", worker_id)
    
    worker.status = "draining"
    await session.commit()
    
    logger.info(f"Worker {worker_id} set to draining")
    return {"status": "draining", "worker_id": worker_id}


@router.delete("/{worker_id}")
async def unregister_worker(
    worker_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Remove a worker from the registry"""
    result = await session.execute(
        select(Worker).where(Worker.worker_id == worker_id)
    )
    worker = result.scalar_one_or_none()
    
    if not worker:
        raise not_found("Worker", worker_id)
    
    await session.delete(worker)
    await session.commit()
    
    logger.info(f"Worker {worker_id} unregistered")
    return {"status": "unregistered", "worker_id": worker_id}
