"""
System Tasks - Worker heartbeat, cleanup, and maintenance
"""
import os
import socket
from datetime import datetime, timedelta
from typing import Dict, Any
from celery import shared_task
from loguru import logger

from worker.capabilities import probe_capabilities


# Store last heartbeat info
_last_heartbeat: Dict[str, Any] = {}


@shared_task(
    bind=True,
    name="worker.tasks.system.heartbeat",
    ignore_result=True,
)
def heartbeat(self) -> Dict[str, Any]:
    """
    Worker heartbeat - report status and capabilities to orchestrator.
    Called periodically by Celery Beat.
    """
    global _last_heartbeat
    
    worker_id = f"{socket.gethostname()}-{os.getpid()}"
    
    # Only re-probe capabilities every 5 minutes
    now = datetime.utcnow()
    last_probe = _last_heartbeat.get("probed_at")
    
    if not last_probe or (now - datetime.fromisoformat(last_probe)) > timedelta(minutes=5):
        caps = probe_capabilities(worker_id)
        _last_heartbeat = caps.to_dict()
    
    _last_heartbeat["heartbeat_at"] = now.isoformat()
    _last_heartbeat["worker_id"] = worker_id
    
    # In production, this would POST to orchestrator API
    # For now, just log
    logger.debug(f"Heartbeat from {worker_id}")
    
    return _last_heartbeat


@shared_task(
    bind=True,
    name="worker.tasks.system.cleanup_old_jobs",
    ignore_result=True,
)
def cleanup_old_jobs(self, max_age_days: int = 7) -> Dict[str, Any]:
    """
    Clean up old job artifacts and temporary files.
    Called periodically by Celery Beat.
    """
    import shutil
    from pathlib import Path
    from config import settings
    
    logger.info(f"Running cleanup for files older than {max_age_days} days")
    
    cleaned = {
        "temp_files": 0,
        "old_logs": 0,
        "bytes_freed": 0,
    }
    
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)
    
    # Clean temp directory
    temp_dir = Path("/tmp/flowml") if os.name != "nt" else Path(os.environ.get("TEMP", "C:/Temp")) / "flowml"
    if temp_dir.exists():
        for f in temp_dir.iterdir():
            try:
                stat = f.stat()
                modified = datetime.fromtimestamp(stat.st_mtime)
                if modified < cutoff:
                    if f.is_file():
                        size = stat.st_size
                        f.unlink()
                        cleaned["temp_files"] += 1
                        cleaned["bytes_freed"] += size
                    elif f.is_dir():
                        size = sum(p.stat().st_size for p in f.rglob("*") if p.is_file())
                        shutil.rmtree(f)
                        cleaned["temp_files"] += 1
                        cleaned["bytes_freed"] += size
            except Exception as e:
                logger.warning(f"Failed to clean {f}: {e}")
    
    # Clean old log files
    logs_dir = settings.LOGS_DIR
    if logs_dir.exists():
        for f in logs_dir.glob("*.log*"):
            try:
                stat = f.stat()
                modified = datetime.fromtimestamp(stat.st_mtime)
                if modified < cutoff:
                    size = stat.st_size
                    f.unlink()
                    cleaned["old_logs"] += 1
                    cleaned["bytes_freed"] += size
            except Exception as e:
                logger.warning(f"Failed to clean log {f}: {e}")
    
    logger.info(f"Cleanup complete: {cleaned['temp_files']} temp files, "
                f"{cleaned['old_logs']} logs, {cleaned['bytes_freed'] / 1024 / 1024:.1f}MB freed")
    
    return cleaned


@shared_task(
    bind=True,
    name="worker.tasks.system.health_check",
)
def health_check(self) -> Dict[str, Any]:
    """
    Perform a health check on this worker.
    Returns current resource utilization and status.
    """
    import psutil
    
    cpu_percent = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    
    return {
        "worker_id": f"{socket.gethostname()}-{os.getpid()}",
        "status": "healthy",
        "cpu_percent": cpu_percent,
        "memory_percent": mem.percent,
        "memory_available_gb": round(mem.available / (1024**3), 2),
        "disk_percent": disk.percent,
        "disk_free_gb": round(disk.free / (1024**3), 2),
        "timestamp": datetime.utcnow().isoformat(),
    }
