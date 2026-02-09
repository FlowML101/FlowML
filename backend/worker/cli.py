"""
FlowML Worker CLI
Entry point for starting Celery workers with capability detection
"""
import os
import sys
import argparse
import json
import time
import threading
from pathlib import Path

# Ensure the backend directory is in path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from loguru import logger


def configure_logging(log_level: str = "INFO"):
    """Configure loguru logging"""
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Also log to file
    log_file = backend_dir / "logs" / "worker.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_file),
        rotation="10 MB",
        retention="7 days",
        level=log_level,
    )


def _start_heartbeat_thread(orchestrator_url: str, worker_id: str, interval: int = 30):
    """
    Start a daemon thread that POSTs heartbeats to the master every `interval` seconds.
    This keeps the worker marked as 'online' in the master's DB (TTL is 90s).
    """
    import psutil
    
    def _heartbeat_loop():
        import httpx
        url = f"{orchestrator_url.rstrip('/')}/api/workers/heartbeat"
        while True:
            try:
                ram = psutil.virtual_memory()
                payload = {
                    "worker_id": worker_id,
                    "status": "online",
                    "available_ram_gb": round(ram.available / (1024 ** 3), 2),
                    "current_tasks": 0,  # Could be enhanced to count active Celery tasks
                }
                resp = httpx.post(url, json=payload, timeout=5.0)
                if resp.status_code != 200:
                    logger.warning(f"Heartbeat returned {resp.status_code}")
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")
            time.sleep(interval)

    t = threading.Thread(target=_heartbeat_loop, daemon=True, name="heartbeat")
    t.start()
    logger.info(f"Heartbeat thread started (every {interval}s -> {orchestrator_url})")


def main():
    parser = argparse.ArgumentParser(description="FlowML Worker")
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=None,
        help="Number of concurrent tasks (default: auto based on CPU)"
    )
    parser.add_argument(
        "--queues", "-Q",
        type=str,
        default=None,
        help="Comma-separated list of queues to subscribe to (default: auto based on capabilities)"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Worker name (default: auto-generated)"
    )
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)"
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Only probe and print capabilities, don't start worker"
    )
    parser.add_argument(
        "--register",
        type=str,
        default=None,
        help="Orchestrator URL to register with (e.g., http://localhost:8000)"
    )
    
    args = parser.parse_args()
    
    configure_logging(args.log_level)
    
    # Probe capabilities
    from worker.capabilities import probe_capabilities
    
    caps = probe_capabilities(args.name)
    
    if args.probe_only:
        print(json.dumps(caps.to_dict(), indent=2))
        return
    
    # Determine queues
    if args.queues:
        queues = args.queues.split(",")
    else:
        queues = caps.get_queues()
    
    # Determine concurrency
    concurrency = args.concurrency or caps.max_concurrency
    
    logger.info(f"Starting worker: {caps.worker_id}")
    logger.info(f"Queues: {queues}")
    logger.info(f"Concurrency: {concurrency}")
    
    # Register with orchestrator if URL provided
    orchestrator_url = args.register
    if orchestrator_url:
        try:
            import httpx
            
            # Build registration payload
            payload = {
                "worker_id": caps.worker_id,
                "hostname": caps.hostname,
                "ip_address": caps.ip_address,
                "cpu_count": caps.cpu_count,
                "cpu_count_logical": caps.cpu_count_logical,
                "total_ram_gb": caps.total_ram_gb,
                "available_ram_gb": caps.available_ram_gb,
                "has_gpu": caps.has_gpu,
                "gpu_count": caps.gpu_count,
                "gpu_names": json.dumps([g.name for g in caps.gpus]) if caps.gpus else None,
                "total_vram_gb": caps.total_vram_gb,
                "cuda_available": caps.cuda_available,
                "cuda_version": caps.gpus[0].cuda_version if caps.gpus else None,
                "python_version": caps.python_version,
                "torch_version": caps.torch_version,
                "sklearn_version": caps.sklearn_version,
                "xgboost_version": caps.xgboost_version,
                "lightgbm_version": caps.lightgbm_version,
                "max_concurrency": concurrency,
                "tags": json.dumps(queues),
            }
            
            response = httpx.post(
                f"{orchestrator_url.rstrip('/')}/api/workers/register",
                json=payload,
                timeout=10.0,
            )
            
            if response.status_code == 200:
                logger.info(f"Registered with orchestrator: {orchestrator_url}")
            else:
                logger.warning(f"Failed to register: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.warning(f"Failed to register with orchestrator: {e}")
    
    # Start heartbeat background thread (keeps worker visible on master)
    if orchestrator_url:
        _start_heartbeat_thread(orchestrator_url, caps.worker_id)
    
    # Start Celery worker
    from worker.celery_app import celery_app
    
    worker = celery_app.Worker(
        hostname=caps.worker_id,
        queues=queues,
        concurrency=concurrency,
        loglevel=args.log_level,
        optimization="fair",  # Fair task distribution
        pool="prefork",  # Multiprocessing pool for CPU-bound tasks
    )
    
    logger.info("Starting Celery worker...")
    worker.start()


if __name__ == "__main__":
    main()
