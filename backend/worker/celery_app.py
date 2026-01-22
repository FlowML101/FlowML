"""
Celery Application Configuration
Central Celery app with queue routing and task discovery
"""
import os
from celery import Celery
from kombu import Queue, Exchange
from loguru import logger

# Redis URL from environment or default
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
celery_app = Celery(
    "flowml",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "worker.tasks.training",
        "worker.tasks.preprocessing",
        "worker.tasks.inference",
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task execution
    task_acks_late=True,  # Ack after task completes (reliability)
    task_reject_on_worker_lost=True,
    task_time_limit=3600 * 4,  # 4 hour hard limit
    task_soft_time_limit=3600 * 3.5,  # 3.5 hour soft limit
    
    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time for resource-heavy ML
    worker_concurrency=1,  # Default, overridden per-worker based on capabilities
    
    # Result backend
    result_expires=86400 * 7,  # 7 days
    result_extended=True,  # Store task args/kwargs
    
    # Task routing
    task_default_queue="cpu",
    task_default_exchange="flowml",
    task_default_routing_key="cpu",
    
    # Queues - capability-based routing
    task_queues=(
        # CPU-only tasks (all workers)
        Queue("cpu", Exchange("flowml"), routing_key="cpu"),
        
        # GPU tasks by VRAM tier
        Queue("gpu", Exchange("flowml"), routing_key="gpu"),
        Queue("gpu.vram6", Exchange("flowml"), routing_key="gpu.vram6"),
        Queue("gpu.vram8", Exchange("flowml"), routing_key="gpu.vram8"),
        Queue("gpu.vram12", Exchange("flowml"), routing_key="gpu.vram12"),
        Queue("gpu.vram24", Exchange("flowml"), routing_key="gpu.vram24"),
        
        # LLM tasks (Ollama)
        Queue("llm", Exchange("flowml"), routing_key="llm"),
        
        # High priority queue
        Queue("priority", Exchange("flowml"), routing_key="priority"),
    ),
    
    # Task routes
    task_routes={
        "worker.tasks.training.train_model": {"queue": "cpu"},
        "worker.tasks.training.train_model_gpu": {"queue": "gpu"},
        "worker.tasks.preprocessing.preprocess_dataset": {"queue": "cpu"},
        "worker.tasks.inference.run_inference": {"queue": "cpu"},
        "worker.tasks.inference.run_inference_gpu": {"queue": "gpu"},
    },
    
    # Beat scheduler (for periodic tasks)
    beat_schedule={
        "worker-heartbeat": {
            "task": "worker.tasks.system.heartbeat",
            "schedule": 30.0,  # Every 30 seconds
        },
        "cleanup-old-jobs": {
            "task": "worker.tasks.system.cleanup_old_jobs",
            "schedule": 3600.0,  # Every hour
        },
    },
)


def get_celery_app() -> Celery:
    """Get the Celery app instance"""
    return celery_app


# Task state constants
class TaskState:
    PENDING = "PENDING"
    STARTED = "STARTED"
    PROGRESS = "PROGRESS"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"


logger.info(f"Celery app configured with broker: {REDIS_URL}")
