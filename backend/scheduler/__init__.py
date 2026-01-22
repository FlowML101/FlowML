"""
Scheduler Abstraction Layer
Provides a unified interface for task scheduling.
MVP: Celery implementation
Phase 2: Swap to Ray without changing API
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from loguru import logger


class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class TaskResult:
    """Unified task result across schedulers"""
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: int = 0
    stage: Optional[str] = None


@dataclass  
class TaskConstraints:
    """Resource constraints for task routing"""
    min_ram_gb: float = 0
    max_ram_gb: Optional[float] = None
    min_vram_gb: float = 0
    require_gpu: bool = False
    require_llm: bool = False
    timeout_seconds: int = 3600
    priority: int = 5  # 1-10, higher = more urgent


class BaseScheduler(ABC):
    """Abstract scheduler interface"""
    
    @abstractmethod
    async def submit_training(
        self,
        job_id: str,
        dataset_path: str,
        target_column: str,
        problem_type: str,
        time_budget: int,
        constraints: Optional[TaskConstraints] = None,
    ) -> str:
        """Submit a training task, returns task_id"""
        pass
    
    @abstractmethod
    async def submit_preprocessing(
        self,
        dataset_id: str,
        dataset_path: str,
        operations: List[Dict[str, Any]],
    ) -> str:
        """Submit a preprocessing task, returns task_id"""
        pass
    
    @abstractmethod
    async def submit_inference(
        self,
        model_path: str,
        data: Any,
        output_path: Optional[str] = None,
        constraints: Optional[TaskConstraints] = None,
    ) -> str:
        """Submit an inference task, returns task_id"""
        pass
    
    @abstractmethod
    async def get_task_status(self, task_id: str) -> TaskResult:
        """Get current status of a task"""
        pass
    
    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        pass
    
    @abstractmethod
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        pass


class CeleryScheduler(BaseScheduler):
    """Celery-based scheduler implementation"""
    
    def __init__(self):
        from worker.celery_app import celery_app
        self.app = celery_app
        logger.info("CeleryScheduler initialized")
    
    def _route_to_queue(self, constraints: Optional[TaskConstraints]) -> str:
        """Determine queue based on constraints"""
        if not constraints:
            return "cpu"
        
        if constraints.require_llm:
            return "llm"
        
        if constraints.require_gpu or constraints.min_vram_gb > 0:
            if constraints.min_vram_gb >= 24:
                return "gpu.vram24"
            elif constraints.min_vram_gb >= 12:
                return "gpu.vram12"
            elif constraints.min_vram_gb >= 8:
                return "gpu.vram8"
            elif constraints.min_vram_gb >= 6:
                return "gpu.vram6"
            else:
                return "gpu"
        
        if constraints.priority >= 8:
            return "priority"
        
        return "cpu"
    
    async def submit_training(
        self,
        job_id: str,
        dataset_path: str,
        target_column: str,
        problem_type: str = "auto",
        time_budget: int = 5,
        constraints: Optional[TaskConstraints] = None,
        model_types: Optional[List[str]] = None,
    ) -> str:
        """Submit training task to Celery"""
        from worker.tasks.training import train_model, train_model_gpu
        
        queue = self._route_to_queue(constraints)
        
        # Choose task based on GPU requirement
        if constraints and (constraints.require_gpu or constraints.min_vram_gb > 0):
            task = train_model_gpu
        else:
            task = train_model
        
        result = task.apply_async(
            kwargs={
                "job_id": job_id,
                "dataset_path": dataset_path,
                "target_column": target_column,
                "problem_type": problem_type,
                "time_budget": time_budget,
                "model_types": model_types,
            },
            queue=queue,
            priority=constraints.priority if constraints else 5,
            soft_time_limit=constraints.timeout_seconds - 60 if constraints else 3540,
            time_limit=constraints.timeout_seconds if constraints else 3600,
        )
        
        logger.info(f"Training task {result.id} submitted to queue '{queue}' for job {job_id}")
        return result.id
    
    async def submit_preprocessing(
        self,
        dataset_id: str,
        dataset_path: str,
        operations: List[Dict[str, Any]],
    ) -> str:
        """Submit preprocessing task to Celery"""
        from worker.tasks.preprocessing import preprocess_dataset
        
        result = preprocess_dataset.apply_async(
            kwargs={
                "dataset_id": dataset_id,
                "dataset_path": dataset_path,
                "operations": operations,
            },
            queue="cpu",
        )
        
        logger.info(f"Preprocessing task {result.id} submitted for dataset {dataset_id}")
        return result.id
    
    async def submit_inference(
        self,
        model_path: str,
        data: Any,
        output_path: Optional[str] = None,
        constraints: Optional[TaskConstraints] = None,
    ) -> str:
        """Submit inference task to Celery"""
        from worker.tasks.inference import run_inference, run_inference_gpu
        
        queue = self._route_to_queue(constraints)
        
        if constraints and (constraints.require_gpu or constraints.min_vram_gb > 0):
            task = run_inference_gpu
        else:
            task = run_inference
        
        result = task.apply_async(
            kwargs={
                "model_path": model_path,
                "data": data,
                "output_path": output_path,
            },
            queue=queue,
        )
        
        logger.info(f"Inference task {result.id} submitted to queue '{queue}'")
        return result.id
    
    async def get_task_status(self, task_id: str) -> TaskResult:
        """Get Celery task status"""
        from celery.result import AsyncResult
        
        result = AsyncResult(task_id, app=self.app)
        
        # Map Celery states to TaskStatus
        state_map = {
            "PENDING": TaskStatus.QUEUED,
            "STARTED": TaskStatus.RUNNING,
            "PROGRESS": TaskStatus.RUNNING,
            "SUCCESS": TaskStatus.COMPLETED,
            "FAILURE": TaskStatus.FAILED,
            "REVOKED": TaskStatus.CANCELLED,
            "RETRY": TaskStatus.QUEUED,
        }
        
        status = state_map.get(result.state, TaskStatus.PENDING)
        
        task_result = TaskResult(
            task_id=task_id,
            status=status,
        )
        
        if result.state == "PROGRESS":
            meta = result.info or {}
            task_result.progress = meta.get("progress", 0)
            task_result.stage = meta.get("stage")
        elif result.state == "SUCCESS":
            task_result.result = result.result
            task_result.progress = 100
        elif result.state == "FAILURE":
            task_result.error = str(result.result) if result.result else "Unknown error"
        
        return task_result
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel/revoke a Celery task"""
        from celery.result import AsyncResult
        
        result = AsyncResult(task_id, app=self.app)
        result.revoke(terminate=True)
        logger.info(f"Task {task_id} cancelled")
        return True
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get Celery queue statistics"""
        inspect = self.app.control.inspect()
        
        stats = {
            "queues": {},
            "workers": {},
            "active_tasks": 0,
            "reserved_tasks": 0,
        }
        
        try:
            # Active tasks
            active = inspect.active() or {}
            for worker, tasks in active.items():
                stats["workers"][worker] = {"active": len(tasks)}
                stats["active_tasks"] += len(tasks)
            
            # Reserved (queued) tasks
            reserved = inspect.reserved() or {}
            for worker, tasks in reserved.items():
                if worker in stats["workers"]:
                    stats["workers"][worker]["reserved"] = len(tasks)
                else:
                    stats["workers"][worker] = {"reserved": len(tasks)}
                stats["reserved_tasks"] += len(tasks)
            
            # Queue lengths (requires Redis inspection)
            from worker.celery_app import REDIS_URL
            import redis
            
            r = redis.from_url(REDIS_URL)
            for queue in ["cpu", "gpu", "gpu.vram6", "gpu.vram8", "gpu.vram12", "gpu.vram24", "llm", "priority"]:
                length = r.llen(queue)
                stats["queues"][queue] = {"length": length}
            
        except Exception as e:
            logger.warning(f"Failed to get queue stats: {e}")
        
        return stats


# Factory function to get the appropriate scheduler
def get_scheduler(backend: str = "celery") -> BaseScheduler:
    """
    Get a scheduler instance.
    
    Args:
        backend: "celery" (MVP) or "ray" (Phase 2)
    
    Returns:
        Scheduler instance
    """
    if backend == "celery":
        return CeleryScheduler()
    elif backend == "ray":
        # TODO: Phase 2 - RayScheduler implementation
        raise NotImplementedError("Ray scheduler not yet implemented")
    else:
        raise ValueError(f"Unknown scheduler backend: {backend}")


# Default scheduler instance
_scheduler: Optional[BaseScheduler] = None


def get_default_scheduler() -> BaseScheduler:
    """Get the default scheduler instance (singleton)"""
    global _scheduler
    if _scheduler is None:
        _scheduler = get_scheduler("celery")
    return _scheduler
