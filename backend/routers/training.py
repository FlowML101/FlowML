"""
Training routes - start jobs, get status, cancel
Supports distributed execution via Celery with local fallback
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from datetime import datetime
import asyncio
import json
import threading
from typing import Dict, Optional
from loguru import logger

from database import get_session, get_session_context
from models.job import Job, JobCreate, JobRead, JobStatus, JobProgress
from models.dataset import Dataset
from services.optuna_automl import optuna_automl, OptunaAutoML
from services.websocket_manager import manager
from exceptions import not_found, bad_request, conflict
from config import settings

router = APIRouter(prefix="/training", tags=["training"])

# Track running jobs for cancellation (local execution only)
_running_jobs: Dict[str, threading.Event] = {}
# Track Celery async-result IDs for distributed jobs
_celery_tasks: Dict[str, str] = {}


def _celery_available() -> bool:
    """Check if Celery broker (Redis) is reachable and at least one worker is online."""
    try:
        from worker.celery_app import celery_app
        conn = celery_app.connection()
        conn.connect()
        conn.close()
        # Check if any workers are actively consuming
        inspect = celery_app.control.inspect(timeout=1.0)
        active_queues = inspect.active_queues()
        return active_queues is not None and len(active_queues) > 0
    except Exception:
        return False


def _get_worker_count() -> int:
    """Get the number of active Celery workers."""
    try:
        from worker.celery_app import celery_app
        inspect = celery_app.control.inspect(timeout=1.0)
        active_queues = inspect.active_queues()
        return len(active_queues) if active_queues else 0
    except Exception:
        return 0


@router.post("/start", response_model=JobRead)
async def start_training(
    job_data: JobCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session)
):
    """
    Start an AutoML training job.
    """
    # Check concurrent job limit
    result = await session.execute(
        select(Job).where(Job.status == JobStatus.RUNNING)
    )
    running_jobs = result.scalars().all()
    if len(running_jobs) >= settings.MAX_CONCURRENT_JOBS:
        raise conflict(f"Maximum {settings.MAX_CONCURRENT_JOBS} concurrent jobs allowed")
    
    # Verify dataset exists
    result = await session.execute(
        select(Dataset).where(Dataset.id == job_data.dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise not_found("Dataset", job_data.dataset_id)
    
    # Verify target column exists
    columns = json.loads(dataset.columns)
    if job_data.target_column not in columns:
        raise bad_request(
            f"Target column '{job_data.target_column}' not found in dataset",
            {"available_columns": columns}
        )
    
    # Validate time budget
    if job_data.time_budget > settings.MAX_TIME_BUDGET:
        raise bad_request(
            f"Time budget exceeds maximum of {settings.MAX_TIME_BUDGET} minutes"
        )
    
    # Create job
    job_name = job_data.name or f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    model_types = job_data.model_types
    if isinstance(model_types, list):
        model_types = json.dumps(model_types)
    
    job = Job(
        name=job_name,
        dataset_id=job_data.dataset_id,
        target_column=job_data.target_column,
        time_budget=job_data.time_budget,
        model_types=model_types,
        problem_type=job_data.problem_type,
        status=JobStatus.PENDING
    )
    
    session.add(job)
    await session.commit()
    await session.refresh(job)
    
    # Get dataset path
    result2 = await session.execute(
        select(Dataset).where(Dataset.id == job_data.dataset_id)
    )
    dataset = result2.scalar_one()
    
    # --- Dispatch: prefer Celery workers, fall back to local ---
    use_celery = _celery_available()
    worker_count = _get_worker_count() if use_celery else 0
    
    if use_celery and worker_count >= 2:
        # Multiple workers available - use distributed training (split compute)
        from worker.tasks.training import dispatch_distributed_training
        
        model_types_list = None
        if job_data.model_types:
            model_types_list = job_data.model_types if isinstance(job_data.model_types, list) else None
        
        # Detect problem type early for distributed dispatch
        problem_type = job_data.problem_type or "auto"
        if problem_type == "auto":
            # Quick detection based on target column
            import polars as pl
            try:
                df = pl.read_csv(dataset.file_path, n_rows=1000)
                target_col = df[job_data.target_column]
                n_unique = target_col.n_unique()
                # Heuristic: if <= 20 unique values, likely classification
                problem_type = "classification" if n_unique <= 20 else "regression"
            except:
                problem_type = "classification"  # default
        
        chord_task_id = dispatch_distributed_training(
            job_id=job.id,
            dataset_path=dataset.file_path,
            target_column=job_data.target_column,
            problem_type=problem_type,
            model_types=model_types_list,
            n_trials_per_model=10,
            time_budget_minutes=job_data.time_budget,
        )
        _celery_tasks[job.id] = chord_task_id
        
        # Start a background poller that watches Celery state and updates DB + WS
        background_tasks.add_task(
            _poll_celery_job,
            job.id,
            chord_task_id,
        )
        
        logger.info(f"Dispatched DISTRIBUTED training job {job.id} across {worker_count} workers (chord={chord_task_id})")
    
    elif use_celery:
        # Single worker - dispatch to Celery but not distributed
        from worker.tasks.training import train_automl
        
        model_types_list = None
        if job_data.model_types:
            model_types_list = job_data.model_types if isinstance(job_data.model_types, list) else None
        
        async_result = train_automl.apply_async(
            kwargs={
                "job_id": job.id,
                "dataset_path": dataset.file_path,
                "target_column": job_data.target_column,
                "problem_type": job_data.problem_type or "auto",
                "time_budget": job_data.time_budget,
                "model_types": model_types_list,
                "n_trials_per_model": 10,
            },
            task_id=job.id,  # use job_id as Celery task_id for easy lookup
            queue="cpu",
        )
        _celery_tasks[job.id] = async_result.id
        
        # Start a background poller that watches Celery state and updates DB + WS
        background_tasks.add_task(
            _poll_celery_job,
            job.id,
            async_result.id,
        )
        
        logger.info(f"Dispatched training job {job.id} to Celery worker (task={async_result.id})")
    else:
        # No Celery workers available – run locally
        cancel_event = threading.Event()
        _running_jobs[job.id] = cancel_event
        
        background_tasks.add_task(
            run_training_job,
            job.id,
            dataset.file_path,
            job_data.target_column,
            job_data.time_budget,
            job_data.problem_type,
            cancel_event,
        )
        
        logger.info(f"Started training job {job.id} locally (no Celery workers available)")
    
    return job


async def run_training_job(
    job_id: str,
    dataset_path: str,
    target_column: str,
    time_budget: int,
    problem_type: str | None,
    cancel_event: threading.Event
):
    """Background task to run Optuna-based AutoML training"""
    import asyncio
    
    # Helper to update job progress with fresh session
    async def update_job_progress(progress: int, stage: str, message: str):
        """Update job progress in database with a fresh session"""
        try:
            async with get_session_context() as update_session:
                result = await update_session.execute(select(Job).where(Job.id == job_id))
                job_to_update = result.scalar_one_or_none()
                if job_to_update:
                    job_to_update.progress = progress
                    job_to_update.current_model = message
                    await update_session.commit()
            
            # Broadcast via WebSocket
            await manager.broadcast({
                "type": "job_update",
                "payload": {
                    "jobId": job_id, 
                    "progress": progress,
                    "stage": stage,
                    "message": message
                }
            })
        except Exception as e:
            logger.error(f"Failed to update progress for job {job_id}: {e}")
    
    async with get_session_context() as session:
        # Get job
        result = await session.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one()
        
        # Update status to running
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        await session.commit()
        
        # Broadcast start
        await manager.broadcast({
            "type": "job_update",
            "payload": {
                "jobId": job_id,
                "status": "running",
                "progress": 0,
                "message": "Training started"
            }
        })
        
        try:
            # Check for early cancellation
            if cancel_event.is_set():
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.utcnow()
                await session.commit()
                return
            
            # Capture the event loop before running in executor
            main_loop = asyncio.get_running_loop()
            
            # Parse model_types from job (stored as JSON string)
            model_types_list = None
            if job.model_types:
                try:
                    model_types_list = json.loads(job.model_types) if isinstance(job.model_types, str) else job.model_types
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Failed to parse model_types for job {job_id}, using all models")
                    model_types_list = None
            
            # Sync progress callback for Optuna (runs in thread)
            def progress_callback(progress: int, stage: str, message: str):
                future = asyncio.run_coroutine_threadsafe(
                    update_job_progress(progress, stage, message),
                    main_loop
                )
                # Don't block, but log any errors
                try:
                    future.result(timeout=1.0)  # Wait briefly for completion
                except Exception as e:
                    logger.warning(f"Progress update async: {e}")
            
            # Run Optuna AutoML in executor (CPU-bound)
            results = await main_loop.run_in_executor(
                None,
                lambda: optuna_automl.train(
                    dataset_path=dataset_path,
                    target_column=target_column,
                    job_id=job_id,
                    time_budget_minutes=time_budget,
                    problem_type=problem_type,
                    model_types=model_types_list,  # Use selected models from job
                    n_trials_per_model=10,
                    cv_folds=5,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                    output_dir=settings.MODELS_DIR / job_id,
                )
            )
            
            # Save trained models to DB
            from models.trained_model import TrainedModel
            
            for model_data in results.models:
                metrics = model_data.get("metrics", {})
                trained_model = TrainedModel(
                    name=model_data["algorithm"],
                    job_id=job_id,
                    dataset_id=job.dataset_id,
                    accuracy=metrics.get("accuracy"),
                    f1_score=metrics.get("f1"),
                    precision=metrics.get("precision"),
                    recall=metrics.get("recall"),
                    auc=metrics.get("auc"),
                    rmse=metrics.get("rmse"),
                    mae=metrics.get("mae"),
                    r2=metrics.get("r2"),
                    training_time=model_data.get("training_time", 0),
                    rank=model_data.get("rank", 0),
                    model_path=model_data.get("model_path"),
                    hyperparameters=json.dumps(model_data.get("params", {})),
                    feature_importance=json.dumps(model_data.get("feature_importance", {}))
                )
                session.add(trained_model)
            
            # Update job as completed
            job.status = JobStatus.COMPLETED
            job.progress = 100
            job.completed_at = datetime.utcnow()
            job.models_completed = len(results.models)
            job.total_models = len(results.models)
            await session.commit()
            
            # Get best model metrics for broadcast
            best_metrics = results.models[0].get("metrics", {}) if results.models else {}
            best_accuracy = best_metrics.get("accuracy", best_metrics.get("r2", 0))
            
            # Broadcast completion
            await manager.broadcast({
                "type": "job_update",
                "payload": {
                    "jobId": job_id,
                    "status": "completed",
                    "progress": 100,
                    "jobName": job.name,
                    "bestModel": results.best_model,
                    "accuracy": best_accuracy * 100 if best_accuracy else 0
                }
            })
            
            logger.info(f"Training job {job_id} completed: {len(results.models)} models, best={results.best_model}")
            
        except Exception as e:
            logger.error(f"Training job {job_id} failed: {e}")
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            await session.commit()
            
            await manager.broadcast({
                "type": "job_update",
                "payload": {
                    "jobId": job_id,
                    "status": "failed",
                    "error": str(e)
                }
            })
        finally:
            # Clean up job tracking
            _running_jobs.pop(job_id, None)


@router.post("/cleanup-stale")
async def cleanup_stale_jobs(
    session: AsyncSession = Depends(get_session)
):
    """Mark stale running/pending jobs as failed (useful after server restart)"""
    result = await session.execute(
        select(Job).where(Job.status.in_([JobStatus.RUNNING, JobStatus.PENDING]))
    )
    stale_jobs = result.scalars().all()
    
    cleaned = []
    for job in stale_jobs:
        # Check if job is actually running (has a cancel_event)
        if job.id not in _running_jobs:
            job.status = JobStatus.FAILED
            job.error_message = "Job interrupted - no active training process"
            job.completed_at = datetime.utcnow()
            cleaned.append(job.id)
            logger.info(f"Cleaned up stale job: {job.id}")
    
    await session.commit()
    return {"cleaned_jobs": cleaned, "count": len(cleaned)}


@router.get("", response_model=list[JobRead])
async def list_jobs(
    session: AsyncSession = Depends(get_session),
    status: JobStatus | None = None,
    limit: int = 100
):
    """List all training jobs"""
    query = select(Job).order_by(Job.created_at.desc()).limit(limit)
    if status:
        query = query.where(Job.status == status)
    
    result = await session.execute(query)
    return result.scalars().all()


@router.get("/{job_id}", response_model=JobRead)
async def get_job(
    job_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Get a single job by ID"""
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise not_found("Job", job_id)
    return job


@router.get("/{job_id}/progress", response_model=JobProgress)
async def get_job_progress(
    job_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Get real-time job progress"""
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise not_found("Job", job_id)
    
    return JobProgress(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        current_model=job.current_model,
        models_completed=job.models_completed,
        total_models=job.total_models,
        logs=[],  # TODO: Implement log streaming
        metrics=None
    )


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Cancel a running job"""
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise not_found("Job", job_id)
    
    if job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
        raise bad_request("Job cannot be cancelled - already completed or failed")
    
    # Signal cancellation – local or Celery
    cancel_event = _running_jobs.get(job_id)
    if cancel_event:
        cancel_event.set()
        logger.info(f"Cancellation signal sent to local job {job_id}")
    
    celery_task_id = _celery_tasks.pop(job_id, None)
    if celery_task_id:
        try:
            from worker.celery_app import celery_app
            celery_app.control.revoke(celery_task_id, terminate=True, signal="SIGTERM")
            logger.info(f"Revoked Celery task {celery_task_id} for job {job_id}")
        except Exception as e:
            logger.warning(f"Failed to revoke Celery task: {e}")
    
    job.status = JobStatus.CANCELLED
    job.completed_at = datetime.utcnow()
    await session.commit()
    
    await manager.broadcast({
        "type": "job_update",
        "payload": {"jobId": job_id, "status": "cancelled"}
    })
    
    return {"status": "cancelled", "id": job_id}


async def start_training_internal(
    dataset_id: str,
    target_column: str,
    problem_type: str,
    time_budget: int,
    job_name: str,
    model_types: list | None,
    session: AsyncSession
) -> Job:
    """
    Internal function to start training - used by scheduled jobs.
    Does not use BackgroundTasks (caller manages background execution).
    """
    from asyncio import get_event_loop
    
    # Verify dataset exists
    result = await session.execute(
        select(Dataset).where(Dataset.id == dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise not_found("Dataset", dataset_id)
    
    # Create job
    job = Job(
        name=job_name,
        dataset_id=dataset_id,
        target_column=target_column,
        time_budget=time_budget,
        model_types=json.dumps(model_types) if model_types else None,
        problem_type=problem_type,
        status=JobStatus.PENDING
    )
    
    session.add(job)
    await session.commit()
    await session.refresh(job)
    
    # Start training in background thread
    cancel_event = threading.Event()
    _running_jobs[job.id] = cancel_event
    
    # Use thread to run training
    import asyncio
    loop = get_event_loop()
    loop.run_in_executor(
        None,
        lambda: asyncio.run(_run_training_wrapper(
            job.id, dataset.file_path, target_column, time_budget, problem_type, cancel_event
        ))
    )
    
    logger.info(f"Started scheduled training job {job.id} for dataset {dataset.name}")
    return job


async def _run_training_wrapper(
    job_id: str,
    dataset_path: str,
    target_column: str,
    time_budget: int,
    problem_type: str,
    cancel_event: threading.Event
):
    """Wrapper for running training in executor"""
    await run_training_job(job_id, dataset_path, target_column, time_budget, problem_type, cancel_event)


async def _poll_celery_job(job_id: str, celery_task_id: str):
    """
    Background poller: watch Celery task state and mirror progress
    back into the Job DB row + WebSocket broadcasts.
    Runs on the FastAPI server, NOT on the worker.
    """
    from celery.result import AsyncResult
    from models.trained_model import TrainedModel
    
    # Mark job as running immediately
    async with get_session_context() as session:
        result = await session.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        if job:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()
            await session.commit()
    
    await manager.broadcast({
        "type": "job_update",
        "payload": {"jobId": job_id, "status": "running", "progress": 0, "message": "Dispatched to worker"}
    })
    
    ar = AsyncResult(celery_task_id)
    last_progress = -1
    
    while True:
        await asyncio.sleep(2)  # poll every 2 seconds
        
        state = ar.state  # PENDING, STARTED, PROGRESS, SUCCESS, FAILURE, REVOKED
        meta = ar.info if isinstance(ar.info, dict) else {}
        
        # Relay progress updates
        if state == "PROGRESS":
            progress = meta.get("progress", 0)
            if progress != last_progress:
                last_progress = progress
                async with get_session_context() as session:
                    res = await session.execute(select(Job).where(Job.id == job_id))
                    j = res.scalar_one_or_none()
                    if j:
                        j.progress = progress
                        j.current_model = meta.get("message", "")
                        await session.commit()
                
                await manager.broadcast({
                    "type": "job_update",
                    "payload": {
                        "jobId": job_id,
                        "progress": progress,
                        "stage": meta.get("stage", ""),
                        "message": meta.get("message", ""),
                    }
                })
        
        elif state == "SUCCESS":
            task_result = ar.result  # dict returned by the Celery task
            task_status = task_result.get("status", "completed")
            
            async with get_session_context() as session:
                res = await session.execute(select(Job).where(Job.id == job_id))
                j = res.scalar_one_or_none()
                if j:
                    if task_status == "completed":
                        # Save trained models from Celery result
                        models = task_result.get("models", [])
                        for model_data in models:
                            metrics = model_data.get("metrics", {})
                            trained_model = TrainedModel(
                                name=model_data.get("algorithm", "unknown"),
                                job_id=job_id,
                                dataset_id=j.dataset_id,
                                accuracy=metrics.get("accuracy"),
                                f1_score=metrics.get("f1"),
                                precision=metrics.get("precision"),
                                recall=metrics.get("recall"),
                                auc=metrics.get("auc"),
                                rmse=metrics.get("rmse"),
                                mae=metrics.get("mae"),
                                r2=metrics.get("r2"),
                                training_time=model_data.get("training_time", 0),
                                rank=model_data.get("rank", 0),
                                model_path=model_data.get("model_path"),
                                hyperparameters=json.dumps(model_data.get("params", {})),
                                feature_importance=json.dumps(model_data.get("feature_importance", {})),
                            )
                            session.add(trained_model)
                        
                        j.status = JobStatus.COMPLETED
                        j.progress = 100
                        j.models_completed = len(models)
                        j.total_models = len(models)
                    elif task_status == "cancelled":
                        j.status = JobStatus.CANCELLED
                    else:  # failed / timeout
                        j.status = JobStatus.FAILED
                        j.error_message = task_result.get("error", "Unknown error")
                    
                    j.completed_at = datetime.utcnow()
                    await session.commit()
            
            best_model = task_result.get("best_model", "")
            await manager.broadcast({
                "type": "job_update",
                "payload": {
                    "jobId": job_id,
                    "status": task_status,
                    "progress": 100,
                    "bestModel": best_model,
                }
            })
            logger.info(f"Celery job {job_id} finished: {task_status}")
            break
        
        elif state == "FAILURE":
            error_msg = str(ar.info) if ar.info else "Worker task failed"
            async with get_session_context() as session:
                res = await session.execute(select(Job).where(Job.id == job_id))
                j = res.scalar_one_or_none()
                if j:
                    j.status = JobStatus.FAILED
                    j.error_message = error_msg
                    j.completed_at = datetime.utcnow()
                    await session.commit()
            
            await manager.broadcast({
                "type": "job_update",
                "payload": {"jobId": job_id, "status": "failed", "error": error_msg}
            })
            logger.error(f"Celery job {job_id} failed: {error_msg}")
            break
        
        elif state == "REVOKED":
            async with get_session_context() as session:
                res = await session.execute(select(Job).where(Job.id == job_id))
                j = res.scalar_one_or_none()
                if j:
                    j.status = JobStatus.CANCELLED
                    j.completed_at = datetime.utcnow()
                    await session.commit()
            
            await manager.broadcast({
                "type": "job_update",
                "payload": {"jobId": job_id, "status": "cancelled"}
            })
            logger.info(f"Celery job {job_id} was revoked")
            break
    
    # Cleanup
    _celery_tasks.pop(job_id, None)
