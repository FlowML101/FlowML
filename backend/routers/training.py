"""
Training routes - start jobs, get status, cancel
Optimized with proper async handling and job management
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from datetime import datetime
import json
import threading
from typing import Dict
from loguru import logger

from database import get_session, get_session_context
from models.job import Job, JobCreate, JobRead, JobStatus, JobProgress
from models.dataset import Dataset
from services.optuna_automl import optuna_automl, OptunaAutoML
from services.websocket_manager import manager
from exceptions import not_found, bad_request, conflict
from config import settings

router = APIRouter(prefix="/training", tags=["training"])

# Track running jobs for cancellation
_running_jobs: Dict[str, threading.Event] = {}


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
    
    # Start training in background
    cancel_event = threading.Event()
    _running_jobs[job.id] = cancel_event
    
    background_tasks.add_task(
        run_training_job,
        job.id,
        dataset.file_path,
        job_data.target_column,
        job_data.time_budget,
        job_data.problem_type,
        cancel_event
    )
    
    logger.info(f"Started training job {job.id} for dataset {dataset.name}")
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
                    model_types=None,  # All models
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
    
    # Signal cancellation to the running task
    cancel_event = _running_jobs.get(job_id)
    if cancel_event:
        cancel_event.set()
        logger.info(f"Cancellation signal sent to job {job_id}")
    
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
