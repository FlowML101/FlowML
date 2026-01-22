"""
Results routes - get trained models, export, predict
Optimized with LRU caching for model loading
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
import joblib
import time
from pathlib import Path
from functools import lru_cache
from typing import Any
import asyncio

from database import get_session
from models.trained_model import (
    TrainedModel, TrainedModelRead, ModelLeaderboard,
    PredictionRequest, PredictionResponse
)
from models.job import Job

router = APIRouter(prefix="/results", tags=["results"])


# ============ Model Cache ============
# LRU cache for loaded models - keeps last 10 models in memory
@lru_cache(maxsize=10)
def _load_model_cached(model_path: str, mtime: float) -> Any:
    """
    Load model with caching. mtime is used to invalidate cache if file changes.
    """
    return joblib.load(model_path)


def get_cached_model(model_path: str) -> Any:
    """Get model from cache or load it"""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    mtime = path.stat().st_mtime
    return _load_model_cached(model_path, mtime)


def clear_model_cache():
    """Clear the model cache (call when models are retrained)"""
    _load_model_cached.cache_clear()


@router.get("/job/{job_id}", response_model=list[TrainedModelRead])
async def get_job_results(
    job_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Get all trained models for a job"""
    # Verify job exists
    job_result = await session.execute(select(Job).where(Job.id == job_id))
    job = job_result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get models
    result = await session.execute(
        select(TrainedModel)
        .where(TrainedModel.job_id == job_id)
        .order_by(TrainedModel.rank)
    )
    return result.scalars().all()


@router.get("/job/{job_id}/leaderboard", response_model=list[ModelLeaderboard])
async def get_leaderboard(
    job_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Get model leaderboard for a job"""
    result = await session.execute(
        select(TrainedModel)
        .where(TrainedModel.job_id == job_id)
        .order_by(TrainedModel.rank)
    )
    models = result.scalars().all()
    
    return [
        ModelLeaderboard(
            rank=m.rank,
            id=m.id,
            name=m.name,
            accuracy=m.accuracy,
            f1_score=m.f1_score,
            precision=m.precision,
            recall=m.recall,
            auc=m.auc,
            training_time=m.training_time
        )
        for m in models
    ]


@router.get("/model/{model_id}", response_model=TrainedModelRead)
async def get_model(
    model_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Get a single trained model"""
    result = await session.execute(
        select(TrainedModel).where(TrainedModel.id == model_id)
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.get("/model/{model_id}/download")
async def download_model(
    model_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Download a trained model file"""
    result = await session.execute(
        select(TrainedModel).where(TrainedModel.id == model_id)
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not model.model_path or not Path(model.model_path).exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    
    return FileResponse(
        model.model_path,
        filename=f"{model.name}_{model.id}.pkl",
        media_type="application/octet-stream"
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    session: AsyncSession = Depends(get_session)
):
    """Make a prediction using a trained model (with caching)"""
    result = await session.execute(
        select(TrainedModel).where(TrainedModel.id == request.model_id)
    )
    model_record = result.scalar_one_or_none()
    if not model_record:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not model_record.model_path or not Path(model_record.model_path).exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    
    # Load and predict with caching
    start_time = time.time()
    try:
        # Use cached model loader
        model = get_cached_model(model_record.model_path)
        
        # PyCaret models expect DataFrame - import once at module level would be better
        # but keeping here for clarity
        import pandas as pd
        input_df = pd.DataFrame([request.features])
        
        # Run prediction in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(
            None, lambda: model.predict(input_df)[0]
        )
        
        # Get probability if classification
        probability = None
        if hasattr(model, 'predict_proba'):
            proba = await loop.run_in_executor(
                None, lambda: model.predict_proba(input_df)[0]
            )
            probability = float(max(proba))
        
        latency = (time.time() - start_time) * 1000
        
        # Convert numpy types to Python types
        if hasattr(prediction, 'item'):
            prediction = prediction.item()
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            confidence=probability,
            model_name=model_record.name,
            latency_ms=round(latency, 2)
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/models", response_model=list[TrainedModelRead])
async def list_all_models(
    session: AsyncSession = Depends(get_session),
    limit: int = 100
):
    """List all trained models across all jobs"""
    result = await session.execute(
        select(TrainedModel)
        .order_by(TrainedModel.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()
