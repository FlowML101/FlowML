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


@router.get("/model/{model_id}/metadata")
async def get_model_metadata(
    model_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Get model metadata including features, types, and preprocessing info"""
    result = await session.execute(
        select(TrainedModel).where(TrainedModel.id == model_id)
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not model.model_path or not Path(model.model_path).exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    
    # Try to load metadata.json file
    model_path = Path(model.model_path)
    metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
    
    if metadata_path.exists():
        import json
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return metadata
    else:
        # Fallback: extract basic info from model file
        try:
            model_data = get_cached_model(str(model_path))
            if isinstance(model_data, dict):
                preprocessor = model_data.get("preprocessor")
                return {
                    "model_id": model_id,
                    "problem_type": model_data.get("problem_type", "unknown"),
                    "feature_names": preprocessor.original_features if preprocessor else model_data.get("feature_names", []),
                    "numeric_features": preprocessor.numeric_cols if preprocessor else [],
                    "categorical_features": preprocessor.categorical_cols if preprocessor else [],
                    "low_cardinality_features": preprocessor.low_card_cols if preprocessor else [],
                    "categorical_modes": preprocessor.categorical_modes if preprocessor else {},
                    "onehot_categories": preprocessor.onehot_categories if preprocessor else {},
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read metadata: {str(e)}")


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
        # Use cached model loader - returns dict with model, preprocessor, feature_names, etc.
        model_data = get_cached_model(model_record.model_path)
        
        # Extract components from the dict
        if isinstance(model_data, dict):
            model = model_data["model"]
            preprocessor = model_data.get("preprocessor", None)
            feature_names = model_data.get("feature_names", [])
            label_encoder = model_data.get("label_encoder", None)
        else:
            # Legacy support for models saved directly
            model = model_data
            preprocessor = None
            feature_names = []
            label_encoder = None
        
        import pandas as pd
        
        # Create DataFrame from input features
        if preprocessor:
            # Use original feature names (before one-hot encoding)
            input_df = pd.DataFrame([request.features])
            
            # Apply the same preprocessing as during training
            loop = asyncio.get_event_loop()
            input_processed = await loop.run_in_executor(
                None, lambda: preprocessor.transform(input_df)
            )
        else:
            # Legacy: no preprocessor available, use features as-is
            if feature_names:
                input_processed = pd.DataFrame([{k: request.features.get(k) for k in feature_names}])
            else:
                input_processed = pd.DataFrame([request.features])
        
        # Run prediction in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(
            None, lambda: model.predict(input_processed)[0]
        )
        
        # Get probability if classification
        probability = None
        if hasattr(model, 'predict_proba'):
            proba = await loop.run_in_executor(
                None, lambda: model.predict_proba(input_processed)[0]
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


@router.post("/models/{model_id}/predict", response_model=PredictionResponse)
async def predict_by_model_id(
    model_id: str,
    features: dict,
    session: AsyncSession = Depends(get_session)
):
    """
    Make a prediction using a trained model (frontend-friendly endpoint).
    This endpoint takes model_id in the URL path instead of request body.
    """
    # Reuse the existing predict function by creating a PredictionRequest
    request = PredictionRequest(
        model_id=model_id,
        features=features
    )
    return await predict(request, session)


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


@router.get("/model/{model_id}/feature-importance")
async def get_feature_importance(
    model_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Get feature importance for a trained model"""
    result = await session.execute(
        select(TrainedModel).where(TrainedModel.id == model_id)
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Check if we have stored feature importance
    if model.feature_importance:
        import json
        try:
            importance_data = json.loads(model.feature_importance)
            # Convert to standardized format
            if isinstance(importance_data, dict):
                features = [
                    {"feature": k, "importance": v, "rank": i + 1}
                    for i, (k, v) in enumerate(
                        sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
                    )
                ]
                return {"model_id": model_id, "model_name": model.name, "features": features}
        except json.JSONDecodeError:
            pass
    
    # Try to extract from the actual model file
    if model.model_path and Path(model.model_path).exists():
        try:
            loaded_model = get_cached_model(model.model_path)
            
            # Check for feature_importances_ attribute (tree-based models)
            if hasattr(loaded_model, 'feature_importances_'):
                importances = loaded_model.feature_importances_
                # Try to get feature names
                feature_names = None
                if hasattr(loaded_model, 'feature_names_in_'):
                    feature_names = list(loaded_model.feature_names_in_)
                elif hasattr(loaded_model, 'feature_name_'):
                    feature_names = list(loaded_model.feature_name_)
                
                if feature_names is None:
                    feature_names = [f"Feature_{i}" for i in range(len(importances))]
                
                features = [
                    {"feature": name, "importance": float(imp), "rank": i + 1}
                    for i, (name, imp) in enumerate(
                        sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
                    )
                ]
                return {"model_id": model_id, "model_name": model.name, "features": features}
            
            # For linear models, use coefficients
            if hasattr(loaded_model, 'coef_'):
                coefs = loaded_model.coef_
                if len(coefs.shape) > 1:
                    coefs = coefs[0]  # For multi-class, use first class
                
                feature_names = None
                if hasattr(loaded_model, 'feature_names_in_'):
                    feature_names = list(loaded_model.feature_names_in_)
                
                if feature_names is None:
                    feature_names = [f"Feature_{i}" for i in range(len(coefs))]
                
                # Use absolute value for importance
                features = [
                    {"feature": name, "importance": float(abs(c)), "rank": i + 1}
                    for i, (name, c) in enumerate(
                        sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
                    )
                ]
                return {"model_id": model_id, "model_name": model.name, "features": features}
                
        except Exception as e:
            pass  # Fall through to default
    
    # Return empty if no importance available
    return {"model_id": model_id, "model_name": model.name, "features": [], "message": "Feature importance not available for this model"}


@router.get("/model/{model_id}/confusion-matrix")
async def get_confusion_matrix(
    model_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Get confusion matrix for a classification model"""
    result = await session.execute(
        select(TrainedModel).where(TrainedModel.id == model_id)
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Check if we have stored confusion matrix
    if model.confusion_matrix:
        import json
        try:
            cm_data = json.loads(model.confusion_matrix)
            return {
                "model_id": model_id,
                "model_name": model.name,
                "matrix": cm_data,
                "labels": ["Negative", "Positive"]  # Default binary
            }
        except json.JSONDecodeError:
            pass
    
    # Generate approximation from metrics if no stored matrix
    if model.accuracy is not None:
        acc = model.accuracy
        # Approximate confusion matrix from accuracy
        # Assumes balanced classes - this is a fallback
        tp = int(acc * 50)
        tn = int(acc * 50)
        fp = int((1 - acc) * 25)
        fn = int((1 - acc) * 25)
        
        return {
            "model_id": model_id,
            "model_name": model.name,
            "matrix": [[tn, fp], [fn, tp]],
            "labels": ["Negative", "Positive"],
            "note": "Approximated from accuracy metric"
        }
    
    return {
        "model_id": model_id,
        "model_name": model.name,
        "matrix": None,
        "message": "Confusion matrix not available"
    }


@router.post("/compare")
async def compare_models(
    model_ids: list[str],
    session: AsyncSession = Depends(get_session)
):
    """Compare multiple models side by side"""
    if len(model_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 models to compare")
    if len(model_ids) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 models can be compared")
    
    result = await session.execute(
        select(TrainedModel).where(TrainedModel.id.in_(model_ids))
    )
    models = result.scalars().all()
    
    if len(models) != len(model_ids):
        raise HTTPException(status_code=404, detail="One or more models not found")
    
    # Build comparison data
    comparison = []
    for model in models:
        model_data = {
            "id": model.id,
            "name": model.name,
            "metrics": {
                "accuracy": model.accuracy,
                "f1_score": model.f1_score,
                "precision": model.precision,
                "recall": model.recall,
                "auc": model.auc,
                "rmse": model.rmse,
                "mae": model.mae,
                "r2": model.r2
            },
            "training_time": model.training_time,
            "rank": model.rank,
            "created_at": model.created_at.isoformat()
        }
        
        # Add hyperparameters if available
        if model.hyperparameters:
            import json
            try:
                model_data["hyperparameters"] = json.loads(model.hyperparameters)
            except:
                pass
        
        comparison.append(model_data)
    
    # Calculate best model for each metric
    metrics_ranking = {}
    metric_names = ["accuracy", "f1_score", "precision", "recall", "auc"]
    
    for metric in metric_names:
        values = [(m["id"], m["metrics"].get(metric)) for m in comparison if m["metrics"].get(metric) is not None]
        if values:
            best = max(values, key=lambda x: x[1])
            metrics_ranking[metric] = {"best_model_id": best[0], "best_value": best[1]}
    
    return {
        "models": comparison,
        "metrics_ranking": metrics_ranking,
        "recommendation": comparison[0]["id"] if comparison else None  # First is usually best ranked
    }


@router.get("/model/{model_id}/hyperparameters")
async def get_hyperparameters(
    model_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Get hyperparameters for a trained model"""
    result = await session.execute(
        select(TrainedModel).where(TrainedModel.id == model_id)
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if model.hyperparameters:
        import json
        try:
            return {
                "model_id": model_id,
                "model_name": model.name,
                "hyperparameters": json.loads(model.hyperparameters)
            }
        except json.JSONDecodeError:
            pass
    
    # Try to extract from model file
    if model.model_path and Path(model.model_path).exists():
        try:
            loaded_model = get_cached_model(model.model_path)
            if hasattr(loaded_model, 'get_params'):
                params = loaded_model.get_params()
                # Filter out None and callable params
                params = {k: v for k, v in params.items() 
                         if v is not None and not callable(v)}
                return {
                    "model_id": model_id,
                    "model_name": model.name,
                    "hyperparameters": params
                }
        except:
            pass
    
    return {
        "model_id": model_id,
        "model_name": model.name,
        "hyperparameters": {},
        "message": "Hyperparameters not available"
    }
