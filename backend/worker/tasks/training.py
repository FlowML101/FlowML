"""
Training Tasks - Celery tasks for ML model training using Optuna

This module provides Celery tasks for:
- CPU-based AutoML training
- GPU-accelerated training (future)
- Model optimization with Optuna
"""
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from celery import shared_task, current_task
from celery.exceptions import SoftTimeLimitExceeded
from loguru import logger
import threading

from config import settings


# Task state tracking
_active_tasks: Dict[str, threading.Event] = {}


def get_cancel_event(job_id: str) -> threading.Event:
    """Get or create cancellation event for a job"""
    if job_id not in _active_tasks:
        _active_tasks[job_id] = threading.Event()
    return _active_tasks[job_id]


def cancel_job(job_id: str) -> bool:
    """Signal a job to cancel"""
    if job_id in _active_tasks:
        _active_tasks[job_id].set()
        return True
    return False


def cleanup_job(job_id: str):
    """Cleanup cancellation event after job completes"""
    _active_tasks.pop(job_id, None)


@shared_task(
    bind=True,
    name="worker.tasks.training.train_automl",
    max_retries=1,
    default_retry_delay=60,
    track_started=True,
    soft_time_limit=3600,  # 1 hour soft limit
    time_limit=3900,  # 1 hour 5 min hard limit
)
def train_automl(
    self,
    job_id: str,
    dataset_path: str,
    target_column: str,
    problem_type: str = "auto",
    time_budget: int = 5,
    model_types: Optional[List[str]] = None,
    n_trials_per_model: int = 10,
) -> Dict[str, Any]:
    """
    Train ML models using Optuna-based AutoML.
    
    This is the main training task that:
    1. Loads the dataset
    2. Auto-detects problem type (if needed)
    3. Runs HPO with Optuna for each model type
    4. Saves the best models
    5. Returns metrics and model paths
    
    Args:
        job_id: Unique job identifier
        dataset_path: Path to training CSV/Parquet file
        target_column: Target variable name
        problem_type: "classification", "regression", or "auto"
        time_budget: Training time budget in minutes
        model_types: List of model names to try, or None for all
        n_trials_per_model: Optuna trials per model type
    
    Returns:
        Dict with trained models info, metrics, and paths
    """
    from services.optuna_automl import optuna_automl
    
    start_time = time.time()
    logger.info(f"[{job_id}] Starting AutoML training task")
    logger.info(f"[{job_id}] Dataset: {dataset_path}")
    logger.info(f"[{job_id}] Target: {target_column}")
    logger.info(f"[{job_id}] Time budget: {time_budget} minutes")
    
    # Get cancellation event
    cancel_event = get_cancel_event(job_id)
    
    # Progress callback to update Celery state
    def progress_callback(progress: int, stage: str, message: str):
        self.update_state(
            state="PROGRESS",
            meta={
                "job_id": job_id,
                "progress": progress,
                "stage": stage,
                "message": message,
            }
        )
    
    try:
        # Update initial state
        self.update_state(
            state="PROGRESS",
            meta={"job_id": job_id, "progress": 0, "stage": "starting"}
        )
        
        # Verify dataset exists
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Output directory for models
        output_dir = settings.MODELS_DIR / job_id
        
        # Run AutoML training
        result = optuna_automl.train(
            dataset_path=dataset_path,
            target_column=target_column,
            job_id=job_id,
            time_budget_minutes=time_budget,
            problem_type=problem_type if problem_type != "auto" else None,
            model_types=model_types,
            n_trials_per_model=n_trials_per_model,
            cv_folds=5,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            output_dir=output_dir,
        )
        
        # Final update
        self.update_state(
            state="PROGRESS",
            meta={"job_id": job_id, "progress": 100, "stage": "completed"}
        )
        
        elapsed = time.time() - start_time
        logger.info(f"[{job_id}] Training completed in {elapsed:.1f}s")
        logger.info(f"[{job_id}] Best model: {result.best_model}")
        logger.info(f"[{job_id}] Models trained: {len(result.models)}")
        
        return {
            "job_id": job_id,
            "status": "completed",
            "problem_type": result.problem_type,
            "best_model": result.best_model,
            "models": result.models,
            "n_trials": result.n_trials,
            "dataset_info": result.dataset_info,
            "training_time_seconds": elapsed,
            "completed_at": datetime.utcnow().isoformat(),
        }
        
    except InterruptedError:
        logger.warning(f"[{job_id}] Training cancelled by user")
        return {
            "job_id": job_id,
            "status": "cancelled",
            "message": "Training cancelled by user",
            "training_time_seconds": time.time() - start_time,
        }
        
    except SoftTimeLimitExceeded:
        logger.warning(f"[{job_id}] Training exceeded time limit")
        return {
            "job_id": job_id,
            "status": "timeout",
            "error": "Training exceeded time limit",
            "training_time_seconds": time.time() - start_time,
        }
        
    except FileNotFoundError as e:
        logger.error(f"[{job_id}] Dataset not found: {e}")
        return {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
        }
        
    except Exception as e:
        logger.error(f"[{job_id}] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
        }
        
    finally:
        cleanup_job(job_id)


@shared_task(
    bind=True,
    name="worker.tasks.training.train_single_model",
    track_started=True,
)
def train_single_model(
    self,
    job_id: str,
    dataset_path: str,
    target_column: str,
    algorithm: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    problem_type: str = "auto",
) -> Dict[str, Any]:
    """
    Train a single model with specific hyperparameters (no HPO).
    
    Useful when you know exactly what model and params you want.
    """
    import pandas as pd
    import polars as pl
    import joblib
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
    
    logger.info(f"[{job_id}] Training single model: {algorithm}")
    start_time = time.time()
    
    try:
        # Load data
        if dataset_path.endswith('.parquet'):
            df = pl.read_parquet(dataset_path).to_pandas()
        else:
            df = pl.read_csv(dataset_path).to_pandas()
        
        # Sample if dataset is too large (prevent memory issues)
        MAX_ROWS = 100000
        MAX_ONEHOT_CARDINALITY = 50
        
        if len(df) > MAX_ROWS:
            logger.info(f"[{job_id}] Sampling {MAX_ROWS} rows from {len(df)} total rows")
            df = df.sample(n=MAX_ROWS, random_state=42)
        
        # Basic preprocessing
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical target
        from sklearn.preprocessing import LabelEncoder
        label_encoder = None
        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        
        # Smart preprocessing for features - handle high cardinality columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Fill missing values
        for col in numeric_cols:
            X[col] = X[col].fillna(X[col].median())
        
        # Handle categorical columns with cardinality limit
        low_cardinality_cols = []
        for col in categorical_cols:
            X[col] = X[col].fillna("MISSING")
            n_unique = X[col].nunique()
            if n_unique <= MAX_ONEHOT_CARDINALITY:
                low_cardinality_cols.append(col)
            else:
                # Label encode high cardinality columns
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                logger.info(f"[{job_id}] Column '{col}' has {n_unique} unique values - label encoded")
        
        # One-hot encode only low cardinality columns
        if low_cardinality_cols:
            X = pd.get_dummies(X, columns=low_cardinality_cols, drop_first=True)
        
        X = X.fillna(0)
        
        # Auto-detect problem type
        if problem_type == "auto":
            if label_encoder or df[target_column].nunique() <= 10:
                problem_type = "classification"
            else:
                problem_type = "regression"
        
        # Get model class
        model = _get_model_by_name(algorithm, problem_type, hyperparameters or {})
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics
        if problem_type == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred, average='weighted')),
            }
        else:
            metrics = {
                "r2": float(r2_score(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            }
        
        # Save model
        output_dir = settings.MODELS_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"model_{algorithm}.joblib"
        
        joblib.dump({
            "model": model,
            "feature_names": X.columns.tolist(),
            "label_encoder": label_encoder,
            "problem_type": problem_type,
        }, model_path)
        
        elapsed = time.time() - start_time
        
        return {
            "job_id": job_id,
            "status": "completed",
            "algorithm": algorithm,
            "problem_type": problem_type,
            "metrics": metrics,
            "model_path": str(model_path),
            "training_time_seconds": elapsed,
        }
        
    except Exception as e:
        logger.error(f"[{job_id}] Single model training failed: {e}")
        return {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
        }


def _get_model_by_name(name: str, problem_type: str, params: dict):
    """Get a sklearn model instance by name"""
    from sklearn.linear_model import LogisticRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    
    models = {
        "classification": {
            "logistic_regression": LogisticRegression,
            "random_forest": RandomForestClassifier,
            "gradient_boosting": GradientBoostingClassifier,
        },
        "regression": {
            "ridge": Ridge,
            "lasso": Lasso,
            "random_forest": RandomForestRegressor,
            "gradient_boosting": GradientBoostingRegressor,
        }
    }
    
    try:
        import xgboost as xgb
        models["classification"]["xgboost"] = xgb.XGBClassifier
        models["regression"]["xgboost"] = xgb.XGBRegressor
    except ImportError:
        pass
    
    try:
        import lightgbm as lgb
        models["classification"]["lightgbm"] = lgb.LGBMClassifier
        models["regression"]["lightgbm"] = lgb.LGBMRegressor
    except ImportError:
        pass
    
    model_class = models.get(problem_type, {}).get(name)
    if model_class is None:
        raise ValueError(f"Unknown model: {name} for {problem_type}")
    
    return model_class(**params)


@shared_task(name="worker.tasks.training.get_training_status")
def get_training_status(job_id: str) -> Dict[str, Any]:
    """Get the current status of a training job"""
    from celery.result import AsyncResult
    
    result = AsyncResult(job_id)
    
    return {
        "job_id": job_id,
        "state": result.state,
        "info": result.info if result.info else {},
    }
