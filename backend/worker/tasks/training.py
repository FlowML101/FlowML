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
            df = pl.read_csv(
                dataset_path,
                infer_schema_length=None,
                ignore_errors=True
            ).to_pandas()
        
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


# =============================================================================
# DISTRIBUTED TRAINING - Split compute across workers
# =============================================================================

@shared_task(
    bind=True,
    name="worker.tasks.training.train_one_model",
    track_started=True,
    time_limit=1800,  # 30 min max per model
)
def train_one_model(
    self,
    job_id: str,
    dataset_path: str,
    target_column: str,
    model_name: str,
    problem_type: str,
    n_trials: int = 10,
    cv_folds: int = 5,
    time_budget_seconds: float = 120,
) -> Dict[str, Any]:
    """
    Train ONE model type - designed to run on a distributed worker.
    
    Each worker picks up one model to train, enabling parallel training
    across multiple machines.
    
    Args:
        job_id: Parent job ID
        dataset_path: Path to training data
        target_column: Target column name
        model_name: e.g. "random_forest", "xgboost"
        problem_type: "classification" or "regression"
        n_trials: Optuna trials for HPO
        cv_folds: Cross-validation folds
        time_budget_seconds: Max time for this model
    
    Returns:
        Dict with model metrics and info
    """
    import socket
    from services.optuna_automl import optuna_automl
    
    worker_hostname = socket.gethostname()
    start_time = time.time()
    
    logger.info(f"[{job_id}] Worker {worker_hostname} training: {model_name}")
    
    try:
        # Update task state
        self.update_state(
            state="PROGRESS",
            meta={
                "job_id": job_id,
                "model": model_name,
                "worker": worker_hostname,
                "stage": "loading",
            }
        )
        
        # Load and preprocess data (each worker does this independently)
        import polars as pl
        import numpy as np
        
        if dataset_path.endswith('.parquet'):
            df = pl.read_parquet(dataset_path).to_pandas()
        else:
            df = pl.read_csv(
                dataset_path,
                infer_schema_length=None,
                ignore_errors=True
            ).to_pandas()
        
        # Preprocess
        X, y, feature_names, label_encoder, preprocessor = optuna_automl.preprocess_data(
            df, target_column
        )
        
        # Get model definition
        model_defs = (
            optuna_automl.CLASSIFICATION_MODELS if problem_type == "classification" 
            else optuna_automl.REGRESSION_MODELS
        )
        
        if model_name not in model_defs:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_def = model_defs[model_name]
        
        # Scoring
        if problem_type == "classification":
            scoring = "accuracy"
            direction = "maximize"
        else:
            scoring = "neg_mean_squared_error"
            direction = "maximize"
        
        self.update_state(
            state="PROGRESS",
            meta={
                "job_id": job_id,
                "model": model_name,
                "worker": worker_hostname,
                "stage": "training",
            }
        )
        
        # Train the model
        result = optuna_automl._train_single_model(
            model_name=model_name,
            model_def=model_def,
            X=X,
            y=y,
            feature_names=feature_names,
            scoring=scoring,
            direction=direction,
            n_trials=n_trials,
            cv_folds=cv_folds,
            time_budget=time_budget_seconds,
            cancel_event=None,
        )
        
        elapsed = time.time() - start_time
        logger.info(f"[{job_id}] Worker {worker_hostname} completed {model_name} in {elapsed:.1f}s")
        
        return {
            "status": "completed",
            "job_id": job_id,
            "model_name": model_name,
            "algorithm": result.algorithm,
            "metrics": result.metrics,
            "params": result.params,
            "cv_scores": result.cv_scores,
            "training_time": result.training_time,
            "feature_importance": result.feature_importance,
            "worker": worker_hostname,
        }
        
    except Exception as e:
        logger.error(f"[{job_id}] Worker {worker_hostname} failed on {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "failed",
            "job_id": job_id,
            "model_name": model_name,
            "error": str(e),
            "worker": worker_hostname,
        }


@shared_task(
    bind=True,
    name="worker.tasks.training.aggregate_distributed_results",
    track_started=True,
)
def aggregate_distributed_results(
    self,
    results: List[Dict[str, Any]],
    job_id: str,
    problem_type: str,
) -> Dict[str, Any]:
    """
    Aggregate results from distributed model training.
    Called after all train_one_model tasks complete.
    
    Args:
        results: List of results from train_one_model tasks
        job_id: Parent job ID
        problem_type: "classification" or "regression"
    
    Returns:
        Aggregated result like train_automl returns
    """
    import joblib
    
    logger.info(f"[{job_id}] Aggregating {len(results)} model results")
    
    # Filter successful results
    successful = [r for r in results if r.get("status") == "completed"]
    failed = [r for r in results if r.get("status") == "failed"]
    
    if not successful:
        return {
            "job_id": job_id,
            "status": "failed",
            "error": "All model training tasks failed",
            "failed_models": [r.get("model_name") for r in failed],
        }
    
    # Rank by primary metric
    if problem_type == "classification":
        successful.sort(key=lambda r: r.get("metrics", {}).get("accuracy", 0), reverse=True)
    else:
        successful.sort(key=lambda r: r.get("metrics", {}).get("r2", 0), reverse=True)
    
    # Prepare output
    models = []
    for rank, r in enumerate(successful, 1):
        models.append({
            "rank": rank,
            "algorithm": r.get("algorithm", r.get("model_name")),
            "metrics": r.get("metrics", {}),
            "params": r.get("params", {}),
            "cv_scores": r.get("cv_scores", []),
            "training_time": r.get("training_time", 0),
            "worker": r.get("worker", "unknown"),
            "feature_importance": r.get("feature_importance"),
        })
    
    best_model = models[0]["algorithm"] if models else None
    total_time = sum(r.get("training_time", 0) for r in successful)
    
    logger.info(f"[{job_id}] Best model: {best_model}")
    logger.info(f"[{job_id}] Successful: {len(successful)}, Failed: {len(failed)}")
    
    return {
        "job_id": job_id,
        "status": "completed",
        "problem_type": problem_type,
        "best_model": best_model,
        "models": models,
        "n_models_trained": len(successful),
        "n_models_failed": len(failed),
        "failed_models": [r.get("model_name") for r in failed],
        "total_training_time": total_time,
        "distributed": True,
        "completed_at": datetime.utcnow().isoformat(),
    }


def dispatch_distributed_training(
    job_id: str,
    dataset_path: str,
    target_column: str,
    problem_type: str,
    model_types: Optional[List[str]] = None,
    n_trials_per_model: int = 10,
    time_budget_minutes: int = 5,
) -> str:
    """
    Dispatch distributed training across available workers.
    
    Uses Celery's chord: runs train_one_model in parallel, 
    then aggregates with aggregate_distributed_results.
    
    Args:
        job_id: Job identifier  
        dataset_path: Path to dataset
        target_column: Target column
        problem_type: "classification" or "regression"
        model_types: List of models to train, or None for all
        n_trials_per_model: Optuna trials per model
        time_budget_minutes: Total time budget
    
    Returns:
        Chord task ID for tracking
    """
    from celery import chord
    
    # Determine which models to train
    from services.optuna_automl import optuna_automl
    
    if problem_type == "classification":
        all_models = list(optuna_automl.CLASSIFICATION_MODELS.keys())
    else:
        all_models = list(optuna_automl.REGRESSION_MODELS.keys())
    
    if model_types:
        models_to_train = [m for m in model_types if m in all_models]
    else:
        models_to_train = all_models
    
    # Calculate time budget per model
    time_per_model = (time_budget_minutes * 60) / len(models_to_train)
    
    logger.info(f"[{job_id}] Dispatching distributed training for {len(models_to_train)} models")
    logger.info(f"[{job_id}] Models: {models_to_train}")
    logger.info(f"[{job_id}] Time per model: {time_per_model:.0f}s")
    
    # Create tasks for each model
    tasks = [
        train_one_model.s(
            job_id=job_id,
            dataset_path=dataset_path,
            target_column=target_column,
            model_name=model_name,
            problem_type=problem_type,
            n_trials=n_trials_per_model,
            cv_folds=5,
            time_budget_seconds=time_per_model,
        )
        for model_name in models_to_train
    ]
    
    # Create chord: run all tasks in parallel, then aggregate
    callback = aggregate_distributed_results.s(
        job_id=job_id,
        problem_type=problem_type,
    )
    
    result = chord(tasks)(callback)
    
    logger.info(f"[{job_id}] Chord dispatched with task ID: {result.id}")
    
    return result.id
