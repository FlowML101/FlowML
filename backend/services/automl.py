"""
AutoML Service - PyCaret wrapper for automated machine learning
Optimized with cancellation support and better error handling
"""
import asyncio
from pathlib import Path
from typing import Callable, Optional
from datetime import datetime
from loguru import logger
import json
import threading

from config import settings


class AutoMLService:
    """
    AutoML service using PyCaret.
    Handles both classification and regression tasks automatically.
    """
    
    def __init__(self):
        self.setup_complete = False
        self.experiment = None
        self.problem_type = None
    
    async def train(
        self,
        dataset_path: str,
        target_column: str,
        time_budget: int = 5,
        problem_type: Optional[str] = None,
        job_id: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        cancel_event: Optional[threading.Event] = None
    ) -> dict:
        """
        Run AutoML training on a dataset.
        
        Args:
            dataset_path: Path to CSV file
            target_column: Column to predict
            time_budget: Training time in minutes
            problem_type: "classification", "regression", or None (auto-detect)
            job_id: Job ID for saving models
            progress_callback: Async function to report progress
            cancel_event: Threading event to check for cancellation
        
        Returns:
            dict with "models" list containing trained model info
        """
        # Run in executor to not block event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._train_sync,
            dataset_path,
            target_column,
            time_budget,
            problem_type,
            job_id,
            cancel_event
        )
    
    def _train_sync(
        self,
        dataset_path: str,
        target_column: str,
        time_budget: int,
        problem_type: Optional[str],
        job_id: Optional[str],
        cancel_event: Optional[threading.Event] = None
    ) -> dict:
        """Synchronous training logic with cancellation support"""
        import pandas as pd
        import polars as pl
        
        def check_cancelled():
            """Check if job was cancelled"""
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Training cancelled by user")
        
        logger.info(f"Starting AutoML training for job {job_id}")
        logger.info(f"Dataset: {dataset_path}, Target: {target_column}")
        
        check_cancelled()
        
        # Load data with Polars (fast), convert to Pandas (PyCaret needs it)
        df = pl.read_csv(dataset_path).to_pandas()
        logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
        check_cancelled()
        
        # Load data with Polars (fast), convert to Pandas (PyCaret needs it)
        df = pl.read_csv(dataset_path).to_pandas()
        logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Auto-detect problem type if not specified
        if problem_type is None or problem_type == "auto":
            target_dtype = df[target_column].dtype
            unique_values = df[target_column].nunique()
            
            if target_dtype == 'object' or unique_values <= 10:
                problem_type = "classification"
            else:
                problem_type = "regression"
            
            logger.info(f"Auto-detected problem type: {problem_type}")
        
        self.problem_type = problem_type
        
        # Import appropriate PyCaret module
        if problem_type == "classification":
            from pycaret.classification import (
                setup, compare_models, pull, save_model, 
                get_config, create_model
            )
        else:
            from pycaret.regression import (
                setup, compare_models, pull, save_model,
                get_config, create_model
            )
        
        # Setup PyCaret experiment
        logger.info("Setting up PyCaret experiment...")
        setup(
            data=df,
            target=target_column,
            session_id=42,
            verbose=False,
            html=False,
            log_experiment=False,
            system_log=False
        )
        self.setup_complete = True
        logger.info("PyCaret setup complete")
        
        # Compare models with time budget
        logger.info(f"Comparing models (budget: {time_budget} min)...")
        best_models = compare_models(
            n_select=5,  # Top 5 models
            budget_time=time_budget,
            verbose=False
        )
        
        # Handle single model return
        if not isinstance(best_models, list):
            best_models = [best_models]
        
        # Get leaderboard
        leaderboard = pull()
        logger.info(f"Training complete. {len(best_models)} models trained.")
        
        # Process results
        results = []
        model_dir = settings.MODELS_DIR / (job_id or "default")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(best_models):
            model_name = type(model).__name__
            
            # Get metrics from leaderboard
            if i < len(leaderboard):
                metrics = leaderboard.iloc[i].to_dict()
            else:
                metrics = {}
            
            # Save model
            model_path = model_dir / f"model_{i+1}_{model_name}"
            save_model(model, str(model_path))
            saved_path = str(model_path) + ".pkl"
            
            # Get feature importance if available
            feature_importance = {}
            try:
                if hasattr(model, 'feature_importances_'):
                    feature_names = get_config('X_train').columns.tolist()
                    importances = model.feature_importances_
                    feature_importance = dict(zip(feature_names, importances.tolist()))
            except:
                pass
            
            result = {
                "name": model_name,
                "model_path": saved_path,
                "rank": i + 1,
                "hyperparameters": self._get_hyperparameters(model),
                "feature_importance": feature_importance,
                **metrics
            }
            results.append(result)
            
            logger.info(f"Model {i+1}: {model_name} - Accuracy: {metrics.get('Accuracy', 'N/A')}")
        
        return {
            "models": results,
            "problem_type": problem_type,
            "best_model": results[0]["name"] if results else None,
            "training_time": time_budget
        }
    
    def _get_hyperparameters(self, model) -> dict:
        """Extract hyperparameters from a model"""
        try:
            return model.get_params()
        except:
            return {}


class QuickAutoML:
    """
    Simplified AutoML for quick predictions.
    Use this for the inference endpoint.
    """
    
    @staticmethod
    def load_and_predict(model_path: str, data: dict) -> dict:
        """Load a saved model and make predictions"""
        import joblib
        import pandas as pd
        
        model = joblib.load(model_path)
        input_df = pd.DataFrame([data])
        
        prediction = model.predict(input_df)[0]
        
        result = {"prediction": prediction}
        
        # Get probabilities for classification
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(input_df)[0]
            result["probabilities"] = probas.tolist()
            result["confidence"] = float(max(probas))
        
        return result
