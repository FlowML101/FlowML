"""
Optuna-based AutoML Service for FlowML

This is a lightweight, production-ready AutoML implementation using:
- Optuna for hyperparameter optimization
- scikit-learn for classical ML models
- XGBoost and LightGBM for gradient boosting

No heavy PyCaret dependency - just the essentials.
"""
import time
import json
import joblib
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading

import numpy as np
import pandas as pd
import polars as pl
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from loguru import logger

# Import models
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Optional: XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class ProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class ModelResult:
    """Result from training a single model"""
    algorithm: str
    metrics: Dict[str, float]
    params: Dict[str, Any]
    training_time: float
    cv_scores: List[float]
    model_path: Optional[str] = None
    feature_importance: Optional[Dict[str, float]] = None
    rank: int = 0


@dataclass
class AutoMLResult:
    """Complete AutoML training result"""
    job_id: str
    problem_type: str
    best_model: str
    models: List[Dict]
    total_time: float
    n_trials: int
    dataset_info: Dict[str, Any]
    created_at: str


class OptunaAutoML:
    """
    Production-ready AutoML using Optuna for HPO.
    
    Features:
    - Automatic problem type detection
    - Multiple model algorithms with HPO
    - Cross-validation for robust evaluation
    - Feature importance extraction
    - Model persistence
    - Progress callbacks
    - Cancellation support
    """
    
    # Model definitions for classification
    CLASSIFICATION_MODELS = {
        "logistic_regression": {
            "class": LogisticRegression,
            "params": lambda trial: {
                "C": trial.suggest_float("C", 1e-4, 10, log=True),
                "max_iter": 1000,
                "solver": "lbfgs",
            }
        },
        "random_forest": {
            "class": RandomForestClassifier,
            "params": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "n_jobs": -1,
            }
        },
        "gradient_boosting": {
            "class": GradientBoostingClassifier,
            "params": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            }
        },
        "knn": {
            "class": KNeighborsClassifier,
            "params": lambda trial: {
                "n_neighbors": trial.suggest_int("n_neighbors", 3, 25),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            }
        },
    }
    
    # Model definitions for regression
    REGRESSION_MODELS = {
        "ridge": {
            "class": Ridge,
            "params": lambda trial: {
                "alpha": trial.suggest_float("alpha", 1e-4, 10, log=True),
            }
        },
        "lasso": {
            "class": Lasso,
            "params": lambda trial: {
                "alpha": trial.suggest_float("alpha", 1e-4, 10, log=True),
                "max_iter": 2000,
            }
        },
        "random_forest": {
            "class": RandomForestRegressor,
            "params": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "n_jobs": -1,
            }
        },
        "gradient_boosting": {
            "class": GradientBoostingRegressor,
            "params": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
            }
        },
    }
    
    def __init__(self):
        self._add_xgboost_models()
        self._add_lightgbm_models()
    
    def _add_xgboost_models(self):
        """Add XGBoost models if available"""
        if not XGBOOST_AVAILABLE:
            return
        
        self.CLASSIFICATION_MODELS["xgboost"] = {
            "class": xgb.XGBClassifier,
            "params": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "verbosity": 0,
            }
        }
        
        self.REGRESSION_MODELS["xgboost"] = {
            "class": xgb.XGBRegressor,
            "params": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "verbosity": 0,
            }
        }
    
    def _add_lightgbm_models(self):
        """Add LightGBM models if available"""
        if not LIGHTGBM_AVAILABLE:
            return
        
        self.CLASSIFICATION_MODELS["lightgbm"] = {
            "class": lgb.LGBMClassifier,
            "params": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "verbosity": -1,
            }
        }
        
        self.REGRESSION_MODELS["lightgbm"] = {
            "class": lgb.LGBMRegressor,
            "params": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "verbosity": -1,
            }
        }
    
    def detect_problem_type(self, y: pd.Series) -> ProblemType:
        """Auto-detect whether this is classification or regression"""
        if y.dtype == 'object' or y.dtype.name == 'category':
            return ProblemType.CLASSIFICATION
        
        n_unique = y.nunique()
        n_total = len(y)
        
        # Heuristic: if few unique values relative to total, likely classification
        if n_unique <= 20 and n_unique / n_total < 0.05:
            return ProblemType.CLASSIFICATION
        
        return ProblemType.REGRESSION
    
    def preprocess_data(
        self, 
        df: pd.DataFrame, 
        target_column: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[LabelEncoder]]:
        """
        Preprocess data for ML training.
        
        Returns:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            label_encoder: LabelEncoder if classification, else None
        """
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column].copy()
        
        # Encode target if categorical
        label_encoder = None
        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        
        # Identify column types
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Simple preprocessing: fill numeric NaN with median, categorical with mode
        for col in numeric_cols:
            X[col] = X[col].fillna(X[col].median())
        
        for col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else "MISSING")
            # One-hot encode
            X = pd.get_dummies(X, columns=[col], prefix=col, drop_first=True)
        
        feature_names = X.columns.tolist()
        
        return X.values, np.array(y), feature_names, label_encoder
    
    def train(
        self,
        dataset_path: str,
        target_column: str,
        job_id: str,
        time_budget_minutes: int = 5,
        problem_type: Optional[str] = None,
        model_types: Optional[List[str]] = None,
        n_trials_per_model: int = 10,
        cv_folds: int = 5,
        progress_callback: Optional[Callable[[int, str, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        output_dir: Optional[Path] = None,
    ) -> AutoMLResult:
        """
        Run AutoML training with Optuna HPO.
        
        Args:
            dataset_path: Path to CSV/Parquet file
            target_column: Name of target column
            job_id: Unique job identifier
            time_budget_minutes: Max training time in minutes
            problem_type: "classification", "regression", or None (auto-detect)
            model_types: List of model names to try, or None for all
            n_trials_per_model: Optuna trials per model type
            cv_folds: Cross-validation folds
            progress_callback: Callback(progress_pct, stage, message)
            cancel_event: Event to check for cancellation
            output_dir: Directory to save models
            
        Returns:
            AutoMLResult with all trained models and metrics
        """
        start_time = time.time()
        time_budget_seconds = time_budget_minutes * 60
        
        def check_cancelled():
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Training cancelled by user")
        
        def report_progress(pct: int, stage: str, msg: str = ""):
            if progress_callback:
                progress_callback(pct, stage, msg)
            logger.info(f"[{job_id}] {stage}: {msg} ({pct}%)")
        
        # Load data
        report_progress(5, "loading", f"Loading {dataset_path}")
        check_cancelled()
        
        if dataset_path.endswith('.parquet'):
            df = pl.read_parquet(dataset_path).to_pandas()
        else:
            df = pl.read_csv(dataset_path).to_pandas()
        
        dataset_info = {
            "rows": len(df),
            "columns": len(df.columns),
            "features": list(df.columns),
            "target": target_column,
        }
        
        report_progress(10, "preprocessing", f"Loaded {len(df)} rows")
        check_cancelled()
        
        # Preprocess
        X, y, feature_names, label_encoder = self.preprocess_data(df, target_column)
        
        # Detect problem type
        if problem_type is None or problem_type == "auto":
            detected = self.detect_problem_type(df[target_column])
            problem_type = detected.value
        
        report_progress(15, "setup", f"Problem type: {problem_type}")
        
        # Select models
        model_defs = (
            self.CLASSIFICATION_MODELS if problem_type == "classification" 
            else self.REGRESSION_MODELS
        )
        
        if model_types:
            model_defs = {k: v for k, v in model_defs.items() if k in model_types}
        
        # Scoring metric
        if problem_type == "classification":
            scoring = "accuracy"
            direction = "maximize"
        else:
            scoring = "neg_mean_squared_error"
            direction = "maximize"  # neg_mse, so maximize
        
        # Train models
        results: List[ModelResult] = []
        n_models = len(model_defs)
        progress_per_model = 70 // n_models  # 15% to 85% for training
        
        for i, (model_name, model_def) in enumerate(model_defs.items()):
            check_cancelled()
            
            # Check time budget
            elapsed = time.time() - start_time
            if elapsed > time_budget_seconds:
                logger.warning(f"[{job_id}] Time budget exceeded, stopping")
                break
            
            remaining_time = time_budget_seconds - elapsed
            model_time_budget = min(remaining_time / (n_models - i), remaining_time)
            
            base_progress = 15 + i * progress_per_model
            report_progress(base_progress, "training", f"Training {model_name}")
            
            try:
                result = self._train_single_model(
                    model_name=model_name,
                    model_def=model_def,
                    X=X,
                    y=y,
                    feature_names=feature_names,
                    scoring=scoring,
                    direction=direction,
                    n_trials=n_trials_per_model,
                    cv_folds=cv_folds,
                    time_budget=model_time_budget,
                    cancel_event=cancel_event,
                )
                results.append(result)
                
                report_progress(
                    base_progress + progress_per_model - 5, 
                    "trained", 
                    f"{model_name}: score={result.cv_scores[-1]:.4f}"
                )
            except Exception as e:
                logger.error(f"[{job_id}] Failed to train {model_name}: {e}")
                continue
        
        check_cancelled()
        
        # Rank results
        if problem_type == "classification":
            results.sort(key=lambda r: r.metrics.get("accuracy", 0), reverse=True)
        else:
            results.sort(key=lambda r: r.metrics.get("r2", 0), reverse=True)
        
        for rank, result in enumerate(results, 1):
            result.rank = rank
        
        # Save models
        report_progress(85, "saving", "Saving trained models")
        output_dir = output_dir or Path("./trained_models") / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for result in results:
            check_cancelled()
            model_path = output_dir / f"model_{result.rank}_{result.algorithm}.joblib"
            # Re-train best params on full data for saving
            best_model = self._create_model(
                model_defs[result.algorithm], 
                result.params
            )
            best_model.fit(X, y)
            joblib.dump({
                "model": best_model,
                "feature_names": feature_names,
                "label_encoder": label_encoder,
                "problem_type": problem_type,
            }, model_path)
            result.model_path = str(model_path)
        
        # Build result
        total_time = time.time() - start_time
        report_progress(95, "finalizing", f"Completed in {total_time:.1f}s")
        
        return AutoMLResult(
            job_id=job_id,
            problem_type=problem_type,
            best_model=results[0].algorithm if results else "none",
            models=[asdict(r) for r in results],
            total_time=total_time,
            n_trials=len(results) * n_trials_per_model,
            dataset_info=dataset_info,
            created_at=datetime.utcnow().isoformat(),
        )
    
    def _create_model(self, model_def: dict, params: dict):
        """Create a model instance with given params"""
        model_class = model_def["class"]
        return model_class(**params)
    
    def _train_single_model(
        self,
        model_name: str,
        model_def: dict,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        scoring: str,
        direction: str,
        n_trials: int,
        cv_folds: int,
        time_budget: float,
        cancel_event: Optional[threading.Event],
    ) -> ModelResult:
        """Train a single model type with Optuna HPO"""
        
        start_time = time.time()
        
        # Create Optuna study
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=42),
        )
        
        def objective(trial):
            if cancel_event and cancel_event.is_set():
                raise optuna.TrialPruned()
            
            # Get hyperparameters from trial
            params = model_def["params"](trial)
            
            # Create and evaluate model
            model = model_def["class"](**params)
            
            try:
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
                return scores.mean()
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('-inf') if direction == "maximize" else float('inf')
        
        # Run optimization with time budget
        study.optimize(
            objective, 
            n_trials=n_trials, 
            timeout=time_budget,
            show_progress_bar=False,
        )
        
        # Get best trial
        best_trial = study.best_trial
        best_params = model_def["params"](best_trial)
        
        # Train final model with best params
        best_model = model_def["class"](**best_params)
        
        # Split for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        if scoring == "accuracy":
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred, average='weighted'),
                "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            }
            # AUC for binary classification
            if len(np.unique(y)) == 2 and hasattr(best_model, 'predict_proba'):
                try:
                    y_proba = best_model.predict_proba(X_test)[:, 1]
                    metrics["auc"] = roc_auc_score(y_test, y_proba)
                except:
                    pass
        else:
            metrics = {
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
            }
        
        # Feature importance
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_
            feature_importance = dict(zip(feature_names, importance.tolist()))
        elif hasattr(best_model, 'coef_'):
            coef = np.abs(best_model.coef_).flatten()
            if len(coef) == len(feature_names):
                feature_importance = dict(zip(feature_names, coef.tolist()))
        
        training_time = time.time() - start_time
        
        return ModelResult(
            algorithm=model_name,
            metrics=metrics,
            params=best_params,
            training_time=training_time,
            cv_scores=[best_trial.value],
            feature_importance=feature_importance,
        )
    
    @staticmethod
    def load_and_predict(model_path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load a saved model and make predictions"""
        saved = joblib.load(model_path)
        model = saved["model"]
        feature_names = saved["feature_names"]
        label_encoder = saved.get("label_encoder")
        problem_type = saved.get("problem_type", "classification")
        
        # Prepare input
        input_df = pd.DataFrame([data])
        
        # Ensure columns match (add missing, remove extra)
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]
        
        # Predict
        prediction = model.predict(input_df.values)[0]
        
        result = {
            "prediction": prediction,
            "problem_type": problem_type,
        }
        
        # Decode label if classifier
        if label_encoder is not None:
            result["prediction"] = label_encoder.inverse_transform([int(prediction)])[0]
        
        # Probabilities for classification
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(input_df.values)[0]
            result["probabilities"] = probas.tolist()
            result["confidence"] = float(max(probas))
            
            if label_encoder is not None:
                result["class_labels"] = label_encoder.classes_.tolist()
        
        return result


# Global instance
optuna_automl = OptunaAutoML()
