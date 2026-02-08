"""
Inference Tasks - Model predictions and batch inference
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from celery import shared_task
from loguru import logger

from config import settings


@shared_task(
    bind=True,
    name="worker.tasks.inference.run_inference",
    max_retries=3,
    default_retry_delay=30,
    track_started=True,
)
def run_inference(
    self,
    model_path: str,
    data: Union[str, List[Dict[str, Any]]],
    output_path: Optional[str] = None,
    batch_size: int = 1000,
) -> Dict[str, Any]:
    """
    Run batch inference using a trained model.
    
    Args:
        model_path: Path to saved model (.pkl)
        data: Either path to CSV or list of row dicts
        output_path: Where to save predictions (CSV)
        batch_size: Batch size for large datasets
    
    Returns:
        Dict with predictions and metadata
    """
    import pandas as pd
    import polars as pl
    import joblib
    
    logger.info(f"Running inference with model: {model_path}")
    
    self.update_state(
        state="PROGRESS",
        meta={"progress": 10, "stage": "loading_model"}
    )
    
    # Load model
    model = joblib.load(model_path)
    
    self.update_state(
        state="PROGRESS",
        meta={"progress": 20, "stage": "loading_data"}
    )
    
    # Load data
    if isinstance(data, str):
        # Path to CSV
        df = pl.read_csv(
            data,
            infer_schema_length=None,
            ignore_errors=True
        ).to_pandas()
    else:
        # List of dicts
        df = pd.DataFrame(data)
    
    total_rows = len(df)
    logger.info(f"Loaded {total_rows} rows for inference")
    
    self.update_state(
        state="PROGRESS",
        meta={"progress": 30, "stage": "predicting", "total_rows": total_rows}
    )
    
    # Run predictions in batches
    predictions = []
    probabilities = []
    
    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i + batch_size]
        batch_preds = model.predict(batch)
        predictions.extend(batch_preds.tolist())
        
        # Get probabilities if available
        if hasattr(model, "predict_proba"):
            try:
                batch_probs = model.predict_proba(batch)
                probabilities.extend(batch_probs.tolist())
            except:
                pass
        
        progress = 30 + int((i + batch_size) / total_rows * 60)
        self.update_state(
            state="PROGRESS",
            meta={"progress": min(progress, 90), "stage": "predicting", "processed": min(i + batch_size, total_rows)}
        )
    
    self.update_state(
        state="PROGRESS",
        meta={"progress": 95, "stage": "saving_results"}
    )
    
    # Prepare results
    result_df = df.copy()
    result_df["prediction"] = predictions
    
    if probabilities:
        # Add probability columns
        prob_array = pd.DataFrame(probabilities)
        prob_array.columns = [f"prob_class_{i}" for i in range(prob_array.shape[1])]
        result_df = pd.concat([result_df, prob_array], axis=1)
    
    # Save if output path provided
    if output_path:
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to: {output_path}")
    
    return {
        "status": "completed",
        "total_rows": total_rows,
        "predictions": predictions[:100] if len(predictions) > 100 else predictions,  # Truncate for response
        "has_probabilities": len(probabilities) > 0,
        "output_path": output_path,
        "completed_at": datetime.utcnow().isoformat(),
    }


@shared_task(
    bind=True,
    name="worker.tasks.inference.run_inference_gpu",
    max_retries=1,
    track_started=True,
)
def run_inference_gpu(
    self,
    model_path: str,
    data: Union[str, List[Dict[str, Any]]],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run GPU-accelerated inference.
    For GPU-trained models (PyTorch, XGBoost with GPU).
    """
    # For now, delegate to CPU inference
    # GPU inference will be added in Phase 1
    return run_inference(
        model_path=model_path,
        data=data,
        output_path=output_path,
    )
