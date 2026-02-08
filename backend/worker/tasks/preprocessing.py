"""
Preprocessing Tasks - Data preparation and feature engineering
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from celery import shared_task, current_task
from loguru import logger

from config import settings


@shared_task(
    bind=True,
    name="worker.tasks.preprocessing.preprocess_dataset",
    max_retries=2,
    track_started=True,
)
def preprocess_dataset(
    self,
    dataset_id: str,
    dataset_path: str,
    operations: List[Dict[str, Any]],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Preprocess a dataset with specified operations.
    
    Args:
        dataset_id: Dataset identifier
        dataset_path: Path to source CSV/Parquet
        operations: List of preprocessing operations, e.g.:
            [
                {"op": "drop_nulls", "columns": ["col1", "col2"]},
                {"op": "fill_nulls", "column": "col3", "value": 0},
                {"op": "encode_categorical", "columns": ["cat1"]},
                {"op": "scale", "columns": ["num1", "num2"], "method": "standard"},
            ]
        output_path: Where to save result (default: auto-generated)
    
    Returns:
        Dict with output path, stats, and operations applied
    """
    import polars as pl
    
    logger.info(f"[{dataset_id}] Starting preprocessing with {len(operations)} operations")
    
    self.update_state(
        state="PROGRESS",
        meta={"dataset_id": dataset_id, "progress": 0, "stage": "loading"}
    )
    
    # Load dataset
    df = pl.read_csv(
        dataset_path,
        infer_schema_length=None,
        ignore_errors=True
    )
    original_shape = df.shape
    logger.info(f"[{dataset_id}] Loaded: {original_shape[0]} rows, {original_shape[1]} columns")
    
    applied_ops = []
    
    for i, op in enumerate(operations):
        op_type = op.get("op")
        progress = int((i + 1) / len(operations) * 80) + 10
        
        self.update_state(
            state="PROGRESS",
            meta={"dataset_id": dataset_id, "progress": progress, "stage": f"applying_{op_type}"}
        )
        
        try:
            if op_type == "drop_nulls":
                columns = op.get("columns")
                if columns:
                    df = df.drop_nulls(subset=columns)
                else:
                    df = df.drop_nulls()
                applied_ops.append({"op": op_type, "rows_before": original_shape[0], "rows_after": len(df)})
                
            elif op_type == "fill_nulls":
                column = op["column"]
                value = op["value"]
                df = df.with_columns(pl.col(column).fill_null(value))
                applied_ops.append({"op": op_type, "column": column, "value": value})
                
            elif op_type == "drop_columns":
                columns = op["columns"]
                df = df.drop(columns)
                applied_ops.append({"op": op_type, "columns": columns})
                
            elif op_type == "rename_column":
                old_name = op["old_name"]
                new_name = op["new_name"]
                df = df.rename({old_name: new_name})
                applied_ops.append({"op": op_type, "old_name": old_name, "new_name": new_name})
                
            elif op_type == "cast_type":
                column = op["column"]
                dtype = op["dtype"]
                dtype_map = {
                    "int": pl.Int64,
                    "float": pl.Float64,
                    "str": pl.Utf8,
                    "bool": pl.Boolean,
                }
                if dtype in dtype_map:
                    df = df.with_columns(pl.col(column).cast(dtype_map[dtype]))
                applied_ops.append({"op": op_type, "column": column, "dtype": dtype})
                
            elif op_type == "filter":
                column = op["column"]
                condition = op["condition"]  # "gt", "lt", "eq", "ne", "contains"
                value = op["value"]
                
                if condition == "gt":
                    df = df.filter(pl.col(column) > value)
                elif condition == "lt":
                    df = df.filter(pl.col(column) < value)
                elif condition == "eq":
                    df = df.filter(pl.col(column) == value)
                elif condition == "ne":
                    df = df.filter(pl.col(column) != value)
                elif condition == "contains":
                    df = df.filter(pl.col(column).str.contains(value))
                    
                applied_ops.append({"op": op_type, "column": column, "condition": condition, "value": value})
                
            else:
                logger.warning(f"[{dataset_id}] Unknown operation: {op_type}")
                
        except Exception as e:
            logger.error(f"[{dataset_id}] Operation {op_type} failed: {e}")
            applied_ops.append({"op": op_type, "error": str(e)})
    
    # Generate output path if not provided
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(settings.UPLOAD_DIR / f"{dataset_id}_preprocessed_{timestamp}.csv")
    
    # Save result
    self.update_state(
        state="PROGRESS",
        meta={"dataset_id": dataset_id, "progress": 95, "stage": "saving"}
    )
    
    df.write_csv(output_path)
    final_shape = df.shape
    
    logger.info(f"[{dataset_id}] Preprocessing complete: {original_shape} -> {final_shape}")
    
    return {
        "dataset_id": dataset_id,
        "status": "completed",
        "output_path": output_path,
        "original_shape": {"rows": original_shape[0], "columns": original_shape[1]},
        "final_shape": {"rows": final_shape[0], "columns": final_shape[1]},
        "operations_applied": applied_ops,
        "completed_at": datetime.utcnow().isoformat(),
    }
