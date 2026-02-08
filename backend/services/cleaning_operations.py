"""
Safe data cleaning operations for FlowML.

All operations are predefined, tested, and safe to execute.
No arbitrary code execution - only parameterized operations.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, validator, Field
import polars as pl
from datetime import datetime
import re


class OperationType(str, Enum):
    """Available cleaning operation types."""
    FILL_MISSING = "fill_missing"
    DROP_MISSING = "drop_missing"
    DROP_DUPLICATES = "drop_duplicates"
    CAST_TYPE = "cast_type"
    REMOVE_OUTLIERS = "remove_outliers"
    STANDARDIZE = "standardize"
    NORMALIZE = "normalize"
    STRING_CLEAN = "string_clean"
    RENAME_COLUMN = "rename_column"
    DROP_COLUMN = "drop_column"
    CREATE_FEATURE = "create_feature"
    FILTER_ROWS = "filter_rows"
    ENCODE_CATEGORICAL = "encode_categorical"


class FillStrategy(str, Enum):
    """Strategies for filling missing values."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    FORWARD = "forward"
    BACKWARD = "backward"
    CONSTANT = "constant"
    INTERPOLATE = "interpolate"


class OutlierMethod(str, Enum):
    """Methods for detecting outliers."""
    IQR = "iqr"
    ZSCORE = "zscore"
    PERCENTILE = "percentile"


class StringOperation(str, Enum):
    """String cleaning operations."""
    LOWERCASE = "lowercase"
    UPPERCASE = "uppercase"
    STRIP = "strip"
    REMOVE_SPECIAL = "remove_special"
    REMOVE_NUMBERS = "remove_numbers"
    REMOVE_WHITESPACE = "remove_whitespace"


class CleaningOperation(BaseModel):
    """Base model for a cleaning operation."""
    operation: OperationType
    column: Optional[str] = None
    columns: Optional[List[str]] = None
    description: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    
    @validator('confidence')
    def validate_confidence(cls, v):
        return max(0.0, min(1.0, v))


class FillMissingOperation(CleaningOperation):
    """Fill missing values in a column."""
    operation: OperationType = OperationType.FILL_MISSING
    column: str
    strategy: FillStrategy
    fill_value: Optional[Union[int, float, str]] = None
    
    @validator('fill_value')
    def validate_fill_value(cls, v, values):
        if values.get('strategy') == FillStrategy.CONSTANT and v is None:
            raise ValueError("fill_value required for CONSTANT strategy")
        return v


class DropMissingOperation(CleaningOperation):
    """Drop rows or columns with missing values."""
    operation: OperationType = OperationType.DROP_MISSING
    axis: str = "rows"  # "rows" or "columns"
    threshold: Optional[float] = None  # Drop if % missing > threshold
    columns: Optional[List[str]] = None  # Specific columns to check
    
    @validator('axis')
    def validate_axis(cls, v):
        if v not in ["rows", "columns"]:
            raise ValueError("axis must be 'rows' or 'columns'")
        return v


class DropDuplicatesOperation(CleaningOperation):
    """Remove duplicate rows."""
    operation: OperationType = OperationType.DROP_DUPLICATES
    columns: Optional[List[str]] = None  # Subset of columns to check
    keep: str = "first"  # "first", "last", or "none"


class CastTypeOperation(CleaningOperation):
    """Convert column data type."""
    operation: OperationType = OperationType.CAST_TYPE
    column: str
    target_type: str  # "int", "float", "string", "datetime", "boolean"
    date_format: Optional[str] = None  # For datetime conversion


class RemoveOutliersOperation(CleaningOperation):
    """Remove outliers from numeric columns."""
    operation: OperationType = OperationType.REMOVE_OUTLIERS
    column: str
    method: OutlierMethod
    threshold: float = 1.5  # IQR multiplier or Z-score threshold
    percentile_range: Optional[tuple] = None  # (lower, upper) for percentile method


class StandardizeOperation(CleaningOperation):
    """Standardize numeric column (z-score normalization)."""
    operation: OperationType = OperationType.STANDARDIZE
    column: str


class NormalizeOperation(CleaningOperation):
    """Normalize numeric column to [0, 1] range."""
    operation: OperationType = OperationType.NORMALIZE
    column: str


class StringCleanOperation(CleaningOperation):
    """Clean string values in a column."""
    operation: OperationType = OperationType.STRING_CLEAN
    column: str
    operations: List[StringOperation]


class RenameColumnOperation(CleaningOperation):
    """Rename a column."""
    operation: OperationType = OperationType.RENAME_COLUMN
    column: str
    new_name: str


class DropColumnOperation(CleaningOperation):
    """Drop a column."""
    operation: OperationType = OperationType.DROP_COLUMN
    column: str


class EncodeCategoricalOperation(CleaningOperation):
    """Encode categorical column."""
    operation: OperationType = OperationType.ENCODE_CATEGORICAL
    column: str
    method: str = "label"  # "label" or "onehot"


# Operation execution functions

class CleaningExecutor:
    """Executes cleaning operations safely on Polars DataFrames."""
    
    @staticmethod
    def execute(df: pl.DataFrame, operation: CleaningOperation) -> pl.DataFrame:
        """Execute a cleaning operation on a DataFrame."""
        
        if operation.operation == OperationType.FILL_MISSING:
            return CleaningExecutor._fill_missing(df, operation)
        elif operation.operation == OperationType.DROP_MISSING:
            return CleaningExecutor._drop_missing(df, operation)
        elif operation.operation == OperationType.DROP_DUPLICATES:
            return CleaningExecutor._drop_duplicates(df, operation)
        elif operation.operation == OperationType.CAST_TYPE:
            return CleaningExecutor._cast_type(df, operation)
        elif operation.operation == OperationType.REMOVE_OUTLIERS:
            return CleaningExecutor._remove_outliers(df, operation)
        elif operation.operation == OperationType.STANDARDIZE:
            return CleaningExecutor._standardize(df, operation)
        elif operation.operation == OperationType.NORMALIZE:
            return CleaningExecutor._normalize(df, operation)
        elif operation.operation == OperationType.STRING_CLEAN:
            return CleaningExecutor._string_clean(df, operation)
        elif operation.operation == OperationType.RENAME_COLUMN:
            return CleaningExecutor._rename_column(df, operation)
        elif operation.operation == OperationType.DROP_COLUMN:
            return CleaningExecutor._drop_column(df, operation)
        elif operation.operation == OperationType.ENCODE_CATEGORICAL:
            return CleaningExecutor._encode_categorical(df, operation)
        else:
            raise ValueError(f"Unknown operation: {operation.operation}")
    
    @staticmethod
    def _fill_missing(df: pl.DataFrame, op: FillMissingOperation) -> pl.DataFrame:
        """Fill missing values."""
        if op.column not in df.columns:
            raise ValueError(f"Column '{op.column}' not found")
        
        col = pl.col(op.column)
        
        if op.strategy == FillStrategy.MEAN:
            return df.with_columns(col.fill_null(col.mean()))
        elif op.strategy == FillStrategy.MEDIAN:
            return df.with_columns(col.fill_null(col.median()))
        elif op.strategy == FillStrategy.MODE:
            mode_val = df[op.column].mode().first()
            return df.with_columns(col.fill_null(mode_val))
        elif op.strategy == FillStrategy.FORWARD:
            return df.with_columns(col.fill_null(strategy="forward"))
        elif op.strategy == FillStrategy.BACKWARD:
            return df.with_columns(col.fill_null(strategy="backward"))
        elif op.strategy == FillStrategy.CONSTANT:
            return df.with_columns(col.fill_null(op.fill_value))
        elif op.strategy == FillStrategy.INTERPOLATE:
            return df.with_columns(col.interpolate())
        
        return df
    
    @staticmethod
    def _drop_missing(df: pl.DataFrame, op: DropMissingOperation) -> pl.DataFrame:
        """Drop rows or columns with missing values."""
        if op.axis == "rows":
            if op.columns:
                # Drop rows with nulls in specific columns
                return df.drop_nulls(subset=op.columns)
            else:
                # Drop any row with nulls
                return df.drop_nulls()
        else:  # columns
            if op.threshold is not None:
                # Drop columns with % missing > threshold
                total_rows = len(df)
                cols_to_keep = []
                for col in df.columns:
                    null_pct = df[col].null_count() / total_rows
                    if null_pct <= op.threshold:
                        cols_to_keep.append(col)
                return df.select(cols_to_keep)
            else:
                # Drop columns with any nulls
                return df.select([c for c in df.columns if df[c].null_count() == 0])
    
    @staticmethod
    def _drop_duplicates(df: pl.DataFrame, op: DropDuplicatesOperation) -> pl.DataFrame:
        """Remove duplicate rows."""
        if op.columns:
            return df.unique(subset=op.columns, keep=op.keep)
        return df.unique(keep=op.keep)
    
    @staticmethod
    def _cast_type(df: pl.DataFrame, op: CastTypeOperation) -> pl.DataFrame:
        """Convert column data type."""
        if op.column not in df.columns:
            raise ValueError(f"Column '{op.column}' not found")
        
        type_map = {
            "int": pl.Int64,
            "float": pl.Float64,
            "string": pl.Utf8,
            "boolean": pl.Boolean,
        }
        
        if op.target_type == "datetime":
            if op.date_format:
                return df.with_columns(
                    pl.col(op.column).str.strptime(pl.Datetime, format=op.date_format)
                )
            else:
                return df.with_columns(pl.col(op.column).str.strptime(pl.Datetime))
        else:
            target_dtype = type_map.get(op.target_type)
            if not target_dtype:
                raise ValueError(f"Unknown target type: {op.target_type}")
            return df.with_columns(pl.col(op.column).cast(target_dtype))
    
    @staticmethod
    def _remove_outliers(df: pl.DataFrame, op: RemoveOutliersOperation) -> pl.DataFrame:
        """Remove outliers from numeric column."""
        if op.column not in df.columns:
            raise ValueError(f"Column '{op.column}' not found")
        
        col = df[op.column]
        
        if op.method == OutlierMethod.IQR:
            q1 = col.quantile(0.25)
            q3 = col.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - op.threshold * iqr
            upper = q3 + op.threshold * iqr
            return df.filter((pl.col(op.column) >= lower) & (pl.col(op.column) <= upper))
        
        elif op.method == OutlierMethod.ZSCORE:
            mean = col.mean()
            std = col.std()
            return df.filter(
                ((pl.col(op.column) - mean).abs() / std) <= op.threshold
            )
        
        elif op.method == OutlierMethod.PERCENTILE:
            if not op.percentile_range:
                op.percentile_range = (0.01, 0.99)
            lower = col.quantile(op.percentile_range[0])
            upper = col.quantile(op.percentile_range[1])
            return df.filter((pl.col(op.column) >= lower) & (pl.col(op.column) <= upper))
        
        return df
    
    @staticmethod
    def _standardize(df: pl.DataFrame, op: StandardizeOperation) -> pl.DataFrame:
        """Standardize column (z-score)."""
        if op.column not in df.columns:
            raise ValueError(f"Column '{op.column}' not found")
        
        col = pl.col(op.column)
        return df.with_columns(
            ((col - col.mean()) / col.std()).alias(op.column)
        )
    
    @staticmethod
    def _normalize(df: pl.DataFrame, op: NormalizeOperation) -> pl.DataFrame:
        """Normalize column to [0, 1]."""
        if op.column not in df.columns:
            raise ValueError(f"Column '{op.column}' not found")
        
        col = pl.col(op.column)
        col_min = col.min()
        col_max = col.max()
        return df.with_columns(
            ((col - col_min) / (col_max - col_min)).alias(op.column)
        )
    
    @staticmethod
    def _string_clean(df: pl.DataFrame, op: StringCleanOperation) -> pl.DataFrame:
        """Clean string column."""
        if op.column not in df.columns:
            raise ValueError(f"Column '{op.column}' not found")
        
        col = pl.col(op.column)
        
        for string_op in op.operations:
            if string_op == StringOperation.LOWERCASE:
                col = col.str.to_lowercase()
            elif string_op == StringOperation.UPPERCASE:
                col = col.str.to_uppercase()
            elif string_op == StringOperation.STRIP:
                col = col.str.strip_chars()
            elif string_op == StringOperation.REMOVE_SPECIAL:
                col = col.str.replace_all(r'[^a-zA-Z0-9\s]', '')
            elif string_op == StringOperation.REMOVE_NUMBERS:
                col = col.str.replace_all(r'\d', '')
            elif string_op == StringOperation.REMOVE_WHITESPACE:
                col = col.str.replace_all(r'\s+', ' ')
        
        return df.with_columns(col.alias(op.column))
    
    @staticmethod
    def _rename_column(df: pl.DataFrame, op: RenameColumnOperation) -> pl.DataFrame:
        """Rename a column."""
        if op.column not in df.columns:
            raise ValueError(f"Column '{op.column}' not found")
        return df.rename({op.column: op.new_name})
    
    @staticmethod
    def _drop_column(df: pl.DataFrame, op: DropColumnOperation) -> pl.DataFrame:
        """Drop a column."""
        if op.column not in df.columns:
            raise ValueError(f"Column '{op.column}' not found")
        return df.drop(op.column)
    
    @staticmethod
    def _encode_categorical(df: pl.DataFrame, op: EncodeCategoricalOperation) -> pl.DataFrame:
        """Encode categorical column."""
        if op.column not in df.columns:
            raise ValueError(f"Column '{op.column}' not found")
        
        if op.method == "label":
            # Label encoding: convert categories to integers
            categories = df[op.column].unique().sort()
            mapping = {cat: idx for idx, cat in enumerate(categories.to_list())}
            return df.with_columns(
                pl.col(op.column).replace(mapping).alias(op.column)
            )
        elif op.method == "onehot":
            # One-hot encoding
            return df.to_dummies(columns=[op.column])
        
        return df


def analyze_dataset(df: pl.DataFrame) -> Dict[str, Any]:
    """
    Analyze a dataset and return statistics for LLM analysis.
    
    Returns comprehensive dataset statistics that can be used
    to generate cleaning suggestions.
    """
    total_rows = len(df)
    columns_info = []
    
    for col in df.columns:
        col_data = df[col]
        dtype = str(col_data.dtype)
        null_count = col_data.null_count()
        null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0
        unique_count = col_data.n_unique()
        
        info = {
            "name": col,
            "dtype": dtype,
            "null_count": null_count,
            "null_percentage": round(null_pct, 2),
            "unique_count": unique_count,
            "is_numeric": dtype in ["Int64", "Int32", "Float64", "Float32"],
            "is_categorical": unique_count < total_rows * 0.05 if total_rows > 0 else False,
        }
        
        # Add numeric statistics
        if info["is_numeric"]:
            try:
                info["mean"] = float(col_data.mean()) if not col_data.is_null().all() else None
                info["median"] = float(col_data.median()) if not col_data.is_null().all() else None
                info["std"] = float(col_data.std()) if not col_data.is_null().all() else None
                info["min"] = float(col_data.min()) if not col_data.is_null().all() else None
                info["max"] = float(col_data.max()) if not col_data.is_null().all() else None
            except:
                pass
        
        # Add sample values
        try:
            non_null_values = col_data.drop_nulls()
            if len(non_null_values) > 0:
                info["sample_values"] = non_null_values.head(5).to_list()
        except:
            info["sample_values"] = []
        
        columns_info.append(info)
    
    # Dataset-level statistics
    duplicate_count = len(df) - len(df.unique())
    
    return {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "columns": columns_info,
        "duplicate_rows": duplicate_count,
        "memory_usage_mb": df.estimated_size() / 1024 / 1024,
    }
