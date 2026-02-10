"""
Dataset routes - upload, list, preview, stats
Supports multiple formats: CSV, Excel, Parquet, JSON, Feather
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
import polars as pl
import json
from pathlib import Path
from datetime import datetime
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from database import get_session
from config import settings
from models.dataset import Dataset, DatasetRead, DatasetPreview, DatasetStats
from exceptions import not_found, bad_request, file_too_large, invalid_file_type
from services.data_formats import DataReader, ReadOptions, DataFormat, EXTENSION_MAP

router = APIRouter(prefix="/datasets", tags=["datasets"])

# Thread pool for CPU-bound operations
_executor = ThreadPoolExecutor(max_workers=4)


def _validate_file(filename: str, file_size: int) -> None:
    """Validate file type and size"""
    ext = Path(filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise invalid_file_type(settings.ALLOWED_EXTENSIONS)
    if file_size > settings.max_upload_bytes:
        raise file_too_large(settings.MAX_UPLOAD_SIZE_MB)


def _read_data_sync(file_path: Path) -> pl.DataFrame:
    """Read data file in thread pool (CPU-bound) - supports multiple formats"""
    return DataReader.read(file_path)


@router.post("/upload", response_model=DatasetRead)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(None),
    description: str = Form(None),
    session: AsyncSession = Depends(get_session)
):
    """
    Upload a CSV/Parquet file and create a dataset record.
    Validates file type and size before processing.
    """
    # Read content first to get size
    content = await file.read()
    file_size = len(content)
    
    # Validate file
    _validate_file(file.filename, file_size)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{file.filename}"
    file_path = settings.UPLOAD_DIR / safe_filename
    
    # Ensure upload dir exists
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Analyze with Polars in thread pool (non-blocking) - supports multiple formats
    loop = asyncio.get_event_loop()
    try:
        df = await loop.run_in_executor(_executor, _read_data_sync, file_path)
    except Exception as e:
        file_path.unlink()  # Clean up
        logger.error(f"Failed to parse file: {e}")
        raise bad_request(f"Failed to parse file: {str(e)}")
    
    # Extract metadata
    columns = df.columns
    dtypes = {col: str(df[col].dtype) for col in columns}
    
    # Create dataset record
    dataset = Dataset(
        name=name or file.filename.replace('.csv', ''),
        filename=file.filename,
        file_path=str(file_path),
        file_size=len(content),
        num_rows=len(df),
        num_columns=len(columns),
        columns=json.dumps(columns),
        dtypes=json.dumps(dtypes),
        description=description
    )
    
    session.add(dataset)
    await session.commit()
    await session.refresh(dataset)
    
    return dataset


@router.get("", response_model=list[DatasetRead])
async def list_datasets(
    session: AsyncSession = Depends(get_session),
    limit: int = 100,
    offset: int = 0
):
    """List all datasets"""
    result = await session.execute(
        select(Dataset)
        .order_by(Dataset.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()


@router.get("/{dataset_id}", response_model=DatasetRead)
async def get_dataset(
    dataset_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Get a single dataset by ID"""
    result = await session.execute(
        select(Dataset).where(Dataset.id == dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise not_found("Dataset", dataset_id)
    return dataset


@router.get("/{dataset_id}/download")
async def download_dataset(
    dataset_id: str,
    session: AsyncSession = Depends(get_session)
):
    """
    Download the raw dataset file.
    Used by distributed workers to fetch data from master.
    """
    from fastapi.responses import FileResponse
    
    result = await session.execute(
        select(Dataset).where(Dataset.id == dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise not_found("Dataset", dataset_id)
    
    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")
    
    return FileResponse(
        path=file_path,
        filename=dataset.filename,
        media_type="application/octet-stream"
    )


@router.get("/download-by-path/{path:path}")
async def download_by_path(path: str):
    """
    Download a file by path (for distributed workers).
    Path should be relative to backend directory.
    """
    from fastapi.responses import FileResponse
    
    # Security: only allow files in uploads directory
    file_path = Path(path)
    uploads_dir = Path("uploads").resolve()
    
    # Resolve to absolute path
    if not file_path.is_absolute():
        file_path = Path(".") / file_path
    file_path = file_path.resolve()
    
    # Check it's within uploads
    try:
        file_path.relative_to(uploads_dir)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="application/octet-stream"
    )


def _get_preview_sync(file_path: str, rows: int) -> tuple[list, list, dict]:
    """Get preview data in thread pool - supports multiple formats"""
    df = DataReader.read(file_path, ReadOptions(sample_rows=rows + 10))
    preview_df = df.head(rows)
    return df.columns, preview_df.to_dicts(), {col: str(df[col].dtype) for col in df.columns}


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
async def preview_dataset(
    dataset_id: str,
    rows: int = 100,
    session: AsyncSession = Depends(get_session)
):
    """Get preview rows from a dataset"""
    # Limit preview rows for performance
    rows = min(rows, 500)
    
    result = await session.execute(
        select(Dataset).where(Dataset.id == dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise not_found("Dataset", dataset_id)
    
    # Read in thread pool
    loop = asyncio.get_event_loop()
    columns, preview_rows, dtypes = await loop.run_in_executor(
        _executor, _get_preview_sync, dataset.file_path, rows
    )
    
    return DatasetPreview(
        id=dataset.id,
        name=dataset.name,
        columns=columns,
        dtypes=dtypes,
        preview_rows=preview_rows,
        num_rows=dataset.num_rows,
        num_columns=dataset.num_columns
    )


def _get_stats_sync(file_path: str) -> list[dict]:
    """Calculate column statistics in thread pool - supports multiple formats"""
    df = DataReader.read(file_path)
    stats = []
    
    for col in df.columns:
        col_data = df[col]
        dtype = str(col_data.dtype)
        
        stat = {
            "column": col,
            "dtype": dtype,
            "non_null_count": col_data.drop_nulls().len(),
            "null_count": col_data.null_count(),
            "unique_count": col_data.n_unique(),
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "top_values": None
        }
        
        # Numeric stats
        if col_data.dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
            stat["mean"] = float(col_data.mean()) if col_data.mean() is not None else None
            stat["std"] = float(col_data.std()) if col_data.std() is not None else None
            stat["min"] = float(col_data.min()) if col_data.min() is not None else None
            stat["max"] = float(col_data.max()) if col_data.max() is not None else None
        
        # Top values for categorical
        if col_data.dtype == pl.Utf8 or col_data.n_unique() < 20:
            value_counts = col_data.value_counts().sort("count", descending=True).head(5)
            stat["top_values"] = value_counts.to_dicts()
        
        stats.append(stat)
    
    return stats


@router.get("/{dataset_id}/stats", response_model=list[DatasetStats])
async def get_dataset_stats(
    dataset_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Get column statistics for a dataset"""
    result = await session.execute(
        select(Dataset).where(Dataset.id == dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise not_found("Dataset", dataset_id)
    
    # Calculate stats in thread pool
    loop = asyncio.get_event_loop()
    stats_dicts = await loop.run_in_executor(_executor, _get_stats_sync, dataset.file_path)
    
    return [DatasetStats(**s) for s in stats_dicts]


def _get_correlation_sync(file_path: str) -> dict:
    """Calculate correlation matrix for numeric columns"""
    df = DataReader.read(file_path)
    
    # Get only numeric columns
    numeric_cols = [col for col in df.columns 
                   if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]]
    
    if len(numeric_cols) < 2:
        return {"matrix": [], "columns": []}
    
    # Limit to first 15 numeric columns for performance
    numeric_cols = numeric_cols[:15]
    numeric_df = df.select(numeric_cols)
    
    # Calculate correlation matrix using Polars
    correlations = []
    for col1 in numeric_cols:
        row = {"feature": col1}
        for col2 in numeric_cols:
            # Pearson correlation
            corr = numeric_df[col1].pearson_corr(numeric_df[col2])
            row[col2] = round(corr, 3) if corr is not None else 0.0
        correlations.append(row)
    
    return {"matrix": correlations, "columns": numeric_cols}


@router.get("/{dataset_id}/correlation")
async def get_correlation_matrix(
    dataset_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Get correlation matrix for numeric columns in a dataset"""
    result = await session.execute(
        select(Dataset).where(Dataset.id == dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise not_found("Dataset", dataset_id)
    
    loop = asyncio.get_event_loop()
    correlation_data = await loop.run_in_executor(_executor, _get_correlation_sync, dataset.file_path)
    
    return correlation_data


def _get_distribution_sync(file_path: str, column: str, bins: int = 20) -> dict:
    """Get distribution/histogram data for a column"""
    df = DataReader.read(file_path)
    
    if column not in df.columns:
        return {"error": f"Column {column} not found"}
    
    col_data = df[column].drop_nulls()
    
    # For numeric columns - create histogram
    if col_data.dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
        min_val = float(col_data.min())
        max_val = float(col_data.max())
        
        if min_val == max_val:
            return {"bins": [min_val], "counts": [len(col_data)], "type": "numeric"}
        
        bin_width = (max_val - min_val) / bins
        bin_edges = [min_val + i * bin_width for i in range(bins + 1)]
        
        counts = []
        for i in range(bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]
            if i == bins - 1:  # Include max in last bin
                count = col_data.filter((col_data >= lower) & (col_data <= upper)).len()
            else:
                count = col_data.filter((col_data >= lower) & (col_data < upper)).len()
            counts.append(count)
        
        return {"bins": bin_edges[:-1], "counts": counts, "type": "numeric"}
    
    # For categorical - value counts
    else:
        value_counts = col_data.value_counts().sort("count", descending=True).head(bins)
        return {
            "categories": value_counts[column].to_list(),
            "counts": value_counts["count"].to_list(),
            "type": "categorical"
        }


@router.get("/{dataset_id}/distribution/{column}")
async def get_column_distribution(
    dataset_id: str,
    column: str,
    bins: int = 20,
    session: AsyncSession = Depends(get_session)
):
    """Get distribution/histogram data for a specific column"""
    result = await session.execute(
        select(Dataset).where(Dataset.id == dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise not_found("Dataset", dataset_id)
    
    loop = asyncio.get_event_loop()
    distribution = await loop.run_in_executor(_executor, _get_distribution_sync, dataset.file_path, column, bins)
    
    return distribution


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Delete a dataset and its file"""
    result = await session.execute(
        select(Dataset).where(Dataset.id == dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise not_found("Dataset", dataset_id)
    
    # Delete file
    file_path = Path(dataset.file_path)
    if file_path.exists():
        file_path.unlink()
    
    # Delete record
    await session.delete(dataset)
    await session.commit()
    
    logger.info(f"Deleted dataset {dataset_id}")
    return {"status": "deleted", "id": dataset_id}
