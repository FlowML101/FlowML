"""
Dataset model - represents uploaded CSV files
"""
from datetime import datetime
from sqlmodel import SQLModel, Field
from typing import Optional
import uuid


class DatasetBase(SQLModel):
    """Base dataset fields"""
    name: str = Field(index=True)
    filename: str
    file_path: str
    file_size: int  # bytes
    num_rows: int
    num_columns: int
    columns: str  # JSON string of column names
    dtypes: str  # JSON string of column dtypes
    description: Optional[str] = None


class Dataset(DatasetBase, table=True):
    """Dataset database model"""
    __tablename__ = "datasets"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DatasetCreate(SQLModel):
    """Schema for creating a dataset (from upload)"""
    name: str
    description: Optional[str] = None


class DatasetRead(DatasetBase):
    """Schema for reading a dataset"""
    id: str
    created_at: datetime
    updated_at: datetime


class DatasetPreview(SQLModel):
    """Schema for dataset preview response"""
    id: str
    name: str
    columns: list[str]
    dtypes: dict[str, str]
    preview_rows: list[dict]
    num_rows: int
    num_columns: int


class DatasetStats(SQLModel):
    """Schema for column statistics"""
    column: str
    dtype: str
    non_null_count: int
    null_count: int
    unique_count: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    top_values: Optional[list[dict]] = None
