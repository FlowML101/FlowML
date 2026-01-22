"""
Training Job model - represents AutoML training runs
"""
from datetime import datetime
from sqlmodel import SQLModel, Field
from typing import Optional
from enum import Enum
import uuid


class JobStatus(str, Enum):
    """Job status enum"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobBase(SQLModel):
    """Base job fields"""
    name: str = Field(index=True)
    dataset_id: str = Field(foreign_key="datasets.id", index=True)
    target_column: str
    time_budget: int = 5  # minutes
    model_types: str = "auto"  # JSON list or "auto"
    problem_type: Optional[str] = None  # classification, regression, auto


class Job(JobBase, table=True):
    """Job database model"""
    __tablename__ = "jobs"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    status: JobStatus = Field(default=JobStatus.PENDING, index=True)
    progress: float = Field(default=0.0)  # 0-100
    current_model: Optional[str] = None
    models_completed: int = Field(default=0)
    total_models: int = Field(default=0)
    error_message: Optional[str] = None
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Worker info (Phase 2)
    worker_id: Optional[str] = None


class JobCreate(SQLModel):
    """Schema for creating a job"""
    name: Optional[str] = None  # Auto-generated if not provided
    dataset_id: str
    target_column: str
    time_budget: int = 5
    model_types: list[str] | str = "auto"
    problem_type: Optional[str] = None


class JobRead(JobBase):
    """Schema for reading a job"""
    id: str
    status: JobStatus
    progress: float
    current_model: Optional[str]
    models_completed: int
    total_models: int
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    worker_id: Optional[str]


class JobProgress(SQLModel):
    """Real-time job progress update"""
    job_id: str
    status: JobStatus
    progress: float
    current_model: Optional[str]
    models_completed: int
    total_models: int
    logs: list[str] = []
    metrics: Optional[dict] = None
