"""
Worker Model - Tracks registered workers and their capabilities
"""
from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
import uuid


class WorkerBase(SQLModel):
    """Base worker fields"""
    worker_id: str = Field(index=True, unique=True)
    hostname: str
    ip_address: str
    
    # Hardware - CPU/RAM
    cpu_count: int
    cpu_count_logical: int
    total_ram_gb: float
    available_ram_gb: float
    
    # Hardware - GPU
    has_gpu: bool = False
    gpu_count: int = 0
    gpu_names: Optional[str] = None  # JSON list
    total_vram_gb: float = 0.0
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    
    # Software versions
    python_version: str
    torch_version: Optional[str] = None
    sklearn_version: Optional[str] = None
    xgboost_version: Optional[str] = None
    lightgbm_version: Optional[str] = None
    
    # Runtime
    max_concurrency: int = 1
    current_tasks: int = 0
    tags: str = "[]"  # JSON list of queue tags
    
    # Status
    status: str = "online"  # online, offline, busy, draining


class Worker(WorkerBase, table=True):
    """Worker database model"""
    __tablename__ = "workers"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    

class WorkerCreate(WorkerBase):
    """Schema for worker registration"""
    pass


class WorkerRead(WorkerBase):
    """Schema for reading workers"""
    id: str
    created_at: datetime
    last_heartbeat: datetime


class WorkerHeartbeat(SQLModel):
    """Schema for worker heartbeat updates"""
    worker_id: str
    available_ram_gb: float
    current_tasks: int
    status: str = "online"
