"""
FlowML Backend Configuration - Optimized with validation

Supports multiple deployment modes:
- Development: SQLite + local filesystem
- Production: Postgres + MinIO S3 storage

Set via environment variables or .env file.
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache
from typing import Literal, Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # App
    APP_NAME: str = "FlowML Studio"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # ===================
    # Database
    # ===================
    # SQLite for dev, Postgres for production
    # Example: postgresql+asyncpg://flowml:flowml@localhost:5432/flowml
    DATABASE_URL: str = "sqlite+aiosqlite:///./flowml.db"
    
    # Connection pool settings (for Postgres)
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    
    # ===================
    # Storage
    # ===================
    # "local" for filesystem, "s3" for MinIO/S3
    STORAGE_MODE: Literal["local", "s3"] = "local"
    LOCAL_STORAGE_PATH: Path = Path("./uploads")
    
    # S3/MinIO settings
    S3_ENDPOINT: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "flowml-admin"
    S3_SECRET_KEY: str = "flowml-secret"
    S3_REGION: str = "us-east-1"
    S3_BUCKET_PREFIX: str = "flowml"
    
    # Legacy paths (still used for local mode)
    UPLOAD_DIR: Path = Path("./uploads")
    MODELS_DIR: Path = Path("./trained_models")
    LOGS_DIR: Path = Path("./logs")
    
    # Upload limits
    MAX_UPLOAD_SIZE_MB: int = 500  # Max file size in MB
    ALLOWED_EXTENSIONS: set[str] = {".csv", ".parquet"}
    
    # ===================
    # AutoML
    # ===================
    DEFAULT_TIME_BUDGET: int = 5  # minutes
    MAX_TIME_BUDGET: int = 60  # minutes
    MAX_CONCURRENT_JOBS: int = 3  # Max parallel training jobs
    
    # ===================
    # Redis / Celery
    # ===================
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Scheduler backend: "celery" (MVP) or "ray" (Phase 2)
    SCHEDULER_BACKEND: Literal["celery", "ray"] = "celery"
    
    # Worker settings
    WORKER_TTL_SECONDS: int = 90  # Mark offline after this many seconds
    
    # ===================
    # Ray (Phase 2 - distributed)
    # ===================
    RAY_HEAD_ADDRESS: Optional[str] = None  # None = local mode
    
    # ===================
    # Security
    # ===================
    SECRET_KEY: str = "change-me-in-production-flowml-secret-key"
    
    # ===================
    # CORS
    # ===================
    # Comma-separated in .env, parsed as list
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"
    
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string"""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    @property
    def max_upload_bytes(self) -> int:
        """Max upload size in bytes"""
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    
    @property
    def is_postgres(self) -> bool:
        """Check if using Postgres database"""
        return "postgresql" in self.DATABASE_URL
    
    @property
    def is_s3_storage(self) -> bool:
        """Check if using S3/MinIO storage"""
        return self.STORAGE_MODE == "s3"
    
    @field_validator("UPLOAD_DIR", "MODELS_DIR", "LOGS_DIR", "LOCAL_STORAGE_PATH", mode="before")
    @classmethod
    def ensure_path(cls, v):
        """Convert string to Path"""
        return Path(v) if isinstance(v, str) else v
    
    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance"""
    return Settings()


# Create directories on import
settings = get_settings()
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
