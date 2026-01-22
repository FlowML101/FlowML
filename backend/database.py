"""
Database setup with SQLModel - Optimized with connection pooling
"""
from typing import AsyncGenerator
from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool
from contextlib import asynccontextmanager
from config import settings

# Async engine with connection pooling
# For SQLite, use StaticPool to share connection across threads
if "sqlite" in settings.DATABASE_URL:
    async_engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # Single connection for SQLite
    )
else:
    # For PostgreSQL/MySQL - use proper connection pooling
    async_engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        future=True,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,  # Recycle connections after 30 min
    )

# Async session factory with expire_on_commit=False for better performance
async_session = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,  # Manual flush for better control
)


async def init_db():
    """Initialize database tables"""
    from models import dataset, job, trained_model, worker  # noqa: F401
    
    async with async_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database session"""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_session_context():
    """Context manager for background tasks (not FastAPI dependency)"""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def close_db():
    """Close database connections on shutdown"""
    await async_engine.dispose()
