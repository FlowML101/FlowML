"""
FlowML Studio - Backend API
FastAPI application entry point
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys

from config import settings
from database import init_db
from services.websocket_manager import manager
from routers import (
    datasets_router,
    training_router,
    results_router,
    workers_router,
    stats_router
)


# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG" if settings.DEBUG else "INFO"
)
logger.add(
    "logs/flowml_{time}.log",
    rotation="500 MB",
    retention="10 days",
    level="DEBUG"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown"""
    # Startup
    logger.info("🚀 Starting FlowML Studio Backend...")
    await init_db()
    logger.info("✅ Database initialized")
    logger.info(f"📁 Upload directory: {settings.UPLOAD_DIR}")
    logger.info(f"📁 Models directory: {settings.MODELS_DIR}")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down FlowML Studio Backend...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Privacy-first distributed AutoML platform",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(datasets_router, prefix="/api")
app.include_router(training_router, prefix="/api")
app.include_router(results_router, prefix="/api")
app.include_router(workers_router, prefix="/api")
app.include_router(stats_router, prefix="/api")


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time updates.
    Handles: job updates, worker status, log streaming, system events
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive and handle messages
            data = await websocket.receive_json()
            
            # Handle different message types from client
            msg_type = data.get("type")
            payload = data.get("payload", {})
            
            if msg_type == "handshake":
                logger.info(f"Client handshake: {payload}")
                await manager.send_personal(websocket, {
                    "type": "system_event",
                    "payload": {
                        "level": "success",
                        "title": "Handshake Complete",
                        "description": f"Connected to FlowML v{settings.APP_VERSION}"
                    }
                })
            
            elif msg_type == "subscribe":
                # Subscribe to a room (e.g., job updates)
                room = payload.get("room")
                if room:
                    if room not in manager.rooms:
                        manager.rooms[room] = set()
                    manager.rooms[room].add(websocket)
                    logger.info(f"Client subscribed to room: {room}")
            
            elif msg_type == "ping":
                await manager.send_personal(websocket, {"type": "pong"})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# WebSocket endpoint for specific job updates
@app.websocket("/ws/job/{job_id}")
async def job_websocket(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for job-specific updates"""
    await manager.connect(websocket, room=f"job_{job_id}")
    
    try:
        while True:
            data = await websocket.receive_json()
            # Handle job-specific messages
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
