"""
WebSocket Manager - handles real-time connections and broadcasts
"""
from fastapi import WebSocket
from typing import Dict, Set
import json
from datetime import datetime
from loguru import logger


class WebSocketManager:
    """
    Manages WebSocket connections and message broadcasting.
    Supports multiple clients and room-based messaging.
    """
    
    def __init__(self):
        # All active connections
        self.active_connections: Set[WebSocket] = set()
        
        # Connections by room (e.g., job_id for training updates)
        self.rooms: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, room: str = "global"):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        
        # Add to room
        if room not in self.rooms:
            self.rooms[room] = set()
        self.rooms[room].add(websocket)
        
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
        
        # Send welcome message
        await websocket.send_json({
            "type": "system_event",
            "payload": {
                "level": "info",
                "title": "Connected",
                "description": "WebSocket connection established"
            },
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.active_connections.discard(websocket)
        
        # Remove from all rooms
        for room in self.rooms.values():
            room.discard(websocket)
        
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict, room: str = None):
        """
        Broadcast a message to all connections or a specific room.
        
        Args:
            message: Dict to send (will be JSON serialized)
            room: Optional room name. If None, broadcasts to all.
        """
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()
        
        # Get target connections
        if room and room in self.rooms:
            targets = self.rooms[room]
        else:
            targets = self.active_connections
        
        # Send to all targets
        disconnected = set()
        for connection in targets:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn)
    
    async def send_personal(self, websocket: WebSocket, message: dict):
        """Send a message to a specific connection"""
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()
        
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast_job_update(
        self,
        job_id: str,
        status: str,
        progress: float,
        current_model: str = None,
        message: str = None
    ):
        """Convenience method for job updates"""
        await self.broadcast({
            "type": "job_update",
            "payload": {
                "jobId": job_id,
                "status": status,
                "progress": progress,
                "currentModel": current_model,
                "message": message
            }
        })
    
    async def broadcast_worker_status(
        self,
        worker_id: str,
        worker_name: str,
        status: str
    ):
        """Convenience method for worker status updates"""
        await self.broadcast({
            "type": "worker_status",
            "payload": {
                "workerId": worker_id,
                "workerName": worker_name,
                "status": status
            }
        })
    
    async def broadcast_log(
        self,
        source: str,
        level: str,
        message: str
    ):
        """Convenience method for log streaming"""
        await self.broadcast({
            "type": "log_stream",
            "payload": {
                "source": source,
                "level": level,
                "message": message
            }
        })
    
    @property
    def connection_count(self) -> int:
        """Get current number of active connections"""
        return len(self.active_connections)


# Global manager instance
manager = WebSocketManager()
