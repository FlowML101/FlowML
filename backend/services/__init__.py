"""
Services package
"""
from services.automl import AutoMLService
from services.websocket_manager import manager

__all__ = ["AutoMLService", "manager"]
