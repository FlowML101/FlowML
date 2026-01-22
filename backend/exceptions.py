"""
FlowML Custom Exceptions - Centralized error handling
"""
from fastapi import HTTPException, status
from typing import Any


class FlowMLException(Exception):
    """Base exception for FlowML"""
    def __init__(self, message: str, details: Any = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class DatasetError(FlowMLException):
    """Dataset-related errors"""
    pass


class TrainingError(FlowMLException):
    """Training/AutoML errors"""
    pass


class ModelError(FlowMLException):
    """Model loading/prediction errors"""
    pass


class ValidationError(FlowMLException):
    """Input validation errors"""
    pass


# HTTP Exception helpers for consistent responses
def not_found(resource: str, id: str | int) -> HTTPException:
    """Return 404 Not Found"""
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"{resource} with ID '{id}' not found"
    )


def bad_request(message: str, details: Any = None) -> HTTPException:
    """Return 400 Bad Request"""
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"message": message, "details": details} if details else message
    )


def conflict(message: str) -> HTTPException:
    """Return 409 Conflict"""
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=message
    )


def server_error(message: str = "Internal server error") -> HTTPException:
    """Return 500 Internal Server Error"""
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=message
    )


def file_too_large(max_size_mb: int) -> HTTPException:
    """Return 413 Payload Too Large"""
    return HTTPException(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        detail=f"File too large. Maximum size is {max_size_mb}MB"
    )


def invalid_file_type(allowed: set[str]) -> HTTPException:
    """Return 415 Unsupported Media Type"""
    return HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail=f"Invalid file type. Allowed: {', '.join(allowed)}"
    )


def service_unavailable(service: str) -> HTTPException:
    """Return 503 Service Unavailable"""
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"{service} is currently unavailable"
    )
