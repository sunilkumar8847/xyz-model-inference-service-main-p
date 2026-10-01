# src/api/schemas/common.py
"""Common schema definitions."""

from typing import Optional, Any, Dict
from pydantic import BaseModel, Field
from datetime import datetime


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class PaginationMetadata(BaseModel):
    """Pagination metadata."""
    
    limit: int
    offset: int
    total: int
    has_next: bool = False


class ServiceInfoResponse(BaseModel):
    """Service information response."""
    
    name: str
    version: str
    environment: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)