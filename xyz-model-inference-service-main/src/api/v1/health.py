# src/api/v1/health.py
"""Health check endpoints."""

from fastapi import APIRouter
from datetime import datetime
from typing import Dict, Any

from src.api.v1.schemas import HealthResponse
from src.config.settings import settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Comprehensive health check endpoint.
    
    Checks:
    - Triton Inference Server connectivity
    - Redis cache availability
    - Feature Store (Feast) connectivity
    
    Returns:
    - healthy: All dependencies available
    - degraded: Some dependencies unavailable but service operational
    - unhealthy: Critical dependencies unavailable
    """
    from src.main import health_checker
    
    if not health_checker:
        return HealthResponse(
            status="unhealthy",
            service=settings.service_name,
            version=settings.version,
            timestamp=datetime.now(),
            triton_status="unknown",
            feature_store_status="unknown",
            redis_status="unknown"
        )
    
    health_status = await health_checker.check_all()
    
    return HealthResponse(
        status=health_status['overall'],
        service=settings.service_name,
        version=settings.version,
        timestamp=datetime.now(),
        triton_status=health_status['triton'],
        feature_store_status=health_status['feature_store'],
        redis_status=health_status['redis']
    )


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Kubernetes readiness probe endpoint."""
    from src.main import health_checker
    
    try:
        if health_checker:
            health = await health_checker.check_all()
            is_ready = health['overall'] in ['healthy', 'degraded']
        else:
            is_ready = False
        
        return {
            "ready": is_ready,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "ready": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """Kubernetes liveness probe endpoint."""
    return {
        "alive": True,
        "timestamp": datetime.now().isoformat(),
        "service": settings.service_name,
        "version": settings.version
    }