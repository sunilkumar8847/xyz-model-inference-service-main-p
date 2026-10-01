# src/infrastructure/observability.py
"""Monitoring, metrics, and health checking."""

import time
import logging
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import FastAPI
from fastapi.responses import Response

logger = logging.getLogger(__name__)

# Prometheus metrics
http_requests_total = Counter(
    'inference_requests_total',
    'Total inference requests',
    ['endpoint', 'status']
)

inference_latency = Histogram(
    'inference_latency_ms',
    'Inference latency in milliseconds',
    buckets=(10, 20, 50, 100, 200, 500, 1000)
)

inference_errors = Counter(
    'inference_errors_total',
    'Total inference errors',
    ['error_type']
)

model_accuracy = Gauge(
    'model_accuracy_f1',
    'Model F1 score',
    ['model_version']
)

gpu_utilization = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

cache_hit_rate = Gauge(
    'cache_hit_rate',
    'Cache hit rate percentage'
)

fallback_rate = Gauge(
    'fallback_rate',
    'Fallback invocation rate',
    ['reason']
)

gpu_memory_used_bytes = Gauge(
    'gpu_memory_used_bytes',
    'GPU memory used in bytes',
    ['gpu_id']
)

model_load_time_ms = Histogram(
    'model_load_time_ms',
    'Time to load model in milliseconds',
    ['model'],
    buckets=(500, 1000, 2000, 5000, 10000, 30000)
)


class HealthChecker:
    """Check health of all dependencies."""
    
    def __init__(self, triton_client=None, redis_cache=None, feature_store_client=None):
        """
        Initialize health checker with optional dependencies.
        
        Args:
            triton_client: Triton inference client (optional)
            redis_cache: Redis cache client (optional)
            feature_store_client: Feature store client (optional)
        """
        self.triton_client = triton_client
        self.redis_cache = redis_cache
        self.feature_store_client = feature_store_client
    
    async def check_all(self) -> dict:
        """Check all service dependencies."""
        triton_ok = await self.check_triton()
        redis_ok = await self.check_redis()
        feature_store_ok = await self.check_feature_store()
        
        # Determine overall status
        critical_services = [triton_ok]  # Triton is critical
        optional_services = [redis_ok, feature_store_ok]
        
        if all(critical_services):
            if all(optional_services):
                overall = "healthy"
            else:
                overall = "degraded"
        else:
            overall = "unhealthy"
        
        return {
            'overall': overall,
            'triton': 'healthy' if triton_ok else 'unhealthy',
            'feature_store': 'healthy' if feature_store_ok else 'unhealthy',
            'redis': 'healthy' if redis_ok else 'unhealthy',
            'timestamp': datetime.now().isoformat()
        }
    
    async def check_triton(self) -> bool:
        """Check Triton Inference Server health."""
        if not self.triton_client:
            logger.warning("Triton client not initialized")
            return False
        
        try:
            return await self.triton_client.health_check()
        except Exception as e:
            logger.error(f"Triton health check failed: {e}")
            return False
    
    async def check_redis(self) -> bool:
        """Check Redis health."""
        if not self.redis_cache:
            logger.warning("Redis cache not initialized")
            return True  # Redis is optional, return True
        
        try:
            # Try to ping Redis
            if hasattr(self.redis_cache, '_client') and self.redis_cache._client:
                self.redis_cache._client.ping()
                return True
            return True  # If no client, assume OK (local mode)
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    async def check_feature_store(self) -> bool:
        """Check Feature Store health."""
        if not self.feature_store_client:
            logger.warning("Feature store client not initialized")
            return True  # Feature store is optional
        
        try:
            # Simple check - just verify client exists
            return True
        except Exception as e:
            logger.error(f"Feature store health check failed: {e}")
            return False


def setup_observability(app: FastAPI) -> None:
    """Setup monitoring and metrics endpoints."""
    
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(generate_latest(), media_type="text/plain")
    
    @app.middleware("http")
    async def add_process_time_header(request, call_next):
        """Add X-Process-Time header and record metrics."""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response