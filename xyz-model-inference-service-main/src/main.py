# src/main.py
"""FastAPI application entry point."""

import logging
import os
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


def _setup_otel(service_name: str, endpoint: str) -> None:
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
    except Exception as e:
        logging.getLogger(__name__).warning(f"OTel setup failed (non-fatal): {e}")


def _instrument_fastapi(app: "FastAPI") -> None:
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
    except Exception as e:
        logging.getLogger(__name__).warning(f"OTel FastAPI instrumentation failed: {e}")

from xyz_security import get_current_tenant
from src.config.settings import settings
from src.config.logging import setup_logging
from src.api.v1 import routes as api_routes
from src.api.v1 import health as health_routes
from src.adapters.triton_client import TritonClient
from src.adapters.feature_store_client import FeatureStoreClient
from src.adapters.redis_client import RedisCache
from src.adapters.mlflow_client import MLflowClient
from src.services.model_manager import ModelManager
from src.services.fallback_service import FallbackService
from src.services.inference_service import InferenceService
from src.infrastructure.observability import setup_observability, HealthChecker
from src.infrastructure.errors import register_error_handlers
from src.grpc.server import GrpcServer

# Setup logging
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

# Global service instances
triton_client: TritonClient = None
feature_store_client: FeatureStoreClient = None
redis_cache: RedisCache = None
mlflow_client: MLflowClient = None
model_manager: ModelManager = None
fallback_service: FallbackService = None
inference_service: InferenceService = None
health_checker: HealthChecker = None
grpc_server: GrpcServer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    global triton_client, feature_store_client, redis_cache, mlflow_client
    global model_manager, fallback_service, inference_service, health_checker, grpc_server
    
    # Startup
    logger.info(f"Starting {settings.service_name} v{settings.version}")
    logger.info(f"Environment: {settings.environment}")
    
    try:
        # Initialize infrastructure (with local dev fallbacks)
        logger.info("Initializing Triton client...")
        triton_client = TritonClient()
        try:
            await triton_client.initialize()
            logger.info("✓ Triton client initialized")
        except Exception as e:
            logger.warning(f"Triton initialization failed (using mock): {e}")
        
        logger.info("Initializing Redis cache...")
        redis_cache = RedisCache(settings.redis)
        try:
            await redis_cache.connect()
            logger.info("✓ Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis connection failed (disabled): {e}")
        
        logger.info("Initializing Feature Store client...")
        feature_store_client = FeatureStoreClient()
        logger.info("✓ Feature Store client initialized")
        
        logger.info("Initializing MLflow client...")
        mlflow_client = MLflowClient()
        logger.info("✓ MLflow client initialized")
        
        # Initialize services
        logger.info("Initializing Model Manager...")
        model_manager = ModelManager(redis_cache, mlflow_client)
        await model_manager.initialize()
        logger.info("✓ Model Manager initialized")
        
        logger.info("Initializing Fallback Service...")
        fallback_service = FallbackService(feature_store_client=feature_store_client)
        logger.info("✓ Fallback Service initialized")
        
        logger.info("Initializing Inference Service...")
        inference_service = InferenceService(
            triton_client,
            feature_store_client,
            model_manager,
            fallback_service
        )
        logger.info("✓ Inference Service initialized")
        
        # Initialize health checker with dependencies
        logger.info("Initializing Health Checker...")
        health_checker = HealthChecker(
            triton_client=triton_client,
            redis_cache=redis_cache,
            feature_store_client=feature_store_client
        )
        logger.info("✓ Health Checker initialized")
        
        # Start gRPC server on port 9090
        logger.info("Starting gRPC server on port 9090...")
        grpc_server = GrpcServer(port=9090)
        try:
            await grpc_server.start(inference_service, triton_client)
            logger.info("✓ gRPC server started on port 9090")
        except Exception as e:
            logger.warning(f"gRPC server start failed (non-fatal in local mode): {e}")

        logger.info(f"🚀 {settings.service_name} started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {settings.service_name}")
    if redis_cache:
        try:
            await redis_cache.disconnect()
            logger.info("✓ Redis disconnected")
        except Exception as e:
            logger.warning(f"Redis disconnect error: {e}")
    
    if grpc_server:
        try:
            await grpc_server.stop()
            logger.info("✓ gRPC server stopped")
        except Exception as e:
            logger.warning(f"gRPC server stop error: {e}")

    logger.info("✓ Shutdown complete")


# Setup OTel before app creation
_setup_otel(
    service_name="model-inference-service",
    endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317"),
)

_is_prod = settings.environment in ("production", "staging", "prod")

# Create FastAPI app
app = FastAPI(
    title="XYZ MDM — Model Inference Service",
    version="1.0.0",
    description="AI-powered model inference for entity matching — BERT + GNN + XGBoost ensemble, Triton-backed low-latency scoring",
    contact={"name": "XYZ MDM Platform", "email": "engineering@xyzmdm.com"},
    servers=[{"url": "http://localhost:8035", "description": "Integration"}],
    lifespan=lifespan,
    docs_url=None if _is_prod else "/swagger-ui.html",
    redoc_url=None if _is_prod else "/redoc",
    openapi_url=None if _is_prod else "/openapi.json",
)

_instrument_fastapi(app)

# CORS — restrict to configured origins; never wildcard
_cors_origins = [o.strip() for o in os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Verified-Tenant-ID",
                   "X-Verified-User-ID", "X-Verified-Roles", "X-Verified-Platform",
                   "X-Request-Timestamp", "X-Gateway-Signature"],
)

# Setup observability
setup_observability(app)

# Register error handlers
register_error_handlers(app)

# Include API routes — HMAC-verified gateway headers required
app.include_router(api_routes.router, dependencies=[Depends(get_current_tenant)])
app.include_router(health_routes.router)  # health router intentionally has no auth


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.service_name,
        "version": settings.version,
        "status": "running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower()
    )