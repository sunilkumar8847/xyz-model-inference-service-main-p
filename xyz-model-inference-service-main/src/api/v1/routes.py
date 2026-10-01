# src/api/v1/routes.py
"""REST API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
import logging

from xyz_security import TenantContext, get_current_tenant

from datetime import datetime, timezone

from src.api.v1.schemas import (
    MatchPredictionRequest, MatchPredictionResponse,
    BatchMatchRequest, BatchMatchResponse,
    ModelsListResponse, ModelMetadataResponse, HealthResponse
)
from src.services.inference_service import InferenceService
from src.services.model_manager import ModelManager
from src.infrastructure.observability import http_requests_total
from src.config.settings import settings as svc_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["inference"])


async def get_inference_service() -> InferenceService:
    """Dependency injection for inference service."""
    from src.main import inference_service
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return inference_service


async def get_model_manager() -> ModelManager:
    """Dependency injection for model manager."""
    from src.main import model_manager
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return model_manager


@router.post(
    "/matching/predict",
    response_model=MatchPredictionResponse,
    summary="Predict if two entities are duplicates"
)
async def predict_match(
    request: MatchPredictionRequest,
    tenant: TenantContext = Depends(get_current_tenant),
    service: InferenceService = Depends(get_inference_service)
) -> MatchPredictionResponse:
    """
    Predict if two entities are matches (duplicates).
    
    Uses ensemble AI model (Transformer + GNN + XGBoost) with:
    - Sub-200ms latency
    - Confidence intervals
    - SHAP explanations
    - Fallback to rule-based matching
    
    Scores:
    - >= 0.95: AUTO_MATCH (auto-merge)
    - 0.85-0.95: LIKELY_MATCH (human review)
    - 0.50-0.85: UNCERTAIN (expert review)
    - 0.15-0.50: LIKELY_NON (verification)
    - < 0.15: AUTO_NON_MATCH (reject)
    """
    try:
        http_requests_total.labels(endpoint="predict_match", status="started").inc()
        
        result = await service.predict_match(
            entity_id_1=request.entity_id_1,
            entity_id_2=request.entity_id_2,
            model_version=request.model_version,
            include_shap=request.include_shap,
            timeout_ms=request.timeout_ms
        )
        
        http_requests_total.labels(endpoint="predict_match", status="success").inc()
        
        return MatchPredictionResponse(
            score=result.score,
            confidence_interval=result.confidence_interval,
            decision=result.decision,
            model_version=result.model_version,
            inference_ms=result.inference_ms,
            explanation=result.explanation,
            fallback_used=result.fallback_used,
            fallback_reason=result.fallback_reason
        )
    
    except TimeoutError:
        http_requests_total.labels(endpoint="predict_match", status="timeout").inc()
        raise HTTPException(status_code=504, detail="Inference timeout")
    
    except Exception as e:
        logger.exception("Prediction failed")
        http_requests_total.labels(endpoint="predict_match", status="error").inc()
        raise HTTPException(status_code=500, detail="Inference failed")


@router.post(
    "/matching/batch",
    response_model=BatchMatchResponse,
    summary="Batch predict matches"
)
async def batch_predict(
    request: BatchMatchRequest,
    tenant: TenantContext = Depends(get_current_tenant),
    service: InferenceService = Depends(get_inference_service)
) -> BatchMatchResponse:
    """
    Predict matches for up to 1000 entity pairs in parallel.
    
    Optimized for throughput with batched Triton inference.
    """
    try:
        http_requests_total.labels(endpoint="batch_predict", status="started").inc()
        
        results = await service.batch_predict(
            entity_pairs=request.entity_pairs,
            timeout_ms=request.timeout_ms
        )
        
        http_requests_total.labels(endpoint="batch_predict", status="success").inc()
        
        return BatchMatchResponse(
            results=[
                MatchPredictionResponse(
                    score=r.score,
                    confidence_interval=r.confidence_interval,
                    decision=r.decision,
                    model_version=r.model_version,
                    inference_ms=r.inference_ms,
                    fallback_used=r.fallback_used,
                    fallback_reason=r.fallback_reason
                )
                for r in results
            ],
            total=len(results)
        )
    
    except Exception as e:
        logger.exception("Batch prediction failed")
        http_requests_total.labels(endpoint="batch_predict", status="error").inc()
        raise HTTPException(status_code=500, detail="Batch inference failed")


@router.get(
    "/models",
    response_model=ModelsListResponse,
    summary="List available models"
)
async def list_models(
    model_manager: ModelManager = Depends(get_model_manager),
    tenant: TenantContext = Depends(get_current_tenant),
) -> ModelsListResponse:
    """List all available models with metadata."""
    active_models = model_manager.get_active_models()

    # Only surface versions actually loaded (local mode has just "local",
    # while settings.champion_version may default to "v5").
    rollout_by_version = {v: pct for v, pct in active_models}
    versions = [v for v, _ in active_models if v in model_manager._models]
    if not versions:
        versions = list(model_manager._models.keys())

    def _to_response(meta, rollout_pct: float) -> ModelMetadataResponse:
        return ModelMetadataResponse(
            name=meta.name,
            version=meta.version,
            framework=meta.framework,
            f1_score=meta.f1_score,
            status=getattr(meta.status, "value", str(meta.status)),
            loaded_at=meta.created_at or datetime.now(timezone.utc),
            rollout_pct=int(meta.rollout_pct if meta.rollout_pct is not None else rollout_pct),
        )

    models = {
        v: _to_response(model_manager._models[v], rollout_by_version.get(v, 100.0))
        for v in versions
    }

    champion_key = (
        svc_settings.champion_version
        if svc_settings.champion_version in model_manager._models
        else next(iter(model_manager._models), "local")
    )

    return ModelsListResponse(
        models=models,
        champion_version=champion_key,
        ab_test_enabled=svc_settings.ab_test_enabled,
    )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns status of all dependencies:
    - Triton Inference Server
    - Feature Store (Feast)
    - Redis Cache
    - Model Registry (MLflow)
    """
    from src.main import health_checker
    
    health = await health_checker.check_all()
    
    return HealthResponse(
        status=health['overall'],
        service="model-inference-service",
        version="3.0.0",
        timestamp=datetime.now(),
        triton_status=health['triton'],
        feature_store_status=health['feature_store'],
        redis_status=health['redis']
    )