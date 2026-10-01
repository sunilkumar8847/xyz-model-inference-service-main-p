# src/api/schemas/responses.py
"""Response schemas for API endpoints."""

from typing import Optional, List, Dict, Tuple
from pydantic import BaseModel, Field
from datetime import datetime


class FieldContributionResponse(BaseModel):
    """Per-field contribution to match score (SHAP value)."""
    
    field_name: str = Field(..., description="Field name")
    contribution: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Contribution to match score (0-1)"
    )
    similarity: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Field similarity score"
    )
    reasoning: Optional[str] = Field(
        None,
        description="Human-readable explanation"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "field_name": "email",
                "contribution": 0.35,
                "similarity": 1.0,
                "reasoning": "Exact email match"
            }
        }


class ExplanationResponse(BaseModel):
    """SHAP/LIME explanation for match decision."""
    
    field_contributions: Dict[str, FieldContributionResponse] = Field(
        ...,
        description="Per-field contributions"
    )
    top_k_fields: List[str] = Field(
        ...,
        description="Top K fields by contribution",
        max_items=10
    )
    human_readable: str = Field(
        ...,
        description="Human-readable explanation"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "field_contributions": {
                    "email": {
                        "field_name": "email",
                        "contribution": 0.35,
                        "similarity": 1.0
                    }
                },
                "top_k_fields": ["email", "name", "phone"],
                "human_readable": "High confidence match: email exact, name 94% similar"
            }
        }


class MatchPredictionResponse(BaseModel):
    """Response schema for match prediction."""
    
    score: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Match probability score"
    )
    confidence_interval: Tuple[float, float] = Field(
        ...,
        description="95% confidence interval [lower, upper]"
    )
    decision: str = Field(
        ...,
        description="Match classification decision"
    )
    model_version: str = Field(
        ...,
        description="Model version used for inference"
    )
    inference_ms: int = Field(
        ...,
        ge=0,
        description="Inference latency in milliseconds"
    )
    explanation: Optional[ExplanationResponse] = Field(
        None,
        description="SHAP explanation (if requested)"
    )
    fallback_used: bool = Field(
        False,
        description="Whether rule-based fallback was used"
    )
    fallback_reason: Optional[str] = Field(
        None,
        description="Reason for fallback activation"
    )
    cache_status: Optional[str] = Field(
        None,
        description="Cache hit/miss status"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "score": 0.96,
                "confidence_interval": [0.94, 0.98],
                "decision": "AUTO_MATCH",
                "model_version": "v5",
                "inference_ms": 145,
                "fallback_used": False
            }
        }


class BatchMatchResponse(BaseModel):
    """Response schema for batch match prediction."""
    
    results: List[MatchPredictionResponse] = Field(
        ...,
        description="Match results for each pair"
    )
    total: int = Field(
        ...,
        ge=0,
        description="Total number of results"
    )
    successful: int = Field(
        default=0,
        ge=0,
        description="Number of successful predictions"
    )
    failed: int = Field(
        default=0,
        ge=0,
        description="Number of failed predictions"
    )
    avg_inference_ms: float = Field(
        default=0,
        ge=0,
        description="Average inference time"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "score": 0.96,
                        "decision": "AUTO_MATCH",
                        "model_version": "v5",
                        "inference_ms": 145
                    }
                ],
                "total": 2,
                "successful": 2,
                "failed": 0,
                "avg_inference_ms": 142.5
            }
        }


class ModelMetadataResponse(BaseModel):
    """Response schema for model metadata."""
    
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    framework: str = Field(..., description="ML framework (PyTorch, ONNX, etc.)")
    f1_score: float = Field(..., ge=0, le=1, description="Model F1 score")
    status: str = Field(..., description="Model status (loading, ready, unavailable)")
    loaded_at: datetime = Field(..., description="When model was loaded")
    rollout_pct: int = Field(..., ge=0, le=100, description="Traffic percentage")
    is_champion: bool = Field(False, description="Is this the champion model")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "customer_matcher",
                "version": "v5",
                "framework": "ONNX",
                "f1_score": 0.947,
                "status": "ready",
                "loaded_at": "2025-01-18T10:30:00Z",
                "rollout_pct": 100,
                "is_champion": True
            }
        }


class ModelsListResponse(BaseModel):
    """Response schema for listing available models."""
    
    models: Dict[str, ModelMetadataResponse] = Field(
        ...,
        description="Available models by version"
    )
    champion_version: str = Field(
        ...,
        description="Current champion model version"
    )
    challenger_version: Optional[str] = Field(
        None,
        description="Challenger model version (if A/B testing)"
    )
    ab_test_enabled: bool = Field(
        False,
        description="Whether A/B testing is active"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "models": {
                    "v5": {
                        "name": "customer_matcher",
                        "version": "v5",
                        "f1_score": 0.947,
                        "status": "ready"
                    }
                },
                "champion_version": "v5",
                "ab_test_enabled": False
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(
        ...,
        description="Overall health status (healthy, degraded, unhealthy)"
    )
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(..., description="Current timestamp")
    triton_status: str = Field(..., description="Triton server status")
    feature_store_status: str = Field(..., description="Feature store status")
    redis_status: str = Field(..., description="Redis cache status")
    dependencies: Optional[Dict[str, str]] = Field(
        None,
        description="Additional dependency statuses"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "service": "model-inference-service",
                "version": "3.0.0",
                "timestamp": "2025-01-18T10:30:00Z",
                "triton_status": "healthy",
                "feature_store_status": "healthy",
                "redis_status": "healthy"
            }
        }


class MetricsResponse(BaseModel):
    """Response schema for service metrics."""
    
    total_requests: int = Field(..., ge=0)
    successful_requests: int = Field(..., ge=0)
    failed_requests: int = Field(..., ge=0)
    avg_latency_ms: float = Field(..., ge=0)
    p95_latency_ms: float = Field(..., ge=0)
    p99_latency_ms: float = Field(..., ge=0)
    cache_hit_rate: float = Field(..., ge=0, le=100)
    model_accuracy: float = Field(..., ge=0, le=1)
    
    class Config:
        schema_extra = {
            "example": {
                "total_requests": 10000,
                "successful_requests": 9950,
                "failed_requests": 50,
                "avg_latency_ms": 145.2,
                "p95_latency_ms": 185.0,
                "p99_latency_ms": 220.0,
                "cache_hit_rate": 96.5,
                "model_accuracy": 0.947
            }
        }