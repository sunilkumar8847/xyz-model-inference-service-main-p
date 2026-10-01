# src/api/v1/schemas.py
"""Pydantic request/response schemas for API v1."""

from typing import Optional, List, Dict, Tuple
from pydantic import BaseModel, Field, validator
from datetime import datetime


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class MatchPredictionRequest(BaseModel):
    """Request schema for match prediction."""
    
    entity_id_1: str = Field(..., description="First entity ID", min_length=1, max_length=255)
    entity_id_2: str = Field(..., description="Second entity ID", min_length=1, max_length=255)
    model_version: Optional[str] = Field(None, description="Specific model version")
    include_shap: bool = Field(True, description="Include SHAP explanations")
    timeout_ms: Optional[int] = Field(None, description="Timeout in milliseconds", ge=50, le=5000)
    
    @validator('entity_id_1', 'entity_id_2')
    def validate_entity_id(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Entity ID cannot be empty")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "entity_id_1": "CUST_001",
                "entity_id_2": "CUST_002",
                "include_shap": True
            }
        }


class BatchMatchRequest(BaseModel):
    """Request schema for batch match prediction."""
    
    entity_pairs: List[Tuple[str, str]] = Field(..., min_items=1, max_items=1000)
    timeout_ms: Optional[int] = Field(None, ge=100, le=10000)
    
    @validator('entity_pairs')
    def validate_pairs(cls, v):
        if not v:
            raise ValueError("Must provide at least one entity pair")
        if len(v) > 1000:
            raise ValueError("Maximum 1000 pairs allowed")
        for pair in v:
            if len(pair) != 2 or not pair[0] or not pair[1]:
                raise ValueError("Each pair must contain 2 non-empty entity IDs")
        return v


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class FieldContributionResponse(BaseModel):
    """Per-field SHAP contribution."""
    
    field_name: str
    contribution: float = Field(..., ge=0, le=1)
    similarity: Optional[float] = Field(None, ge=0, le=1)
    reasoning: Optional[str] = None


class ExplanationResponse(BaseModel):
    """SHAP explanation for match decision."""
    
    field_contributions: Dict[str, FieldContributionResponse]
    top_k_fields: List[str]
    human_readable: str


class MatchPredictionResponse(BaseModel):
    """Response schema for match prediction."""
    
    score: float = Field(..., ge=0, le=1)
    confidence_interval: Tuple[float, float]
    decision: str
    model_version: str
    inference_ms: int = Field(..., ge=0)
    explanation: Optional[ExplanationResponse] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    cache_status: Optional[str] = None


class BatchMatchResponse(BaseModel):
    """Response schema for batch prediction."""
    
    results: List[MatchPredictionResponse]
    total: int = Field(..., ge=0)


class ModelMetadataResponse(BaseModel):
    """Model metadata response."""
    
    name: str
    version: str
    framework: str
    f1_score: float = Field(..., ge=0, le=1)
    status: str
    loaded_at: datetime
    rollout_pct: int = Field(..., ge=0, le=100)


class ModelsListResponse(BaseModel):
    """Available models list."""
    
    models: Dict[str, ModelMetadataResponse]
    champion_version: str
    challenger_version: Optional[str] = None
    ab_test_enabled: bool


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    service: str
    version: str
    timestamp: datetime
    triton_status: str
    feature_store_status: str
    redis_status: str