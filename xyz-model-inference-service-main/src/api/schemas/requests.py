# src/api/schemas/requests.py
"""Request schemas for API endpoints."""

from typing import Optional, List, Tuple
from pydantic import BaseModel, Field, validator


class MatchPredictionRequest(BaseModel):
    """Request schema for match prediction."""
    
    entity_id_1: str = Field(
        ...,
        description="First entity ID",
        example="CUST_001",
        min_length=1,
        max_length=255
    )
    entity_id_2: str = Field(
        ...,
        description="Second entity ID",
        example="CUST_002",
        min_length=1,
        max_length=255
    )
    model_version: Optional[str] = Field(
        None,
        description="Specific model version to use (or use A/B test routing)",
        example="v5"
    )
    include_shap: bool = Field(
        True,
        description="Include SHAP explanations in response"
    )
    timeout_ms: Optional[int] = Field(
        None,
        description="Inference timeout in milliseconds",
        ge=50,
        le=5000
    )
    
    @validator('entity_id_1', 'entity_id_2')
    def validate_entity_id(cls, v):
        """Validate entity ID format."""
        if not v or v.strip() == "":
            raise ValueError("Entity ID cannot be empty")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "entity_id_1": "CUST_001",
                "entity_id_2": "CUST_002",
                "model_version": "v5",
                "include_shap": True,
                "timeout_ms": 200
            }
        }


class BatchMatchRequest(BaseModel):
    """Request schema for batch match prediction."""
    
    entity_pairs: List[Tuple[str, str]] = Field(
        ...,
        description="List of entity ID pairs to match",
        min_items=1,
        max_items=1000
    )
    timeout_ms: Optional[int] = Field(
        None,
        description="Inference timeout in milliseconds per pair",
        ge=100,
        le=10000
    )
    
    @validator('entity_pairs')
    def validate_pairs(cls, v):
        """Validate entity pairs."""
        if not v:
            raise ValueError("Must provide at least one entity pair")
        if len(v) > 1000:
            raise ValueError("Maximum 1000 pairs allowed per batch")
        
        for pair in v:
            if len(pair) != 2:
                raise ValueError("Each pair must contain exactly 2 entity IDs")
            if not pair[0] or not pair[1]:
                raise ValueError("Entity IDs cannot be empty")
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "entity_pairs": [
                    ["CUST_001", "CUST_002"],
                    ["CUST_003", "CUST_004"]
                ],
                "timeout_ms": 500
            }
        }


class EmbeddingRequest(BaseModel):
    """Request schema for generating embeddings."""
    
    entity_id: str = Field(
        ...,
        description="Entity ID to generate embedding for",
        example="CUST_001"
    )
    model_version: Optional[str] = Field(
        None,
        description="Embedding model version",
        example="v1"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "entity_id": "CUST_001",
                "model_version": "v1"
            }
        }


class ExplainRequest(BaseModel):
    """Request schema for explaining a match decision."""
    
    entity_id_1: str = Field(..., description="First entity ID")
    entity_id_2: str = Field(..., description="Second entity ID")
    model_version: Optional[str] = Field(None, description="Model version")
    
    class Config:
        schema_extra = {
            "example": {
                "entity_id_1": "CUST_001",
                "entity_id_2": "CUST_002",
                "model_version": "v5"
            }
        }