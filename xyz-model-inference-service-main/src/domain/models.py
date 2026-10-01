# src/domain/models.py
"""Domain models for inference service."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple




@dataclass
class FeatureVector:
    """Feature vector for matching."""
    entity_id_1: str
    entity_id_2: str
    features: List[float]
    feature_names: List[str]
    timestamp: datetime
    
    @property
    def dimension(self) -> int:
        return len(self.features)


@dataclass
class MatchScore:
    """Match scoring result."""
    score: float
    confidence_lower: float
    confidence_upper: float
    model_version: str
    inference_ms: int


@dataclass
class FieldContribution:
    """Per-field contribution to match score."""
    field_name: str
    contribution: float  # Percentage
    similarity: Optional[float] = None
    reasoning: Optional[str] = None


@dataclass
class ExplanationResult:
    """SHAP/LIME explanation result."""
    field_contributions: Dict[str, FieldContribution]
    top_k_fields: List[str]
    human_readable: str


@dataclass
class PredictionResult:
    """Complete inference result."""
    score: float
    confidence_interval: Tuple[float, float]
    decision: str
    model_version: str
    inference_ms: int
    explanation: Optional[ExplanationResult] = None
    cache_status: Optional[str] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None


@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking."""
    status: str  # closed, open, half_open
    error_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    open_until: Optional[datetime] = None
    failure_threshold: float = 0.5
    recovery_timeout_s: int = 30

class InferenceDecision(str, Enum):
    AUTO_MATCH = "AUTO_MATCH"
    LIKELY_MATCH = "LIKELY_MATCH"
    UNCERTAIN = "UNCERTAIN"
    LIKELY_NON = "LIKELY_NON"
    AUTO_NON_MATCH = "AUTO_NON_MATCH"


@dataclass
class MatchScore:
    score: float
    confidence_lower: float
    confidence_upper: float
    model_version: str
    inference_ms: int


@dataclass
class InferenceResult:
    score: float
    confidence_interval: Tuple[float, float]
    decision: str
    model_version: str
    inference_ms: int
    explanation: Optional[dict] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None

class FallbackReason(str, Enum):
    TIMEOUT = "timeout"
    TRITON_UNAVAILABLE = "triton_unavailable"
    FEATURE_STORE_ERROR = "feature_store_error"
    MODEL_NOT_LOADED = "model_not_loaded"
    MODEL_ERROR = "model_error"
    INTERNAL_ERROR = "internal_error"

@dataclass
class EntityAttributes:
    """Entity attributes used by the rule-based fallback engine."""
    entity_id: str
    email: Optional[str] = None
    phone: Optional[str] = None       # E.164 normalised, digits only
    full_name: Optional[str] = None
    address_line: Optional[str] = None
    city: Optional[str] = None
    postal_code: Optional[str] = None


class ModelStatus(str, Enum):
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    DEPRECATED = "deprecated"


@dataclass
class ModelMetadata:
    name: str
    status: ModelStatus
    version: str
    framework: str
    f1_score: float
    rollout_pct: int = 100
    accuracy: Optional[float] = None
    created_at: Optional[str] = None