# src/domain/enums.py
"""Domain enumerations."""

from enum import Enum


class ModelStatus(str, Enum):
    """Model deployment status."""
    LOADING = "loading"
    READY = "ready"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"


class InferenceDecision(str, Enum):
    """Match classification decision."""
    AUTO_MATCH = "AUTO_MATCH"
    LIKELY_MATCH = "LIKELY_MATCH"
    UNCERTAIN = "UNCERTAIN"
    LIKELY_NON = "LIKELY_NON"
    AUTO_NON_MATCH = "AUTO_NON_MATCH"


class FallbackReason(str, Enum):
    """Why fallback was triggered."""
    TIMEOUT = "timeout"
    MODEL_ERROR = "model_error"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    GPU_OOM = "gpu_oom"
    INTENTIONAL = "intentional"


class CacheStatus(str, Enum):
    """Cache lookup result."""
    HIT = "hit"
    MISS = "miss"
    EXPIRED = "expired"
    ERROR = "error"