# src/adapters/exceptions.py
"""Service exceptions."""


class InferenceException(Exception):
    """Base inference exception."""
    pass


class InferenceTimeoutError(InferenceException):
    """Inference timed out."""
    pass


class ModelNotFoundError(InferenceException):
    """Model version not found."""
    pass


class TritonConnectionError(InferenceException):
    """Failed to connect to Triton."""
    pass


class FeatureStoreError(InferenceException):
    """Feature store operation failed."""
    pass


class CircuitBreakerError(InferenceException):
    """Circuit breaker is open."""
    pass
