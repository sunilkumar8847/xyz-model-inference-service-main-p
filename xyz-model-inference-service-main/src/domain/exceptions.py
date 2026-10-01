# src/domain/exceptions.py
"""Domain-level exceptions."""


class InferenceException(Exception):
    """Base exception for inference operations."""
    
    def __init__(self, message: str, error_code: str = "INFERENCE_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class ModelNotFoundError(InferenceException):
    """Raised when requested model version is not found."""
    
    def __init__(self, model_version: str):
        super().__init__(
            f"Model version '{model_version}' not found in registry",
            error_code="MODEL_NOT_FOUND"
        )


class InferenceTimeoutError(InferenceException):
    """Raised when inference exceeds timeout."""
    
    def __init__(self, timeout_ms: int, entity_id_1: str = None, entity_id_2: str = None):
        msg = f"Inference exceeded timeout ({timeout_ms}ms)"
        if entity_id_1 and entity_id_2:
            msg += f" for pair ({entity_id_1}, {entity_id_2})"
        super().__init__(msg, error_code="INFERENCE_TIMEOUT")


class TritonConnectionError(InferenceException):
    """Raised when unable to connect to Triton server."""
    
    def __init__(self, host: str, port: int):
        super().__init__(
            f"Failed to connect to Triton at {host}:{port}",
            error_code="TRITON_CONNECTION_ERROR"
        )


class FeatureStoreError(InferenceException):
    """Raised when feature store operation fails."""
    
    def __init__(self, operation: str, entity_id_1: str = None, entity_id_2: str = None):
        msg = f"Feature store operation '{operation}' failed"
        if entity_id_1 and entity_id_2:
            msg += f" for pair ({entity_id_1}, {entity_id_2})"
        super().__init__(msg, error_code="FEATURE_STORE_ERROR")


class CircuitBreakerOpenError(InferenceException):
    """Raised when circuit breaker is open."""
    
    def __init__(self, service_name: str):
        super().__init__(
            f"Circuit breaker for '{service_name}' is open",
            error_code="CIRCUIT_BREAKER_OPEN"
        )


class InvalidConfigurationError(InferenceException):
    """Raised when configuration is invalid."""
    
    def __init__(self, config_key: str, reason: str):
        super().__init__(
            f"Invalid configuration for '{config_key}': {reason}",
            error_code="INVALID_CONFIGURATION"
        )
