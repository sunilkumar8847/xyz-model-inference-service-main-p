# src/infrastructure/errors.py
"""HTTP error handlers for FastAPI."""

from datetime import datetime, timezone
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

from src.domain.exceptions import (
    InferenceException,
    InferenceTimeoutError,
    ModelNotFoundError,
    TritonConnectionError,
    CircuitBreakerOpenError
)

logger = logging.getLogger(__name__)


def _trace_id(request: Request) -> str:
    """Extract trace-id from OTel context, falling back to request header."""
    try:
        from opentelemetry import trace
        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:
            return format(ctx.trace_id, '032x')
    except Exception:
        pass
    return request.headers.get("x-request-id", "")


async def inference_exception_handler(
    request: Request, 
    exc: InferenceException
) -> JSONResponse:
    """Handle domain-level inference exceptions."""
    logger.warning(f"Inference exception: {exc.message}", exc_info=True)
    
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    # Map specific exceptions to HTTP status codes
    if isinstance(exc, InferenceTimeoutError):
        status_code = status.HTTP_504_GATEWAY_TIMEOUT
    elif isinstance(exc, ModelNotFoundError):
        status_code = status.HTTP_404_NOT_FOUND
    elif isinstance(exc, TritonConnectionError):
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(exc, CircuitBreakerOpenError):
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error_code": exc.error_code,
            "message": exc.message,
            "trace_id": _trace_id(request),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    logger.warning(f"Validation error: {exc.errors()}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error_code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": exc.errors(),
            "trace_id": _trace_id(request),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "trace_id": _trace_id(request),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


def register_error_handlers(app) -> None:
    """Register all error handlers with FastAPI app."""
    from fastapi.exceptions import RequestValidationError
    
    app.add_exception_handler(InferenceException, inference_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)