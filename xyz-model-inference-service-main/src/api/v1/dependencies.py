# src/api/v1/dependencies.py
"""FastAPI dependency injection."""

from typing import Optional
import logging

from src.services.inference_service import InferenceService
from src.services.model_manager import ModelManager

logger = logging.getLogger(__name__)


# Global service instances (set during app startup)
_inference_service: Optional[InferenceService] = None
_model_manager: Optional[ModelManager] = None


def set_inference_service(service: InferenceService) -> None:
    """Set global inference service instance."""
    global _inference_service
    _inference_service = service


def set_model_manager(manager: ModelManager) -> None:
    """Set global model manager instance."""
    global _model_manager
    _model_manager = manager


async def get_inference_service() -> InferenceService:
    """Get inference service dependency."""
    if _inference_service is None:
        raise RuntimeError("Inference service not initialized")
    return _inference_service


async def get_model_manager() -> ModelManager:
    """Get model manager dependency."""
    if _model_manager is None:
        raise RuntimeError("Model manager not initialized")
    return _model_manager
