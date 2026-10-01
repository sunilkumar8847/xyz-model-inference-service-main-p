# src/tests/test_model_manager.py
"""Tests for ModelManager."""

import pytest

from src.services.model_manager import ModelManager
from src.tests.mocks.mock_mlflow import MockMLflowClient


@pytest.fixture
async def model_manager():
    """Create model manager with mocks."""
    mlflow = MockMLflowClient()
    return ModelManager(None, mlflow)


@pytest.mark.asyncio
async def test_get_active_models_champion_only(model_manager):
    """Test getting active models when A/B testing disabled."""
    model_manager._models = {
        'v5': type('obj', (object,), {'f1_score': 0.95})(),
        'v4': type('obj', (object,), {'f1_score': 0.93})()
    }
    
    from src.config.settings import settings
    original_ab_test = settings.ab_test_enabled
    settings.ab_test_enabled = False
    
    try:
        active_models = model_manager.get_active_models()
        assert len(active_models) == 1
        assert active_models[0][0] == settings.champion_version
    finally:
        settings.ab_test_enabled = original_ab_test


@pytest.mark.asyncio
async def test_get_model_version_routing(model_manager):
    """Test model version selection."""
    model_manager._models = {'v5': type('obj', (object,), {})()}
    
    version = model_manager.get_model_version()
    assert version in model_manager._models
