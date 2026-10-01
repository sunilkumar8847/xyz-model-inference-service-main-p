# src/tests/conftest.py
"""Pytest configuration and fixtures."""

import pytest
import asyncio
from typing import Generator

from src.config.settings import Settings
from src.services.model_manager import ModelManager
from src.services.inference_service import InferenceService
from src.services.fallback_service import FallbackService


@pytest.fixture
def event_loop() -> Generator:
    """Provide event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> Settings:
    """Test configuration."""
    return Settings(
        environment="test",
        triton_timeout_ms=200,
        inference_timeout_ms=200,
        fallback_enabled=True,
        ab_test_enabled=False
    )


@pytest.fixture
async def mock_triton_client():
    """Mock Triton client for testing."""
    from src.adapters.triton_client import TritonClient
    from src.domain.models import MatchScore
    
    class MockTritonClient(TritonClient):
        async def predict(self, features, model_version, timeout_ms):
            # Return fixed score for testing
            return MatchScore(
                score=0.96,
                confidence_lower=0.94,
                confidence_upper=0.98,
                model_version=model_version,
                inference_ms=50
            )
        
        async def health_check(self):
            return True
    
    return MockTritonClient()


@pytest.fixture
async def mock_feature_store():
    """Mock Feature Store client."""
    from src.adapters.feature_store_client import FeatureStoreClient
    from src.domain.models import FeatureVector
    from datetime import datetime
    
    class MockFeatureStore(FeatureStoreClient):
        async def get_features(self, entity_id_1, entity_id_2):
            return FeatureVector(
                entity_id_1=entity_id_1,
                entity_id_2=entity_id_2,
                features=[0.5] * 50,
                feature_names=[f"f_{i}" for i in range(50)],
                timestamp=datetime.now()
            )
    
    return MockFeatureStore()


@pytest.fixture
async def fallback_service():
    """Fallback service instance."""
    return FallbackService()

