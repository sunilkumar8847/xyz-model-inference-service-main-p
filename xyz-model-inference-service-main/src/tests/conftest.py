# src/tests/conftest.py
"""Pytest configuration and fixtures."""

import pytest
import asyncio
from typing import Generator
from datetime import datetime

from src.config.settings import Settings
from src.domain.models import (
    ModelMetadata,
    FeatureVector,
    MatchScore,
    PredictionResult
)
from src.services.model_manager import ModelManager
from src.services.inference_service import InferenceService
from src.services.fallback_service import FallbackService
from src.adapters.triton_client import TritonClient
from src.adapters.feature_store_client import FeatureStoreClient
from src.adapters.mlflow_client import MLflowClient


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Provide event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> Settings:
    """Test configuration settings."""
    return Settings(
        environment="test",
        service_host="localhost",
        service_port=8000,
        triton_host="localhost",
        triton_port=8001,
        triton_timeout_ms=200,
        inference_timeout_ms=200,
        fallback_enabled=True,
        ab_test_enabled=False,
        enable_prometheus=False,
        enable_opentelemetry=False,
        log_level="DEBUG"
    )


@pytest.fixture
def mock_feature_vector() -> FeatureVector:
    """Mock feature vector for testing."""
    return FeatureVector(
        entity_id_1="TEST_001",
        entity_id_2="TEST_002",
        features=[0.5] * 50,
        feature_names=[f"feature_{i}" for i in range(50)],
        timestamp=datetime.now()
    )


@pytest.fixture
def mock_match_score() -> MatchScore:
    """Mock match score for testing."""
    return MatchScore(
        score=0.96,
        confidence_lower=0.94,
        confidence_upper=0.98,
        model_version="v5",
        inference_ms=145
    )


@pytest.fixture
def mock_model_metadata() -> ModelMetadata:
    """Mock model metadata for testing."""
    return ModelMetadata(
        name="customer_matcher",
        version="v5",
        framework="ONNX",
        f1_score=0.947,
        status="ready",
        loaded_at=datetime.now(),
        weight=1.0,
        rollout_pct=100,
        is_champion=True
    )


@pytest.fixture
async def mock_triton_client():
    """Mock Triton client for testing."""
    class MockTritonClient(TritonClient):
        async def initialize(self):
            pass
        
        async def predict(self, features, model_version, timeout_ms):
            return MatchScore(
                score=0.96,
                confidence_lower=0.94,
                confidence_upper=0.98,
                model_version=model_version,
                inference_ms=50
            )
        
        async def explain(self, features, model_version, field_names):
            return {
                "field_contributions": {
                    field: {"contribution": 0.1}
                    for field in field_names[:5]
                },
                "top_k_fields": field_names[:5],
                "human_readable": "Mock explanation"
            }
        
        async def health_check(self):
            return True
    
    client = MockTritonClient()
    await client.initialize()
    return client


@pytest.fixture
async def mock_feature_store_client():
    """Mock Feature Store client for testing."""
    class MockFeatureStoreClient(FeatureStoreClient):
        async def get_features(self, entity_id_1, entity_id_2):
            return FeatureVector(
                entity_id_1=entity_id_1,
                entity_id_2=entity_id_2,
                features=[0.5] * 50,
                feature_names=[f"feature_{i}" for i in range(50)],
                timestamp=datetime.now()
            )
    
    return MockFeatureStoreClient()


@pytest.fixture
async def mock_mlflow_client():
    """Mock MLflow client for testing."""
    class MockMLflowClient(MLflowClient):
        async def get_model_versions(self, model_name):
            return [
                {
                    'name': model_name,
                    'version': 'v5',
                    'framework': 'ONNX',
                    'f1_score': 0.947,
                    'stage': 'Production',
                    'rollout_pct': 100
                }
            ]
        
        async def download_model(self, version, local_path):
            return local_path
        
        async def log_metrics(self, run_id, metrics):
            pass
    
    return MockMLflowClient()


@pytest.fixture
async def fallback_service():
    """Fallback service instance for testing."""
    return FallbackService()


@pytest.fixture
async def model_manager(mock_mlflow_client):
    """Model manager instance for testing."""
    manager = ModelManager(None, mock_mlflow_client)
    manager._models = {
        'v5': ModelMetadata(
            name="customer_matcher",
            version="v5",
            framework="ONNX",
            f1_score=0.947,
            status="ready",
            loaded_at=datetime.now()
        )
    }
    return manager


@pytest.fixture
async def inference_service(
    mock_triton_client,
    mock_feature_store_client,
    model_manager,
    fallback_service
):
    """Inference service instance for testing."""
    return InferenceService(
        mock_triton_client,
        mock_feature_store_client,
        model_manager,
        fallback_service
    )


@pytest.fixture
def sample_entity_pairs():
    """Sample entity pairs for batch testing."""
    return [
        ("E1", "E2"),
        ("E3", "E4"),
        ("E5", "E6")
    ]
