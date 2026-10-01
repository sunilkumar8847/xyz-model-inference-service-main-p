# src/tests/test_inference_service.py
"""Tests for InferenceService."""

import pytest

from src.services.inference_service import InferenceService
from src.services.fallback_service import FallbackService
from src.services.model_manager import ModelManager
from src.tests.mocks.mock_triton import MockTritonClient
from src.tests.mocks.mock_feature_store import MockFeatureStoreClient
from src.tests.mocks.mock_mlflow import MockMLflowClient


@pytest.fixture
async def inference_service():
    """Create inference service with mocks."""
    triton = MockTritonClient()
    await triton.initialize()
    
    feature_store = MockFeatureStoreClient()
    fallback = FallbackService()
    mlflow = MockMLflowClient()
    
    model_manager = ModelManager(None, mlflow)
    model_manager._models = {
        'v5': type('obj', (object,), {'f1_score': 0.95, 'status': 'ready'})()
    }
    
    return InferenceService(triton, feature_store, model_manager, fallback)


@pytest.mark.asyncio
async def test_predict_match_success(inference_service):
    """Test successful match prediction."""
    result = await inference_service.predict_match(
        entity_id_1="E1",
        entity_id_2="E2"
    )
    
    assert 0 <= result.score <= 1
    assert result.decision in [
        "AUTO_MATCH", "LIKELY_MATCH", "UNCERTAIN", "LIKELY_NON", "AUTO_NON_MATCH"
    ]
    assert result.model_version == 'v5'


@pytest.mark.asyncio
async def test_batch_predict_success(inference_service):
    """Test batch prediction."""
    pairs = [("E1", "E2"), ("E3", "E4"), ("E5", "E6")]
    results = await inference_service.batch_predict(pairs)
    
    assert len(results) == 3
    for result in results:
        assert 0 <= result.score <= 1


@pytest.mark.asyncio
async def test_classify_decision(inference_service):
    """Test decision classification."""
    test_cases = [
        (0.96, "AUTO_MATCH"),
        (0.90, "LIKELY_MATCH"),
        (0.70, "UNCERTAIN"),
        (0.30, "LIKELY_NON"),
        (0.10, "AUTO_NON_MATCH"),
    ]
    
    for score, expected_decision in test_cases:
        decision = inference_service._classify_decision(score)
        assert decision == expected_decision
