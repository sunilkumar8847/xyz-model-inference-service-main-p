# src/tests/test_fallback.py
"""Tests for FallbackService."""

import pytest

from src.services.fallback_service import FallbackService
from src.domain.models import FallbackReason


@pytest.fixture
def fallback_service():
    """Create fallback service."""
    return FallbackService()


@pytest.mark.asyncio
async def test_fallback_predict_match(fallback_service):
    """Test fallback match prediction."""
    result = await fallback_service.predict_match(
        entity_id_1="E1",
        entity_id_2="E2",
        reason=FallbackReason.TIMEOUT
    )
    
    assert 0 <= result.score <= 1
    assert result.fallback_used is True
    assert result.fallback_reason == "timeout"


@pytest.mark.asyncio
async def test_fallback_classify_decision(fallback_service):
    """Test decision classification."""
    test_cases = [
        (0.96, "AUTO_MATCH"),
        (0.87, "LIKELY_MATCH"),
        (0.70, "UNCERTAIN"),
        (0.10, "AUTO_NON_MATCH"),
    ]
    
    for score, expected_decision in test_cases:
        decision = fallback_service._classify_decision(score)
        assert decision == expected_decision