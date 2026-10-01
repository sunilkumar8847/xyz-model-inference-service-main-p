# src/tests/mocks/mock_triton.py
"""Mock Triton client for testing."""

from typing import List, Optional

from src.domain.models import MatchScore


class MockTritonClient:
    """Mock Triton client that returns test data."""
    
    async def initialize(self) -> None:
        """Mock initialization (no-op)."""
        pass
    
    async def predict(
        self,
        features: List[float],
        model_version: str,
        timeout_ms: int
    ) -> MatchScore:
        """Return mock match score."""
        # Deterministic scoring based on first feature
        base_score = features[0] if features else 0.5
        return MatchScore(
            score=base_score,
            confidence_lower=max(0.0, base_score - 0.05),
            confidence_upper=min(1.0, base_score + 0.05),
            model_version=model_version,
            inference_ms=50
        )
    
    async def explain(
        self,
        features: List[float],
        model_version: str,
        field_names: List[str]
    ) -> Optional[dict]:
        """Return mock explanation."""
        return {
            "field_contributions": {field: {"contribution": 0.1} for field in field_names[:5]},
            "top_k_fields": field_names[:5],
            "human_readable": "Mock explanation"
        }
    
    async def health_check(self) -> bool:
        """Return healthy status."""
        return True
