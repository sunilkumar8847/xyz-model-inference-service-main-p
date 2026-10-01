# src/tests/mocks/mock_feature_store.py
"""Mock Feature Store client for testing."""

from datetime import datetime

from src.domain.models import FeatureVector


class MockFeatureStoreClient:
    """Mock Feature Store client that returns test data."""
    
    async def get_features(
        self, entity_id_1: str, entity_id_2: str
    ) -> FeatureVector:
        """Return mock feature vector."""
        return FeatureVector(
            entity_id_1=entity_id_1,
            entity_id_2=entity_id_2,
            features=[0.5] * 50,  # Default feature values
            feature_names=[f"feature_{i}" for i in range(50)],
            timestamp=datetime.utcnow()
        )
