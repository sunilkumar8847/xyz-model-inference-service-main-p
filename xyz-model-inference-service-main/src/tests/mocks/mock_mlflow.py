# src/tests/mocks/mock_mlflow.py
"""Mock MLflow client for testing."""

from typing import List, Dict, Any
from datetime import datetime

from src.domain.models import ModelMetadata


class MockMLflowClient:
    """Mock MLflow client that returns test data."""
    
    def __init__(self):
        self.models = {
            'v5': {
                'name': 'customer_matcher',
                'version': 'v5',
                'framework': 'ONNX',
                'f1_score': 0.947,
                'stage': 'Production',
                'rollout_pct': 100
            },
            'v4': {
                'name': 'customer_matcher',
                'version': 'v4',
                'framework': 'ONNX',
                'f1_score': 0.935,
                'stage': 'Archived',
                'rollout_pct': 0
            }
        }
    
    async def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Return mock model versions."""
        return list(self.models.values())
    
    async def download_model(self, version: str, local_path: str) -> str:
        """Mock model download."""
        return local_path
    
    async def log_metrics(self, run_id: str, metrics: Dict[str, float]) -> None:
        """Mock metrics logging."""
        pass
