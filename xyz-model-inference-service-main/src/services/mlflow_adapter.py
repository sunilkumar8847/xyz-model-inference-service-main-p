# src/services/mlflow_adapter.py
"""MLflow adapter for model registry operations."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.adapters.mlflow_client import MLflowClient
from src.domain.models import ModelMetadata

logger = logging.getLogger(__name__)


class MLflowAdapter:
    """
    High-level adapter for MLflow model registry.
    
    Responsibilities:
    - Fetch model versions and metadata
    - Track model performance metrics
    - Manage model lifecycle (staging, production, archived)
    - Handle registry failures gracefully
    """
    
    def __init__(self, mlflow_client: MLflowClient):
        self.client = mlflow_client
        self._model_cache: Dict[str, ModelMetadata] = {}
        self._last_refresh: Optional[datetime] = None
    
    async def get_production_models(
        self,
        model_name: str = "customer_matcher"
    ) -> List[ModelMetadata]:
        """
        Get all production-ready models.
        
        Args:
            model_name: Model name in MLflow registry
        
        Returns:
            List of production ModelMetadata
        """
        versions = await self.client.get_model_versions(model_name)
        
        production_models = []
        for version_info in versions:
            if version_info.get('stage') in ['Production', 'Staging']:
                model_meta = self._parse_model_metadata(version_info)
                production_models.append(model_meta)
                self._model_cache[model_meta.version] = model_meta
        
        self._last_refresh = datetime.now()
        logger.info(f"Loaded {len(production_models)} production models")
        
        return production_models
    
    async def get_model_by_version(
        self,
        version: str,
        model_name: str = "customer_matcher"
    ) -> Optional[ModelMetadata]:
        """
        Get specific model version metadata.
        
        Args:
            version: Model version (e.g., "v5")
            model_name: Model name
        
        Returns:
            ModelMetadata or None if not found
        """
        # Check cache first
        if version in self._model_cache:
            return self._model_cache[version]
        
        # Fetch from registry
        versions = await self.client.get_model_versions(model_name)
        for version_info in versions:
            if version_info['version'] == version:
                model_meta = self._parse_model_metadata(version_info)
                self._model_cache[version] = model_meta
                return model_meta
        
        logger.warning(f"Model version {version} not found in registry")
        return None
    
    async def log_inference_metrics(
        self,
        model_version: str,
        metrics: Dict[str, float],
        run_id: Optional[str] = None
    ) -> None:
        """
        Log inference metrics to MLflow.
        
        Args:
            model_version: Model version being evaluated
            metrics: Metrics to log (e.g., {"latency_ms": 145, "accuracy": 0.96})
            run_id: MLflow run ID (optional)
        """
        try:
            await self.client.log_metrics(
                run_id or f"inference_{model_version}_{datetime.now().isoformat()}",
                metrics
            )
            logger.debug(f"Logged metrics for {model_version}: {metrics}")
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")
    
    async def compare_models(
        self,
        version_1: str,
        version_2: str,
        metric_name: str = "f1_score"
    ) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            version_1: First model version
            version_2: Second model version
            metric_name: Metric to compare
        
        Returns:
            Comparison result dict
        """
        model_1 = await self.get_model_by_version(version_1)
        model_2 = await self.get_model_by_version(version_2)
        
        if not model_1 or not model_2:
            raise ValueError("One or both model versions not found")
        
        metric_1 = getattr(model_1, metric_name, None)
        metric_2 = getattr(model_2, metric_name, None)
        
        return {
            "version_1": version_1,
            "version_2": version_2,
            "metric": metric_name,
            "value_1": metric_1,
            "value_2": metric_2,
            "winner": version_1 if metric_1 > metric_2 else version_2,
            "difference": abs(metric_1 - metric_2) if metric_1 and metric_2 else None
        }
    
    def _parse_model_metadata(self, version_info: Dict[str, Any]) -> ModelMetadata:
        """Parse MLflow version info into ModelMetadata."""
        return ModelMetadata(
            name=version_info.get('name', 'unknown'),
            version=version_info.get('version', 'unknown'),
            framework=version_info.get('framework', 'unknown'),
            f1_score=version_info.get('f1_score', 0.0),
            status='ready',
            loaded_at=datetime.now(),
            weight=1.0,
            rollout_pct=version_info.get('rollout_pct', 100),
            is_champion=version_info.get('stage') == 'Production'
        )
    
    def should_refresh(self, max_age_seconds: int = 300) -> bool:
        """Check if model cache should be refreshed."""
        if not self._last_refresh:
            return True
        
        age = (datetime.now() - self._last_refresh).total_seconds()
        return age > max_age_seconds