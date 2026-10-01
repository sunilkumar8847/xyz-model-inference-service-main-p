# src/adapters/mlflow_client.py
"""MLflow model registry client."""

# import logging
# from typing import List, Dict, Any, Optional
# from datetime import datetime
# import asyncio

# from src.config.settings import settings

# logger = logging.getLogger(__name__)


# class MLflowClient:
#     """MLflow model registry integration."""
    
#     def __init__(self):
#         self.tracking_uri = settings.mlflow.tracking_uri
#         self.model_name = settings.mlflow.model_name
#         self._cached_versions: Dict[str, Any] = {}
#         self._cache_time: Optional[datetime] = None
    
#     async def get_model_versions(
#         self, model_name: str
#     ) -> List[Dict[str, Any]]:
#         """
#         Get all versions of a model from MLflow registry.
        
#         Returns list of version info dicts with:
#         - name: Model name
#         - version: Version string
#         - framework: Framework type
#         - f1_score: Evaluation metric
#         - stage: Stage (Production, Staging, Archived)
#         """
#         try:
#             # In production, would use mlflow.client.MlflowClient
#             # For now, return mock data
#             versions = [
#                 {
#                     'name': model_name,
#                     'version': 'v5',
#                     'framework': 'ONNX',
#                     'f1_score': 0.947,
#                     'stage': 'Production',
#                     'rollout_pct': 100,
#                     'trained_at': datetime.now().isoformat()
#                 },
#                 {
#                     'name': model_name,
#                     'version': 'v4',
#                     'framework': 'ONNX',
#                     'f1_score': 0.935,
#                     'stage': 'Archived',
#                     'rollout_pct': 0,
#                     'trained_at': datetime.now().isoformat()
#                 }
#             ]
            
#             self._cached_versions = {v['version']: v for v in versions}
#             self._cache_time = datetime.now()
            
#             logger.info(f"Fetched {len(versions)} versions for {model_name}")
#             return versions
        
#         except Exception as e:
#             logger.error(f"Failed to fetch model versions: {e}")
#             raise
    
#     async def download_model(self, version: str, local_path: str) -> str:
#         """
#         Download model artifact from MLflow.
        
#         Returns path to downloaded model.
#         """
#         try:
#             # In production: use mlflow.artifacts.download_artifacts()
#             logger.info(f"Downloaded {version} to {local_path}")
#             return local_path
#         except Exception as e:
#             logger.error(f"Failed to download model: {e}")
#             raise
    
#     async def log_metrics(
#         self,
#         run_id: str,
#         metrics: Dict[str, float]
#     ) -> None:
#         """Log metrics to MLflow tracking."""
#         try:
#             # In production: use mlflow.log_metrics()
#             logger.info(f"Logged metrics for run {run_id}: {metrics}")
#         except Exception as e:
#             logger.warning(f"Failed to log metrics: {e}")

from typing import Optional
import logging

from src.config.settings import settings

logger = logging.getLogger(__name__)


class MLflowClient:
    """MLflow adapter."""

    def __init__(self):
        if settings.mlflow is None:
            self.enabled = False
            self.tracking_uri = None
            logger.warning("MLflow is DISABLED (local development mode)")
            return

        self.enabled = True
        self.tracking_uri = settings.mlflow.tracking_uri

        logger.info(f"MLflow tracking URI: {self.tracking_uri}")

    async def get_model_versions(self, model_name: str):
        if not self.enabled:
            return []

        # real implementation later
        raise NotImplementedError

    async def download_model(self, version: str, local_path: str):
        if not self.enabled:
            return local_path

        raise NotImplementedError

    async def log_metrics(self, run_id: str, metrics: dict):
        if not self.enabled:
            return

        raise NotImplementedError

