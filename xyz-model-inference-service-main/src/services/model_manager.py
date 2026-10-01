# # src/services/model_manager.py
# """Model lifecycle management: loading, caching, versioning."""

import logging
import json
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import hashlib
from functools import lru_cache

from src.config.settings import settings
from src.domain.models import ModelMetadata, ModelStatus
from src.adapters.redis_client import RedisCache
from src.adapters.mlflow_client import MLflowClient

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model lifecycle: loading, caching, versioning, A/B testing."""
    
    # def __init__(self, redis_cache: RedisCache, mlflow_client: MLflowClient):
    #     self.redis = redis_cache
    #     self.mlflow = mlflow_client
    #     self._models: Dict[str, ModelMetadata] = {}
    #     self._model_cache_key = "models:metadata"
    #     self._refresh_interval = settings.mlflow.refresh_interval_s
    #     self._last_refresh = datetime.now()

    def __init__(self, redis_cache: RedisCache, mlflow_client: MLflowClient):
        self.redis_cache = redis_cache
        self.mlflow_client = mlflow_client
        self._models: Dict[str, ModelMetadata] = {}

        # MLflow DISABLED → local static mode
        if settings.mlflow is None or not getattr(mlflow_client, "enabled", False):
            logger.warning("ModelManager running in LOCAL STATIC MODE")

            self._refresh_interval = None

            # Register a dummy local model
            self._models["local"] = ModelMetadata(
                name="local-model",
                version="local",
                framework="mock",
                f1_score=0.0,
                status=ModelStatus.READY,
                rollout_pct=100,
            )
            return
    
    # async def initialize(self) -> None:
    #     """Initialize model manager at startup."""
    #     logger.info("Initializing model manager")
    #     await self._load_available_models()
    #     self._validate_champion_challenger()

    async def initialize(self) -> None:
        """
        Initialize model registry.
        In local mode, models are already loaded.
        """
        if self.mlflow_client is None:
            return
    
    
    async def _load_available_models(self) -> None:
        """Load all available models from MLflow."""
        try:
            # Fetch from MLflow registry
            available_versions = await self.mlflow.get_model_versions(
                settings.mlflow.model_name
            )
            
            for version_info in available_versions:
                model_meta = ModelMetadata(
                    name=version_info['name'],
                    version=version_info['version'],
                    framework=version_info.get('framework', 'unknown'),
                    f1_score=version_info.get('f1_score', 0.0),
                    status=ModelStatus.READY.value,
                    loaded_at=datetime.now(),
                    rollout_pct=version_info.get('rollout_pct', 100)
                )
                self._models[version_info['version']] = model_meta
            
            logger.info(f"Loaded {len(self._models)} model versions from MLflow")
            
            # Cache to Redis
            await self._cache_models()
        
        except Exception as e:
            logger.error(f"Failed to load models from MLflow: {e}")
            # Try to load from cache
            await self._load_from_cache()
    
    async def _cache_models(self) -> None:
        """Cache model metadata to Redis."""
        try:
            models_dict = {
                version: {
                    'name': meta.name,
                    'version': meta.version,
                    'f1_score': meta.f1_score,
                    'status': meta.status,
                    'loaded_at': meta.loaded_at.isoformat()
                }
                for version, meta in self._models.items()
            }
            await self.redis.set(
                self._model_cache_key,
                json.dumps(models_dict),
                ttl=self._refresh_interval
            )
        except Exception as e:
            logger.warning(f"Failed to cache models: {e}")
    
    async def _load_from_cache(self) -> None:
        """Load model metadata from Redis cache."""
        try:
            cached = await self.redis.get(self._model_cache_key)
            if cached:
                models_dict = json.loads(cached)
                for version, data in models_dict.items():
                    self._models[version] = ModelMetadata(
                        name=data['name'],
                        version=data['version'],
                        f1_score=data['f1_score'],
                        status=data['status'],
                        loaded_at=datetime.fromisoformat(data['loaded_at'])
                    )
                logger.info(f"Loaded {len(self._models)} models from cache")
        except Exception as e:
            logger.error(f"Failed to load from cache: {e}")
    
    def _validate_champion_challenger(self) -> None:
        """Validate champion/challenger configuration."""
        champion = settings.champion_version
        challenger = settings.challenger_version
        
        if champion not in self._models:
            logger.error(f"Champion model {champion} not found")
            raise ValueError(f"Champion model {champion} not available")
        
        if challenger and challenger not in self._models:
            logger.warning(f"Challenger model {challenger} not found, disabling A/B test")
            settings.ab_test_enabled = False
        
        logger.info(
            f"Model config: champion={champion} ({self._models[champion].f1_score:.2%}), "
            f"challenger={challenger} ({self._models.get(challenger).f1_score if challenger else 'none'})"
        )
    
    def get_active_models(self) -> List[tuple[str, float]]:
        """Get active models for inference with traffic percentages."""
        if not settings.ab_test_enabled or not settings.challenger_version:
            return [(settings.champion_version, 100.0)]
        
        challenger_pct = settings.challenger_traffic_pct
        champion_pct = 100 - challenger_pct
        
        return [
            (settings.champion_version, champion_pct),
            (settings.challenger_version, challenger_pct)
        ]
    
    # def get_model_version(self, version_hint: Optional[str] = None) -> str:
    #     """Deterministically select model version (for A/B testing)."""
    #     import random
        
    #     if not version_hint:
    #         active_models = self.get_active_models()
    #         versions, percentages = zip(*active_models)
    #         chosen = random.choices(versions, weights=percentages, k=1)[0]
    #         return chosen
        
    #     return version_hint if version_hint in self._models else settings.champion_version
    
    
    def get_model_version(self) -> str:
        if not self._models:
            raise RuntimeError("No models available")

    # Local mode → always return local
        if "local" in self._models:
            return "local"

        return next(iter(self._models.keys()))

    
    def should_refresh_models(self) -> bool:
        """Check if models need refresh from MLflow."""
        elapsed = (datetime.now() - self._last_refresh).total_seconds()
        return elapsed > self._refresh_interval




        
    
