# src/services/feature_adapter.py
"""Feature Store adapter for retrieving features."""

import logging
from typing import List, Optional
from datetime import datetime

from src.adapters.feature_store_client import FeatureStoreClient
from src.adapters.redis_client import RedisCache
from src.domain.models import FeatureVector
from src.config.constants import CACHE_TTL_FEATURES

logger = logging.getLogger(__name__)


class FeatureAdapter:
    """
    High-level adapter for feature retrieval with caching.
    
    Responsibilities:
    - Cache features in Redis (1-min TTL)
    - Handle feature store failures gracefully
    - Provide fallback to default features
    - Track feature retrieval metrics
    """
    
    def __init__(
        self,
        feature_store_client: FeatureStoreClient,
        redis_cache: Optional[RedisCache] = None
    ):
        self.feature_store = feature_store_client
        self.redis = redis_cache
        self._cache_hits = 0
        self._cache_misses = 0
        self._feature_errors = 0
    
    async def get_features(
        self,
        entity_id_1: str,
        entity_id_2: str,
        use_cache: bool = True
    ) -> FeatureVector:
        """
        Get feature vector for entity pair with caching.
        
        Args:
            entity_id_1: First entity ID
            entity_id_2: Second entity ID
            use_cache: Whether to use cache (default: True)
        
        Returns:
            FeatureVector with 50 features
        
        Raises:
            FeatureStoreError: If feature retrieval fails and no fallback
        """
        cache_key = self._make_cache_key(entity_id_1, entity_id_2)
        
        # Try cache first
        if use_cache and self.redis:
            cached = await self._get_from_cache(cache_key)
            if cached:
                self._cache_hits += 1
                logger.debug(f"Feature cache hit: {cache_key}")
                return cached
            self._cache_misses += 1
        
        # Cache miss - fetch from feature store
        try:
            features = await self.feature_store.get_features(
                entity_id_1, entity_id_2
            )
            
            # Cache for future requests
            if self.redis:
                await self._set_in_cache(cache_key, features)
            
            return features
        
        except Exception as e:
            self._feature_errors += 1
            logger.error(f"Feature store error: {e}", exc_info=True)
            
            # Return default features as fallback
            logger.warning("Using default feature vector")
            return self._get_default_features(entity_id_1, entity_id_2)
    
    async def get_features_batch(
        self,
        entity_pairs: List[tuple[str, str]]
    ) -> List[FeatureVector]:
        """
        Batch retrieve features for multiple entity pairs.
        
        Args:
            entity_pairs: List of (entity_id_1, entity_id_2) tuples
        
        Returns:
            List of FeatureVectors
        """
        import asyncio
        
        tasks = [
            self.get_features(e1, e2)
            for e1, e2 in entity_pairs
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    def _make_cache_key(self, entity_id_1: str, entity_id_2: str) -> str:
        """Generate cache key for entity pair."""
        # Sort IDs to ensure consistent key regardless of order
        ids = tuple(sorted([entity_id_1, entity_id_2]))
        return f"features:{ids[0]}:{ids[1]}"
    
    async def _get_from_cache(self, key: str) -> Optional[FeatureVector]:
        """Retrieve feature vector from cache."""
        try:
            import json
            cached_json = await self.redis.get(key)
            if cached_json:
                data = json.loads(cached_json)
                return FeatureVector(
                    entity_id_1=data['entity_id_1'],
                    entity_id_2=data['entity_id_2'],
                    features=data['features'],
                    feature_names=data['feature_names'],
                    timestamp=datetime.fromisoformat(data['timestamp'])
                )
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
        
        return None
    
    async def _set_in_cache(self, key: str, features: FeatureVector) -> None:
        """Store feature vector in cache."""
        try:
            import json
            cache_data = {
                'entity_id_1': features.entity_id_1,
                'entity_id_2': features.entity_id_2,
                'features': features.features,
                'feature_names': features.feature_names,
                'timestamp': features.timestamp.isoformat()
            }
            await self.redis.set(
                key,
                json.dumps(cache_data),
                ttl=CACHE_TTL_FEATURES
            )
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")
    
    def _get_default_features(
        self,
        entity_id_1: str,
        entity_id_2: str
    ) -> FeatureVector:
        """Return default feature vector as fallback."""
        return FeatureVector(
            entity_id_1=entity_id_1,
            entity_id_2=entity_id_2,
            features=[0.5] * 50,  # Neutral default values
            feature_names=[f"feature_{i}" for i in range(50)],
            timestamp=datetime.now()
        )
    
    def get_stats(self) -> dict:
        """Get feature adapter statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate_pct": round(hit_rate, 2),
            "feature_errors": self._feature_errors
        }