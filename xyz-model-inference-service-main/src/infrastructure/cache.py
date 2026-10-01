# src/infrastructure/cache.py
"""Caching abstraction layer."""

import logging
import json
from typing import Optional, Any, Dict
from datetime import timedelta
import hashlib

from src.adapters.redis_client import RedisCache

logger = logging.getLogger(__name__)


class CacheManager:
    """High-level caching abstraction with automatic serialization."""
    
    def __init__(self, redis_cache: RedisCache):
        self.redis = redis_cache
        self._hit_count = 0
        self._miss_count = 0
    
    async def get(
        self, 
        key: str, 
        namespace: str = "default"
    ) -> Optional[Any]:
        """
        Get value from cache with automatic deserialization.
        
        Args:
            key: Cache key
            namespace: Cache namespace for key isolation
        
        Returns:
            Deserialized value or None if not found
        """
        full_key = self._make_key(namespace, key)
        
        try:
            value = await self.redis.get(full_key)
            if value:
                self._hit_count += 1
                return json.loads(value)
            else:
                self._miss_count += 1
                return None
        except Exception as e:
            logger.warning(f"Cache get error for {full_key}: {e}")
            self._miss_count += 1
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int,
        namespace: str = "default"
    ) -> bool:
        """
        Set value in cache with automatic serialization.
        
        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl_seconds: Time-to-live in seconds
            namespace: Cache namespace
        
        Returns:
            True if successful, False otherwise
        """
        full_key = self._make_key(namespace, key)
        
        try:
            serialized = json.dumps(value)
            return await self.redis.set(full_key, serialized, ttl_seconds)
        except Exception as e:
            logger.warning(f"Cache set error for {full_key}: {e}")
            return False
    
    async def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete key from cache."""
        full_key = self._make_key(namespace, key)
        return await self.redis.delete(full_key)
    
    async def get_or_compute(
        self,
        key: str,
        compute_fn,
        ttl_seconds: int,
        namespace: str = "default"
    ) -> Any:
        """
        Get from cache or compute and store.
        
        Pattern: Read-through cache
        
        Args:
            key: Cache key
            compute_fn: Async function to compute value if cache miss
            ttl_seconds: Cache TTL
            namespace: Cache namespace
        
        Returns:
            Cached or computed value
        """
        # Try cache first
        cached = await self.get(key, namespace)
        if cached is not None:
            return cached
        
        # Cache miss - compute value
        value = await compute_fn()
        
        # Store in cache
        await self.set(key, value, ttl_seconds, namespace)
        
        return value
    
    def _make_key(self, namespace: str, key: str) -> str:
        """Create namespaced cache key."""
        return f"{namespace}:{key}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hit_count + self._miss_count
        hit_rate = (self._hit_count / total * 100) if total > 0 else 0
        
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "total_requests": total,
            "hit_rate_pct": round(hit_rate, 2)
        }
    
    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._hit_count = 0
        self._miss_count = 0


def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate deterministic cache key from function arguments.
    
    Usage:
        cache_key = generate_cache_key(entity_id_1, entity_id_2, model_version="v5")
    """
    # Combine args and kwargs into single string
    key_parts = [str(arg) for arg in args]
    key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
    key_string = "|".join(key_parts)
    
    # Hash for consistent length
    return hashlib.sha256(key_string.encode()).hexdigest()[:32]