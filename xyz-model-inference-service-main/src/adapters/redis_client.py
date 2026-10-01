# src/adapters/redis_client.py
"""Redis client wrapper for caching."""

import json
import logging
import redis
from typing import Optional, Any

from src.config.settings import RedisSettings

logger = logging.getLogger(__name__)


# class RedisCache:
#     """Redis cache abstraction."""
    
#     def __init__(self, settings: RedisSettings):
#         self.settings = settings
#         self._client: Optional[redis.Redis] = None
    
#     async def connect(self) -> None:
#         """Connect to Redis."""
#         try:
#             self._client = redis.Redis(
#                 host=self.settings.host,
#                 port=self.settings.port,
#                 db=self.settings.db,
#                 password=self.settings.password,
#                 max_connections=self.settings.max_connections,
#                 decode_responses=True
#             )
#             self._client.ping()
#             logger.info(f"Connected to Redis at {self.settings.host}:{self.settings.port}")
#         except Exception as e:
#             logger.error(f"Failed to connect to Redis: {e}")
#             raise
    
#     async def get(self, key: str) -> Optional[str]:
#         """Get value from cache."""
#         try:
#             return self._client.get(key)
#         except Exception as e:
#             logger.warning(f"Redis get error: {e}")
#             return None
    
#     async def set(self, key: str, value: str, ttl: int) -> bool:
#         """Set value in cache with TTL."""
#         try:
#             self._client.setex(key, ttl, value)
#             return True
#         except Exception as e:
#             logger.warning(f"Redis set error: {e}")
#             return False
    
#     async def delete(self, key: str) -> bool:
#         """Delete key from cache."""
#         try:
#             self._client.delete(key)
#             return True
#         except Exception as e:
#             logger.warning(f"Redis delete error: {e}")
#             return False
    
#     async def disconnect(self) -> None:
#         """Close Redis connection."""
#         if self._client:
#             self._client.close()
#             logger.info("Disconnected from Redis")

class RedisCache:
    """Redis cache abstraction."""

    def __init__(self, settings: Optional[RedisSettings]):
        if settings is None:
            self.enabled = False
            self._client = None
            self.settings = None
            return

        self.enabled = True
        self.settings = settings
        self._client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        if not self.enabled:
            logger.warning("Redis is DISABLED (local development mode)")
            return

        try:
            self._client = redis.Redis(
                host=self.settings.host,
                port=self.settings.port,
                db=self.settings.db,
                password=self.settings.password,
                max_connections=self.settings.max_connections,
                decode_responses=True
            )
            self._client.ping()
            logger.info(
                f"Connected to Redis at {self.settings.host}:{self.settings.port}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
