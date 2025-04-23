"""
Redis Client Manager - Provides a Redis client connection
"""

from redis.asyncio import Redis
import logging
from app.config import settings

logger = logging.getLogger(__name__)
_redis_client = None

async def get_redis_client() -> Redis:
    """Get a Redis client instance"""
    global _redis_client
    
    if _redis_client is None:
        # Get the redis DB, defaulting to 0 if not specified
        redis_db = getattr(settings, "REDIS_DB", 0)
        
        redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{redis_db}"
        
        if hasattr(settings, "REDIS_PASSWORD") and settings.REDIS_PASSWORD:
            redis_url = f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}/{redis_db}"
        
        logger.info(f"Connecting to Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        _redis_client = Redis.from_url(redis_url, decode_responses=True)
        
    return _redis_client

async def release_redis_client():
    """Close the Redis client if it exists"""
    global _redis_client
    
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis client connection closed") 