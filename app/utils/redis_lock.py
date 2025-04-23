"""
Redis Lock Utility - Provides distributed locking mechanisms using Redis
"""

import asyncio
import uuid
import logging
from redis.asyncio import Redis
from typing import Optional

logger = logging.getLogger(__name__)

class RedisLock:
    """Simple distributed lock using Redis"""
    
    def __init__(self, redis_client: Redis, lock_name: str, expire_seconds: int = 30):
        """
        Initialize a Redis lock
        
        Args:
            redis_client: Redis client instance
            lock_name: Name of the lock (used as Redis key)
            expire_seconds: Lock expiration time in seconds
        """
        self.redis = redis_client
        self.lock_name = lock_name
        self.lock_value = str(uuid.uuid4())
        self.expire_seconds = expire_seconds
        self._locked = False
    
    async def acquire(self, retry_count: int = 3, retry_delay: float = 0.5) -> bool:
        """
        Acquire the lock with retries
        
        Args:
            retry_count: Number of acquisition attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            True if lock acquired, False otherwise
        """
        for attempt in range(retry_count):
            acquired = await self.redis.set(
                self.lock_name, 
                self.lock_value,
                nx=True,  # Only set if key doesn't exist
                ex=self.expire_seconds
            )
            
            if acquired:
                self._locked = True
                logger.debug(f"Lock acquired: {self.lock_name}")
                return True
                
            if attempt < retry_count - 1:
                logger.debug(f"Lock acquisition failed, retrying in {retry_delay}s: {self.lock_name}")
                await asyncio.sleep(retry_delay)
        
        logger.warning(f"Failed to acquire lock after {retry_count} attempts: {self.lock_name}")
        return False
    
    async def release(self) -> bool:
        """
        Release the lock if we own it
        
        Returns:
            True if lock released successfully, False otherwise
        """
        if not self._locked:
            return True
        
        # For Redis versions that don't support eval with keys/args parameters
        try:    
            # Try direct delete - less safe but more compatible
            current_value = await self.redis.get(self.lock_name)
            if current_value == self.lock_value:
                await self.redis.delete(self.lock_name)
                self._locked = False
                logger.debug(f"Lock released: {self.lock_name}")
                return True
            else:
                logger.warning(f"Failed to release lock (not owner): {self.lock_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error releasing lock: {str(e)}")
            return False
    
    async def refresh(self) -> bool:
        """
        Refresh the lock's expiration time
        
        Returns:
            True if lock refreshed successfully, False otherwise
        """
        if not self._locked:
            return False
            
        try:
            # For Redis versions that don't support eval with keys/args parameters
            # Try direct expire - less safe but more compatible
            current_value = await self.redis.get(self.lock_name)
            if current_value == self.lock_value:
                await self.redis.expire(self.lock_name, self.expire_seconds)
                logger.debug(f"Lock refreshed: {self.lock_name}")
                return True
            else:
                logger.warning(f"Failed to refresh lock (not owner): {self.lock_name}")
                self._locked = False
                return False
                
        except Exception as e:
            logger.error(f"Error refreshing lock: {str(e)}")
            self._locked = False
            return False
    
    async def __aenter__(self):
        """Async context manager support for acquiring the lock"""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager support for releasing the lock"""
        await self.release()
        
async def obtain_lock(redis_client, lock_name, expire_seconds=30):
    """
    Helper function to obtain a Redis lock
    
    Args:
        redis_client: Redis client instance
        lock_name: Name of the lock
        expire_seconds: Lock expiration time in seconds
        
    Returns:
        A RedisLock instance
    """
    lock = RedisLock(redis_client, lock_name, expire_seconds)
    success = await lock.acquire()
    if not success:
        return None
    return lock 