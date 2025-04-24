"""
Redis Lock Utility - Provides distributed locking mechanisms using Redis
"""

import asyncio
import uuid
import logging
from redis.asyncio import Redis
from typing import Optional, Any

logger = logging.getLogger(__name__)

class RedisLock:
    """
    基于Redis的分布式锁实现
    
    提供异步操作和上下文管理器支持
    """
    
    def __init__(self, redis_client, lock_name, expire_seconds=30, owner=None):
        """
        初始化锁对象
        
        Args:
            redis_client: Redis客户端实例
            lock_name: 锁的名称/键
            expire_seconds: 锁的过期时间（秒）
            owner: 锁的拥有者标识（默认生成UUID）
        """
        self.redis = redis_client
        self.lock_name = lock_name
        self.expire_seconds = expire_seconds
        self.owner = owner or str(uuid.uuid4())
        self._acquired = False
        self._refresh_task = None
    
    async def acquire(self, retry_count=3, retry_delay=0.5):
        """
        尝试获取锁，支持重试
        
        Args:
            retry_count: 重试次数
            retry_delay: 重试延迟（秒）
            
        Returns:
            bool: 是否成功获取锁
        """
        for attempt in range(retry_count):
            acquired = await self.redis.set(
                self.lock_name, self.owner, 
                nx=True, ex=self.expire_seconds
            )
            
            if acquired:
                self._acquired = True
                # 启动自动刷新任务延长锁寿命
                self._start_refresh_task()
                logger.info(f"获取锁成功: {self.lock_name} (所有者: {self.owner})")
                return True
            
            if attempt < retry_count - 1:
                logger.warning(f"获取锁失败，尝试重试 ({attempt+1}/{retry_count}): {self.lock_name}")
                await asyncio.sleep(retry_delay)
        
        logger.error(f"所有重试都失败，无法获取锁: {self.lock_name}")
        return False
    
    async def release(self):
        """
        释放锁，只允许所有者释放
        
        Returns:
            bool: 是否成功释放锁
        """
        if not self._acquired:
            logger.warning(f"尝试释放未获取的锁: {self.lock_name}")
            return False
        
        # 停止自动刷新任务
        self._stop_refresh_task()
        
        # 使用Lua脚本确保只有锁的所有者可以释放锁
        script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('del', KEYS[1])
        else
            return 0
        end
        """
        
        try:
            result = await self.redis.eval(script, 1, self.lock_name, self.owner)
            if result:
                self._acquired = False
                logger.info(f"成功释放锁: {self.lock_name} (所有者: {self.owner})")
                return True
            else:
                logger.warning(f"锁不存在或不属于此所有者: {self.lock_name} (所有者: {self.owner})")
                return False
        except Exception as e:
            logger.error(f"释放锁时出错: {str(e)}")
            self._acquired = False
            return False
    
    async def refresh(self):
        """
        刷新锁的过期时间
        
        Returns:
            bool: 是否成功刷新
        """
        if not self._acquired:
            return False
        
        # 使用Lua脚本确保只有所有者可以刷新
        script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('expire', KEYS[1], ARGV[2])
        else
            return 0
        end
        """
        
        try:
            result = await self.redis.eval(
                script, 1, self.lock_name, self.owner, self.expire_seconds
            )
            
            if result:
                logger.debug(f"刷新锁过期时间: {self.lock_name}")
                return True
            else:
                logger.warning(f"刷新锁失败，锁可能已被释放: {self.lock_name}")
                self._acquired = False
                return False
        except Exception as e:
            logger.error(f"刷新锁时出错: {str(e)}")
            return False
    
    def _start_refresh_task(self):
        """启动自动刷新任务"""
        async def refresh_periodically():
            try:
                while self._acquired:
                    # 在过期时间的一半时刷新
                    await asyncio.sleep(self.expire_seconds / 2)
                    if self._acquired:
                        await self.refresh()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"锁刷新任务出错: {str(e)}")
        
        self._refresh_task = asyncio.create_task(refresh_periodically())
    
    def _stop_refresh_task(self):
        """停止自动刷新任务"""
        if self._refresh_task:
            self._refresh_task.cancel()
            self._refresh_task = None
    
    async def __aenter__(self):
        """异步上下文管理器支持"""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        """异步上下文管理器退出时释放锁"""
        await self.release()


async def obtain_lock(redis_client, lock_key, expire_seconds=30):
    """
    获取锁的辅助函数
    
    Args:
        redis_client: Redis客户端
        lock_key: 锁名称
        expire_seconds: 过期时间（秒）
        
    Returns:
        Optional[RedisLock]: 获取的锁对象，获取失败返回None
    """
    lock = RedisLock(redis_client, lock_key, expire_seconds)
    if await lock.acquire():
        return lock
    return None 