"""
Redis 客户端 - 提供统一的 Redis 连接管理
"""

import redis
from redis.asyncio import Redis
from redis.exceptions import RedisError
import os
import logging

logger = logging.getLogger(__name__)

class RedisClient:
    """
    Redis客户端类 - 提供连接Redis的功能
    """
    
    def __init__(self):
        """初始化Redis客户端"""
        # 从环境变量或配置中获取Redis参数
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6378"))
        self.redis_db = int(os.getenv("REDIS_DB", "0"))
        self.redis_password = os.getenv("REDIS_PASSWORD", "!qaz2wsX")
        
        # 构建Redis连接URL
        self.redis_url = f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
        if self.redis_password:
            self.redis_url = f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        
        # 异步Redis客户端
        self._async_redis = None
        
        # 同步Redis客户端
        self._sync_redis = None
        
        logger.info(f"Redis客户端初始化完成, URL: {self.redis_url}")
    
    async def get_async_client(self) -> Redis:
        """
        获取异步Redis客户端
        
        Returns:
            Redis: 异步Redis客户端
        """
        if self._async_redis is None:
            try:
                self._async_redis = await Redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                
                # 测试连接
                await self._async_redis.ping()
                logger.info("异步Redis客户端连接成功")
            except RedisError as e:
                logger.error(f"异步Redis客户端连接失败: {str(e)}")
                raise
        
        return self._async_redis
    
    def get_sync_client(self) -> redis.Redis:
        """
        获取同步Redis客户端
        
        Returns:
            redis.Redis: 同步Redis客户端
        """
        if self._sync_redis is None:
            try:
                self._sync_redis = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    db=self.redis_db,
                    password=self.redis_password,
                    decode_responses=True
                )
                
                # 测试连接
                self._sync_redis.ping()
                logger.info("同步Redis客户端连接成功")
            except RedisError as e:
                logger.error(f"同步Redis客户端连接失败: {str(e)}")
                return None
        
        return self._sync_redis
    
    async def close(self):
        """关闭Redis连接"""
        if self._async_redis:
            await self._async_redis.close()
            self._async_redis = None
            logger.info("异步Redis连接已关闭")
            
        if self._sync_redis:
            self._sync_redis.close()
            self._sync_redis = None
            logger.info("同步Redis连接已关闭")

# 全局Redis客户端实例
_redis_client = None

async def get_redis_client() -> Redis:
    """
    获取Redis客户端实例
    
    Returns:
        Redis: Redis客户端
    """
    global _redis_client
    
    if _redis_client is None:
        _redis_client = RedisClient()
    
    return await _redis_client.get_async_client() 