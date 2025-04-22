#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Redis服务 - 提供统一的Redis连接管理
"""

import logging
import os
import redis
from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

class RedisService:
    """
    Redis服务类 - 提供同步和异步访问Redis的功能
    """
    
    def __init__(self):
        """初始化Redis服务"""
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
        self.async_redis = None
        
        # 同步Redis客户端
        self.sync_redis = None
        
        logger.info(f"Redis服务初始化完成, URL: {self.redis_url}")
    
    async def get_redis(self) -> Redis:
        """
        获取异步Redis客户端
        
        Returns:
            Redis: 异步Redis客户端
        """
        if self.async_redis is None:
            try:
                self.async_redis = await Redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                
                # 测试连接
                await self.async_redis.ping()
                logger.info("异步Redis客户端连接成功")
            except RedisError as e:
                logger.error(f"异步Redis客户端连接失败: {str(e)}")
                raise
        
        return self.async_redis
    
    def get_redis_sync(self) -> redis.Redis:
        """
        获取同步Redis客户端
        
        Returns:
            redis.Redis: 同步Redis客户端
        """
        if self.sync_redis is None:
            try:
                self.sync_redis = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    db=self.redis_db,
                    password=self.redis_password,
                    decode_responses=True
                )
                
                # 测试连接
                self.sync_redis.ping()
                logger.info("同步Redis客户端连接成功")
            except RedisError as e:
                logger.error(f"同步Redis客户端连接失败: {str(e)}")
                return None
        
        return self.sync_redis
    
    async def close(self):
        """关闭Redis连接"""
        if self.async_redis:
            await self.async_redis.close()
            self.async_redis = None
            logger.info("异步Redis连接已关闭")
            
        if self.sync_redis:
            self.sync_redis.close()
            self.sync_redis = None
            logger.info("同步Redis连接已关闭")

# 创建全局Redis服务实例
redis_service = RedisService() 