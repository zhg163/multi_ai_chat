"""
数据库连接模块 - 提供数据库连接和相关功能
"""

import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorDatabase

from .connection import get_client

# 创建logger
logger = logging.getLogger(__name__)

async def get_database() -> AsyncIOMotorDatabase:
    """
    获取MongoDB数据库连接
    
    Returns:
        AsyncIOMotorDatabase: MongoDB数据库连接
    """
    try:
        client = await get_client()
        db = client.get_database()
        return db
    except Exception as e:
        logger.error(f"获取数据库连接失败: {str(e)}")
        raise 