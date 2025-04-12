from motor.motor_asyncio import AsyncIOMotorDatabase
from .connection import Database, get_database
import logging

logger = logging.getLogger(__name__)

# Re-export the get_database function for backward compatibility
__all__ = ["get_database", "get_collection", "get_db"]

async def get_collection(db: AsyncIOMotorDatabase, collection_name: str):
    """获取指定的集合"""
    return db[collection_name]

async def get_db() -> AsyncIOMotorDatabase:
    """获取数据库连接，作为依赖项提供给路由处理器"""
    return await get_database() 