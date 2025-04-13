from motor.motor_asyncio import AsyncIOMotorDatabase
from .connection import Database, get_database
import logging

logger = logging.getLogger(__name__)

# Re-export the get_database function for backward compatibility
__all__ = ["get_database", "get_collection", "get_db", "connect_to_mongodb", "close_mongodb_connection"]

async def get_collection(db: AsyncIOMotorDatabase, collection_name: str):
    """获取指定的集合"""
    return db[collection_name]

async def get_db() -> AsyncIOMotorDatabase:
    """获取数据库连接，作为依赖项提供给路由处理器"""
    return await get_database() 

# 添加兼容函数，用于保持与app/db/mongodb.py的兼容性
async def connect_to_mongodb():
    """连接到MongoDB数据库，兼容旧版接口"""
    logger.info("调用兼容函数connect_to_mongodb")
    return await get_database()

async def close_mongodb_connection():
    """关闭MongoDB连接，兼容旧版接口"""
    logger.info("调用兼容函数close_mongodb_connection")
    await Database.close() 