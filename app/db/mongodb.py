from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.database import Database
from typing import Optional, Callable, Any

# MongoDB连接字符串，实际应用中应放在环境变量中
# 添加正确的认证信息
MONGO_URL = "mongodb://root:example@localhost:27017/"

DB_NAME = "multi_ai_chat"

# 创建全局客户端实例
client: Optional[AsyncIOMotorClient] = None

# 数据库实例
db: Optional[AsyncIOMotorDatabase] = None

async def connect_to_mongodb():
    """连接到MongoDB数据库"""
    global client, db
    if client is None:
        try:
            client = AsyncIOMotorClient(MONGO_URL)
            # 验证连接是否成功
            await client.admin.command('ping')
            print("MongoDB连接成功!")
            db = client[DB_NAME]
        except Exception as e:
            print(f"MongoDB连接失败: {e}")
            # 重试使用没有认证的连接
            try:
                print("尝试使用无认证连接...")
                client = AsyncIOMotorClient("mongodb://localhost:27017")
                db = client[DB_NAME]
            except Exception as e:
                print(f"无认证连接也失败: {e}")
                raise

async def close_mongodb_connection():
    """关闭MongoDB连接"""
    global client
    if client is not None:
        client.close()
        client = None

async def get_db() -> AsyncIOMotorDatabase:
    """获取数据库连接，作为依赖项提供给路由处理器"""
    if db is None:
        await connect_to_mongodb()
    return db

# 添加get_database函数以兼容现有代码
async def get_database() -> AsyncIOMotorDatabase:
    """获取数据库连接，与get_db功能相同，保持向后兼容"""
    return await get_db() 