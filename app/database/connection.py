"""
数据库连接模块
提供MongoDB连接相关功能
"""

import motor.motor_asyncio
from pymongo.errors import ConnectionFailure, OperationFailure
import logging
from pymongo import MongoClient
from pymongo.operations import IndexModel
import os
import asyncio
import time
from typing import Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, OperationFailure, ServerSelectionTimeoutError

logger = logging.getLogger(__name__)

class Database:
    client = None
    db = None

    @classmethod
    async def connect(cls, connection_string, db_name="multi_ai_chat"):
        """建立与MongoDB的连接"""
        try:
            # 设置连接超时和重试选项
            cls.client = motor.motor_asyncio.AsyncIOMotorClient(
                connection_string,
                serverSelectionTimeoutMS=5000,  # 5秒超时
                connectTimeoutMS=5000,
                socketTimeoutMS=5000,
                retryWrites=True
            )
            # 测试连接
            await cls.client.admin.command('ping')
            cls.db = cls.client[db_name]
            logger.info(f"Connected to MongoDB database: {db_name}")
            return cls.db
        except ConnectionFailure as e:
            logger.error(f"MongoDB server not available: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"MongoDB连接异常: {str(e)}")
            raise

    @classmethod
    async def close(cls):
        """关闭数据库连接"""
        if cls.client:
            cls.client.close()
            logger.info("MongoDB connection closed")

    @classmethod
    async def create_index(cls, collection_name, field, unique=False):
        """创建索引"""
        collection = cls.db[collection_name]
        if unique:
            await collection.create_index(field, unique=True)
        else:
            await collection.create_index([(field, 1)])

    @classmethod
    async def create_composite_index(cls, collection_name, fields):
        """创建复合索引"""
        collection = cls.db[collection_name]
        await collection.create_index(fields)

    @classmethod
    async def create_multiple_indexes(cls, collection_name, indexes):
        """创建多个索引"""
        collection = cls.db[collection_name]
        await collection.create_indexes(indexes)

    @classmethod
    async def create_text_index(cls, collection_name, field):
        """创建文本索引"""
        collection = cls.db[collection_name]
        await collection.create_index([(field, 'text')])

    @classmethod
    async def create_ttl_index(cls, collection_name, field, expire_after_seconds):
        """创建TTL索引（自动过期）"""
        collection = cls.db[collection_name]
        await collection.create_index(field, expireAfterSeconds=expire_after_seconds)

    @classmethod
    async def create_vector_index(cls, collection_name, field, dimensions, similarity):
        """创建向量索引（适用于MongoDB 6.0+）"""
        collection = cls.db[collection_name]
        await collection.create_index([(field, 'vector')], vectorOptions={'dimensions': dimensions, 'similarity': similarity})

    @classmethod
    async def list_indexes(cls, collection_name):
        """查看已创建的索引"""
        collection = cls.db[collection_name]
        indexes = collection.list_indexes()
        for index in indexes:
            print(index)

    @classmethod
    async def drop_index(cls, collection_name, index_name):
        """删除指定索引"""
        collection = cls.db[collection_name]
        await collection.drop_index(index_name)

    @classmethod
    async def drop_all_indexes(cls, collection_name):
        """删除所有索引"""
        collection = cls.db[collection_name]
        await collection.drop_indexes()

# 全局缓存连接客户端和数据库
_mongo_client = None
_mongo_db = None

# 重试设置
MAX_RETRIES = 3
RETRY_DELAY = 1  # 秒

async def get_mongo_client() -> AsyncIOMotorClient:
    """
    获取MongoDB客户端连接，带有错误处理和重试机制
    
    Returns:
        AsyncIOMotorClient: MongoDB连接客户端
    """
    global _mongo_client
    
    # 检查是否有现有连接并验证其有效性
    if _mongo_client is not None:
        try:
            # 检查连接是否有效
            await _mongo_client.admin.command('ping')
            return _mongo_client
        except Exception as e:
            logger.warning(f"现有MongoDB连接失效，将重新连接: {str(e)}")
            # 继续尝试新连接
    
    # 从环境变量获取连接信息
    mongo_uri = os.getenv("MONGODB_URL", os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    mongo_user = os.getenv("MONGODB_USERNAME", os.getenv("MONGO_USER", ""))
    mongo_pass = os.getenv("MONGODB_PASSWORD", os.getenv("MONGO_PASSWORD", ""))
    auth_source = os.getenv("MONGODB_AUTH_SOURCE", os.getenv("MONGO_AUTH_SOURCE", "admin"))
    
    logger.info(f"MongoDB连接参数: URI={mongo_uri}, 用户={mongo_user}, 认证源={auth_source}")
    
    # 尝试连接，带重试
    retries = 0
    last_error = None
    
    while retries < MAX_RETRIES:
        try:
            logger.info(f"正在连接MongoDB (尝试 {retries+1}/{MAX_RETRIES})...")
            
            # 使用AsyncIOMotorClient直接提供认证信息，而不是修改URI
            client_options = {
                "serverSelectionTimeoutMS": 5000,
                "connectTimeoutMS": 5000,
                "socketTimeoutMS": 5000
            }
            
            # 只有当用户名和密码都存在时才添加认证信息
            if mongo_user and mongo_pass:
                client_options["username"] = mongo_user
                client_options["password"] = mongo_pass
                client_options["authSource"] = auth_source
                logger.info(f"使用认证信息连接: user={mongo_user}, authSource={auth_source}")
            
            client = AsyncIOMotorClient(mongo_uri, **client_options)
            
            # 验证连接
            await client.admin.command('ping')
            
            logger.info("MongoDB连接成功！")
            _mongo_client = client
            return client
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            retries += 1
            last_error = e
            error_msg = str(e)
            
            # 检查错误类型并提供更详细的日志
            if "Authentication failed" in error_msg:
                logger.error(f"MongoDB认证失败，请检查用户名、密码和认证源: {error_msg}")
                logger.info(f"认证详情: 用户={mongo_user}, 认证源={auth_source}")
                # 尝试不同的认证源
                if auth_source != "admin" and retries == 1:
                    auth_source = "admin"
                    logger.info(f"尝试使用默认认证源 'admin' 重新连接")
                    continue
            elif "requires authentication" in error_msg:
                logger.error(f"MongoDB需要认证: {error_msg}")
                if not mongo_user or not mongo_pass:
                    logger.error("未提供MongoDB用户名或密码，请检查环境变量配置")
            
            if retries < MAX_RETRIES:
                logger.warning(f"MongoDB连接失败 (尝试 {retries}/{MAX_RETRIES}): {error_msg}. 将在 {RETRY_DELAY} 秒后重试...")
                await asyncio.sleep(RETRY_DELAY)  # 使用异步睡眠
            else:
                logger.error(f"MongoDB连接失败，已达到最大重试次数: {error_msg}")
                
        except Exception as e:
            retries += 1
            last_error = e
            
            if retries < MAX_RETRIES:
                logger.warning(f"MongoDB连接出现未预期错误 (尝试 {retries}/{MAX_RETRIES}): {str(e)}. 将在 {RETRY_DELAY} 秒后重试...")
                await asyncio.sleep(RETRY_DELAY)  # 使用异步睡眠
            else:
                logger.error(f"MongoDB连接出现未预期错误，已达到最大重试次数: {str(e)}")
    
    # 如果所有重试都失败，返回None
    logger.error(f"无法连接到MongoDB，已尝试 {MAX_RETRIES} 次: {str(last_error)}")
    return None

async def get_database(db_name: Optional[str] = None) -> Optional[AsyncIOMotorDatabase]:
    """
    获取MongoDB数据库连接，带有错误处理和重试机制
    
    Args:
        db_name: 数据库名称，如果为None则使用环境变量中的配置
        
    Returns:
        AsyncIOMotorDatabase: MongoDB数据库连接，如果连接失败则返回None
    """
    global _mongo_db
    
    if db_name is None:
        db_name = os.getenv("MONGODB_DATABASE", os.getenv("MONGO_DB", "multi_ai_chat"))
    
    # 检查是否有现有数据库连接
    if _mongo_db is not None:
        try:
            # 验证连接
            await _mongo_db.command('ping')
            # 如果所需数据库与现有连接的数据库不同，则切换数据库
            if _mongo_db.name != db_name:
                client = _mongo_db.client
                _mongo_db = client[db_name]
            logger.debug(f"使用现有MongoDB连接: 数据库={db_name}")
            return _mongo_db
        except Exception as e:
            # 连接已失效，继续尝试新连接
            logger.warning(f"现有MongoDB数据库连接已失效，将重新连接: {str(e)}")
            _mongo_db = None
    
    # 获取客户端连接
    client = await get_mongo_client()
    if client is None:
        logger.error(f"无法获取MongoDB客户端连接，无法访问数据库 '{db_name}'")
        return None
    
    try:
        # 获取数据库
        db = client[db_name]
        # 验证连接和权限
        await db.command('ping')
        
        # 检查是否有足够的权限
        try:
            # 尝试列出collection以验证权限
            collections = await db.list_collection_names(limit=1)
            logger.info(f"成功获取{db_name}数据库的集合列表: {collections[:5] if collections else '无集合'}")
        except OperationFailure as e:
            if "not authorized" in str(e):
                logger.warning(f"已连接到{db_name}数据库，但权限受限: {str(e)}")
            else:
                logger.warning(f"列出集合时出现问题: {str(e)}")
        
        _mongo_db = db
        logger.info(f"已连接到MongoDB数据库: {db_name}")
        return db
    except Exception as e:
        logger.error(f"访问数据库'{db_name}'时出错: {str(e)}")
        return None

async def ping_database() -> Dict[str, Any]:
    """
    检查数据库连接状态
    
    Returns:
        Dict: 包含连接状态信息的字典
    """
    result = {
        "status": "error",
        "connected": False,
        "database": os.getenv("MONGO_DB", "multi_ai_chat"),
        "message": "未连接到MongoDB"
    }
    
    try:
        client = await get_mongo_client()
        if client is None:
            result["message"] = "无法建立MongoDB连接"
            return result
            
        # 测试通过ping命令
        await client.admin.command('ping')
        result["connected"] = True
        result["status"] = "success"
        result["message"] = "MongoDB连接正常"
        
        # 获取服务器信息
        try:
            server_info = await client.admin.command('serverStatus')
            result["version"] = server_info.get("version", "unknown")
            result["uptime"] = server_info.get("uptime", 0)
            result["connections"] = server_info.get("connections", {}).get("current", 0)
        except Exception as info_error:
            logger.warning(f"获取MongoDB服务器信息失败: {str(info_error)}")
            
        return result
    except Exception as e:
        result["message"] = f"MongoDB连接检查失败: {str(e)}"
        return result 