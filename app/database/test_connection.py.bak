"""
测试数据库连接模块

执行方法：
python -m app.database.test_connection
"""

import asyncio
import logging
from .mongodb import get_db, connect_to_mongodb, close_mongodb_connection
from .connection import ping_database

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_database_connection():
    """测试数据库连接功能"""
    
    logger.info("=== 开始测试数据库连接模块 ===")
    
    # 测试ping
    logger.info("测试ping_database...")
    ping_result = await ping_database()
    logger.info(f"Ping结果: {ping_result}")
    
    # 测试connect_to_mongodb兼容函数
    logger.info("测试connect_to_mongodb兼容函数...")
    db_from_connect = await connect_to_mongodb()
    logger.info(f"数据库连接获取成功: {db_from_connect.name}")
    
    # 测试get_db
    logger.info("测试get_db函数...")
    db = await get_db()
    logger.info(f"通过get_db获取数据库连接成功: {db.name}")
    
    # 测试列出集合
    try:
        collections = await db.list_collection_names()
        logger.info(f"数据库集合列表: {collections}")
    except Exception as e:
        logger.error(f"列出集合失败: {str(e)}")
    
    # 测试关闭连接
    logger.info("测试close_mongodb_connection函数...")
    await close_mongodb_connection()
    logger.info("数据库连接已关闭")
    
    logger.info("=== 数据库连接模块测试完成 ===")

if __name__ == "__main__":
    asyncio.run(test_database_connection()) 