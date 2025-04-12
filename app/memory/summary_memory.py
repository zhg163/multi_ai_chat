"""
长期记忆模块 - 使用MongoDB实现的摘要记忆
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pymongo import MongoClient, DESCENDING
from bson.objectid import ObjectId
from pymongo.errors import ConnectionFailure, OperationFailure, ServerSelectionTimeoutError
from app.memory.schemas import SessionSummary
from app.config import memory_settings
from app.database.connection import get_database

logger = logging.getLogger(__name__)

class SummaryMemory:
    """长期记忆存储，用于存储聊天摘要和重要消息"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.chat_summaries = None
        self.chat_messages = None
        self.is_connected = False
        self.connection_error = None
        logger.info("长期记忆存储对象创建完成，等待异步初始化")
        
    async def initialize(self) -> 'SummaryMemory':
        """
        异步初始化方法，应在创建对象后手动调用
        
        Returns:
            SummaryMemory: 返回初始化后的对象，方便链式调用
        """
        success = await self.connect()
        if success:
            logger.info("长期记忆存储异步初始化成功")
        else:
            logger.warning("长期记忆存储异步初始化失败，部分功能可能不可用")
        return self
        
    async def connect(self) -> bool:
        """
        连接到MongoDB，设置集合和索引
        
        Returns:
            bool: 连接是否成功
        """
        try:
            logger.info("正在连接到MongoDB数据库...")
            self.db = await get_database()
            if self.db is None:
                logger.error("无法连接到MongoDB数据库")
                return False
                
            # 初始化集合
            self.chat_summaries = self.db.chat_summaries
            self.chat_messages = self.db.chat_messages
            
            # 尝试创建索引
            try:
                await self.chat_summaries.create_index("session_id")
                await self.chat_summaries.create_index("user_id")
                await self.chat_messages.create_index("session_id")
                await self.chat_messages.create_index(
                    [("session_id", 1), ("timestamp", 1)]
                )
                logger.info("MongoDB索引创建成功")
            except OperationFailure as e:
                if "requires authentication" in str(e):
                    logger.warning(f"无法创建MongoDB索引，权限不足: {str(e)}")
                else:
                    logger.warning(f"创建MongoDB索引时出错: {str(e)}")
                # 如果索引创建失败，但连接正常，仍然继续
            
            self.is_connected = True
            self.connection_error = None
            logger.info("成功连接到MongoDB长期记忆存储")
            return True
            
        except Exception as e:
            self.is_connected = False
            self.connection_error = str(e)
            logger.error(f"连接到MongoDB数据库或初始化集合时出错: {str(e)}")
            return False
    
    async def reconnect_if_needed(self) -> bool:
        """
        检查连接状态，如果断开则重新连接
        
        Returns:
            bool: 连接是否正常
        """
        if not self.is_connected or self.client is None:
            logger.info("尝试重新连接到MongoDB")
            return await self.connect()
        
        try:
            # 测试连接是否仍然有效
            await self.db.command("ping")
            return True
        except Exception as e:
            logger.warning(f"MongoDB连接已断开，尝试重新连接: {str(e)}")
            return await self.connect()
    
    async def _execute_with_retry(self, operation, *args, **kwargs):
        """
        执行MongoDB操作，带有重试机制
        
        Args:
            operation: 要执行的异步操作函数
            *args, **kwargs: 传递给操作的参数
            
        Returns:
            操作的结果
            
        Raises:
            Exception: 如果操作在重试后仍然失败
        """
        max_retries = 2
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                # 确保连接正常
                connected = await self.reconnect_if_needed()
                if not connected:
                    retry_count += 1
                    await asyncio.sleep(1)
                    continue
                    
                # 执行操作
                return await operation(*args, **kwargs)
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                last_error = e
                retry_count += 1
                if retry_count <= max_retries:
                    logger.warning(
                        f"MongoDB操作失败 (尝试 {retry_count}/{max_retries}): {str(e)}. 将重试..."
                    )
                    await asyncio.sleep(1)
                else:
                    logger.error(f"MongoDB操作失败，已达到最大重试次数: {str(e)}")
                    raise
                    
            except Exception as e:
                logger.error(f"执行MongoDB操作时出错: {str(e)}")
                raise
                
        if last_error:
            raise last_error
        raise Exception("无法执行MongoDB操作，连接失败")
    
    async def store_chat_summary(self, summary_data: Dict[str, Any]) -> Optional[str]:
        """
        存储聊天摘要
        
        Args:
            summary_data: 包含摘要信息的字典
            
        Returns:
            str: 插入的文档ID，如果失败则返回None
        """
        async def _store():
            result = await self.chat_summaries.insert_one(summary_data)
            return str(result.inserted_id)
            
        return await self._execute_with_retry(_store)
    
    async def store_chat_message(self, message_data: Dict[str, Any]) -> Optional[str]:
        """
        存储聊天消息
        
        Args:
            message_data: 包含消息信息的字典
            
        Returns:
            str: 插入的文档ID，如果失败则返回None
        """
        async def _store():
            result = await self.chat_messages.insert_one(message_data)
            return str(result.inserted_id)
            
        return await self._execute_with_retry(_store)
    
    async def get_session_messages(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取指定会话的消息
        
        Args:
            session_id: 会话ID
            limit: 返回的最大消息数
            
        Returns:
            List[Dict]: 消息列表，如果发生错误则返回空列表
        """
        async def _get():
            cursor = self.chat_messages.find(
                {"session_id": session_id}
            ).sort("timestamp", -1).limit(limit)
            
            messages = []
            async for message in cursor:
                # 将MongoDB的ObjectId转换为字符串
                message["_id"] = str(message["_id"])
                messages.append(message)
                
            return messages
            
        return await self._execute_with_retry(_get)
    
    async def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取指定会话的摘要
        
        Args:
            session_id: 会话ID
            
        Returns:
            Dict: 摘要信息，如果不存在或发生错误则返回None
        """
        async def _get():
            summary = await self.chat_summaries.find_one({"session_id": session_id})
            if summary:
                summary["_id"] = str(summary["_id"])
            return summary
            
        return await self._execute_with_retry(_get)
    
    async def update_session_summary(self, session_id: str, update_data: Dict[str, Any]) -> bool:
        """
        更新会话摘要
        
        Args:
            session_id: 会话ID
            update_data: 要更新的字段
            
        Returns:
            bool: 更新是否成功
        """
        async def _update():
            update_data["updated_at"] = datetime.utcnow()
            result = await self.chat_summaries.update_one(
                {"session_id": session_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
            
        return await self._execute_with_retry(_update)
    
    async def get_user_sessions(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        获取用户的所有会话摘要
        
        Args:
            user_id: 用户ID
            limit: 返回的最大会话数
            
        Returns:
            List[Dict]: 会话摘要列表，如果发生错误则返回空列表
        """
        async def _get():
            cursor = self.chat_summaries.find(
                {"user_id": user_id}
            ).sort("updated_at", -1).limit(limit)
            
            sessions = []
            async for session in cursor:
                session["_id"] = str(session["_id"])
                sessions.append(session)
                
            return sessions
            
        return await self._execute_with_retry(_get)
    
    async def delete_session(self, session_id: str) -> bool:
        """
        删除会话及其所有消息
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 删除是否成功
        """
        async def _delete():
            # 删除会话摘要
            summary_result = await self.chat_summaries.delete_one({"session_id": session_id})
            
            # 删除所有消息
            message_result = await self.chat_messages.delete_many({"session_id": session_id})
            
            return summary_result.deleted_count > 0 or message_result.deleted_count > 0
            
        return await self._execute_with_retry(_delete)
    
    async def check_connection(self) -> Dict[str, Any]:
        """
        检查数据库连接状态
        
        Returns:
            Dict: 包含连接状态的字典
        """
        result = {
            "status": "error",
            "connected": False,
            "collections": {
                "chat_summaries": False,
                "chat_messages": False
            },
            "message": "未连接到MongoDB"
        }
        
        if not await self.reconnect_if_needed():
            result["message"] = "无法连接到MongoDB数据库"
            return result
        
        try:
            # 测试数据库连接
            await self.db.command("ping")
            result["connected"] = True
            result["status"] = "success"
            result["message"] = "MongoDB连接正常"
            
            # 检查集合
            collections = await self.db.list_collection_names()
            result["collections"]["chat_summaries"] = "chat_summaries" in collections
            result["collections"]["chat_messages"] = "chat_messages" in collections
            
            # 获取统计信息
            try:
                result["summary_count"] = await self.chat_summaries.count_documents({})
                result["message_count"] = await self.chat_messages.count_documents({})
            except Exception as e:
                logger.warning(f"获取集合统计信息失败: {str(e)}")
                
            return result
        except Exception as e:
            result["message"] = f"MongoDB连接检查失败: {str(e)}"
            return result

    async def store_session_summary(self, 
                              session_id: str, 
                              user_id: str, 
                              summary: str, 
                              messages_count: int, 
                              embedding: Optional[List[float]] = None) -> str:
        """存储会话摘要到长期记忆"""
        try:
            # 检查数据库连接是否可用
            if self.db is None:
                logger.warning("MongoDB连接不可用，无法存储摘要")
                return None
                
            # 获取数据库连接
            db = await get_database()
            if db is None:
                logger.warning("无法获取MongoDB连接，使用备用方式")
                if self.db is None:
                    logger.error("无法存储摘要：主连接和备用连接都不可用")
                    return None
                db = self.db
            
            # 创建摘要文档
            summary_doc = {
                "session_id": session_id,
                "user_id": user_id,
                "summary": summary,
                "messages_count": messages_count,
                "embedding": embedding,
                "created_at": datetime.utcnow()
            }
            
            # 插入数据库
            try:
                result = await db.chat_summaries.insert_one(summary_doc)
                logger.info(f"已存储会话摘要，ID: {result.inserted_id}, 会话ID: {session_id}")
                return str(result.inserted_id)
            except Exception as insert_error:
                logger.error(f"存储摘要到MongoDB失败: {str(insert_error)}")
                return None
            
        except Exception as e:
            logger.error(f"存储会话摘要失败: {str(e)}")
            return None
            
    async def get_user_summaries(self, user_id: str, limit: int = 20, skip: int = 0) -> List[Dict]:
        """获取用户的所有会话摘要，按时间倒序排列"""
        try:
            db = await get_database()
            cursor = db.chat_summaries.find(
                {"user_id": user_id}
            ).sort(
                "created_at", DESCENDING
            ).skip(skip).limit(limit)
            
            summaries = await cursor.to_list(length=limit)
            
            # 转换ObjectId为字符串
            for summary in summaries:
                summary["_id"] = str(summary["_id"])
                # 移除向量，减少数据量
                if "embedding" in summary:
                    del summary["embedding"]
                    
            return summaries
            
        except Exception as e:
            logger.error(f"获取用户摘要失败: {str(e)}")
            return []
            
    async def search_relevant_summaries(self, user_id: str, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        """向量搜索相关摘要"""
        try:
            db = await get_database()
            
            # 检查向量索引是否可用
            try:
                # 使用向量搜索
                pipeline = [
                    {"$match": {"user_id": user_id}},
                    {
                        "$vectorSearch": {
                            "index": "embedding",
                            "queryVector": query_embedding,
                            "path": "embedding",
                            "numCandidates": top_k * 10,  # 搜索候选数量，通常设置为结果数量的5-10倍
                            "limit": top_k,
                            "filter": {"user_id": user_id}
                        }
                    },
                    {"$project": {"_id": 1, "summary": 1, "session_id": 1, "created_at": 1, "score": {"$meta": "vectorSearchScore"}}}
                ]
                
                cursor = db.chat_summaries.aggregate(pipeline)
                results = await cursor.to_list(length=top_k)
                
                # 转换ObjectId为字符串
                for result in results:
                    result["_id"] = str(result["_id"])
                    
                return results
                
            except Exception as e:
                logger.warning(f"向量搜索失败，回退到基本查询: {str(e)}")
                # 回退到基本查询（最近的摘要）
                return await self.get_user_summaries(user_id, limit=top_k)
                
        except Exception as e:
            logger.error(f"搜索相关摘要失败: {str(e)}")
            return []
            
    async def delete_session_summary(self, session_id: str) -> bool:
        """删除会话摘要"""
        try:
            db = await get_database()
            result = await db.chat_summaries.delete_one({"session_id": session_id})
            
            if result.deleted_count > 0:
                logger.info(f"已删除会话摘要: {session_id}")
                return True
            else:
                logger.warning(f"未找到要删除的会话摘要: {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"删除会话摘要失败: {str(e)}")
            return False
            
    async def count_user_summaries(self, user_id: str) -> int:
        """统计用户的摘要数量"""
        try:
            db = await get_database()
            count = await db.chat_summaries.count_documents({"user_id": user_id})
            return count
        except Exception as e:
            logger.error(f"统计用户摘要数量失败: {str(e)}")
            return 0

# 创建LongTermMemory类作为SummaryMemory的别名，保持向后兼容性
class LongTermMemory(SummaryMemory):
    """长期记忆存储类，继承自SummaryMemory，提供与SummaryMemory相同的功能"""
    
    def __init__(self):
        """初始化长期记忆存储"""
        super().__init__()
        logger.info("长期记忆存储初始化完成") 