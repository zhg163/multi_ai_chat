"""
短期记忆模块 - 管理对话短期记忆(Redis)
"""

import json
import time
import uuid
import logging
from typing import List, Dict, Any, Tuple, Optional
from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

class ShortTermMemory:
    """短期记忆管理器，使用Redis存储最近的对话"""
    
    def __init__(self, redis_client=None):
        """初始化短期记忆管理器"""
        self.redis = redis_client
        self.messages_ttl = 60 * 60 * 24 * 7  # 7天过期
        
    async def initialize(self, redis_client=None):
        """初始化Redis连接"""
        if redis_client:
            self.redis = redis_client
        
        if not self.redis:
            # 如果没有提供redis客户端，从服务中获取
            try:
                from app.common.redis_client import get_redis_client
                self.redis = await get_redis_client()
                logger.info("短期记忆Redis客户端初始化成功")
            except Exception as e:
                logger.error(f"Redis客户端初始化失败: {str(e)}")
                raise
                
        return self
    
    async def start_new_session(self, user_id: str) -> str:
        """
        开始新的会话
        
        Args:
            user_id: 用户ID
            
        Returns:
            str: 会话ID
        """
        session_id = str(uuid.uuid4()).replace("-", "")
        
        # 创建会话记录，包含开始时间
        session_info = {
            "user_id": user_id,
            "start_time": time.time(),
            "messages_count": 0
        }
        
        # 保存会话信息
        try:
            session_key = f"session:{user_id}:{session_id}"
            await self.redis.hmset(session_key, session_info)
            await self.redis.expire(session_key, self.messages_ttl)
            logger.info(f"创建会话: {session_id}, 用户: {user_id}")
        except Exception as e:
            logger.error(f"创建会话失败: {str(e)}")
            raise
            
        return session_id
        
    async def end_session(self, session_id: str, user_id: str) -> bool:
        """
        结束会话
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            bool: 是否成功
        """
        try:
            # 更新会话状态
            session_key = f"session:{user_id}:{session_id}"
            
            # 检查会话是否存在
            exists = await self.redis.exists(session_key)
            if not exists:
                logger.warning(f"要结束的会话不存在: {session_id}")
                return False
                
            # 添加结束时间
            await self.redis.hset(session_key, "end_time", time.time())
            logger.info(f"会话结束: {session_id}")
            return True
        except Exception as e:
            logger.error(f"结束会话失败: {str(e)}")
            return False
            
    async def add_message(
        self, 
        session_id: str, 
        user_id: str, 
        role: str, 
        content: str,
        role_id: str = None,
        message_id: str = None
    ) -> Tuple[bool, str]:
        """
        添加消息到会话
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            role: 消息角色 (user, assistant)
            content: 消息内容
            role_id: 角色ID（可选）
            message_id: 自定义消息ID（可选）
            
        Returns:
            Tuple[bool, str]: (成功标志, 消息ID)
        """
        if not message_id:
            message_id = str(uuid.uuid4()).replace("-", "")
            
        try:
            # 消息时间戳
            timestamp = time.time()
            
            # 消息对象
            message = {
                "message_id": message_id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "timestamp": timestamp
            }
            
            # 如果提供了角色ID，添加到消息中
            if role_id:
                message["roleid"] = role_id
            
            # 构建Redis键
            messages_key = f"messages:{user_id}:{session_id}"
            message_key = f"message:{message_id}"
            
            # 将消息添加到列表
            message_json = json.dumps(message)
            await self.redis.rpush(messages_key, message_json)
            
            # 单独存储消息便于查询
            await self.redis.set(message_key, message_json, ex=self.messages_ttl)
            
            # 更新会话消息计数
            session_key = f"session:{user_id}:{session_id}"
            await self.redis.hincrby(session_key, "messages_count", 1)
            
            # 设置过期时间
            await self.redis.expire(messages_key, self.messages_ttl)
            await self.redis.expire(session_key, self.messages_ttl)
            
            logger.info(f"添加消息: {message_id}, 会话: {session_id}, 角色: {role}")
            return True, message_id
        except Exception as e:
            logger.error(f"添加消息失败: {str(e)}")
            return False, message_id
    
    async def get_session_messages(self, session_id: str, user_id: str) -> List[Dict[str, Any]]:
        """
        获取会话的所有消息
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            List[Dict[str, Any]]: 消息列表
        """
        try:
            messages_key = f"messages:{user_id}:{session_id}"
            
            # 获取所有消息
            message_jsons = await self.redis.lrange(messages_key, 0, -1)
            
            # 解析消息
            messages = []
            for msg_json in message_jsons:
                try:
                    message = json.loads(msg_json)
                    messages.append(message)
                except json.JSONDecodeError:
                    logger.error(f"解析消息JSON失败: {msg_json[:100]}...")
                    continue
                    
            return messages
        except Exception as e:
            logger.error(f"获取会话消息失败: {str(e)}")
            return []
            
    async def get_session_info(self, session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话信息
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            Optional[Dict[str, Any]]: 会话信息
        """
        try:
            session_key = f"session:{user_id}:{session_id}"
            
            # 检查会话是否存在
            exists = await self.redis.exists(session_key)
            if not exists:
                logger.warning(f"会话不存在: {session_id}")
                return None
                
            # 获取会话信息
            session_info = await self.redis.hgetall(session_key)
            
            # 添加会话ID
            session_info["session_id"] = session_id
            
            return session_info
        except Exception as e:
            logger.error(f"获取会话信息时出错: {str(e)}")
            return None
            
    async def get_all_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        获取用户的所有会话
        
        Args:
            user_id: 用户ID
            
        Returns:
            List[Dict[str, Any]]: 会话列表
        """
        try:
            # 获取所有会话键
            session_pattern = f"session:{user_id}:*"
            keys = await self.redis.keys(session_pattern)
            
            # 获取每个会话的信息
            sessions = []
            for key in keys:
                session_id = key.split(":")[-1]
                session_info = await self.get_session_info(session_id, user_id)
                if session_info:
                    sessions.append(session_info)
                    
            # 按开始时间排序
            sessions.sort(key=lambda x: float(x.get("start_time", 0)), reverse=True)
            
            return sessions
        except Exception as e:
            logger.error(f"获取所有会话失败: {str(e)}")
            return []
            
    async def delete_session(self, session_id: str, user_id: str) -> bool:
        """
        删除会话及其所有消息
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            bool: 是否成功
        """
        try:
            # 构建键
            session_key = f"session:{user_id}:{session_id}"
            messages_key = f"messages:{user_id}:{session_id}"
            
            # 获取所有消息
            message_jsons = await self.redis.lrange(messages_key, 0, -1)
            
            # 提取消息ID并删除单独存储的消息
            for msg_json in message_jsons:
                try:
                    message = json.loads(msg_json)
                    message_id = message.get("message_id")
                    if message_id:
                        await self.redis.delete(f"message:{message_id}")
                except json.JSONDecodeError:
                    continue
                    
            # 删除会话和消息列表
            await self.redis.delete(session_key, messages_key)
            
            logger.info(f"删除会话: {session_id}")
            return True
        except Exception as e:
            logger.error(f"删除会话失败: {str(e)}")
            return False 