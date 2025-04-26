"""
记忆管理器 - 管理短期记忆
"""

import time
import logging
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from app.memory.buffer_memory import ShortTermMemory
from app.services.custom_session_service import CustomSessionService
from app.memory.schemas import SessionResponse, MemoryContext
from app.models.custom_session import SessionStatus
from app.config import memory_settings
import os
from datetime import datetime
from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    记忆管理器，负责短期记忆
    """
    
    def __init__(self):
        """初始化记忆管理器"""
        try:
            # 初始化短期记忆
            try:
                # 创建Redis客户端
                from app.config import memory_settings
                
                # 从环境变量或内存设置中获取配置
                redis_host = os.getenv("REDIS_HOST", "localhost")
                redis_port = int(os.getenv("REDIS_PORT", "6378"))
                redis_password = os.getenv("REDIS_PASSWORD", "!qaz2wsX")
                max_chat_rounds = int(os.getenv("MAX_CHAT_ROUNDS", "2"))
                
                self.redis = None
                self.redis_url = f"redis://{redis_host}:{redis_port}"
                if redis_password:
                    self.redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}"
                
                # 初始化其他组件
                self.short_term_memory = ShortTermMemory()
                self.session_service = CustomSessionService()
                self.max_chat_rounds = max_chat_rounds
                
                logger.info("记忆管理器初始化完成")
                
            except Exception as e:
                logger.error(f"初始化短期记忆时出错: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"初始化记忆管理器时出错: {str(e)}")
            raise
            
    async def _ensure_redis_connected(self):
        """确保Redis连接已建立"""
        if self.redis is None:
            self.redis = await Redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
    async def initialize(self):
        """异步初始化服务"""
        try:
            await self._ensure_redis_connected()
            # 测试Redis连接
            await self.redis.ping()
            logger.info(f"成功连接到Redis服务器: {self.redis_url}")
            
            # 初始化短期记忆
            await self.short_term_memory.initialize()
            logger.info("短期记忆模块初始化成功")
            
            logger.info("记忆管理器初始化完成")
            
        except RedisError as e:
            logger.error(f"Redis连接失败: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"初始化记忆管理器时出错: {str(e)}")
            raise
        
    async def start_new_session(self, user_id: str, selected_username: str = None) -> str:
        """
        开始新会话
        
        Args:
            user_id: 用户ID
            selected_username: 前端选中的用户名称（可选）
            
        Returns:
            会话ID
        """
        try:
            # 检查短期记忆是否可用
            if not self.short_term_memory:
                logger.error("短期记忆模块不可用，无法创建新会话")
                # 返回一个临时会话ID，允许应用继续工作
                return f"temp-{int(time.time())}-{str(uuid.uuid4())[:8]}"
                
            # 生成会话ID
            session_id = str(int(time.time())) + "-" + str(uuid.uuid4())[:8]
            
            # 创建会话
            self.short_term_memory.start_session(session_id, user_id, selected_username)
            
            logger.info(f"已开始新会话: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"开始新会话失败: {str(e)}")
            # 返回一个临时会话ID，允许应用继续工作
            return f"temp-{int(time.time())}-{str(uuid.uuid4())[:8]}"
        
    async def end_session(self, session_id: str, user_id: str) -> Dict:
        """
        结束会话
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            结果信息，包含会话ID
        """
        try:
            # 获取会话消息
            messages = self.short_term_memory.get_session_messages(session_id, user_id)
            
            if not messages:
                logger.warning(f"会话为空: {session_id}")
                return SessionResponse(
                    success=False,
                    error="会话为空"
                ).dict()
            
            # 更新会话状态为已归档
            try:
                await CustomSessionService.update_session_status(
                    session_id=session_id,
                    status=2  # 已结束状态
                )
                logger.info(f"已将会话 {session_id} 状态更新为已归档")
            except Exception as status_error:
                logger.error(f"更新会话状态失败: {str(status_error)}")
            
            # 更新会话状态
            self.short_term_memory.end_session(session_id, user_id)
            
            logger.info(f"已结束会话: {session_id}")
            return SessionResponse(
                success=True,
                session_id=session_id
            ).dict()
            
        except Exception as e:
            logger.error(f"结束会话失败: {str(e)}")
            return SessionResponse(
                success=False,
                error=f"结束会话失败: {str(e)}"
            ).dict()
            
    async def add_message(self, session_id: str, user_id: str, role: str, content: str, role_id: str = None, message_id: str = None) -> bool:
        """
        添加消息到会话
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            role: 消息角色（user/assistant/system）
            content: 消息内容
            role_id: 角色ID，对应MongoDB roles表中的_id
            message_id: 消息ID，如果不提供则自动生成
            
        Returns:
            是否成功
        
        Raises:
            ValueError: 如果用户是匿名用户但未选择用户名
        """
        try:
            # 检查短期记忆模块是否可用
            if not self.short_term_memory:
                logger.error("短期记忆模块不可用，无法添加消息")
                return False
                
            # 预处理role_id，确保它不是"null"字符串
            if role_id == "null" or role_id == "":
                role_id = None
                logger.info("MemoryManager: 接收到空字符串或'null'角色ID，已设置为None")
            elif role_id:
                logger.info(f"MemoryManager: 处理消息，角色ID: {role_id}, 类型: {type(role_id).__name__}")
                
            # 添加消息，并获取是否需要归档的消息
            try:
                result, oldest_message = await self.short_term_memory.add_message(
                    session_id=session_id, 
                    user_id=user_id, 
                    role=role, 
                    content=content, 
                    role_id=role_id,
                    message_id=message_id
                )
            except ValueError as e:
                # 重新抛出用户选择错误
                logger.error(f"选择用户错误: {str(e)}")
                raise
            
            return result
        except Exception as e:
            logger.error(f"添加消息失败: {str(e)}")
            return False
            
    async def build_context(self, session_id: str, user_id: str, current_message: str = None) -> Dict:
        """
        构建对话上下文
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            current_message: 当前消息（可选）
            
        Returns:
            上下文信息，包含消息
        """
        try:
            # 获取当前会话消息
            current_messages = await self.short_term_memory.get_session_messages(session_id, user_id)
                
            # 构建上下文
            context = []
            
            # 添加当前会话消息
            for msg in current_messages:
                context.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
                
            # 创建上下文对象
            memory_context = MemoryContext(
                messages=context,
                related_summaries=[]  # 不提供相关摘要
            )
            
            return memory_context.dict()
            
        except Exception as e:
            logger.error(f"构建上下文失败: {str(e)}")
            # 返回基本上下文
            return MemoryContext(
                messages=[{"role": "system", "content": "无法加载完整上下文"}]
            ).dict()
            
    async def get_user_sessions(self, user_id: str, limit: int = 20, skip: int = 0) -> List[Dict]:
        """
        获取用户的会话列表
        
        Args:
            user_id: 用户ID
            limit: 结果限制
            skip: 跳过数量
            
        Returns:
            会话列表
        """
        try:
            # 获取活跃会话
            active_sessions = await self.short_term_memory.list_active_sessions(user_id)
            
            # 返回活跃会话
            return active_sessions
            
        except Exception as e:
            logger.error(f"获取用户会话列表失败: {str(e)}")
            return []
            
    async def get_session_detail(self, session_id: str, user_id: str) -> Dict:
        """
        获取短期记忆聊天详情
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            会话详情
        """
        try:
            # 检查会话是否活跃
            session_info = await self.short_term_memory.get_session_info(session_id, user_id)
            
            if session_info:
                # 获取消息
                messages = await self.short_term_memory.get_session_messages(session_id, user_id)
                
                return {
                    "session_id": session_id,
                    "user_id": user_id,
                    "status": session_info.get("status", "unknown"),
                    "start_time": session_info.get("start_time"),
                    "end_time": session_info.get("end_time"),
                    "messages": messages,
                    "summary": None
                    }
                    
            # 会话不存在
            return None
            
        except Exception as e:
            logger.error(f"获取短期记忆聊天详情失败: {str(e)}")
            return None
            
    def truncate_by_token(self, text: str, max_token: int = 12000) -> str:
        """
        按token数量截断文本
        
        Args:
            text: 文本内容
            max_token: 最大token数量
            
        Returns:
            截断后的文本
        """
        # 简单估算：假设平均每个字符是1.5个token
        char_limit = int(max_token / 1.5)
        
        if len(text) <= char_limit:
            return text
        
        return text[:char_limit]

    async def ensure_session_exists(self, user_id: str, session_id: str = None) -> str:
        """
        确保会话存在，不存在则创建
        
        Args:
            user_id: 用户ID
            session_id: 会话ID（可选）
            
        Returns:
            会话ID
        """
        try:
            # 如果没有提供会话ID，直接创建新会话
            if not session_id:
                return await self.start_new_session(user_id)
                
            # 检查会话是否存在
            session_info = await self.short_term_memory.get_session_info(session_id, user_id)
            if not session_info:
                # 会话不存在，创建新会话
                return await self.start_new_session(user_id)
                
            return session_id
        except Exception as e:
            logger.error(f"确保会话存在失败: {str(e)}")
            # 出错时尝试创建新会话
            try:
                return await self.start_new_session(user_id)
            except Exception as inner_e:
                logger.error(f"创建备用会话失败: {str(inner_e)}")
                raise

    async def add_message_safe(self, session_id: str, user_id: str, role: str, content: str, role_id: str = None, message_id: str = None) -> bool:
        """
        安全地添加消息到会话，包含会话创建和重试机制
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            role: 消息角色（user/assistant/system）
            content: 消息内容
            role_id: 角色ID，对应MongoDB roles表中的_id
            message_id: 消息ID，如果不提供则自动生成
            
        Returns:
            是否成功
        """
        try:
            # 检查短期记忆模块是否可用
            if not self.short_term_memory:
                logger.error("短期记忆模块不可用，无法安全添加消息")
                return False
                
            # 确保会话存在
            actual_session_id = await self.ensure_session_exists(user_id, session_id)
            
            # 使用重试机制添加消息
            result, oldest_message = await self.short_term_memory.add_message_with_retry(
                actual_session_id, 
                user_id, 
                role, 
                content,
                role_id,
                message_id,
                max_retries=3
            )
            
            return result
        except Exception as e:
            logger.error(f"安全添加消息失败: {str(e)}")
            return False

    async def update_session_role_names(self, session_id: str, user_id: str) -> Dict:
        """
        更新会话中的角色名称为中文名
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            更新结果
        """
        try:
            # 检查短期记忆模块是否可用
            if not self.short_term_memory:
                logger.error("短期记忆模块不可用，无法更新角色名称")
                return {"success": False, "error": "短期记忆模块不可用"}
            
            # 调用ShortTermMemory的更新方法
            result = await self.short_term_memory.update_role_names(session_id, user_id)
            
            # 添加成功标记
            result["success"] = len(result.get("errors", [])) == 0
            
            logger.info(f"更新会话角色名称完成: {session_id}, 更新了{result.get('updated_messages', 0)}条消息")
            return result
        except Exception as e:
            logger.error(f"更新会话角色名称失败: {str(e)}")
            return {"success": False, "error": str(e)}

# 全局变量保存单例实例
_memory_manager = None

async def get_memory_manager():
    """
    获取记忆管理器的单例实例
    如果还没有初始化，则创建并初始化一个新实例
    
    Returns:
        MemoryManager: 初始化好的记忆管理器实例
    """
    global _memory_manager
    
    if _memory_manager is None:
        logger.info("正在创建并初始化记忆管理器...")
        manager = MemoryManager()
        await manager.initialize()
        _memory_manager = manager
        logger.info("记忆管理器初始化完成")
    
    return _memory_manager

# 为了兼容性保留这个变量，但在应用启动时应确保调用get_memory_manager()
# memory_manager = MemoryManager()  # 注意：这个对象没有完成异步初始化
# 应该在需要使用memory_manager的地方使用异步方式获取：
# memory_manager = await get_memory_manager() 