"""
记忆管理器 - 管理短期和长期记忆
"""

import time
import logging
import uuid
from typing import List, Dict, Any, Optional
from app.memory.buffer_memory import ShortTermMemory
from app.memory.summary_memory import LongTermMemory
from app.services.summary_service import summary_service
from app.services.session_service import SessionService
from app.memory.schemas import SessionResponse, MemoryContext
from app.models.session import SessionStatus
from app.config import memory_settings
import os

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    记忆管理器，负责协调短期记忆和长期记忆
    """
    
    def __init__(self):
        """初始化记忆管理器"""
        try:
            # 初始化短期记忆
            try:
                # 创建Redis客户端
                import redis
                from app.config import memory_settings
                
                # 从环境变量或内存设置中获取配置
                redis_host = os.getenv("REDIS_HOST", "localhost")
                redis_port = int(os.getenv("REDIS_PORT", "6378"))
                redis_password = os.getenv("REDIS_PASSWORD", "!qaz2wsX")
                max_chat_rounds = int(os.getenv("MAX_CHAT_ROUNDS", "2"))
                
                redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    password=redis_password,
                    decode_responses=True
                )
                
                # 测试Redis连接
                try:
                    redis_client.ping()
                    logger.info(f"成功连接到Redis服务器: {redis_host}:{redis_port}")
                except redis.ConnectionError as conn_err:
                    logger.error(f"Redis连接失败: {str(conn_err)}")
                    raise
                
                # 使用创建的Redis客户端初始化ShortTermMemory
                self.short_term = ShortTermMemory(redis_client=redis_client, max_rounds=max_chat_rounds)
                logger.info("短期记忆模块初始化成功")
            except Exception as redis_error:
                logger.error(f"短期记忆模块初始化失败: {str(redis_error)}")
                self.short_term = None
            
            # 初始化长期记忆对象（但不连接数据库）
            try:
                self.long_term = LongTermMemory()  # 注意：此时尚未连接数据库
                logger.info("长期记忆模块对象创建成功，等待异步初始化")
            except Exception as mongo_error:
                logger.error(f"长期记忆模块对象创建失败: {str(mongo_error)}")
                self.long_term = None
            
            # 检查至少一个记忆模块初始化成功
            if not self.short_term and not self.long_term:
                logger.error("所有记忆模块初始化失败，部分功能可能不可用")
            
            logger.info("记忆管理器初始化完成")
        except Exception as e:
            logger.error(f"记忆管理器初始化失败: {str(e)}")
            # 确保对象至少有属性定义
            self.short_term = None
            self.long_term = None
        
    async def initialize(self):
        """
        异步初始化记忆管理器
        必须在创建MemoryManager对象后调用
        """
        try:
            # 如果短期记忆模块初始化失败，尝试重新初始化
            if not self.short_term:
                try:
                    # 创建Redis客户端
                    import redis
                    from app.config import memory_settings
                    
                    # 从环境变量或内存设置中获取配置
                    redis_host = os.getenv("REDIS_HOST", "localhost")
                    redis_port = int(os.getenv("REDIS_PORT", "6378"))
                    redis_password = os.getenv("REDIS_PASSWORD", "!qaz2wsX")
                    max_chat_rounds = int(os.getenv("MAX_CHAT_ROUNDS", "2"))
                    
                    redis_client = redis.Redis(
                        host=redis_host,
                        port=redis_port,
                        password=redis_password,
                        decode_responses=True
                    )
                    
                    # 测试Redis连接
                    try:
                        redis_client.ping()
                        logger.info(f"异步初始化：成功连接到Redis服务器: {redis_host}:{redis_port}")
                    except redis.ConnectionError as conn_err:
                        logger.error(f"异步初始化：Redis连接失败: {str(conn_err)}")
                        raise
                    
                    # 使用创建的Redis客户端初始化ShortTermMemory
                    self.short_term = ShortTermMemory(redis_client=redis_client, max_rounds=max_chat_rounds)
                    logger.info("异步初始化：短期记忆模块初始化成功")
                except Exception as redis_error:
                    logger.error(f"异步初始化：短期记忆模块初始化失败: {str(redis_error)}")
                    self.short_term = None
            
            # 异步初始化长期记忆
            if self.long_term:
                try:
                    await self.long_term.initialize()
                    logger.info("长期记忆模块异步初始化成功")
                except Exception as mongo_error:
                    logger.error(f"长期记忆模块异步初始化失败: {str(mongo_error)}")
                    # 保留对象，部分功能可能仍然可用
            return self
        except Exception as e:
            logger.error(f"记忆管理器异步初始化失败: {str(e)}")
            return self
        
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
            if not self.short_term:
                logger.error("短期记忆模块不可用，无法创建新会话")
                # 返回一个临时会话ID，允许应用继续工作
                return f"temp-{int(time.time())}-{str(uuid.uuid4())[:8]}"
                
            # 生成会话ID
            session_id = str(int(time.time())) + "-" + str(uuid.uuid4())[:8]
            
            # 创建会话
            self.short_term.start_session(session_id, user_id, selected_username)
            
            logger.info(f"已开始新会话: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"开始新会话失败: {str(e)}")
            # 返回一个临时会话ID，允许应用继续工作
            return f"temp-{int(time.time())}-{str(uuid.uuid4())[:8]}"
        
    async def end_session(self, session_id: str, user_id: str) -> Dict:
        """
        结束会话并生成摘要
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            结果信息，包含会话ID和摘要
        """
        try:
            # 获取会话消息
            messages = self.short_term.get_session_messages(session_id, user_id)
            
            if not messages:
                logger.warning(f"会话为空: {session_id}")
                return SessionResponse(
                    success=False,
                    error="会话为空，无法生成摘要"
                ).dict()
                
            # 生成摘要
            summary = await summary_service.generate_summary(messages)
            
            # 生成嵌入向量
            embedding = summary_service.generate_embedding(summary)
            
            # 存储摘要
            summary_id = await self.long_term.store_session_summary(
                session_id=session_id,
                user_id=user_id,
                summary=summary,
                messages_count=len(messages),
                embedding=embedding
            )
            
            # 更新会话状态为已归档
            try:
                await SessionService.change_session_status(
                    session_id=session_id,
                    user_id=user_id,
                    new_status=SessionStatus.ARCHIVED
                )
                logger.info(f"已将会话 {session_id} 状态更新为已归档")
            except Exception as status_error:
                logger.error(f"更新会话状态失败: {str(status_error)}")
            
            # 更新会话状态
            self.short_term.end_session(session_id, user_id)
            
            logger.info(f"已结束会话: {session_id}, 生成摘要: {summary[:100]}...")
            return SessionResponse(
                success=True,
                session_id=session_id,
                summary=summary
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
            if not self.short_term:
                logger.error("短期记忆模块不可用，无法添加消息")
                return False
                
            # 预处理role_id，确保它不是"null"字符串
            if role_id == "null" or role_id == "":
                role_id = None
                logger.info("MemoryManager: 接收到空字符串或'null'角色ID，已设置为None")
            elif role_id:
                logger.info(f"MemoryManager: 处理消息，角色ID: {role_id}, 类型: {type(role_id).__name__}")
                
                # 尝试预先获取角色名称，用于日志记录
                try:
                    from app.database.connection import get_database
                    from bson.objectid import ObjectId
                    
                    if role_id and str(role_id).strip():
                        try:
                            db = await get_database()
                            if db is not None:
                                object_id = ObjectId(role_id)
                                role_info = await db.roles.find_one({"_id": object_id})
                                
                                if role_info and "name" in role_info:
                                    logger.info(f"MemoryManager: 找到角色名称: {role_info['name']}")
                        except Exception as e:
                            logger.info(f"MemoryManager: 预先获取角色名称失败 (非关键错误): {str(e)}")
                except Exception:
                    # 这只是用于日志记录的非关键操作，忽略错误
                    pass
            
            # 添加消息，并获取是否需要归档的消息
            try:
                result, oldest_message = await self.short_term.add_message(
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
            
            # 如果有需要归档的消息，执行归档操作
            if result and oldest_message and self.long_term and self.long_term.db is not None:  # 修复: 使用identity比较而非布尔测试
                try:
                    # 记录传递给归档功能的消息信息
                    logger.info(f"准备归档消息: role={oldest_message.get('role', '未知')}, roleid={oldest_message.get('roleid', '无')}")
                    archived = await self.archive_message(session_id, user_id, oldest_message)
                    if archived:
                        logger.info(f"已成功将消息归档到MongoDB: 会话{session_id}")
                    else:
                        logger.warning(f"消息归档失败: 会话{session_id}")
                except Exception as archive_error:
                    logger.error(f"归档消息时出错: {str(archive_error)}")
                    # 继续执行，不影响主流程
            
            return result
        except Exception as e:
            logger.error(f"添加消息失败: {str(e)}")
            return False
            
    async def archive_message(self, session_id: str, user_id: str, message: dict) -> bool:
        """
        将单条消息归档到MongoDB
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            message: 消息数据
            
        Returns:
            是否成功归档
        """
        try:
            # 检查长期记忆是否可用
            if not self.long_term or self.long_term.db is None:
                logger.warning("长期记忆模块不可用，无法归档消息")
                return False
                
            from app.database.connection import get_database
            db = await get_database()
            
            if db is not None:
                # 构建消息文档
                message_doc = {
                    "session_id": session_id,
                    "user_id": user_id,
                    "role": message.get("role", "unknown"),
                    "content": message.get("content", ""),
                    "timestamp": message.get("timestamp", time.time()),
                    "archived_at": time.time(),
                    "archived_from": "redis_buffer_memory"
                }
                
                # 添加roleid字段，如果原消息中存在
                if "roleid" in message and message["roleid"] is not None:
                    message_doc["roleid"] = message["roleid"]
                    logger.info(f"消息归档: 包含roleid = {message['roleid']}")
                
                # 插入MongoDB
                try:
                    if hasattr(db, 'messages'):
                        result = await db.messages.insert_one(message_doc)
                        if result and result.inserted_id:
                            logger.info(f"消息已归档到MongoDB, ID: {result.inserted_id}")
                            return True
                        else:
                            logger.warning("消息归档到MongoDB失败，未返回插入ID")
                            return False
                    else:
                        # 尝试使用self.long_term.db作为备用
                        if hasattr(self.long_term.db, 'messages'):
                            result = await self.long_term.db.messages.insert_one(message_doc)
                            if result and result.inserted_id:
                                logger.info(f"使用备用连接归档消息到MongoDB, ID: {result.inserted_id}")
                                return True
                        
                        logger.error("数据库连接中没有messages集合")
                        return False
                except Exception as insert_error:
                    logger.error(f"插入MongoDB失败: {str(insert_error)}")
                    return False
                
            else:
                logger.error("无法获取数据库连接")
                return False
                
        except Exception as e:
            logger.error(f"归档消息到MongoDB失败: {str(e)}")
            return False
            
    async def build_context(self, session_id: str, user_id: str, current_message: str = None) -> Dict:
        """
        构建对话上下文
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            current_message: 当前消息（可选）
            
        Returns:
            上下文信息，包含消息和相关摘要
        """
        try:
            # 获取当前会话消息
            current_messages = self.short_term.get_session_messages(session_id, user_id)
            
            # 检查是否需要生成摘要
            token_count = self.short_term.count_tokens(session_id, user_id)
            if summary_service.should_generate_summary(current_messages, token_count):
                # 生成中间摘要
                summary = await summary_service.generate_summary(current_messages)
                embedding = summary_service.generate_embedding(summary)
                
                # 存储摘要
                await self.long_term.store_session_summary(
                    session_id=session_id,
                    user_id=user_id,
                    summary=summary,
                    messages_count=len(current_messages),
                    embedding=embedding
                )
                
                logger.info(f"已生成中间摘要: {summary[:100]}...")
            
            # 生成当前消息的嵌入向量（如果提供）
            query_embedding = None
            if current_message:
                query_embedding = summary_service.generate_embedding(current_message)
            elif current_messages:
                # 使用最后一条消息作为查询
                query_embedding = summary_service.generate_embedding(current_messages[-1]["content"])
                
            # 检索相关历史摘要
            relevant_summaries = []
            if query_embedding:
                summaries = await self.long_term.search_relevant_summaries(user_id, query_embedding)
                relevant_summaries = [summary["summary"] for summary in summaries]
                
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
                related_summaries=relevant_summaries
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
            active_sessions = self.short_term.list_active_sessions(user_id)
            
            # 获取已完成会话
            completed_sessions = await self.long_term.get_user_summaries(user_id, limit, skip)
            
            # 合并会话列表
            return active_sessions + completed_sessions
            
        except Exception as e:
            logger.error(f"获取用户会话列表失败: {str(e)}")
            return []
            
    async def get_session_detail(self, session_id: str, user_id: str) -> Dict:
        """
        获取会话详情
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            会话详情
        """
        try:
            # 检查会话是否活跃
            session_info = self.short_term.get_session_info(session_id, user_id)
            
            if session_info:
                # 获取消息
                messages = self.short_term.get_session_messages(session_id, user_id)
                
                return {
                    "session_id": session_id,
                    "user_id": user_id,
                    "status": session_info.get("status", "unknown"),
                    "start_time": session_info.get("start_time"),
                    "end_time": session_info.get("end_time"),
                    "messages": messages,
                    "summary": None  # 活跃会话没有摘要
                }
            else:
                # 尝试从长期记忆获取
                summary = await self.long_term.get_session_summary(session_id)
                
                if summary:
                    return {
                        "session_id": session_id,
                        "user_id": user_id,
                        "status": "completed",
                        "summary": summary.get("summary"),
                        "created_at": summary.get("created_at"),
                        "messages_count": summary.get("messages_count")
                    }
                    
            # 会话不存在
            return None
            
        except Exception as e:
            logger.error(f"获取会话详情失败: {str(e)}")
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
            session_info = self.short_term.get_session_info(session_id, user_id)
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
            if not self.short_term:
                logger.error("短期记忆模块不可用，无法安全添加消息")
                return False
                
            # 确保会话存在
            actual_session_id = await self.ensure_session_exists(user_id, session_id)
            
            # 使用重试机制添加消息
            result, oldest_message = await self.short_term.add_message_with_retry(
                actual_session_id, 
                user_id, 
                role, 
                content,
                role_id,
                message_id,
                max_retries=3
            )
            
            # 如果有需要归档的消息，执行归档操作
            if result and oldest_message and self.long_term and self.long_term.db is not None:  # 修复: 使用identity比较而非布尔测试
                try:
                    await self.archive_message(actual_session_id, user_id, oldest_message)
                except Exception as archive_error:
                    logger.error(f"安全添加消息中归档步骤出错: {str(archive_error)}")
                    # 继续执行，不影响主流程
            
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
            if not self.short_term:
                logger.error("短期记忆模块不可用，无法更新角色名称")
                return {"success": False, "error": "短期记忆模块不可用"}
            
            # 调用ShortTermMemory的更新方法
            result = await self.short_term.update_role_names(session_id, user_id)
            
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