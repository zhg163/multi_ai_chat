"""
消息处理服务

负责处理用户消息并协调AI回复流程，包括角色选择和回复生成
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import asyncio
from bson import ObjectId

from app.services.message_service import MessageService
from app.services.session_service import SessionService
from app.services.role_service import RoleService
from app.services.role_selection_engine import RoleSelectionEngine
from app.services.ai_service import AIService
from app.models.message import MessageType, MessageStatus, MessageResponse

logger = logging.getLogger(__name__)

class MessageProcessingService:
    """
    消息处理服务
    
    协调消息处理流程，包括：
    1. 接收并存储用户消息
    2. 选择合适的角色进行回复
    3. 生成并存储AI回复
    4. 处理多轮对话逻辑
    """
    
    def __init__(self, 
                 message_service=None,
                 session_service=None,
                 role_service=None,
                 role_selection_engine=None,
                 ai_service=None):
        """
        初始化消息处理服务
        
        Args:
            message_service: 消息服务
            session_service: 会话服务
            role_service: 角色服务
            role_selection_engine: 角色选择引擎
            ai_service: AI服务
        """
        self.message_service = message_service or MessageService()
        self.session_service = session_service or SessionService()
        self.role_service = role_service or RoleService()
        self.role_selection_engine = role_selection_engine or RoleSelectionEngine(
            session_service=self.session_service,
            role_service=self.role_service
        )
        self.ai_service = ai_service or AIService()
    
    async def process_user_message(self,
                                  session_id: str,
                                  content: str,
                                  user_id: str,
                                  parent_id: Optional[str] = None,
                                  preferred_role_id: Optional[str] = None,
                                  selection_mode: str = "auto",
                                  metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理用户消息并生成AI回复
        
        Args:
            session_id: 会话ID
            content: 消息内容
            user_id: 用户ID
            parent_id: 父消息ID
            preferred_role_id: 首选角色ID（如果用户指定）
            selection_mode: 角色选择模式
            metadata: 元数据
            
        Returns:
            包含用户消息和AI回复的字典
        """
        # 1. 创建并存储用户消息
        user_message = await self.message_service.create_message(
            session_id=session_id,
            content=content,
            message_type=MessageType.USER,
            user_id=user_id,
            parent_id=parent_id,
            metadata=metadata
        )
        
        # 检查会话是否存在
        session = await self.session_service.get_session_by_id(session_id, user_id)
        if not session:
            logger.error(f"Session {session_id} not found for user {user_id}")
            return {
                "user_message": user_message,
                "ai_response": None,
                "error": "Session not found"
            }
        
        # 2. 获取对话历史
        conversation_history = await self.get_conversation_context(session_id, limit=10)
        
        # 3. 选择角色
        start_selection = datetime.utcnow()
        selected_role, match_score, selection_reason = await self.role_selection_engine.select_role_for_message(
            session_id=session_id,
            message=content,
            user_id=user_id,
            context_messages=conversation_history,
            preferred_role_id=preferred_role_id,
            selection_mode=selection_mode
        )
        selection_time = (datetime.utcnow() - start_selection).total_seconds()
        
        if not selected_role:
            logger.error(f"No suitable role found for message in session {session_id}")
            # 创建错误消息
            error_message = await self.message_service.create_message(
                session_id=session_id,
                content="抱歉，无法为您的消息找到合适的回复角色。请确保会话中添加了有效的角色，或联系管理员获取帮助。",
                message_type=MessageType.ASSISTANT,
                parent_id=str(user_message.id),
                status=MessageStatus.ERROR,
                metadata={
                    "error": "No suitable role found",
                    "selection_time": selection_time
                }
            )
            return {
                "user_message": user_message,
                "ai_response": error_message,
                "error": "No suitable role found"
            }
        
        # 4. 记录角色选择结果
        await self.role_selection_engine.log_selection(
            session_id=session_id,
            message_id=str(user_message.id),
            selected_role_id=str(selected_role["_id"]),
            score=match_score,
            selection_reason=selection_reason
        )
        
        # 5. 创建AI助手消息（状态为处理中）
        assistant_message = await self.message_service.create_message(
            session_id=session_id,
            content="",  # 初始为空，后续更新
            message_type=MessageType.ASSISTANT,
            role_id=str(selected_role["_id"]),
            parent_id=str(user_message.id),
            status=MessageStatus.PROCESSING,
            metadata={
                "role_name": selected_role.get("name", "Assistant"),
                "match_score": match_score,
                "selection_reason": selection_reason,
                "selection_time": selection_time
            }
        )
        
        # 6. 获取角色提示
        role_prompt = await self.role_selection_engine.get_role_prompt(
            role_id=str(selected_role["_id"]),
            message=content,
            conversation_history=conversation_history
        )
        
        # 7. 异步生成AI回复
        # 注意：这里启动异步任务但不等待它完成
        asyncio.create_task(
            self._generate_ai_response(
                message_id=str(assistant_message.id),
                role_id=str(selected_role["_id"]),
                user_message=content,
                role_prompt=role_prompt,
                conversation_history=conversation_history,
                temperature=selected_role.get("temperature", 0.7)
            )
        )
        
        # 8. 返回用户消息和初始AI回复
        return {
            "user_message": user_message,
            "ai_response": assistant_message,
            "selected_role": {
                "id": str(selected_role["_id"]),
                "name": selected_role.get("name", "Assistant"),
                "match_score": match_score,
                "selection_reason": selection_reason
            }
        }
    
    async def _generate_ai_response(self,
                                  message_id: str,
                                  role_id: str,
                                  user_message: str,
                                  role_prompt: str,
                                  conversation_history: List[Dict[str, Any]],
                                  temperature: float = 0.7) -> None:
        """
        异步生成AI回复
        
        Args:
            message_id: 回复消息ID
            role_id: 角色ID
            user_message: 用户消息
            role_prompt: 角色提示
            conversation_history: 对话历史
            temperature: 生成温度参数
        """
        try:
            # 使用AI服务生成回复
            ai_response = await self.ai_service.generate_response(
                user_message=user_message,
                role_prompt=role_prompt,
                conversation_history=conversation_history,
                temperature=temperature
            )
            
            # 更新消息状态为已发送，并添加生成的内容
            await self.message_service.update_message(
                message_id=message_id,
                update_data={
                    "content": ai_response,
                    "status": MessageStatus.SENT,
                    "metadata": {
                        "completion_time": datetime.utcnow().isoformat(),
                        "temperature": temperature
                    }
                }
            )
            
            logger.info(f"Successfully generated AI response for message {message_id}")
            
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            
            # 更新消息状态为错误
            await self.message_service.update_message(
                message_id=message_id,
                update_data={
                    "content": f"生成回复时发生错误，请稍后再试。错误: {str(e)}",
                    "status": MessageStatus.ERROR,
                    "metadata": {
                        "error": str(e),
                        "error_time": datetime.utcnow().isoformat()
                    }
                }
            )
    
    async def get_conversation_context(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取对话上下文
        
        Args:
            session_id: 会话ID
            limit: 最大消息数量
            
        Returns:
            对话历史消息列表
        """
        # 获取最近的消息
        message_history = await self.message_service.get_session_messages(
            session_id=session_id,
            limit=limit,
            sort_order="asc"  # 按时间升序，确保正确的对话顺序
        )
        
        # 转换为简单列表格式
        messages = []
        for msg in message_history.items:
            message_dict = {
                "id": str(msg.id),
                "content": msg.content,
                "message_type": msg.message_type,
                "created_at": msg.created_at.isoformat(),
            }
            
            if msg.message_type == MessageType.USER and hasattr(msg, "user_id"):
                message_dict["user_id"] = msg.user_id
            elif msg.message_type == MessageType.ASSISTANT and hasattr(msg, "role_id"):
                message_dict["role_id"] = msg.role_id
            
            messages.append(message_dict)
            
        return messages
    
    async def regenerate_response(self,
                                message_id: str,
                                user_id: str,
                                preferred_role_id: Optional[str] = None) -> Optional[MessageResponse]:
        """
        重新生成AI回复
        
        Args:
            message_id: 需要重新生成的消息ID
            user_id: 用户ID
            preferred_role_id: 首选角色ID（如果要更换角色）
            
        Returns:
            重新生成的消息，如果失败则返回None
        """
        # 获取原始消息
        original_message = await self.message_service.get_message_by_id(message_id)
        if not original_message:
            logger.error(f"Message {message_id} not found")
            return None
        
        # 检查消息类型和权限
        if original_message.message_type != MessageType.ASSISTANT:
            logger.error(f"Message {message_id} is not an assistant message")
            return None
        
        # 获取父消息（用户消息）
        parent_id = getattr(original_message, "parent_id", None)
        if not parent_id:
            logger.error(f"Assistant message {message_id} has no parent message")
            return None
            
        parent_message = await self.message_service.get_message_by_id(parent_id)
        if not parent_message:
            logger.error(f"Parent message {parent_id} not found")
            return None
        
        # 获取会话信息
        session_id = original_message.session_id
        session = await self.session_service.get_session_by_id(session_id, user_id)
        if not session:
            logger.error(f"Session {session_id} not found for user {user_id}")
            return None
        
        # 确定是否需要更换角色
        role_id = preferred_role_id or getattr(original_message, "role_id", None)
        
        # 获取对话上下文
        conversation_history = await self.get_conversation_context(session_id)
        
        # 如果未指定角色，使用角色选择引擎选择角色
        if not role_id:
            selected_role, match_score, selection_reason = await self.role_selection_engine.select_role_for_message(
                session_id=session_id,
                message=parent_message.content,
                user_id=user_id,
                context_messages=conversation_history,
                selection_mode="highest_score"  # 使用得分最高的角色
            )
            
            if not selected_role:
                logger.error(f"No suitable role found for regeneration in session {session_id}")
                return None
                
            role_id = str(selected_role["_id"])
        
        # 获取角色信息
        role = await self.role_service.get_role_by_id(role_id)
        if not role:
            logger.error(f"Role {role_id} not found")
            return None
        
        # 获取角色提示
        role_prompt = await self.role_selection_engine.get_role_prompt(
            role_id=role_id,
            message=parent_message.content,
            conversation_history=conversation_history
        )
        
        # 更新消息状态为处理中
        await self.message_service.update_message(
            message_id=message_id,
            update_data={
                "status": MessageStatus.PROCESSING,
                "role_id": role_id,
                "metadata": {
                    "regenerated": True,
                    "original_role_id": getattr(original_message, "role_id", None),
                    "regeneration_time": datetime.utcnow().isoformat()
                }
            }
        )
        
        # 异步生成新回复
        asyncio.create_task(
            self._generate_ai_response(
                message_id=message_id,
                role_id=role_id,
                user_message=parent_message.content,
                role_prompt=role_prompt,
                conversation_history=conversation_history,
                temperature=role.get("temperature", 0.7)
            )
        )
        
        # 返回更新后的消息
        return await self.message_service.get_message_by_id(message_id) 