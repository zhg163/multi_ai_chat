from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import logging
from bson.objectid import ObjectId

from ..database.connection import get_database
from ..models.message import MessageType, MessageStatus, MessageResponse, MessageHistoryResponse

logger = logging.getLogger(__name__)

class MessageService:
    """消息服务类，处理消息的CRUD操作"""
    
    def __init__(self, db=None):
        """初始化，存储数据库连接（如果提供）"""
        self.db = db
        self.collection = None
        self._initialized = False
    
    async def initialize(self):
        """异步初始化，确保数据库连接可用"""
        if self._initialized:
            return
            
        if self.db is None:
            from ..database.connection import get_database
            self.db = await get_database()
            
        if self.db is not None:
            self.collection = self.db.messages
            self._initialized = True
        else:
            logger.error("无法获取数据库连接")
            raise ConnectionError("无法连接到数据库")
            
    async def _ensure_initialized(self):
        """确保服务已初始化"""
        if not self._initialized:
            await self.initialize()
    
    async def create_message(
        self, 
        session_id: str, 
        content: str, 
        message_type: MessageType, 
        user_id: Optional[str] = None, 
        role_id: Optional[str] = None, 
        parent_id: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None,
        status: MessageStatus = MessageStatus.SENT
    ) -> MessageResponse:
        """
        创建新消息
        
        Args:
            session_id: 会话ID
            content: 消息内容
            message_type: 消息类型(用户/助手)
            user_id: 用户ID (用户消息必填)
            role_id: 角色ID (助手消息必填)
            parent_id: 父消息ID
            metadata: 元数据
            status: 消息状态
            
        Returns:
            创建的消息对象
        """
        await self._ensure_initialized()
        
        now = datetime.utcnow()
        message_data = {
            "session_id": session_id,
            "content": content,
            "message_type": message_type,
            "status": status,
            "created_at": now,
            "updated_at": now,
            "metadata": metadata or {}
        }
        
        # 根据消息类型添加不同字段
        if message_type == MessageType.USER and user_id:
            message_data["user_id"] = user_id
        elif message_type == MessageType.ASSISTANT and role_id:
            message_data["role_id"] = role_id
        
        if parent_id:
            message_data["parent_id"] = parent_id
        
        try:
            result = await self.collection.insert_one(message_data)
            message_data["id"] = str(result.inserted_id)
            return MessageResponse(**message_data)
        except Exception as e:
            logger.error(f"Failed to create message: {str(e)}")
            raise
    
    async def get_message_by_id(self, message_id: str) -> Optional[MessageResponse]:
        """
        根据ID获取消息
        
        Args:
            message_id: 消息ID
            
        Returns:
            找到的消息对象，如果不存在则返回None
        """
        await self._ensure_initialized()
        
        try:
            message = await self.collection.find_one({"_id": ObjectId(message_id)})
            if not message:
                return None
            
            message["id"] = str(message.pop("_id"))
            return MessageResponse(**message)
        except Exception as e:
            logger.error(f"Failed to get message by ID {message_id}: {str(e)}")
            raise
    
    async def update_message(self, message_id: str, update_data: Dict[str, Any]) -> Optional[MessageResponse]:
        """
        更新消息
        
        Args:
            message_id: 消息ID
            update_data: 要更新的字段
            
        Returns:
            更新后的消息对象，如果不存在则返回None
        """
        await self._ensure_initialized()
        
        try:
            update_data["updated_at"] = datetime.utcnow()
            result = await self.collection.find_one_and_update(
                {"_id": ObjectId(message_id)},
                {"$set": update_data},
                return_document=True
            )
            
            if not result:
                return None
                
            result["id"] = str(result.pop("_id"))
            return MessageResponse(**result)
        except Exception as e:
            logger.error(f"Failed to update message {message_id}: {str(e)}")
            raise
    
    async def get_session_messages(
        self, 
        session_id: str, 
        limit: int = 50, 
        offset: int = 0, 
        sort_order: str = "desc"
    ) -> MessageHistoryResponse:
        """
        获取会话消息历史
        
        Args:
            session_id: 会话ID
            limit: 限制返回的消息数量
            offset: 跳过的消息数量
            sort_order: 排序方式，asc 或 desc
            
        Returns:
            消息历史对象，包含总数和消息列表
        """
        await self._ensure_initialized()
        
        try:
            # 统计总数
            total = await self.collection.count_documents({"session_id": session_id})
            
            # 设置排序
            sort_direction = -1 if sort_order.lower() == "desc" else 1
            
            # 获取消息
            cursor = self.collection.find({"session_id": session_id})
            cursor = cursor.sort("created_at", sort_direction).skip(offset).limit(limit)
            
            messages = []
            async for message in cursor:
                message["id"] = str(message.pop("_id"))
                messages.append(MessageResponse(**message))
            
            return MessageHistoryResponse(total=total, items=messages)
        except Exception as e:
            logger.error(f"Failed to get session messages for session {session_id}: {str(e)}")
            raise
    
    async def delete_message(self, message_id: str) -> bool:
        """
        删除消息（标记为已删除状态）
        
        Args:
            message_id: 消息ID
            
        Returns:
            是否成功删除
        """
        await self._ensure_initialized()
        
        try:
            result = await self.collection.update_one(
                {"_id": ObjectId(message_id)},
                {"$set": {"status": MessageStatus.DELETED, "updated_at": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to delete message {message_id}: {str(e)}")
            raise
            
    async def permanently_delete_message(self, message_id: str) -> bool:
        """
        永久删除消息（从数据库中移除）
        
        Args:
            message_id: 消息ID
            
        Returns:
            是否成功删除
        """
        await self._ensure_initialized()
        
        try:
            result = await self.collection.delete_one({"_id": ObjectId(message_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to permanently delete message {message_id}: {str(e)}")
            raise
    
    async def delete_session_messages(self, session_id: str) -> int:
        """
        删除会话中的所有消息
        
        Args:
            session_id: 会话ID
            
        Returns:
            删除的消息数量
        """
        await self._ensure_initialized()
        
        try:
            result = await self.collection.update_many(
                {"session_id": session_id},
                {"$set": {"status": MessageStatus.DELETED, "updated_at": datetime.utcnow()}}
            )
            return result.modified_count
        except Exception as e:
            logger.error(f"Failed to delete messages for session {session_id}: {str(e)}")
            raise
            
    async def permanently_delete_session_messages(self, session_id: str) -> int:
        """
        永久删除会话中的所有消息
        
        Args:
            session_id: 会话ID
            
        Returns:
            删除的消息数量
        """
        await self._ensure_initialized()
        
        try:
            result = await self.collection.delete_many({"session_id": session_id})
            return result.deleted_count
        except Exception as e:
            logger.error(f"Failed to permanently delete messages for session {session_id}: {str(e)}")
            raise