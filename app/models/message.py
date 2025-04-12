from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from bson.objectid import ObjectId
from app.database.connection import Database
from pydantic import BaseModel, Field, validator
from enum import Enum

class MessageType(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class MessageStatus(str, Enum):
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    ERROR = "error"
    DELETED = "deleted"
    PROCESSING = "processing"  # 添加处理中状态

class MessageBase(BaseModel):
    """消息基础类"""
    content: str = Field(..., description="消息内容")
    metadata: Optional[Dict[str, Any]] = Field(None, description="消息元数据")

class UserMessageCreate(MessageBase):
    """用户发送消息请求模型"""
    session_id: str = Field(..., description="会话ID")
    parent_id: Optional[str] = Field(None, description="父消息ID（用于消息树结构）")

class AssistantMessageCreate(MessageBase):
    """助手消息创建请求模型"""
    session_id: str = Field(..., description="会话ID")
    role_id: str = Field(..., description="角色ID")
    parent_id: Optional[str] = Field(None, description="父消息ID（用于消息树结构）")

class MessageUpdate(BaseModel):
    """消息更新请求模型"""
    content: Optional[str] = Field(None, description="消息内容")
    status: Optional[MessageStatus] = Field(None, description="消息状态")
    metadata: Optional[Dict[str, Any]] = Field(None, description="消息元数据")

class MessageResponse(BaseModel):
    """单条消息响应模型"""
    id: str = Field(..., description="消息ID")
    session_id: str = Field(..., description="会话ID")
    user_id: Optional[str] = Field(None, description="用户ID，仅用户消息有此字段")
    role_id: Optional[str] = Field(None, description="角色ID，仅助手消息有此字段")
    message_type: MessageType = Field(..., description="消息类型")
    content: str = Field(..., description="消息内容")
    parent_id: Optional[str] = Field(None, description="父消息ID")
    status: MessageStatus = Field(..., description="消息状态")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    metadata: Optional[Dict[str, Any]] = Field(None, description="消息元数据")
    
    class Config:
        """配置类"""
        orm_mode = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class MessageHistoryResponse(BaseModel):
    """消息历史列表响应模型"""
    total: int = Field(..., description="总消息数")
    items: List[MessageResponse] = Field(..., description="消息列表")
    
    class Config:
        """配置类"""
        orm_mode = True

class Message:
    """消息模型类，提供数据库操作接口"""
    
    collection_name = "messages"
    
    @classmethod
    async def create(cls, db, session_id, content, message_type, status="sent", role_id=None, parent_id=None, metadata=None):
        """创建新消息"""
        now = datetime.utcnow()
        message = {
            "session_id": session_id,
            "role_id": role_id,
            "content": content,
            "message_type": message_type,
            "status": status,
            "parent_id": parent_id,
            "created_at": now,
            "updated_at": now,
            "metadata": metadata or {}
        }
        
        result = await db[cls.collection_name].insert_one(message)
        message["_id"] = result.inserted_id
        return message
    
    @classmethod
    async def get_by_id(cls, db, message_id):
        """通过ID获取消息"""
        return await db[cls.collection_name].find_one({"_id": message_id})
    
    @classmethod
    async def list_by_session(cls, db, session_id, limit=50, offset=0, sort_direction=1):
        """获取会话消息列表"""
        cursor = db[cls.collection_name].find(
            {"session_id": session_id}
        ).sort(
            "created_at", sort_direction
        ).skip(offset).limit(limit)
        
        return await cursor.to_list(length=limit)
    
    @classmethod
    async def count_by_session(cls, db, session_id):
        """获取会话消息数量"""
        return await db[cls.collection_name].count_documents({"session_id": session_id})
    
    @classmethod
    async def update(cls, db, message_id, update_data):
        """更新消息"""
        update_data["updated_at"] = datetime.utcnow()
        return await db[cls.collection_name].update_one(
            {"_id": message_id},
            {"$set": update_data}
        )
    
    @classmethod
    async def delete(cls, db, message_id):
        """删除消息"""
        return await db[cls.collection_name].delete_one({"_id": message_id})
    
    @classmethod
    async def delete_by_session(cls, db, session_id):
        """删除会话所有消息"""
        result = await db[cls.collection_name].delete_many({"session_id": session_id})
        return result.deleted_count

    @classmethod
    async def get_session_history(cls, session_id, limit=20):
        """获取会话的最近历史记录"""
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)
            
        cursor = cls.get_collection().find({"session_id": session_id})\
            .sort("created_at", -1)\
            .limit(limit)
            
        messages = await cursor.to_list(length=limit)
        messages.reverse()  # 按时间正序排列
        
        return messages 