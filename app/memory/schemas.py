"""
记忆模块的数据模型定义
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class Message(BaseModel):
    """聊天消息模型"""
    role: str
    content: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    roleid: Optional[str] = None

class ChatSession(BaseModel):
    """聊天会话模型"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    status: str = "active"  # active, completed
    start_time: float = Field(default_factory=lambda: datetime.now().timestamp())
    end_time: Optional[float] = None
    
class SessionSummary(BaseModel):
    """会话摘要模型"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_id: str
    summary: str
    messages_count: int
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class MemoryContext(BaseModel):
    """记忆上下文模型，用于构建LLM输入"""
    messages: List[Dict[str, str]]
    related_summaries: List[str] = []
    
class SessionResponse(BaseModel):
    """会话响应模型"""
    success: bool
    session_id: Optional[str] = None
    summary: Optional[str] = None
    error: Optional[str] = None 