from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

class RAGChatRequest(BaseModel):
    """RAG增强聊天请求模型"""
    model: str = Field(..., description="LLM模型名称")
    messages: List[Dict[str, Any]] = Field(..., description="聊天消息列表")
    session_id: Optional[str] = Field(None, description="会话ID")
    role_id: Optional[str] = Field(None, description="角色ID")
    stream: bool = Field(True, description="是否流式响应")
    temperature: float = Field(0.7, description="温度参数")
    max_tokens: Optional[int] = Field(None, description="最大token数")
    auto_title: bool = Field(False, description="是否自动生成标题")
    message_id: Optional[str] = Field(None, description="消息ID，用于继续生成")
    context_limit: Optional[int] = Field(None, description="上下文窗口大小限制")
    
class RAGChatResponse(BaseModel):
    """RAG增强聊天响应"""
    choices: List[Dict[str, Any]] = Field(..., description="选择列表")
    message_id: Optional[str] = Field(None, description="消息ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    usage: Optional[Dict[str, int]] = Field(None, description="使用情况统计")
    references: Optional[List[Dict[str, Any]]] = Field(None, description="引用列表")

class RAGStopRequest(BaseModel):
    """停止生成请求"""
    message_id: str = Field(..., description="要停止的消息ID")
    session_id: Optional[str] = Field(None, description="会话ID") 