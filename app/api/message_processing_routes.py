"""
消息处理API路由 - 简化版

包含处理用户消息、生成AI回复、重新生成回复等功能
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import Dict, List, Optional, Any, AsyncIterator
from pydantic import BaseModel, Field
import logging
import asyncio
import json
from datetime import datetime

# 导入auth模块
from app.auth.auth_bearer import JWTBearer
from app.auth.auth_handler import get_current_user

logger = logging.getLogger(__name__)

# 创建路由
router = APIRouter(prefix="/api/messages", tags=["message-processing"])

# 请求和响应模型
class StreamMessageRequest(BaseModel):
    content: str = Field(..., description="用户消息内容")
    session_id: str = Field(..., description="会话ID")
    role_id: Optional[str] = Field(None, description="角色ID，如不提供则自动选择")
    temperature: Optional[float] = Field(0.7, description="生成温度参数")

# 模拟角色数据
MOCK_ROLES = {
    "role_id_1": {"name": "助手", "description": "通用助手角色"},
    "role_id_2": {"name": "专家", "description": "专业知识专家"},
    "role_id_3": {"name": "诗人", "description": "富有创造力的诗人"}
}

@router.post("/stream")
async def stream_message(
    request: StreamMessageRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    流式生成AI回复 - 简化演示版
    
    接收用户消息并以流式方式返回AI回复，适用于实时交互场景
    """
    
    async def generate_stream() -> AsyncIterator[str]:
        try:
            # 记录请求
            logger.info(f"收到流式请求: {request.content[:30]}...")
            
            # 获取角色信息
            role_id = request.role_id or "role_id_1"  # 默认使用助手角色
            role_name = MOCK_ROLES.get(role_id, {"name": "AI助手"})["name"]
            
            # 发送开始信号
            yield json.dumps({
                "type": "start",
                "message_id": f"msg_{datetime.now().timestamp()}",
                "role_id": role_id
            })
            
            # 根据角色生成不同的响应
            responses = {
                "role_id_1": f"我是{role_name}，很高兴为您服务！您问的是\"{request.content}\"，这是一个很好的问题。让我来详细回答...",
                "role_id_2": f"作为{role_name}，我可以专业地解答您关于\"{request.content}\"的问题。从技术角度来看...",
                "role_id_3": f"灵感闪现，{role_name}为您作答：\n关于\"{request.content}\"\n思绪万千如细雨落下..."
            }
            
            response = responses.get(role_id, responses["role_id_1"])
            
            # 模拟流式输出，将回复分成小块
            chunks = [response[i:i+10] for i in range(0, len(response), 10)]
            
            for chunk in chunks:
                yield json.dumps({
                    "type": "chunk",
                    "content": chunk
                })
                await asyncio.sleep(0.1)  # 模拟延迟
            
            # 发送完成信号
            yield json.dumps({
                "type": "end",
                "message_id": f"msg_{datetime.now().timestamp()}",
                "role_id": role_id
            })
            
        except Exception as e:
            logger.error(f"流生成错误: {str(e)}")
            yield json.dumps({
                "type": "error",
                "content": f"生成回复时出错: {str(e)}"
            })
    
    return StreamingResponse(
        generate_stream(),
        media_type="application/json"
    ) 