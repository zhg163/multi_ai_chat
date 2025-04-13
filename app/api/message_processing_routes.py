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
from app.auth.auth_handler import get_current_user, get_current_user_optional

# 导入角色服务
from app.services.role_service import RoleService
# 导入默认角色配置
from app.config.defaults import DEFAULT_ROLES

logger = logging.getLogger(__name__)

# 创建路由
router = APIRouter(prefix="/api/messages", tags=["message-processing"])

# 请求和响应模型
class StreamMessageRequest(BaseModel):
    content: str = Field(..., description="用户消息内容")
    session_id: str = Field(..., description="会话ID")
    role_id: Optional[str] = Field(None, description="角色ID，如不提供则自动选择")
    temperature: Optional[float] = Field(0.7, description="生成温度参数")

class MessageRequest(BaseModel):
    content: str = Field(..., description="用户消息内容")
    session_id: str = Field(..., description="会话ID")
    role_id: Optional[str] = Field(None, description="角色ID")
    
@router.post("")
async def process_message(
    request: MessageRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """
    处理用户消息 - 将请求转发到stream端点
    """
    try:
        logger.info(f"收到消息处理请求: {request.content[:30]}...")
        
        # 构建StreamMessageRequest
        stream_request = StreamMessageRequest(
            content=request.content,
            session_id=request.session_id,
            role_id=request.role_id,
            temperature=0.7  # 使用默认温度
        )
        
        # 重用stream_message的逻辑
        from app.memory.memory_manager import get_memory_manager
        memory_manager = await get_memory_manager()
        
        # 记录用户消息
        user_id = current_user.get("id", "anonymous_user") if current_user else "anonymous_user"
        await memory_manager.add_message(
            request.session_id,
            user_id,
            "user",
            request.content,
            role_id=request.role_id
        )
        
        # 使用角色生成响应
        role_id = request.role_id or "role_id_1"  # 默认使用助手角色
        role = DEFAULT_ROLES.get(role_id, DEFAULT_ROLES["default"])
        role_name = role["name"]
        
        # 生成回复内容
        responses = {
            "role_id_1": f"我是{role_name}，很高兴为您服务！您问的是\"{request.content}\"，这是一个很好的问题。让我来详细回答...",
            "role_id_2": f"作为{role_name}，我可以专业地解答您关于\"{request.content}\"的问题。从技术角度来看...",
            "role_id_3": f"灵感闪现，{role_name}为您作答：\n关于\"{request.content}\"\n思绪万千如细雨落下..."
        }
        
        response_content = responses.get(role_id, responses["role_id_1"])
        
        # 记录AI回复
        message_id = f"msg_{datetime.now().timestamp()}"
        await memory_manager.add_message(
            request.session_id,
            user_id,
            "assistant",
            response_content,
            role_id=role_id,
            message_id=message_id
        )
        
        return {
            "content": response_content,
            "message_id": message_id,
            "role_id": role_id
        }
    except Exception as e:
        logger.error(f"处理消息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理消息失败: {str(e)}")

@router.post("/stream")
async def stream_message(
    request: StreamMessageRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
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
            
            # 获取角色名称 - 从配置获取，不再使用硬编码
            role = DEFAULT_ROLES.get(role_id, DEFAULT_ROLES["default"])
            role_name = role["name"]
            
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