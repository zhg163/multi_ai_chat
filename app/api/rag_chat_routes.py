from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, Any, List, Optional
import json
import asyncio
import logging

from ..services.rag_enhanced_service import RAGEnhancedService
from ..services.llm_service import LLMService
from ..services.message_service import MessageService
from ..services.session_service import SessionService
from ..services.role_service import RoleService
from ..auth.auth_handler import get_current_user, get_current_user_optional

# 创建日志记录器
logger = logging.getLogger(__name__)

router = APIRouter()

# 创建RAGEnhancedService依赖
def get_rag_service(
    llm_service: LLMService = Depends(),
    message_service: MessageService = Depends(),
    session_service: SessionService = Depends(),
    role_service: RoleService = Depends()
) -> RAGEnhancedService:
    return RAGEnhancedService(
        llm_service=llm_service,
        message_service=message_service,
        session_service=session_service,
        role_service=role_service
    )

@router.post("/api/llm/chatrag")
async def chat_with_rag(
    request: Request,
    current_user: Dict = Depends(get_current_user_optional),
    rag_service: RAGEnhancedService = Depends(get_rag_service)
):
    """
    RAG增强聊天接口
    
    - 自动判断是否需要检索知识
    - 展示思考过程和检索结果
    - 支持各种参数控制，包括流式响应、温度、最大token等
    """
    try:
        # 解析请求体
        request_data = await request.json()
        
        # 验证请求参数
        if "messages" not in request_data or not request_data["messages"] or len(request_data["messages"]) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "消息列表不能为空"}
            )
            
        for msg in request_data["messages"]:
            if "role" not in msg or "content" not in msg:
                return JSONResponse(
                    status_code=400,
                    content={"error": "消息格式错误，必须包含role和content字段"}
                )
        
        # 确保RAG服务初始化 - 使用_ensure_initialized内部方法代替initialize
        await rag_service._ensure_initialized()
        
        # 获取请求参数，提供默认值
        messages = request_data["messages"]
        model = "deepseek-chat"  # 忽略请求中的模型参数，固定使用deepseek-chat
        session_id = request_data.get("session_id")
        role_id = request_data.get("role_id")
        stream = request_data.get("stream", True)
        temperature = float(request_data.get("temperature", 0.7))
        max_tokens = request_data.get("max_tokens")
        message_id = request_data.get("message_id")
        context_limit = request_data.get("context_limit")
        auto_title = request_data.get("auto_title", False)
        
        # 获取用户ID
        user_id = current_user.get("id") if current_user else None
        
        if stream:
            # 流式响应
            async def generate_stream():
                async for chunk in rag_service.process_chat(
                    messages=messages,
                    model=model,
                    session_id=session_id,
                    user_id=user_id,
                    role_id=role_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    message_id=message_id,
                    context_limit=context_limit,
                    auto_title=auto_title
                ):
                    if isinstance(chunk, dict):
                        yield f"data: {json.dumps(chunk)}\n\n"
                    else:
                        # 文本块转换为符合SSE格式的JSON
                        sse_data = {
                            "choices": [{
                                "delta": {"content": chunk},
                                "index": 0,
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(sse_data)}\n\n"
                
                # 结束标记
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream"
            )
        else:
            # 非流式响应
            full_response = ""
            last_dict_response = None
            
            async for chunk in rag_service.process_chat(
                messages=messages,
                model=model,
                session_id=session_id,
                user_id=user_id,
                role_id=role_id,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                message_id=message_id,
                context_limit=context_limit,
                auto_title=auto_title
            ):
                if isinstance(chunk, str):
                    full_response += chunk
                elif isinstance(chunk, dict):
                    last_dict_response = chunk
            
            # 格式化为与标准chat接口一致的响应格式
            response = {
                "choices": [{
                    "message": {"content": full_response, "role": "assistant"},
                    "index": 0,
                    "finish_reason": "stop"
                }]
            }
            
            # 添加可能的额外信息
            if last_dict_response:
                response.update({k: v for k, v in last_dict_response.items() 
                                if k not in ["choices"]})
            
            return response
    except Exception as e:
        logger.error(f"RAG聊天处理出错: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"服务器错误: {str(e)}"}
        )

@router.post("/api/llm/chatrag/stop")
async def stop_generation(
    request: Request,
    rag_service: RAGEnhancedService = Depends(get_rag_service)
):
    """停止消息生成"""
    try:
        request_data = await request.json()
        
        if "message_id" not in request_data:
            return JSONResponse(
                status_code=400,
                content={"error": "缺少必须的message_id参数"}
            )
            
        message_id = request_data["message_id"]
        # 使用_ensure_initialized内部方法代替显式初始化
        await rag_service._ensure_initialized()
        success = await rag_service.stop_message_generation(message_id)
        return {"success": success}
    except Exception as e:
        logger.error(f"停止生成失败: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"停止生成失败: {str(e)}"}
        ) 