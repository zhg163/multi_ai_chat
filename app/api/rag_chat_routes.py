from fastapi import APIRouter, Depends, Body, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List, Optional
import json
import asyncio
from ..models.rag_models import RAGChatRequest, RAGStopRequest
from ..services.rag_enhanced_service import RAGEnhancedService
from ..services.llm_service import LLMService
from ..services.message_service import MessageService
from ..services.session_service import SessionService
from ..services.role_service import RoleService
from ..auth.auth_handler import get_current_user, get_current_user_optional

router = APIRouter()

# 添加RAGEnhancedService依赖
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
    request: RAGChatRequest = Body(...),
    current_user: Dict = Depends(get_current_user_optional),
    llm_service: LLMService = Depends(),
    message_service: MessageService = Depends(),
    session_service: SessionService = Depends(),
    role_service: RoleService = Depends()
):
    """
    RAG增强聊天接口
    
    - 自动判断是否需要检索知识
    - 展示思考过程和检索结果
    - 支持各种参数控制，包括流式响应、温度、最大token等
    """
    # 创建RAG服务
    rag_service = RAGEnhancedService(
        llm_service=llm_service,
        message_service=message_service,
        session_service=session_service,
        role_service=role_service
    )
    
    # 忽略请求中的模型参数，固定使用deepseek-chat
    model = "deepseek-chat"
    
    if request.stream:
        # 流式响应
        async def generate_stream():
            async for chunk in rag_service.process_chat(
                messages=request.messages,
                model=model,
                session_id=request.session_id,
                user_id=current_user.get("id") if current_user else None,
                role_id=request.role_id,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
                message_id=request.message_id,
                context_limit=request.context_limit,
                auto_title=request.auto_title
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
            messages=request.messages,
            model=model,
            session_id=request.session_id,
            user_id=current_user.get("id") if current_user else None,
            role_id=request.role_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False,
            message_id=request.message_id,
            context_limit=request.context_limit,
            auto_title=request.auto_title
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

@router.post("/api/llm/chatrag/stop")
async def stop_generation(
    request: RAGStopRequest,
    rag_service: RAGEnhancedService = Depends(get_rag_service)
):
    """停止消息生成"""
    success = await rag_service.stop_message_generation(request.message_id)
    return {"success": success} 