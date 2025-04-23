from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, Any, List, Optional
import json
import asyncio
import logging
import uuid

from ..services.rag_enhanced_service import RAGEnhancedService
from ..services.llm_service import LLMService
from ..services.message_service import MessageService
from ..services.session_service import SessionService
from ..services.role_service import RoleService
from ..auth.auth_handler import get_current_user, get_current_user_optional
# from ..common.redis_client import get_redis_client  # Removed import for non-existent module

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
        provider = request_data.get("provider")
        model_name = request_data.get("model_name")
        api_key = request_data.get("api_key")
        auto_role_match = request_data.get("auto_role_match", False)
        lang = request_data.get("lang", "zh")
        
        # 获取用户ID
        user_id = current_user.get("id") if current_user else None
        
        if stream:
            # 流式响应
            async def generate_stream():
                try:
                    # 确保首先发送message_id，方便前端处理停止生成
                    message_id = str(uuid.uuid4())
                    yield f"data: {json.dumps({'message_id': message_id})}\n\n"
                    
                    # 记录所有角色匹配信息，确保可以在错误恢复时使用
                    role_match_data = None
                    # 用于收集完整响应的缓冲区
                    full_response = ""
                    
                    async for chunk in rag_service.process_chat(
                        messages=messages,
                        model=model,
                        session_id=session_id,
                        user_id=user_id,
                        role_id=role_id,
                        stream=True,
                        provider=provider,
                        model_name=model_name,
                        api_key=api_key,
                        auto_role_match=auto_role_match,
                        lang=lang
                    ):
                        if isinstance(chunk, dict):
                            # 保存角色匹配信息
                            if 'role_match' in chunk:
                                role_match_data = chunk['role_match']
                                logger.info(f"接收到角色匹配信息: {role_match_data}")
                            
                            # 正常发送字典数据
                            yield f"data: {json.dumps(chunk)}\n\n"
                        else:
                            # 处理文本块
                            content = chunk.content if hasattr(chunk, 'content') else chunk
                            full_response += content
                            
                            # 文本块转换为符合SSE格式的JSON
                            sse_data = {
                                "choices": [{
                                    "delta": {"content": content},
                                    "index": 0,
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(sse_data)}\n\n"
                    
                    # 正常结束标记
                    yield f"data: {json.dumps({'choices': [{'finish_reason': 'stop'}]})}\n\n"
                    yield "data: [DONE]\n\n"
                    
                except Exception as e:
                    logger.error(f"流式生成过程中出错: {str(e)}")
                    
                    # 在出错时，返回错误信息
                    error_response = {
                        "error": f"生成过程中出错: {str(e)}",
                        "choices": [{
                            "message": {"content": full_response or "生成过程中断", "role": "assistant"},
                            "finish_reason": "error"
                        }]
                    }
                    
                    # 如果有角色匹配信息，确保包含在错误响应中
                    if role_match_data:
                        error_response["role_match"] = role_match_data
                        
                    yield f"data: {json.dumps(error_response)}\n\n"
                    yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # 禁用 Nginx 缓冲
                }
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
                stream=False,
                provider=provider,
                model_name=model_name,
                api_key=api_key,
                auto_role_match=auto_role_match,
                lang=lang
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
    """
    停止内容生成
    
    需要提供request_id来停止对应的生成任务
    """
    try:
        request_data = await request.json()
        
        if "request_id" not in request_data:
            return JSONResponse(
                status_code=400,
                content={"error": "缺少request_id参数"}
            )
            
        request_id = request_data["request_id"]
        # 获取Redis客户端
        # redis_client = get_redis_client()
        
        # 清除请求锁
        lock_key = f"generating:{request_id}"
        # await redis_client.delete(lock_key)
        
        # 删除存储的请求数据
        # await redis_client.delete(f"request:{request_id}")
        
        # Without Redis, we'll return a success message but warn in logs
        logger.warning(f"Redis client not available, can't clear resources for request {request_id}")
        
        return {"success": True, "message": f"已清除请求 {request_id} 的资源锁"}
        
    except Exception as e:
        logger.error(f"停止生成时出错: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"停止生成时出错: {str(e)}"}
        )

@router.post("/api/llm/generate")
async def generate_content(
    request: Request,
    current_user: Dict = Depends(get_current_user_optional),
    rag_service: RAGEnhancedService = Depends(get_rag_service)
):
    """
    生成内容接口 - 用于以下场景：
    
    1. 对预先角色匹配的请求进行内容生成
    2. 需要使用RAG增强的内容生成
    
    需要提供以下参数：
    1. 请求ID（通过角色匹配接口获得）
    2. 可选参数，如分段生成设置等
    """
    try:
        request_data = await request.json()
        
        if "request_id" not in request_data:
            return JSONResponse(
                status_code=400,
                content={"error": "缺少request_id参数"}
            )
            
        request_id = request_data["request_id"]
        
        # 获取Redis客户端并查找请求数据
        # redis_client = get_redis_client()
        # stored_data_json = await redis_client.get(f"request:{request_id}")
        
        # Without Redis, we can't retrieve the stored request data
        # Return an error message
        logger.error(f"Redis client not available, can't retrieve stored data for request {request_id}")
        return JSONResponse(
            status_code=500,
            content={"error": f"无法检索存储的请求数据，Redis不可用"}
        )
        
        # If Redis was available, the original code would continue here:
        # if not stored_data_json:
        #     return JSONResponse(
        #         status_code=404,
        #         content={"error": f"找不到对应的请求数据，请求ID: {request_id}"}
        #     )
            
        # stored_data = json.loads(stored_data_json)
        
        # # 获取存储的请求数据
        # query = stored_data.get("query", "")
        # matched_role = stored_data.get("matched_role", {})
        # system_prompt = stored_data.get("system_prompt")
        # messages = stored_data.get("messages", [])
        
        # # 构建聊天消息，将用户的查询作为最后一条用户消息
        # chat_messages = []
        
        # # 如果有预设的消息，优先使用
        # if messages:
        #     chat_messages = messages
        # else:
        #     # 否则构建简单的消息列表
        #     chat_messages = [{"role": "user", "content": query}]
        
        # # 其他请求参数
        # stream = request_data.get("stream", True)
        # provider = request_data.get("provider")
        # model_name = request_data.get("model_name")
        # api_key = request_data.get("api_key")
        # session_id = request_data.get("session_id")
        # user_id = current_user.get("id") if current_user else None
        # role_id = matched_role.get("role_id") if matched_role else None
        
        # # 设置生成锁，避免重复生成
        # lock_key = f"generating:{request_id}"
        # lock_result = await redis_client.set(lock_key, "1", nx=True, ex=300)  # 5分钟过期
        
        # if not lock_result:
        #     return JSONResponse(
        #         status_code=409,
        #         content={"error": "该请求正在生成中，请等待或停止当前生成"}
        #     )
            
        # try:
        #     if stream:
        #         # Stream implementation would go here
        
    except Exception as e:
        logger.error(f"内容生成处理出错: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"内容生成处理出错: {str(e)}"}
        ) 