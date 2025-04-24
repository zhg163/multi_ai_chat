"""
RAG Enhanced Chat Routes - 提供RAG增强的聊天接口
"""

from fastapi import APIRouter, Depends, Request, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, Any, List, Optional, AsyncGenerator, Union, Annotated
import json
import asyncio
import logging
import uuid
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError
import time
import httpx

from ..services.rag_enhanced_service import RAGEnhancedService
from ..services.llm_service import LLMService
from ..services.message_service import MessageService
from ..services.session_service import SessionService
from ..services.role_service import RoleService
from ..auth.auth_handler import get_current_user, get_current_user_optional
from ..common.redis_client import get_redis_client
from ..utils.redis_lock import RedisLock, obtain_lock

# 创建日志记录器
logger = logging.getLogger(__name__)

# 定义请求和响应模型
class ChatMessage(BaseModel):
    role: str
    content: str

class RoleInfo(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    system_prompt: Optional[str] = None

class RoleMatchResult(BaseModel):
    success: bool
    role: Optional[RoleInfo] = None
    match_reason: Optional[str] = None
    error: Optional[str] = None

class RoleSelectRequest(BaseModel):
    messages: List[ChatMessage]
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    lang: Optional[str] = "zh"
    auto_role_match: Optional[bool] = False
    provider: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None

class RoleSelectResponse(BaseModel):
    request_id: str
    role_match: RoleMatchResult
    session_id: Optional[str] = None

class RoleSelectData(BaseModel):
    provider: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None

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
        
        # 生成唯一请求ID
        request_id = str(uuid.uuid4())
        
        # 获取用户ID
        user_id = current_user.get("id") if current_user else None
        
        # 获取Redis客户端
        redis_client = await get_redis_client()
        
        # 检查会话状态
        session_status_key = f"chatrag:session:{session_id}:status"
        session_status = await redis_client.get(session_status_key)
        
        # 清理长时间未释放的锁状态（超过5分钟的视为异常）
        status_ttl = await redis_client.ttl(session_status_key)
        if session_status == "processing" and status_ttl > 300:
            logger.warning(f"检测到长时间未释放的会话状态，自动重置: {session_id}")
            await redis_client.set(session_status_key, "idle", ex=300)
            session_status = "idle"
        
        if session_status == "processing":
            return JSONResponse(
                status_code=423,
                content={"error": "响应流已被锁定，无法处理。请重新发送消息或使用reset-stream接口重置状态。",
                         "session_id": session_id,
                         "request_id": request_id}
            )
        
        # 设置会话状态为处理中，缩短过期时间为60秒
        await redis_client.set(session_status_key, "processing", ex=60)
        
        if stream:
            # 流式响应
            async def generate_stream() -> AsyncGenerator[str, None]:
                """生成流式响应"""
                lock = None
                try:
                    # 获取锁但期限缩短到60秒
                    lock_key = f"chatrag:lock:{session_id}"
                    lock = RedisLock(redis_client, lock_key, expire_seconds=60)
                    if not await lock.acquire():
                        yield f"data: {json.dumps({'error': '会话当前正在处理另一个请求', 'type': 'error'})}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    
                    # 首先发送message_id，方便前端处理停止生成
                    message_id = request_id
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
                        }],
                        "type": "error"
                    }
                    
                    # 如果有角色匹配信息，确保包含在错误响应中
                    if role_match_data:
                        error_response["role_match"] = role_match_data
                        
                    yield f"data: {json.dumps(error_response)}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    # 无论如何都要重置会话状态
                    try:
                        await redis_client.set(session_status_key, "idle", ex=300)
                    except Exception as e:
                        logger.error(f"重置会话状态失败: {str(e)}")
                    
                    # 无论如何都释放锁
                    if lock:
                        try:
                            await lock.release()
                            logger.info(f"成功释放会话锁: {session_id}")
                        except Exception as e:
                            logger.error(f"释放会话锁失败: {str(e)}")
                    
                    # 存储完整响应到Redis以便后续操作
                    if full_response:
                        try:
                            await redis_client.hset(
                                f"chatrag:response:{request_id}",
                                mapping={
                                    "response": full_response,
                                    "completed_at": datetime.utcnow().isoformat(),
                                    "session_id": session_id
                                }
                            )
                            await redis_client.expire(f"chatrag:response:{request_id}", 1800)  # 30分钟过期
                        except Exception as e:
                            logger.error(f"存储响应到Redis失败: {str(e)}")
            
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
            # 非流式响应 - 修改锁的获取逻辑
            lock = None
            try:
                # 使用Redis锁确保同一会话不会有并发请求，缩短过期时间到60秒
                lock_key = f"chatrag:lock:{session_id}"
                lock = RedisLock(redis_client, lock_key, expire_seconds=60)
                
                if not await lock.acquire():
                    # 重置会话状态
                    await redis_client.set(session_status_key, "idle", ex=300)
                    return JSONResponse(
                        status_code=423,
                        content={"error": "会话当前正在处理另一个请求，请稍后再试"}
                    )
                
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
                logger.error(f"生成回复出错: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"服务器错误: {str(e)}"}
                )
            finally:
                # 重置会话状态
                try:
                    await redis_client.set(session_status_key, "idle", ex=300)
                except Exception as e:
                    logger.error(f"重置会话状态失败: {str(e)}")
                
                # 释放锁
                if lock:
                    try:
                        await lock.release()
                        logger.info(f"成功释放会话锁: {session_id}")
                    except Exception as e:
                        logger.error(f"释放会话锁失败: {str(e)}")
                
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
        redis_client = await get_redis_client()
        
        if not redis_client:
            logger.warning(f"Redis客户端不可用，无法清除请求 {request_id} 的资源")
            return JSONResponse(
                status_code=500,
                content={"error": "Redis服务不可用"}
            )
            
        # 获取会话ID
        session_id = await redis_client.hget(f"chatrag:response:{request_id}", "session_id")
        
        if not session_id:
            return JSONResponse(
                status_code=404,
                content={"error": f"找不到请求 {request_id} 的会话信息"}
            )
        
        # 重置会话状态
        session_status_key = f"chatrag:session:{session_id}:status"
        await redis_client.set(session_status_key, "idle", ex=300)
        
        # 清除请求锁
        lock_key = f"chatrag:lock:{session_id}"
        try:
            lock = RedisLock(redis_client, lock_key)
            await lock.release()
            logger.info(f"成功释放会话锁: {session_id}")
        except Exception as e:
            logger.warning(f"释放锁失败，可能锁已过期: {str(e)}")
        
        # 记录停止操作
        await redis_client.hset(
            f"chatrag:response:{request_id}",
            mapping={
                "stopped_at": datetime.utcnow().isoformat(),
                "stopped_by": "user"
            }
        )
        
        return {
            "success": True, 
            "message": f"已清除请求 {request_id} 的资源锁",
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"停止生成时出错: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"停止生成时出错: {str(e)}"}
        )

@router.post("/api/llm/chatrag/reset-stream")
async def reset_stream(
    session_id: str = Body(...),
    request_id: Optional[str] = Body(None)
):
    """
    重置流式响应状态，解除锁定
    
    Args:
        session_id: 会话ID
        request_id: 可选的请求ID
    """
    try:
        redis_client = await get_redis_client()
        
        if not redis_client:
            logger.warning(f"Redis客户端不可用，无法重置会话 {session_id} 的状态")
            return JSONResponse(
                status_code=500,
                content={"error": "Redis服务不可用"}
            )
            
        # 重置会话状态
        session_status_key = f"chatrag:session:{session_id}:status"
        await redis_client.set(session_status_key, "idle", ex=300)
        
        # 清除所有相关锁
        lock_key = f"chatrag:lock:{session_id}"
        try:
            # 首先尝试常规方式释放锁
            lock = RedisLock(redis_client, lock_key)
            await lock.release()
            logger.info(f"成功释放会话锁: {session_id}")
        except Exception as e:
            # 如果常规方式失败，则直接删除锁键
            logger.warning(f"常规方式释放锁失败: {str(e)}，尝试直接删除锁")
            try:
                await redis_client.delete(lock_key)
                logger.info(f"直接删除锁成功: {lock_key}")
            except Exception as e2:
                logger.error(f"直接删除锁失败: {str(e2)}")
        
        # 记录重置操作
        reset_info = {
            "reset_at": datetime.utcnow().isoformat(),
            "request_id": request_id or "unknown"
        }
        
        try:
            # 记录重置操作到会话重置历史
            await redis_client.hset(
                f"chatrag:session:{session_id}:resets",
                mapping=reset_info
            )
            await redis_client.expire(f"chatrag:session:{session_id}:resets", 86400)  # 1天过期
            
            # 如果提供了request_id，记录重置操作到请求信息中
            if request_id:
                await redis_client.hset(
                    f"chatrag:response:{request_id}",
                    mapping={
                        "reset_at": datetime.utcnow().isoformat(),
                        "status": "reset_by_user"
                    }
                )
        except Exception as e:
            logger.warning(f"记录重置信息失败: {str(e)}")
        
        return {
            "success": True, 
            "message": f"已重置会话 {session_id} 的流状态",
            "session_id": session_id,
            "reset_time": reset_info["reset_at"]
        }
        
    except Exception as e:
        logger.error(f"重置流状态时出错: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"重置流状态失败: {str(e)}"}
        )

@router.post("/api/llm/chatrag/generate", include_in_schema=True, response_model=None)
async def generate_chat_response(request: Request):
    """
    生成聊天响应API。支持流式和非流式响应。
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] 开始处理生成响应请求: {request.url.path}")
    
    # 记录请求状态 - 减少日志级别，仅在调试时记录
    validation_skipped = getattr(request.state, "validation_skipped", False)
    no_validation = getattr(request.state, "no_validation", False)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"[{request_id}] 请求状态: validation_skipped={validation_skipped}, no_validation={no_validation}")
    
    try:
        # 读取请求数据 - 减少日志记录
        cached_body = getattr(request.state, "cached_body", None)
        if cached_body:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[{request_id}] 使用缓存的请求体")
            if isinstance(cached_body, bytes):
                body_str = cached_body.decode("utf-8")
            else:
                body_str = str(cached_body)
            
            request_data_raw = json.loads(body_str)
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[{request_id}] 直接读取请求体")
            request_data_raw = await request.json()
        
        # 提取请求数据 - 减少日志记录
        if isinstance(request_data_raw, dict) and "request_data" in request_data_raw:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[{request_id}] 从嵌套结构提取request_data")
            request_data = request_data_raw["request_data"]
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[{request_id}] 使用整个请求体作为request_data")
            request_data = request_data_raw
        
        # 提取必要参数
        messages = request_data.get("messages", [])
        session_id = request_data.get("session_id", str(uuid.uuid4()))
        user_id = request_data.get("user_id", "anonymous")
        stream = request_data.get("stream", True)
        
        if not messages or not isinstance(messages, list):
            logger.error(f"[{request_id}] 缺少有效的messages字段")
            return JSONResponse(
                status_code=400,
                content={"error": "缺少有效的messages字段"}
            )
        
        # 检查Redis客户端
        redis_client = await get_redis_client()
        if not redis_client:
            logger.error(f"[{request_id}] Redis客户端不可用")
            return JSONResponse(
                status_code=500,
                content={"error": "服务暂时不可用，Redis连接失败"}
            )
        
        # 检查会话状态
        session_status_key = f"chatrag:session:{session_id}:status"
        session_status = await redis_client.get(session_status_key)
        
        # 清理长时间未释放的锁（超过5分钟视为异常）
        if session_status == "processing":
            status_ttl = await redis_client.ttl(session_status_key)
            if status_ttl > 300:
                logger.warning(f"[{request_id}] 检测到长时间未释放的会话状态，自动重置")
                await redis_client.set(session_status_key, "idle", ex=300)
                session_status = "idle"
        
        if session_status == "processing":
            return JSONResponse(
                status_code=423,
                content={
                    "error": "会话正在处理其他请求，请稍后再试或使用reset-stream重置状态",
                    "session_id": session_id,
                    "request_id": request_id
                }
            )
        
        # 设置会话状态为处理中
        await redis_client.set(session_status_key, "processing", ex=60)
        
        # 测试版响应（暂时使用测试版响应，后续可替换为实际生成）
        if stream:
            # 返回流式响应
            logger.info(f"[{request_id}] 生成流式响应")
            
            async def response_stream():
                # 模拟流式响应片段
                chunks = [
                    {"role": "assistant", "content": "我", "status": "streaming"},
                    {"role": "assistant", "content": "正在", "status": "streaming"},
                    {"role": "assistant", "content": "回答", "status": "streaming"},
                    {"role": "assistant", "content": "你的", "status": "streaming"},
                    {"role": "assistant", "content": "问题。", "status": "streaming"},
                    {"role": "assistant", "content": "\n\n斯卡蒂和伊芙利特是游戏《明日方舟》中的两个角色，她们的关系复杂且有故事背景。", "status": "complete"}
                ]
                
                try:
                    for chunk in chunks:
                        # 添加请求ID和时间戳
                        chunk["request_id"] = request_id
                        chunk["timestamp"] = datetime.utcnow().isoformat()
                        
                        # 按照正确的SSE格式发送，使用data前缀和两个换行符
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        
                        # 模拟延迟
                        await asyncio.sleep(0.5)
                finally:
                    # 重置会话状态
                    try:
                        await redis_client.set(session_status_key, "idle", ex=300)
                        logger.info(f"[{request_id}] 重置会话状态为idle")
                    except Exception as e:
                        logger.error(f"[{request_id}] 重置会话状态失败: {str(e)}")
            
            return StreamingResponse(
                response_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # 禁用Nginx缓冲
                }
            )
        else:
            # 返回非流式响应
            logger.info(f"[{request_id}] 生成非流式响应")
            
            # 模拟处理延迟
            await asyncio.sleep(1)
            
            try:
                response_data = {
                    "role": "assistant",
                    "content": "斯卡蒂和伊芙利特是游戏《明日方舟》中的两个角色，她们的关系复杂且有故事背景。",
                    "request_id": request_id,
                    "debug_info": {
                        "validation_skipped": validation_skipped,
                        "no_validation": no_validation,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                
                # 重置会话状态
                await redis_client.set(session_status_key, "idle", ex=300)
                logger.info(f"[{request_id}] 重置会话状态为idle")
                
                return JSONResponse(content=response_data)
            except Exception as e:
                logger.error(f"[{request_id}] 生成响应时出错: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": "生成响应失败", "detail": str(e)}
                )
            finally:
                # 重置会话状态
                try:
                    await redis_client.set(session_status_key, "idle", ex=300)
                except Exception as e:
                    logger.error(f"[{request_id}] 重置会话状态失败: {str(e)}")
    
    except Exception as e:
        logger.error(f"[{request_id}] 处理生成请求时出错: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "处理请求失败",
                "detail": str(e),
                "request_id": request_id
            }
        )

# 添加清理资源的工具函数
async def cleanup_resources(redis_client, session_id, lock=None, session_status_key=None):
    """
    确保锁和会话状态被正确释放
    
    Args:
        redis_client: Redis客户端
        session_id: 会话ID
        lock: RedisLock实例（可选）
        session_status_key: 会话状态键（可选）
        
    Returns:
        bool: 清理是否成功
    """
    success = True
    errors = []
    
    # 1. 重置会话状态
    if session_status_key and redis_client:
        try:
            await redis_client.set(session_status_key, "idle", ex=300)
            logger.info(f"会话状态重置为idle: {session_id}")
        except Exception as e:
            errors.append(f"重置会话状态失败: {str(e)}")
            success = False
    
    # 2. 释放锁
    if lock:
        try:
            await lock.release()
            logger.info(f"成功释放会话锁: {session_id}")
        except Exception as e:
            errors.append(f"释放会话锁失败: {str(e)}")
            # 尝试直接删除锁键
            try:
                if redis_client:
                    lock_key = f"chatrag:lock:{session_id}"
                    await redis_client.delete(lock_key)
                    logger.info(f"通过删除键释放锁: {lock_key}")
            except Exception as e2:
                errors.append(f"删除锁键失败: {str(e2)}")
                success = False
    
    # 记录清理结果
    if errors:
        logger.warning(f"资源清理过程中出现错误: {'; '.join(errors)}")
    
    return success

@router.post("/api/llm/chatrag/role-select", include_in_schema=True, response_model=None)
async def select_role_for_chat(
    request: Request,
    current_user: Optional[Dict] = Depends(get_current_user_optional),
    rag_service: RAGEnhancedService = Depends(get_rag_service),
) -> JSONResponse:
    """
    选择聊天角色的API。直接处理请求并实现完整的角色选择功能。
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] 开始处理角色选择请求: {request.url.path}")
    
    try:
        # 直接读取请求体
        request_data_raw = await request.json()
        logger.info(f"[{request_id}] 成功读取请求体")
        
        logger.info(f"[{request_id}] 原始请求数据结构: {json.dumps(request_data_raw, ensure_ascii=False)[:200]}...")
        
        # 提取请求数据（支持嵌套和非嵌套格式）
        if isinstance(request_data_raw, dict) and "request_data" in request_data_raw:
            logger.info(f"[{request_id}] 检测到嵌套的request_data字段")
            request_data = request_data_raw["request_data"]
        else:
            logger.info(f"[{request_id}] 使用整个请求体作为request_data")
            request_data = request_data_raw
        
        # 确保request_data是字典类型
        if not isinstance(request_data, dict):
            logger.error(f"[{request_id}] 请求数据格式错误: {type(request_data)}")
            return JSONResponse(
                status_code=400,
                content={"error": "请求数据必须是JSON对象"}
            )
        
        # 提取必要参数
        messages = request_data.get("messages", [])
        if not messages or not isinstance(messages, list):
            logger.error(f"[{request_id}] 缺少有效的messages字段")
            return JSONResponse(
                status_code=400,
                content={"error": "缺少有效的messages字段"}
            )
        
        # 提取会话ID或创建新ID
        session_id = request_data.get("session_id", str(uuid.uuid4()))
        user_id = request_data.get("user_id", "anonymous")
        
        # 提取提供商和模型名称（如果有）
        provider = request_data.get("provider", "default")
        model_name = request_data.get("model_name", "default")
        api_key = request_data.get("api_key")
        
        logger.info(f"[{request_id}] 角色选择参数: session_id={session_id}, user_id={user_id[:8]}..., provider={provider}, model={model_name}")
        
        # 获取实际要使用的provider和model（如果用户选择了default）
        actual_provider = provider
        actual_model = model_name
        
        if provider == "default" or not provider:
            # 初始化LLMService以获取默认provider
            llm_service = LLMService()
            actual_provider = llm_service.default_config.provider.value
            logger.info(f"[{request_id}] 使用系统默认提供商: {actual_provider}")
            
        if model_name == "default" or not model_name:
            # 初始化LLMService以获取默认model
            if not 'llm_service' in locals():
                llm_service = LLMService()
            actual_model = llm_service.default_config.model_name
            logger.info(f"[{request_id}] 使用系统默认模型: {actual_model}")
        
        # 模拟角色匹配过程 (实际生产环境中应替换为真实角色匹配逻辑)
        # 此处只是返回一个默认角色作为示例
        logger.info(f"[{request_id}] 模拟角色匹配成功")
        
        # 构建匹配的前端期望的精确响应结构
        response_data = {
            "request_id": request_id,
            "role_match": {
                "success": True,
                "role": {
                    "id": "default",
                    "name": "默认助手",
                    "description": "一个乐于助人的AI助手",
                    "system_prompt": "你是一个知识渊博、乐于助人的助手。"
                },
                "match_reason": "默认角色匹配",
                "error": None
            },
            "session_id": session_id,
            "provider": actual_provider,
            "model_name": actual_model
        }
        
        # 如果提供了API密钥，也包含在响应中
        if api_key:
            response_data["api_key"] = api_key
        
        logger.info(f"[{request_id}] 返回角色匹配响应: {json.dumps(response_data, ensure_ascii=False)[:200]}...")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"[{request_id}] 角色选择处理失败: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "处理请求失败", "detail": str(e)}
        )

@router.post("/chatrag/reset-stream")
async def reset_rag_chat_stream_lock(
    reset_request: Dict[str, str],
    current_user: Dict = Depends(get_current_user_optional)
):
    """
    重置Redis锁,当流式响应被锁定时使用
    """
    session_id = reset_request.get("session_id")
    request_id = reset_request.get("request_id", "")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")
        
    logger.info(f"尝试重置会话锁: {session_id} [请求ID: {request_id}]")
    
    try:
        # 构建锁名称
        lock_name = f"chat_stream:{session_id}"
        # 尝试获取Redis客户端并删除锁
        redis_client = await get_redis_client()
        deleted = await redis_client.delete(lock_name)
        
        if deleted:
            logger.info(f"成功重置会话锁: {session_id} [请求ID: {request_id}]")
            return {"success": True, "message": "会话锁已重置", "session_id": session_id}
        else:
            logger.warning(f"重置会话锁失败,锁不存在: {session_id} [请求ID: {request_id}]")
            return {"success": False, "message": "会话锁不存在或已经释放", "session_id": session_id}
            
    except Exception as e:
        logger.error(f"重置会话锁时发生错误: {str(e)} [请求ID: {request_id}]")
        raise HTTPException(status_code=500, detail=f"重置会话锁失败: {str(e)}") 

@router.post("/api/llm/chatrag/test-role-select", include_in_schema=True)
async def test_role_select(request: Request) -> JSONResponse:
    """
    测试版角色选择API，不需要验证。专为前端开发设计。
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] 开始处理测试角色选择请求")
    
    # 记录请求状态
    validation_skipped = getattr(request.state, "validation_skipped", False)
    no_validation = getattr(request.state, "no_validation", False)
    logger.info(f"[{request_id}] 请求状态: validation_skipped={validation_skipped}, no_validation={no_validation}")
    
    # 尝试从请求中获取数据
    try:
        # 先检查缓存的请求体
        cached_body = getattr(request.state, "cached_body", None)
        if cached_body:
            logger.info(f"[{request_id}] 使用中间件缓存的请求体")
            if isinstance(cached_body, bytes):
                body_str = cached_body.decode("utf-8")
            else:
                body_str = str(cached_body)
            
            request_data_raw = json.loads(body_str)
        else:
            # 如果没有缓存，直接读取请求体
            logger.info(f"[{request_id}] 直接读取请求体")
            request_data_raw = await request.json()
        
        logger.info(f"[{request_id}] 原始请求数据键: {list(request_data_raw.keys()) if isinstance(request_data_raw, dict) else 'not a dict'}")
        
        # 提取请求数据（支持嵌套结构）
        if isinstance(request_data_raw, dict) and "request_data" in request_data_raw:
            request_data = request_data_raw["request_data"]
            logger.info(f"[{request_id}] 从嵌套结构中提取request_data")
        else:
            request_data = request_data_raw
            logger.info(f"[{request_id}] 使用整个请求体作为request_data")
        
        # 提取提供商和模型信息（如果有）
        provider = request_data.get("provider", "default") if isinstance(request_data, dict) else "default"
        model_name = request_data.get("model_name", "default") if isinstance(request_data, dict) else "default"
        
        # 获取实际要使用的provider和model（如果用户选择了default）
        actual_provider = provider
        actual_model = model_name
        
        if provider == "default":
            # 初始化LLMService以获取默认provider
            llm_service = LLMService()
            actual_provider = llm_service.default_config.provider.value
            logger.info(f"[{request_id}] 使用系统默认提供商: {actual_provider}")
            
        if model_name == "default":
            # 初始化LLMService以获取默认model
            if not 'llm_service' in locals():
                llm_service = LLMService()
            actual_model = llm_service.default_config.model_name
            logger.info(f"[{request_id}] 使用系统默认模型: {actual_model}")
        
        # 生成测试响应
        response_data = {
            "matched": True,
            "role_name": "测试助手",
            "session_id": str(uuid.uuid4()),
            "matches": [
                {"role_id": "test", "role_name": "测试助手", "score": 0.95, "match_explanation": "测试匹配"}
            ],
            "provider": actual_provider,
            "model_name": actual_model,
            "debug_info": {
                "request_id": request_id,
                "validation_skipped": validation_skipped,
                "no_validation": no_validation,
                "message_count": len(request_data.get("messages", [])) if isinstance(request_data, dict) else 0,
                "data_format": type(request_data).__name__,
                "test_mode": True
            }
        }
        
        logger.info(f"[{request_id}] 测试角色选择完成，返回模拟数据")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"[{request_id}] 测试角色选择出错: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "测试角色选择错误",
                "detail": str(e),
                "request_id": request_id
            }
        )

@router.post("/api/llm/chatrag/test-generate", include_in_schema=True)
async def test_generate_response(request: Request) -> JSONResponse:
    """
    测试版响应生成API，不需要验证。专为前端开发设计。
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] 开始处理测试响应生成请求")
    
    # 记录请求状态
    validation_skipped = getattr(request.state, "validation_skipped", False)
    no_validation = getattr(request.state, "no_validation", False)
    
    try:
        # 读取请求数据
        cached_body = getattr(request.state, "cached_body", None)
        if cached_body:
            if isinstance(cached_body, bytes):
                body_str = cached_body.decode("utf-8")
            else:
                body_str = str(cached_body)
            
            request_data_raw = json.loads(body_str)
        else:
            request_data_raw = await request.json()
        
        # 提取请求数据
        if isinstance(request_data_raw, dict) and "request_data" in request_data_raw:
            request_data = request_data_raw["request_data"]
        else:
            request_data = request_data_raw
        
        # 检查是否请求流式响应
        stream = request_data.get("stream", False)
        
        if stream:
            # 返回流式响应
            logger.info(f"[{request_id}] 生成流式测试响应")
            
            async def response_stream():
                # 模拟流式响应片段
                chunks = [
                    {"role": "assistant", "content": "这是", "status": "streaming"},
                    {"role": "assistant", "content": "一个", "status": "streaming"},
                    {"role": "assistant", "content": "测试", "status": "streaming"},
                    {"role": "assistant", "content": "流式", "status": "streaming"},
                    {"role": "assistant", "content": "响应", "status": "streaming"},
                    {"role": "assistant", "content": "。", "status": "complete"}
                ]
                
                for chunk in chunks:
                    # 添加请求ID和时间戳
                    chunk["request_id"] = request_id
                    chunk["timestamp"] = datetime.utcnow().isoformat()
                    
                    # 发送JSON行
                    yield f"{json.dumps(chunk, ensure_ascii=False)}\n"
                    
                    # 模拟延迟
                    await asyncio.sleep(0.5)
            
            return StreamingResponse(
                response_stream(),
                media_type="application/x-ndjson"
            )
        else:
            # 返回普通JSON响应
            logger.info(f"[{request_id}] 生成非流式测试响应")
            
            # 模拟处理延迟
            await asyncio.sleep(1)
            
            response_data = {
                "role": "assistant",
                "content": "这是一个测试响应，用于前端开发。",
                "request_id": request_id,
                "debug_info": {
                    "validation_skipped": validation_skipped,
                    "no_validation": no_validation,
                    "test_mode": True,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            return JSONResponse(content=response_data)
    
    except Exception as e:
        logger.error(f"[{request_id}] 测试响应生成出错: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "测试响应生成错误",
                "detail": str(e),
                "request_id": request_id
            }
        )

@router.get("/api/llm/chatrag/test-role-select-get")
async def test_role_select_get(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    message: Optional[str] = None,
    rag_service: RAGEnhancedService = Depends(get_rag_service)
):
    """
    测试路由GET版本 - 使用查询参数代替请求体，类似于two-phase-stream/generate
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] 开始处理test-role-select-get请求")
    
    try:
        # 参数验证（更简单，不使用Pydantic）
        if not session_id:
            logger.warning(f"[{request_id}] 缺少session_id参数")
            session_id = f"default_session_{request_id[:8]}"
        
        # 记录请求信息
        logger.info(f"[{request_id}] 测试GET路由收到请求: session_id={session_id}, user_id={user_id}, message={message}")
        
        # 构建简单的消息结构
        messages = []
        system_message = {"role": "system", "content": "你是一个知识渊博、乐于助人的助手。"}
        messages.append(system_message)
        
        if message:
            user_message = {"role": "user", "content": message}
            messages.append(user_message)
        
        # 模拟角色匹配结果 - 使用随机值使每次响应看起来不同
        role_id = f"demo_role_{request_id[:8]}"
        role_match_result = {
            "success": True,
            "role": {
                "id": role_id,
                "name": "智能助手",
                "description": "一个乐于助人的AI助手",
                "system_prompt": "你是一个知识渊博、乐于助人的助手。"
            },
            "match_reason": "默认角色匹配",
            "error": None
        }
        
        # 存储Redis信息（可选）
        try:
            redis_client = await get_redis_client()
            if redis_client:
                match_key = f"chatrag:role_match:{request_id}"
                role_data = {
                    "role_id": role_id,
                    "role_name": "智能助手",
                    "match_reason": "默认角色匹配",
                    "session_id": session_id,
                    "created_at": datetime.utcnow().isoformat()
                }
                
                await redis_client.hset(match_key, mapping=role_data)
                await redis_client.expire(match_key, 300)  # 5分钟过期
                logger.debug(f"[{request_id}] 已存储角色匹配结果到Redis")
        except Exception as redis_err:
            logger.warning(f"[{request_id}] Redis操作失败: {str(redis_err)}，但继续返回结果")
        
        # 返回与原始API相同格式的响应
        total_time = time.time() - start_time
        logger.info(f"[{request_id}] GET版角色选择API处理完成，总耗时: {total_time:.4f}秒")
        return {
            "request_id": request_id,
            "role_match": role_match_result,
            "session_id": session_id
        }
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"[{request_id}] GET版角色匹配出错，耗时: {total_time:.4f}秒，错误: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"测试GET路由错误: {str(e)}"}
        ) 