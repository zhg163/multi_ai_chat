"""
Two-Phase Streaming API Routes - Implements the streaming two-phase chatbot API

Phase 1: Generate a streaming response based on user input and select an appropriate role
Phase 2: Process feedback on the generated response and stream an improved response
"""

from fastapi import APIRouter, HTTPException, Body, Request, Response
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional, List, AsyncGenerator
import logging
import uuid
import json
import time
import asyncio
import traceback
from datetime import datetime

from app.services.redis_manager import get_redis_client
from app.services.custom_session_service import CustomSessionService
from app.services.llm_service_two_phase_stream import TwoPhaseStreamService
from app.utils.redis_lock import RedisLock, obtain_lock
from app.memory.memory_manager import get_memory_manager
from app.services.llm_service import LLMService, Message, MessageRole, LLMConfig

router = APIRouter(
    prefix="/api/two-phase-stream",
    tags=["two-phase-stream"]
)

logger = logging.getLogger(__name__)

async def streaming_response_generator(
    session_id: str,
    message: str,
    message_id: str,
    selected_role: Dict[str, Any],
    match_score: float,
    match_reason: str
) -> AsyncGenerator[str, None]:
    """
    生成流式响应的协程

    Args:
        session_id: 会话ID
        message: 用户消息
        message_id: 消息ID
        selected_role: 选中的角色
        match_score: 匹配分数
        match_reason: 匹配原因

    Yields:
        SSE格式的数据流
    """
    # 获取Redis客户端
    redis_client = await get_redis_client()
    
    # 发送初始匹配信息
    initial_data = {
        "type": "match_info",
        "message_id": message_id,
        "selected_role": selected_role.get("role_name", "Unknown"),
        "match_score": match_score,
        "match_reason": match_reason
    }
    yield f"data: {json.dumps(initial_data)}\n\n"
    
    # 缓冲区，用于存储完整响应以便后续改进
    response_buffer = ""
    
    # 流式生成响应
    service = TwoPhaseStreamService()
    try:
        async for chunk in service.generate_stream_response(message, selected_role):
            if chunk.error:
                error_data = {
                    "type": "error",
                    "message": chunk.error
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                break
                
            if chunk.content:
                # 将响应片段添加到缓冲区
                response_buffer += chunk.content
                
                # 发送响应片段
                chunk_data = {
                    "type": "content",
                    "content": chunk.content
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                
                # 每积累一定长度就更新Redis中的响应
                if len(response_buffer) % 100 == 0:
                    await redis_client.hset(
                        f"two_phase:request:{message_id}",
                        "partial_response", 
                        response_buffer
                    )
                
        # 发送完成信号
        complete_data = {
            "type": "complete",
            "message_id": message_id
        }
        yield f"data: {json.dumps(complete_data)}\n\n"
        
        # 存储完整响应到Redis
        await redis_client.hset(
            f"two_phase:request:{message_id}",
            mapping={
                "response": response_buffer,
                "selected_role": json.dumps(selected_role),
                "match_score": str(match_score),
                "match_reason": match_reason,
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.exception(f"流式生成响应时出错: {e}")
        error_data = {
            "type": "error",
            "message": str(e)
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        
        # 存储错误信息到Redis
        await redis_client.hset(
            f"two_phase:request:{message_id}",
            mapping={
                "status": "error",
                "error": str(e),
                "error_at": datetime.utcnow().isoformat()
            }
        )

async def streaming_feedback_generator(
    session_id: str,
    message_id: str,
    original_message: str,
    original_response: str,
    selected_role: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    """
    生成改进流式响应的协程

    Args:
        session_id: 会话ID
        message_id: 原始消息ID
        original_message: 原始用户消息
        original_response: 原始响应
        selected_role: 选中的角色

    Yields:
        SSE格式的数据流
    """
    # 获取Redis客户端
    redis_client = await get_redis_client()
    
    # 发送初始信息
    initial_data = {
        "type": "feedback_info",
        "message_id": message_id,
        "status": "improving"
    }
    yield f"data: {json.dumps(initial_data)}\n\n"
    
    # 缓冲区，用于存储完整改进响应
    improved_buffer = ""
    
    # 流式生成改进响应
    service = TwoPhaseStreamService()
    try:
        async for chunk in service.improve_stream_response(
            original_message, 
            original_response, 
            selected_role
        ):
            if chunk.error:
                error_data = {
                    "type": "error",
                    "message": chunk.error
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                break
                
            if chunk.content:
                # 将响应片段添加到缓冲区
                improved_buffer += chunk.content
                
                # 发送响应片段
                chunk_data = {
                    "type": "content",
                    "content": chunk.content
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                
                # 每积累一定长度就更新Redis中的响应
                if len(improved_buffer) % 100 == 0:
                    await redis_client.hset(
                        f"two_phase:request:{message_id}",
                        "partial_improved_response", 
                        improved_buffer
                    )
                
        # 发送完成信号
        complete_data = {
            "type": "complete",
            "message_id": message_id
        }
        yield f"data: {json.dumps(complete_data)}\n\n"
        
        # 存储完整改进响应到Redis
        await redis_client.hset(
            f"two_phase:request:{message_id}",
            mapping={
                "improved_response": improved_buffer,
                "feedback_completed_at": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.exception(f"流式生成改进响应时出错: {e}")
        error_data = {
            "type": "error",
            "message": str(e)
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        
        # 存储错误信息到Redis
        await redis_client.hset(
            f"two_phase:request:{message_id}",
            mapping={
                "feedback_status": "error",
                "feedback_error": str(e),
                "feedback_error_at": datetime.utcnow().isoformat()
            }
        )

@router.get("/generate")
async def generate_stream_response(
    session_id: str,
    message: str,
    user_id: str = None  # 添加用户ID参数
):
    """
    First phase streaming API: Generates a response stream for the given message
    
    Parameters:
        session_id: Session identifier (query param)
        message: User's message (query param)
        user_id: User identifier (query param, optional)
    """
    # 设置默认用户ID
    if not user_id:
        user_id = "anonymous_user"
    
    try:
        # 1. 生成唯一的消息ID
        message_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # 2. 获取记忆管理器实例
        memory_manager = await get_memory_manager()
        
        # 3. 将用户消息添加到短期记忆
        await memory_manager.add_message(
            session_id=session_id,
            user_id=user_id,
            role="user",
            content=message,
            message_id=f"user-{message_id}"
        )
        
        # 4. 获取Redis客户端和会话
        redis_client = await get_redis_client()
        session = await CustomSessionService.get_session(session_id)
        
        if not session or "roles" not in session:
            raise HTTPException(status_code=404, detail="Session not found or has no roles")
        
        roles = session.get("roles", [])
        if not roles:
            raise HTTPException(status_code=400, detail="No roles available in session")
        
        # 5. 获取Redis锁防止并发操作
        lock_name = f"two_phase_stream:lock:{session_id}"
        lock = await obtain_lock(redis_client, lock_name, expire_seconds=60)
        
        if lock is None:
            raise HTTPException(
                status_code=409, 
                detail="Session is busy. Please try again in a moment."
            )
        
        try:
            # 6. 存储请求信息到Redis
            request_key = f"two_phase:request:{message_id}"
            await redis_client.hset(request_key, mapping={
                "session_id": session_id,
                "user_id": user_id,
                "message": message,
                "status": "pending",
                "created_at": timestamp
            })
            await redis_client.expire(request_key, 3600)  # 1 hour expiry
            
            # 7. 选择角色
            service = TwoPhaseStreamService()
            selected_role, match_score, match_reason = await service.select_role(
                message=message,
                roles=roles
            )
            
            # 8. 返回流式响应
            return StreamingResponse(
                streaming_response_generator_with_memory(
                    session_id=session_id,
                    user_id=user_id,
                    message=message,
                    message_id=message_id,
                    selected_role=selected_role,
                    match_score=match_score,
                    match_reason=match_reason,
                    memory_manager=memory_manager
                ),
                media_type="text/event-stream"
            )
            
        finally:
            # 释放锁
            await lock.release()
            
    except HTTPException:
        raise
    except Exception as e:
        error_detail = f"Error generating response: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

@router.get("/feedback")
async def provide_feedback_stream(
    session_id: str,
    message_id: str,
    is_accepted: bool = False,
    user_id: str = None
):
    """
    Second phase streaming API: Processes feedback on the generated response
    
    Parameters:
        session_id: Session identifier (query param)
        message_id: Message identifier from the first phase (query param)
        is_accepted: Whether the response was accepted (query param)
        user_id: User identifier (query param, optional)
    """
    # 设置默认用户ID
    if not user_id:
        user_id = "anonymous_user"
        
    try:
        # 获取记忆管理器实例
        from app.memory.memory_manager import get_memory_manager
        memory_manager = await get_memory_manager()
        
        # 1. Get the Redis client
        redis_client = await get_redis_client()
        
        # 2. Get the stored request/response
        request_key = f"two_phase:request:{message_id}"
        
        stored_data = await redis_client.hgetall(request_key)
        if not stored_data:
            return Response(
                content=json.dumps({"error": f"Request not found: {message_id}"}),
                media_type="application/json",
                status_code=404
            )
        
        # 3. Verify this message belongs to the specified session
        if stored_data.get("session_id") != session_id:
            return Response(
                content=json.dumps({"error": "Session ID mismatch"}),
                media_type="application/json",
                status_code=403
            )
        
        # 4. Acquire a lock for this message to prevent concurrent processing
        lock_name = f"two_phase:lock:{message_id}"
        lock = await obtain_lock(redis_client, lock_name, expire_seconds=30)
        
        if lock is None:
            return Response(
                content=json.dumps({"error": "Message is being processed. Please try again in a moment."}),
                media_type="application/json",
                status_code=409
            )
        
        try:
            # 5. Update the feedback status
            await redis_client.hset(request_key, mapping={
                "feedback": "accepted" if is_accepted else "rejected",
                "feedback_time": datetime.utcnow().isoformat()
            })
            
            # 6. If accepted, just return a success response
            if is_accepted:
                return StreamingResponse(
                    accepted_response_generator(message_id),
                    media_type="text/event-stream"
                )
            
            # 7. If rejected, generate an improved response
            # Get the selected role data
            selected_role_json = stored_data.get("selected_role", "{}")
            selected_role = json.loads(selected_role_json)
            
            # Get the original message and response
            original_message = stored_data.get("message", "")
            original_response = stored_data.get("response", "")
            
            # Return a streaming response with the improved content
            return StreamingResponse(
                feedback_response_generator_with_memory(
                    session_id=session_id,
                    user_id=user_id,
                    message_id=message_id,
                    original_message=original_message,
                    original_response=original_response,
                    selected_role=selected_role,
                    memory_manager=memory_manager
                ),
                media_type="text/event-stream"
            )
            
        finally:
            # Always release the lock
            await lock.release()
            
    except Exception as e:
        error_detail = f"Error processing feedback: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        return Response(
            content=json.dumps({"error": error_detail}),
            media_type="application/json",
            status_code=500
        )

@router.get("/debug/check")
async def debug_check():
    """
    Debug endpoint to check if the two-phase streaming API is functioning
    
    Returns:
        status: Always "ok"
        timestamp: Current time
    """
    return {
        "status": "ok",
        "api": "two-phase-stream",
        "timestamp": datetime.utcnow().isoformat()
    }

async def streaming_response_generator_with_memory(
    session_id: str,
    user_id: str,
    message: str,
    message_id: str,
    selected_role: Dict[str, Any],
    match_score: float,
    match_reason: str,
    memory_manager
) -> AsyncGenerator[str, None]:
    """
    生成带有短期记忆的流式响应
    
    Args:
        session_id: 会话ID
        user_id: 用户ID
        message: 用户消息
        message_id: 消息ID
        selected_role: 选中的角色
        match_score: 匹配分数
        match_reason: 匹配原因
        memory_manager: 记忆管理器实例
        
    Yields:
        SSE格式的数据流
    """
    # 获取Redis客户端
    redis_client = await get_redis_client()
    
    # 发送初始匹配信息
    initial_data = {
        "type": "match_info",
        "message_id": message_id,
        "selected_role": selected_role.get("role_name", "Unknown"),
        "match_score": match_score,
        "match_reason": match_reason
    }
    yield f"data: {json.dumps(initial_data)}\n\n"
    
    # 缓冲区，用于存储完整响应以便后续改进
    response_buffer = ""
    
    # 流式生成响应
    llm_service = LLMService()
    try:
        # 1. 获取历史消息
        history_messages = await memory_manager.short_term_memory.get_session_messages(
            session_id=session_id, 
            user_id=user_id
        )
        
        # 2. 获取角色信息
        role_name = selected_role.get("role_name", "Unknown")
        system_prompt = selected_role.get("system_prompt", "")
        
        if not system_prompt:
            logger.warning(f"角色 {role_name} 没有system_prompt")
            system_prompt = f"你是{role_name}，请以自然的方式回复用户。"
        
        # 3. 准备所有消息，包括系统提示和历史消息
        messages = []
        messages.append(Message(role=MessageRole.SYSTEM, content=system_prompt))
        
        # 添加历史消息(限制最近10条，防止上下文过长)
        recent_messages = history_messages[-10:] if len(history_messages) > 10 else history_messages
        for msg in recent_messages:
            role_type = MessageRole.USER if msg.get("role") == "user" else MessageRole.ASSISTANT
            # 跳过系统消息
            if msg.get("role") == "system":
                continue
            messages.append(Message(role=role_type, content=msg.get("content", "")))
        
        # 4. 开始流式生成
        logger.info(f"开始为角色 {role_name} 生成带有记忆的流式回复，历史消息数：{len(recent_messages)}")
        
        temperature = float(selected_role.get("temperature", 0.7))
        async for chunk in llm_service.generate_stream(
            messages=messages,
            config=LLMConfig(
                provider=llm_service.default_config.provider,
                model_name=llm_service.default_config.model_name,
                api_key=llm_service.default_config.api_key,
                temperature=temperature
            )
        ):
            if chunk.error:
                error_data = {
                    "type": "error",
                    "message": chunk.error
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                break
                
            if chunk.content:
                # 将响应片段添加到缓冲区
                response_buffer += chunk.content
                
                # 发送响应片段
                chunk_data = {
                    "type": "content",
                    "content": chunk.content
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                
                # 每积累一定长度就更新Redis中的响应
                if len(response_buffer) % 100 == 0:
                    await redis_client.hset(
                        f"two_phase:request:{message_id}",
                        "partial_response", 
                        response_buffer
                    )
                
        # 5. 发送完成信号
        complete_data = {
            "type": "complete",
            "message_id": message_id
        }
        yield f"data: {json.dumps(complete_data)}\n\n"
        
        # 6. 存储完整响应到Redis
        await redis_client.hset(
            f"two_phase:request:{message_id}",
            mapping={
                "response": response_buffer,
                "selected_role": json.dumps(selected_role),
                "match_score": str(match_score),
                "match_reason": match_reason,
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat()
            }
        )
        
        # 7. 将AI回复添加到短期记忆
        await memory_manager.add_message(
            session_id=session_id,
            user_id=user_id,
            role="assistant",
            content=response_buffer,
            role_id=selected_role.get("role_id"),
            message_id=f"assistant-{message_id}"
        )
        
        logger.info(f"已将AI回复添加到短期记忆, 会话ID: {session_id}, 回复长度: {len(response_buffer)}")
        
    except Exception as e:
        logger.exception(f"流式生成响应时出错: {e}")
        error_data = {
            "type": "error",
            "message": str(e)
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        
        # 存储错误信息到Redis
        await redis_client.hset(
            f"two_phase:request:{message_id}",
            mapping={
                "status": "error",
                "error": str(e),
                "error_at": datetime.utcnow().isoformat()
            }
        ) 

async def feedback_response_generator_with_memory(
    session_id: str,
    user_id: str,
    message_id: str,
    original_message: str,
    original_response: str,
    selected_role: Dict[str, Any],
    memory_manager
) -> AsyncGenerator[str, None]:
    """
    生成带有短期记忆的改进流式响应
    """
    # 获取Redis客户端
    redis_client = await get_redis_client()
    
    # 发送初始信息
    initial_data = {
        "type": "feedback_info",
        "message_id": message_id,
        "status": "improving"
    }
    yield f"data: {json.dumps(initial_data)}\n\n"
    
    # 缓冲区，用于存储完整改进响应
    improved_buffer = ""
    
    # 流式生成改进响应
    llm_service = LLMService()
    try:
        # 1. 获取历史消息
        history_messages = await memory_manager.short_term_memory.get_session_messages(
            session_id=session_id, 
            user_id=user_id
        )
        
        # 2. 获取角色信息
        role_name = selected_role.get("role_name", "Unknown")
        system_prompt = selected_role.get("system_prompt", "")
        
        if not system_prompt:
            system_prompt = f"你是{role_name}，请以自然的方式回复用户。"
        
        # 3. 构建改进提示
        improvement_prompt = f"""你之前的回复被用户拒绝，需要改进。

原始用户消息: "{original_message}"

你之前的回复: "{original_response}"

请根据角色设定提供一个更好的回复。特别注意:
1. 确保回复与角色人设一致
2. 提供更有帮助、更详细的信息
3. 请考虑历史对话中的所有相关信息
4. 避免重复之前回复中的问题

请直接给出改进后的回复，不要解释你做了什么改变。"""

        # 4. 准备消息，包括系统提示、历史消息和改进请求
        messages = []
        messages.append(Message(role=MessageRole.SYSTEM, content=system_prompt))
        
        # 添加历史消息(限制10条)
        filtered_history = []
        for msg in history_messages[-10:]:
            # 过滤掉要改进的回复
            if (msg.get("role") == "assistant" and msg.get("content") == original_response):
                continue
            filtered_history.append(msg)
        
        for msg in filtered_history:
            role_type = MessageRole.USER if msg.get("role") == "user" else MessageRole.ASSISTANT
            if msg.get("role") == "system":
                continue
            messages.append(Message(role=role_type, content=msg.get("content", "")))
        
        # 添加原始消息和改进提示
        messages.append(Message(role=MessageRole.USER, content=original_message))
        messages.append(Message(role=MessageRole.ASSISTANT, content=original_response))
        messages.append(Message(role=MessageRole.USER, content=improvement_prompt))
        
        # 5. 生成改进回复
        temperature = min(float(selected_role.get("temperature", 0.7)) + 0.1, 1.0)
        async for chunk in llm_service.generate_stream(
            messages=messages,
            config=LLMConfig(
                provider=llm_service.default_config.provider,
                model_name=llm_service.default_config.model_name,
                api_key=llm_service.default_config.api_key,
                temperature=temperature
            )
        ):
            if chunk.error:
                error_data = {
                    "type": "error",
                    "message": chunk.error
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                break
                
            if chunk.content:
                # 将响应片段添加到缓冲区
                improved_buffer += chunk.content
                
                # 发送响应片段
                chunk_data = {
                    "type": "content",
                    "content": chunk.content
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                
                # 每积累一定长度就更新Redis中的响应
                if len(improved_buffer) % 100 == 0:
                    await redis_client.hset(
                        f"two_phase:request:{message_id}",
                        "partial_improved_response", 
                        improved_buffer
                    )
                
        # 6. 发送完成信号
        complete_data = {
            "type": "complete",
            "message_id": message_id
        }
        yield f"data: {json.dumps(complete_data)}\n\n"
        
        # 7. 存储完整改进响应到Redis
        await redis_client.hset(
            f"two_phase:request:{message_id}",
            mapping={
                "improved_response": improved_buffer,
                "feedback_completed_at": datetime.utcnow().isoformat()
            }
        )
        
        # 8. 将改进后的回复添加到短期记忆
        await memory_manager.add_message(
            session_id=session_id,
            user_id=user_id,
            role="assistant",
            content=improved_buffer,
            role_id=selected_role.get("role_id"),
            message_id=f"improved-{message_id}"
        )
        
        # 9. 添加系统消息标记这是改进后的回复
        await memory_manager.add_message(
            session_id=session_id,
            user_id=user_id,
            role="system",
            content="上面是改进后的回复",
            message_id=f"system-{message_id}"
        )
        
        logger.info(f"已将改进的AI回复添加到短期记忆, 会话ID: {session_id}")
        
    except Exception as e:
        logger.exception(f"流式生成改进响应时出错: {e}")
        error_data = {
            "type": "error",
            "message": str(e)
        }
        yield f"data: {json.dumps(error_data)}\n\n" 