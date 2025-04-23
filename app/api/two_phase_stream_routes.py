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
    message: str
):
    """
    First phase streaming API: Generates a response stream for the given message
    
    Parameters:
        session_id: Session identifier (query param)
        message: User's message content (query param)
    
    Returns:
        SSE stream with message_id, selected role info, and content chunks
    """
    if not session_id or not message:
        return Response(
            content=json.dumps({"error": "Missing required parameters"}),
            media_type="application/json",
            status_code=400
        )
    
    try:
        # 1. Generate a unique message ID
        message_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # 2. Get the Redis client
        redis_client = await get_redis_client()
        
        # 3. Acquire a lock for this session to prevent concurrent processing
        lock_name = f"two_phase:lock:{session_id}"
        lock = await obtain_lock(redis_client, lock_name, expire_seconds=60)
        
        if lock is None:
            return Response(
                content=json.dumps({"error": "Session is busy. Please try again in a moment."}),
                media_type="application/json",
                status_code=409
            )
        
        try:
            # 4. Store the message in Redis
            request_key = f"two_phase:request:{message_id}"
            
            await redis_client.hset(request_key, mapping={
                "session_id": session_id,
                "message": message,
                "status": "pending",
                "created_at": timestamp
            })
            await redis_client.expire(request_key, 3600)  # 1 hour expiry
            
            # 5. Get session info and select response role
            session = await CustomSessionService.get_session(session_id)
            
            if not session or "roles" not in session:
                return Response(
                    content=json.dumps({"error": "Session not found or has no roles"}),
                    media_type="application/json",
                    status_code=404
                )
            
            roles = session.get("roles", [])
            if not roles:
                return Response(
                    content=json.dumps({"error": "No roles available in session"}),
                    media_type="application/json",
                    status_code=400
                )
            
            # 6. Select role using the two-phase service
            stream_service = TwoPhaseStreamService()
            selected_role, match_score, match_reason = await stream_service.select_role(
                message=message,
                roles=roles
            )
            
            # 7. Add to session history
            history_key = f"two_phase:history:{session_id}"
            await redis_client.rpush(history_key, message_id)
            await redis_client.expire(history_key, 86400 * 7)  # 7 day expiry
            
            # 8. Create and return the streaming response
            return StreamingResponse(
                streaming_response_generator(
                    session_id=session_id,
                    message=message,
                    message_id=message_id,
                    selected_role=selected_role,
                    match_score=match_score,
                    match_reason=match_reason
                ),
                media_type="text/event-stream"
            )
            
        finally:
            # Always release the lock
            await lock.release()
            
    except Exception as e:
        error_detail = f"Error generating response: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        return Response(
            content=json.dumps({"error": error_detail}),
            media_type="application/json",
            status_code=500
        )

@router.get("/feedback")
async def provide_feedback_stream(
    session_id: str,
    message_id: str,
    is_accepted: bool = False
):
    """
    Second phase streaming API: Provides feedback on a generated response
    
    Parameters:
        session_id: Session identifier (query param)
        message_id: Message identifier from the generate phase (query param)
        is_accepted: Whether the response was accepted or rejected (query param)
    
    Returns:
        If accepted: Simple acknowledgement
        If rejected: SSE stream with improved response
    """
    if not session_id or not message_id:
        return Response(
            content=json.dumps({"error": "Missing required parameters"}),
            media_type="application/json",
            status_code=400
        )
    
    try:
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
                return Response(
                    content=json.dumps({
                        "success": True,
                        "is_accepted": True,
                        "message": "Feedback recorded successfully"
                    }),
                    media_type="application/json"
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
                streaming_feedback_generator(
                    session_id=session_id,
                    message_id=message_id,
                    original_message=original_message,
                    original_response=original_response,
                    selected_role=selected_role
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