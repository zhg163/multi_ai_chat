"""
Two-Phase API Routes - Implements the two-phase chatbot API

Phase 1: Generate a response based on user input and select an appropriate role
Phase 2: Process feedback on the generated response
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional, List
import logging
import uuid
import json
import time
import traceback
from datetime import datetime

from app.services.redis_manager import get_redis_client
from app.services.custom_session_service import CustomSessionService
from app.services.llm_service_two_phase import TwoPhaseService
from app.utils.redis_lock import RedisLock, obtain_lock

router = APIRouter(
    prefix="/api/two-phase",
    tags=["two-phase"]
)

logger = logging.getLogger(__name__)

@router.post("/generate", response_model=Dict[str, Any])
async def generate_response(
    data: Dict[str, Any] = Body(...)
):
    """
    First phase API: Generates a response for the given message
    
    Parameters:
        session_id: Session identifier
        message: User's message content
    
    Returns:
        message_id: Unique identifier for this message
        response: Generated response
        selected_role: Name of the selected role that responded
    """
    session_id = data.get("session_id")
    message = data.get("message")
    
    if not session_id or not message:
        raise HTTPException(status_code=400, detail="Missing required parameters")
    
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
            raise HTTPException(
                status_code=409, 
                detail="Session is busy. Please try again in a moment."
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
                raise HTTPException(status_code=404, detail="Session not found or has no roles")
            
            roles = session.get("roles", [])
            if not roles:
                raise HTTPException(status_code=400, detail="No roles available in session")
            
            # 6. Select role using the two-phase service
            two_phase_service = TwoPhaseService()
            selected_role, match_score, match_reason = await two_phase_service.select_role(
                message=message,
                roles=roles
            )
            
            # 7. Generate response
            response = await two_phase_service.generate_response(
                message=message,
                role=selected_role
            )
            
            # 8. Store the response
            await redis_client.hset(request_key, mapping={
                "response": response,
                "selected_role": json.dumps(selected_role),
                "match_score": str(match_score),
                "match_reason": match_reason,
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat()
            })
            
            # 9. Add to session history
            history_key = f"two_phase:history:{session_id}"
            await redis_client.rpush(history_key, message_id)
            await redis_client.expire(history_key, 86400 * 7)  # 7 day expiry
            
            role_name = selected_role.get("role_name", "Unknown")
            
            return {
                "message_id": message_id,
                "response": response,
                "selected_role": role_name,
                "match_score": match_score,
                "match_reason": match_reason
            }
        finally:
            # Always release the lock
            await lock.release()
            
    except HTTPException:
        # Re-raise HTTP exceptions without wrapping
        raise
    except Exception as e:
        error_detail = f"Error generating response: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

@router.post("/feedback", response_model=Dict[str, Any])
async def provide_feedback(
    data: Dict[str, Any] = Body(...)
):
    """
    Second phase API: Provides feedback on a generated response
    
    Parameters:
        session_id: Session identifier
        message_id: Message identifier from the generate phase
        is_accepted: Whether the response was accepted or rejected
    
    Returns:
        success: Whether feedback was processed successfully
        is_accepted: The feedback value provided
        improved_response: If rejected, an improved response (when implemented)
    """
    session_id = data.get("session_id")
    message_id = data.get("message_id")
    is_accepted = data.get("is_accepted", False)
    
    if not session_id or not message_id:
        raise HTTPException(status_code=400, detail="Missing required parameters")
    
    try:
        # 1. Get the Redis client
        redis_client = await get_redis_client()
        
        # 2. Get the stored request/response
        request_key = f"two_phase:request:{message_id}"
        
        stored_data = await redis_client.hgetall(request_key)
        if not stored_data:
            raise HTTPException(status_code=404, detail=f"Request not found: {message_id}")
        
        # 3. Verify this message belongs to the specified session
        if stored_data.get("session_id") != session_id:
            raise HTTPException(status_code=403, detail="Session ID mismatch")
        
        # 4. Acquire a lock for this message to prevent concurrent processing
        lock_name = f"two_phase:lock:{message_id}"
        lock = await obtain_lock(redis_client, lock_name, expire_seconds=30)
        
        if lock is None:
            raise HTTPException(
                status_code=409, 
                detail="Message is being processed. Please try again in a moment."
            )
        
        try:
            # 5. Update the feedback status
            await redis_client.hset(request_key, mapping={
                "feedback": "accepted" if is_accepted else "rejected",
                "feedback_time": datetime.utcnow().isoformat()
            })
            
            improved_response = None
            
            # 6. If rejected, generate an improved response
            if not is_accepted:
                # Get the selected role data
                selected_role_json = stored_data.get("selected_role", "{}")
                selected_role = json.loads(selected_role_json)
                
                # Get the original message and response
                original_message = stored_data.get("message", "")
                original_response = stored_data.get("response", "")
                
                # Generate improved response using the two-phase service
                two_phase_service = TwoPhaseService()
                improved_response = await two_phase_service.improve_response(
                    original_message=original_message,
                    original_response=original_response,
                    role=selected_role
                )
                
                # Store the improved response
                await redis_client.hset(request_key, "improved_response", improved_response)
            
            return {
                "success": True,
                "is_accepted": is_accepted,
                "improved_response": improved_response
            }
        finally:
            # Always release the lock
            await lock.release()
            
    except HTTPException:
        # Re-raise HTTP exceptions without wrapping
        raise
    except Exception as e:
        error_detail = f"Error processing feedback: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

@router.get("/debug/check")
async def debug_check():
    """
    Debug endpoint to check if the two-phase API is functioning
    
    Returns:
        status: Always "ok"
        timestamp: Current time
    """
    # Check Redis connection
    redis_status = "unknown"
    try:
        redis_client = await get_redis_client()
        await redis_client.set("two_phase:debug:check", "ok", ex=60)
        test_value = await redis_client.get("two_phase:debug:check")
        redis_status = "ok" if test_value == "ok" else f"error: got {test_value}"
    except Exception as e:
        redis_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "redis_status": redis_status
    } 