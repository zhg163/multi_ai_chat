from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from fastapi import APIRouter, Depends, HTTPException, status, Path, Query
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database.mongodb import get_database
from app.models.message import (
    UserMessageCreate, 
    AssistantMessageCreate, 
    MessageResponse, 
    MessageHistoryResponse,
    MessageUpdate
)
from app.services.message_service import MessageService
from app.services.auth_service import AuthService, get_current_user
from app.services.session_service import SessionService

router = APIRouter(
    prefix="/api/messages",
    tags=["messages"],
    responses={404: {"description": "Not found"}},
)

@router.post("/", response_model=MessageResponse)
async def send_user_message(
    message: UserMessageCreate,
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    发送用户消息
    """
    try:
        message_service = MessageService(db)
        result = await message_service.send_user_message(
            user_id=current_user["id"],
            session_id=message.session_id,
            content=message.content,
            metadata=message.metadata,
            parent_id=message.parent_id
        )
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send message: {str(e)}"
        )

@router.post("/assistant", response_model=MessageResponse)
async def create_assistant_message(
    message: AssistantMessageCreate,
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    创建助手消息（内部API或管理员使用）
    """
    try:
        # 先验证用户是否有权限访问这个会话
        session_service = SessionService(db)
        session = await session_service.get_session_by_id(
            session_id=message.session_id,
            user_id=current_user["id"]
        )
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this session"
            )
            
        message_service = MessageService(db)
        result = await message_service.create_assistant_message(
            session_id=message.session_id,
            role_id=message.role_id,
            content=message.content,
            parent_id=message.parent_id,
            metadata=message.metadata
        )
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create assistant message: {str(e)}"
        )

@router.get("/{message_id}", response_model=MessageResponse)
async def get_message(
    message_id: str = Path(..., description="The ID of the message to get"),
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    获取消息详情
    """
    try:
        message_service = MessageService(db)
        result = await message_service.get_message_by_id(
            message_id=message_id,
            user_id=current_user["id"]
        )
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get message: {str(e)}"
        )

@router.get("/session/{session_id}", response_model=MessageHistoryResponse)
async def get_session_messages(
    session_id: str = Path(..., description="The ID of the session"),
    limit: int = Query(50, ge=1, le=100, description="Number of messages to return"),
    offset: int = Query(0, ge=0, description="Number of messages to skip"),
    sort: str = Query("asc", description="Sort order: 'asc' for oldest first, 'desc' for newest first"),
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    获取会话消息历史
    """
    try:
        sort_direction = 1 if sort.lower() == "asc" else -1
        
        message_service = MessageService(db)
        result = await message_service.get_message_history(
            session_id=session_id,
            user_id=current_user["id"],
            limit=limit,
            offset=offset,
            sort_direction=sort_direction
        )
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get message history: {str(e)}"
        )

@router.get("/session/{session_id}/anonymous", response_model=MessageHistoryResponse)
async def get_session_messages_anonymous(
    session_id: str = Path(..., description="The ID of the session"),
    limit: int = Query(50, ge=1, le=100, description="Number of messages to return"),
    offset: int = Query(0, ge=0, description="Number of messages to skip"),
    sort: str = Query("asc", description="Sort order: 'asc' for oldest first, 'desc' for newest first"),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    获取匿名会话消息历史（不需要用户认证）
    """
    try:
        sort_direction = 1 if sort.lower() == "asc" else -1
        
        message_service = MessageService(db)
        # 使用匿名用户ID
        result = await message_service.get_message_history(
            session_id=session_id,
            user_id="anonymous_user",  # 匿名用户ID
            limit=limit,
            offset=offset,
            sort_direction=sort_direction
        )
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get message history: {str(e)}"
        )

@router.delete("/session/{session_id}")
async def delete_session_messages(
    session_id: str = Path(..., description="The ID of the session"),
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    删除会话的所有消息
    """
    try:
        message_service = MessageService(db)
        deleted_count = await message_service.delete_session_messages(
            session_id=session_id,
            user_id=current_user["id"]
        )
        return {"deleted_count": deleted_count}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete messages: {str(e)}"
        )

@router.patch("/{message_id}/status", response_model=Dict[str, bool])
async def update_message_status(
    status_update: Dict[str, str],
    message_id: str = Path(..., description="The ID of the message to update"),
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    更新消息状态
    """
    try:
        # 验证状态值
        valid_statuses = ["sending", "sent", "delivered", "read", "error", "deleted"]
        if status_update.get("status") not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
            )
            
        message_service = MessageService(db)
        success = await message_service.update_message_status(
            message_id=message_id,
            status=status_update["status"],
            error_message=status_update.get("error_message")
        )
        
        if not success:
            return {"success": False}
            
        return {"success": True}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update message status: {str(e)}"
        ) 