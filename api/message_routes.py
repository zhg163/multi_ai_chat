from fastapi import APIRouter, Path, Query, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import List, Optional

from multi_ai_chat.app.services.message_service import MessageService
from multi_ai_chat.app.models.message_models import MessageHistoryResponse
from multi_ai_chat.app.utils.database import get_database

router = APIRouter()

@router.get("/{session_id}/anonymous", response_model=MessageHistoryResponse)
async def get_session_messages_anonymous(
    session_id: str = Path(..., description="会话ID"),
    limit: int = Query(20, description="每页消息数量"),
    offset: int = Query(0, description="分页偏移量"),
    sort: str = Query("desc", description="排序方式，asc或desc"),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    获取匿名会话消息历史
    """
    try:
        message_service = MessageService(db)
        result = await message_service.get_message_history(
            session_id=session_id,
            limit=limit,
            offset=offset,
            sort=sort,
            user_id=None  # 匿名用户无需用户ID
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取会话消息失败: {str(e)}"
        ) 