"""
自定义会话API路由 - 提供基于MD5会话ID生成的会话创建接口
"""

from fastapi import APIRouter, Depends, HTTPException, Body, Path, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

from app.services.custom_session_service import CustomSessionService
from app.auth.auth_bearer import JWTBearer
from app.auth.auth_handler import get_current_user, get_current_user_or_none

# 创建logger
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/custom-sessions",
    tags=["custom-sessions"]
)

# 数据模型
class RoleInfo(BaseModel):
    role_id: str = Field(..., description="角色ID")
    role_name: str = Field(..., description="角色名称")

class SessionCreateRequest(BaseModel):
    class_id: str = Field(..., description="聊天室ID")
    class_name: str = Field(..., description="聊天室名称")
    user_id: str = Field(..., description="用户ID")
    user_name: str = Field(..., description="用户名称")
    roles: List[RoleInfo] = Field(..., description="角色列表")

class SessionResponse(BaseModel):
    session_id: str = Field(..., description="会话ID")
    class_name: str = Field(..., description="聊天室名称")
    user_name: str = Field(..., description="用户名称")

class SessionStatusUpdateRequest(BaseModel):
    status: int = Field(..., description="会话状态 (0-未开始，1-进行中，2-已结束)")

# 路由定义
@router.post("", response_model=SessionResponse)
async def create_custom_session(
    request: SessionCreateRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_or_none)
):
    """
    创建自定义会话
    
    基于class_name、user_name和role_name组合生成MD5会话ID
    """
    try:
        # 记录请求
        logger.info(f"创建自定义会话请求: {request}")
        
        # 转换角色格式
        roles = [{"role_id": role.role_id, "role_name": role.role_name} for role in request.roles]
        
        # 验证请求用户和会话用户是否匹配
        # if current_user and current_user.get("id") != request.user_id:
        #     logger.warning(f"请求用户({current_user.get('id')})与会话用户({request.user_id})不匹配")
        
        # 创建会话
        session = await CustomSessionService.create_custom_session(
            class_id=request.class_id,
            class_name=request.class_name,
            user_id=request.user_id,
            user_name=request.user_name,
            roles=roles
        )
        
        return session
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"创建自定义会话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建会话失败: {str(e)}")

@router.put("/{session_id}/status", response_model=Dict[str, bool])
async def update_session_status(
    session_id: str = Path(..., description="会话ID"),
    request: SessionStatusUpdateRequest = Body(...),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_or_none)
):
    """
    更新会话状态
    
    参数:
        session_id: 会话ID
        status: 会话状态 (0-未开始，1-进行中，2-已结束)
    """
    try:
        # 验证会话是否存在
        session_exists = await CustomSessionService.check_session_exists(session_id)
        if not session_exists:
            raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")
        
        # 更新会话状态
        success = await CustomSessionService.update_session_status(
            session_id=session_id,
            status=request.status
        )
        
        return {"success": success}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"更新会话状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新会话状态失败: {str(e)}")

@router.get("/{session_id}", response_model=Dict[str, Any])
async def get_session(
    session_id: str = Path(..., description="会话ID"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_or_none)
):
    """
    获取会话信息
    
    参数:
        session_id: 会话ID
    """
    try:
        # 获取会话
        session = await CustomSessionService.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")
        
        # 转换ID为字符串
        if "_id" in session:
            session["_id"] = str(session["_id"])
        
        # 转换时间为ISO格式字符串
        if "created_at" in session:
            session["created_at"] = session["created_at"].isoformat()
        if "updated_at" in session:
            session["updated_at"] = session["updated_at"].isoformat()
        
        return session
    except Exception as e:
        logger.error(f"获取会话信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取会话信息失败: {str(e)}") 