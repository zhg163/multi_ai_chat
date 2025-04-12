from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import List, Optional
from pydantic import BaseModel, Field

from ..services.session_service import SessionService
from ..auth.auth_handler import get_current_user

router = APIRouter(
    prefix="/api/sessions",
    tags=["session-roles"],
    responses={404: {"description": "Not found"}},
)

# 模型定义
class SessionRoleResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    status: str
    keywords: Optional[List[str]] = None
    
class SessionRoleAdd(BaseModel):
    role_id: str = Field(..., description="要添加到会话的角色ID")

# API端点
@router.get("/{session_id}/roles", response_model=List[SessionRoleResponse])
async def get_session_roles(
    session_id: str,
    current_user = Depends(get_current_user)
):
    """
    获取会话的角色列表
    """
    try:
        user_id = str(current_user["id"])
        roles = await SessionService.get_session_roles(session_id, user_id)
        
        # 转换为响应模型格式
        response = []
        for role in roles:
            response.append({
                "id": str(role.get("_id")),
                "name": role.get("name"),
                "description": role.get("description"),
                "status": role.get("status"),
                "keywords": role.get("keywords")
            })
        
        return response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取会话角色失败: {str(e)}"
        )

@router.post("/{session_id}/roles", status_code=status.HTTP_201_CREATED)
async def add_session_role(
    session_id: str,
    role_data: SessionRoleAdd,
    current_user = Depends(get_current_user)
):
    """
    向会话添加角色
    """
    try:
        user_id = str(current_user["id"])
        success = await SessionService.add_session_role(
            session_id=session_id,
            user_id=user_id,
            role_id=role_data.role_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="添加角色失败"
            )
            
        return {"message": "角色添加成功"}
    except ValueError as e:
        if "角色不存在" in str(e) or "角色已禁用" in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"添加角色失败: {str(e)}"
        )

@router.delete("/{session_id}/roles/{role_id}")
async def remove_session_role(
    session_id: str,
    role_id: str,
    current_user = Depends(get_current_user)
):
    """
    从会话中移除角色
    """
    try:
        user_id = str(current_user["id"])
        success = await SessionService.remove_session_role(
            session_id=session_id,
            user_id=user_id,
            role_id=role_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="移除角色失败"
            )
            
        return {"message": "角色移除成功"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"移除角色失败: {str(e)}"
        )

@router.get("/{session_id}/roles/{role_id}/check")
async def check_session_role(
    session_id: str,
    role_id: str,
    current_user = Depends(get_current_user)
):
    """
    检查会话是否包含特定角色
    """
    try:
        user_id = str(current_user["id"])
        has_role = await SessionService.has_session_role(
            session_id=session_id,
            user_id=user_id,
            role_id=role_id
        )
        
        return {"has_role": has_role}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"检查角色失败: {str(e)}"
        ) 