from fastapi import APIRouter, Depends, HTTPException, Query, Path, Request
from typing import List, Dict, Optional, Any
from bson import ObjectId
from pydantic import BaseModel, Field, validator
from datetime import datetime

from ..services.session_service import SessionService
from ..auth.auth_bearer import JWTBearer
from ..auth.auth_handler import get_current_user, get_current_user_or_none
from ..models.user import User

router = APIRouter(
    prefix="/api/sessions",
    tags=["sessions"]
)

# 数据模型
class SessionBase(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True

class SessionCreate(SessionBase):
    role_ids: Optional[List[str]] = []
    settings: Optional[Dict[str, Any]] = None

class SessionUpdate(SessionBase):
    pass

class SessionRoleUpdate(BaseModel):
    role_ids: List[str] = Field(..., description="角色ID列表")
    
    @validator('role_ids')
    def validate_role_ids(cls, v):
        if not v:
            raise ValueError("角色ID列表不能为空")
        return v

class SessionSettingsUpdate(BaseModel):
    history_enabled: Optional[bool] = None
    context_window: Optional[int] = None
    memory_enabled: Optional[bool] = None
    system_prompt: Optional[str] = None

class SessionResponse(SessionBase):
    id: str
    user_id: str
    role_ids: List[str]
    settings: Dict[str, Any]
    status: str
    created_at: str
    updated_at: str
    last_message_at: str
    
    class Config:
        orm_mode = True

# 会话路由
@router.post("", response_model=SessionResponse)
async def create_session(
    session_data: SessionCreate,
    current_user: User = Depends(get_current_user)
):
    """创建会话"""
    try:
        session = await SessionService.create_session(
            current_user["id"],
            title=session_data.title,
            description=session_data.description,
            role_ids=session_data.role_ids,
            settings=session_data.settings
        )
        
        # 格式化日期
        session["created_at"] = session["created_at"].isoformat()
        session["updated_at"] = session["updated_at"].isoformat()
        session["last_message_at"] = session["last_message_at"].isoformat()
        session["id"] = str(session["_id"])
        
        return session
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建会话失败: {str(e)}")

@router.get("", response_model=List[SessionResponse])
async def list_sessions(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    skip: Optional[int] = Query(None, ge=0, description="别名，与offset参数相同"),
    status: Optional[str] = Query("active", description="会话状态: active, archived, 或 null 获取所有状态"),
    current_user: Optional[User] = Depends(get_current_user_or_none)
):
    """获取会话列表"""
    try:
        # 处理未认证用户
        if not current_user:
            # 返回空列表或示例会话
            return []
            
        # 处理传入null字符串的情况
        if status == "null":
            status = None
            
        # 使用skip参数值（如果提供）作为offset
        final_offset = skip if skip is not None else offset
            
        sessions = await SessionService.list_sessions(
            current_user["id"],
            limit=limit,
            offset=final_offset,
            status=status
        )
        
        # 格式化日期和ID
        for session in sessions:
            session["created_at"] = session["created_at"].isoformat()
            session["updated_at"] = session["updated_at"].isoformat()
            session["last_message_at"] = session["last_message_at"].isoformat()
            session["id"] = str(session["_id"])
            
        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")

@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str = Path(..., title="会话ID"),
    current_user: User = Depends(get_current_user)
):
    """获取会话详情"""
    try:
        session = await SessionService.get_session(session_id, current_user["id"])
        
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在或无权访问")
            
        # 格式化日期和ID
        session["created_at"] = session["created_at"].isoformat()
        session["updated_at"] = session["updated_at"].isoformat()
        session["last_message_at"] = session["last_message_at"].isoformat()
        session["id"] = str(session["_id"])
        
        return session
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话失败: {str(e)}")

@router.put("/{session_id}", response_model=dict)
async def update_session(
    session_data: SessionUpdate,
    session_id: str = Path(..., title="会话ID"),
    current_user: User = Depends(get_current_user)
):
    """更新会话基本信息"""
    try:
        success = await SessionService.update_session(
            session_id,
            current_user["id"],
            session_data.dict(exclude_unset=True)
        )
        
        return {"success": success}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新会话失败: {str(e)}")

@router.put("/{session_id}/roles", response_model=dict)
async def update_session_roles(
    role_data: SessionRoleUpdate,
    session_id: str = Path(..., title="会话ID"),
    current_user: User = Depends(get_current_user)
):
    """更新会话角色"""
    try:
        success = await SessionService.configure_session_roles(
            session_id,
            current_user["id"],
            role_data.role_ids
        )
        
        return {"success": success}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新会话角色失败: {str(e)}")

@router.put("/{session_id}/settings", response_model=dict)
async def update_session_settings(
    settings_data: SessionSettingsUpdate,
    session_id: str = Path(..., title="会话ID"),
    current_user: User = Depends(get_current_user)
):
    """更新会话设置"""
    try:
        settings_dict = settings_data.dict(exclude_unset=True)
        
        if not settings_dict:
            raise HTTPException(status_code=400, detail="没有提供需要更新的设置")
            
        success = await SessionService.configure_session_settings(
            session_id,
            current_user["id"],
            settings_dict
        )
        
        return {"success": success}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新会话设置失败: {str(e)}")

@router.post("/{session_id}/archive", response_model=dict)
async def archive_session(
    session_id: str = Path(..., title="会话ID"),
    current_user: Optional[User] = Depends(get_current_user_or_none)
):
    """归档会话"""
    try:
        # 使用默认用户ID（如果未认证）
        user_id = current_user["id"] if current_user else "anonymous_user"
        success = await SessionService.archive_session(session_id, user_id)
        return {"success": success}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"归档会话失败: {str(e)}")

@router.put("/{session_id}/archive", response_model=dict)
async def archive_session_put(
    session_id: str = Path(..., title="会话ID"),
    current_user: Optional[User] = Depends(get_current_user_or_none)
):
    """归档会话 (PUT方法)"""
    try:
        # 使用默认用户ID（如果未认证）
        user_id = current_user["id"] if current_user else "anonymous_user"
        success = await SessionService.archive_session(session_id, user_id)
        return {"success": success}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"归档会话失败: {str(e)}")

@router.post("/{session_id}/restore", response_model=dict)
async def restore_session(
    session_id: str = Path(..., title="会话ID"),
    current_user: Optional[User] = Depends(get_current_user_or_none)
):
    """恢复已归档会话"""
    try:
        # 使用默认用户ID（如果未认证）
        user_id = current_user["id"] if current_user else "anonymous_user"
        success = await SessionService.restore_session(session_id, user_id)
        return {"success": success}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"恢复会话失败: {str(e)}")

@router.put("/{session_id}/restore", response_model=dict)
async def restore_session_put(
    session_id: str = Path(..., title="会话ID"),
    current_user: Optional[User] = Depends(get_current_user_or_none)
):
    """恢复已归档会话 (PUT方法)"""
    try:
        # 使用默认用户ID（如果未认证）
        user_id = current_user["id"] if current_user else "anonymous_user"
        success = await SessionService.restore_session(session_id, user_id)
        return {"success": success}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"恢复会话失败: {str(e)}")

@router.delete("/{session_id}", response_model=dict)
async def delete_session(
    session_id: str = Path(..., title="会话ID"),
    current_user: User = Depends(get_current_user)
):
    """删除会话"""
    try:
        success = await SessionService.delete_session(session_id, current_user["id"])
        return {"success": success}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除会话失败: {str(e)}")

# 以下是新增的角色管理端点
@router.get("/{session_id}/roles", response_model=List[Dict[str, Any]])
async def get_session_roles(
    session_id: str = Path(..., title="会话ID"),
    current_user: User = Depends(get_current_user)
):
    """获取会话角色列表"""
    try:
        roles = await SessionService.get_session_roles(session_id, current_user["id"])
        return roles
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话角色失败: {str(e)}")

@router.post("/{session_id}/roles/{role_id}", response_model=dict)
async def add_session_role(
    session_id: str = Path(..., title="会话ID"),
    role_id: str = Path(..., title="角色ID"),
    current_user: User = Depends(get_current_user)
):
    """添加角色到会话"""
    try:
        success = await SessionService.add_session_role(session_id, current_user["id"], role_id)
        return {"success": success}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加会话角色失败: {str(e)}")

@router.delete("/{session_id}/roles/{role_id}", response_model=dict)
async def remove_session_role(
    session_id: str = Path(..., title="会话ID"),
    role_id: str = Path(..., title="角色ID"),
    current_user: User = Depends(get_current_user)
):
    """从会话中移除角色"""
    try:
        success = await SessionService.remove_session_role(session_id, current_user["id"], role_id)
        return {"success": success}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"移除会话角色失败: {str(e)}")

@router.get("/{session_id}/roles/{role_id}", response_model=dict)
async def has_session_role(
    session_id: str = Path(..., title="会话ID"),
    role_id: str = Path(..., title="角色ID"),
    current_user: User = Depends(get_current_user)
):
    """检查会话是否包含特定角色"""
    try:
        has_role = await SessionService.has_session_role(session_id, current_user["id"], role_id)
        return {"has_role": has_role}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检查会话角色失败: {str(e)}")

# 添加永久删除和清理已删除会话的端点
@router.delete("/{session_id}/permanent", response_model=dict)
async def permanently_delete_session(
    session_id: str = Path(..., title="会话ID"),
    current_user: User = Depends(get_current_user)
):
    """永久删除会话（从数据库彻底删除，不可恢复）"""
    try:
        success = await SessionService.permanently_delete_session(session_id, current_user["id"])
        return {"success": success}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"永久删除会话失败: {str(e)}")

@router.delete("/cleanup/deleted", response_model=dict)
async def clean_deleted_sessions(
    older_than_days: Optional[int] = Query(None, description="如果提供，只删除超过指定天数的已删除会话"),
    current_user: User = Depends(get_current_user)
):
    """清理用户的已删除会话"""
    try:
        count = await SessionService.clean_deleted_sessions(current_user["id"], older_than_days)
        return {"success": True, "deleted_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理已删除会话失败: {str(e)}")

@router.put("/{session_id}/status/{new_status}", response_model=dict)
async def change_session_status(
    session_id: str = Path(..., title="会话ID"),
    new_status: str = Path(..., title="新状态", description="可选值: active, archived, deleted"),
    current_user: User = Depends(get_current_user)
):
    """修改会话状态"""
    try:
        success = await SessionService.change_session_status(session_id, current_user["id"], new_status)
        return {"success": success}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"修改会话状态失败: {str(e)}")

# 会话搜索相关路由
class SessionSearchParams(BaseModel):
    query: Optional[str] = None
    status: Optional[str] = None
    role_id: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    updated_after: Optional[datetime] = None
    updated_before: Optional[datetime] = None
    sort_by: str = "updated_at"
    sort_direction: int = -1
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)

@router.post("/search", response_model=Dict[str, Any])
async def search_sessions(
    search_params: SessionSearchParams,
    current_user: User = Depends(get_current_user)
):
    """高级会话搜索"""
    try:
        result = await SessionService.search_sessions(
            current_user["id"],
            query=search_params.query,
            status=search_params.status,
            role_id=search_params.role_id,
            created_after=search_params.created_after,
            created_before=search_params.created_before,
            updated_after=search_params.updated_after,
            updated_before=search_params.updated_before,
            sort_by=search_params.sort_by,
            sort_direction=search_params.sort_direction,
            limit=search_params.limit,
            offset=search_params.offset
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索会话失败: {str(e)}")

@router.get("/search", response_model=Dict[str, Any])
async def search_sessions_get(
    query: Optional[str] = None,
    status: Optional[str] = None,
    role_id: Optional[str] = None,
    sort_by: str = "updated_at",
    sort_direction: int = -1,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user)
):
    """通过GET请求搜索会话（简化版，不包含时间筛选）"""
    try:
        result = await SessionService.search_sessions(
            current_user["id"],
            query=query,
            status=status,
            role_id=role_id,
            sort_by=sort_by,
            sort_direction=sort_direction,
            limit=limit,
            offset=offset
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索会话失败: {str(e)}")

@router.post("/{session_id}/end-and-archive", response_model=dict)
async def end_and_archive_session(
    session_id: str = Path(..., title="会话ID"),
    request: Request = None,
    current_user: Optional[User] = Depends(get_current_user_or_none)
):
    """结束会话并强制归档所有消息"""
    try:
        # 解析请求体
        data = await request.json()
        user_id = data.get("user_id")
        
        # 如果没有提供user_id，使用当前登录用户
        if not user_id:
            if current_user:
                user_id = current_user.get("id", "anonymous_user")
            else:
                user_id = "anonymous_user"
        
        logger.info(f"结束并归档会话: session_id={session_id}, user_id={user_id}")
        
        # 获取记忆管理器
        from app.memory.memory_manager import get_memory_manager
        memory_manager = await get_memory_manager()
        
        # 获取会话中所有消息
        messages = memory_manager.short_term.get_session_messages(session_id, user_id)
        message_count = len(messages)
        
        if not messages:
            logger.warning(f"会话 {session_id} 没有消息可归档")
            return {
                "success": False,
                "archived_messages_count": 0,
                "total_messages": 0,
                "message": "会话没有消息可归档"
            }
        
        # 归档消息计数
        archived_count = 0
        
        # 逐条归档消息到MongoDB
        for message in messages:
            success = await memory_manager.archive_message(session_id, user_id, message)
            if success:
                archived_count += 1
        
        # 调用会话结束函数，生成摘要
        result = await memory_manager.end_session(session_id, user_id)
        
        return {
            "success": True,
            "archived_messages_count": archived_count,
            "total_messages": message_count,
            "summary": result.get("summary", ""),
            "session_id": session_id
        }
    except Exception as e:
        logger.error(f"结束并归档会话失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"结束并归档会话失败: {str(e)}") 