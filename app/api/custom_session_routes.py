"""
自定义会话API路由 - 提供基于MD5会话ID生成的会话创建接口
"""

from fastapi import APIRouter, Depends, HTTPException, Body, Path, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
import traceback
import sys

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

@router.get("/list", response_model=Dict[str, Any])
async def get_all_sessions(
    page: int = Query(1, ge=1, description="页码"),
    limit: int = Query(10, le=100, description="每页数量"),
    status: Optional[int] = Query(None, description="过滤状态 (0/1/2)"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_or_none)
):
    logger.info(f"开始获取会话列表: page={page}, limit={limit}, status={status}")
    try:
        # 记录请求参数
        logger.debug(f"获取会话列表参数: page={page}, limit={limit}, status={status}, user={current_user}")
        
        # 调用服务前记录日志
        logger.info("即将调用CustomSessionService.get_all_sessions方法")
        
        # 获取会话列表
        sessions, total = await CustomSessionService.get_all_sessions(page=page, limit=limit, status=status)
        
        # 记录会话获取结果
        logger.info(f"成功获取会话列表: 共{total}条记录，本次返回{len(sessions)}条")
        logger.debug(f"会话ID列表: {[s.get('session_id', 'unknown') for s in sessions]}")
        
        # 构建响应数据前记录每个会话的字段
        for i, session in enumerate(sessions):
            if not isinstance(session, dict):
                logger.warning(f"会话{i}不是字典类型: {type(session)}")
                continue
                
            missing_fields = []
            for field in ['session_id', 'class_name', 'user_name', 'session_status', 'created_at', 'updated_at']:
                if field not in session:
                    missing_fields.append(field)
                elif field in ['created_at', 'updated_at'] and session[field] is None:
                    missing_fields.append(f"{field}(None)")
                    
            if missing_fields:
                logger.warning(f"会话{i}(ID:{session.get('session_id', 'unknown')})缺少字段: {', '.join(missing_fields)}")
                logger.debug(f"会话{i}完整内容: {session}")
        
        # 构建响应数据
        response_data = {
            "data": [],
            "pagination": {
                "total": total,
                "page": page,
                "limit": limit
            }
        }
        
        # 处理每个会话数据
        for session in sessions:
            try:
                # 验证会话字段
                if not all(key in session for key in ['session_id', 'class_name', 'user_name', 'session_status']):
                    missing = [key for key in ['session_id', 'class_name', 'user_name', 'session_status'] if key not in session]
                    logger.error(f"会话缺少必要字段: {missing}, 会话ID: {session.get('session_id', 'unknown')}")
                    continue
                
                session_data = {
                    "session_id": session["session_id"],
                    "class_name": session["class_name"],
                    "user_name": session["user_name"],
                    "status": session["session_status"],
                }
                
                # 安全处理日期字段
                for date_field in ['created_at', 'updated_at']:
                    if date_field in session and session[date_field] is not None:
                        try:
                            session_data[date_field] = session[date_field].isoformat()
                        except AttributeError as e:
                            logger.error(f"日期字段{date_field}格式错误: {e}, 值: {session[date_field]}, 类型: {type(session[date_field])}")
                            session_data[date_field] = str(session[date_field])
                    else:
                        logger.warning(f"会话缺少{date_field}字段, 会话ID: {session.get('session_id', 'unknown')}")
                        session_data[date_field] = None
                
                response_data["data"].append(session_data)
            except Exception as item_e:
                logger.error(f"处理单个会话数据时出错: {str(item_e)}, 会话ID: {session.get('session_id', 'unknown')}")
                logger.error(f"错误详情: {traceback.format_exc()}")
                continue
        
        logger.info(f"会话列表API响应: {len(response_data['data'])}条记录")
        return response_data
        
    except Exception as e:
        # 记录详细错误信息
        error_msg = f"获取会话列表失败: {str(e)}"
        logger.error(error_msg)
        logger.error(f"错误详情: {traceback.format_exc()}")
        
        # 尝试获取更多关于CustomSessionService的信息
        try:
            logger.debug(f"CustomSessionService类信息: {dir(CustomSessionService)}")
        except Exception as debug_e:
            logger.error(f"无法获取CustomSessionService信息: {str(debug_e)}")
        
        # 更详细的错误返回
        if isinstance(e, (AttributeError, KeyError, TypeError)):
            detail = f"服务数据结构错误: {str(e)}"
        elif isinstance(e, ImportError):
            detail = f"服务加载错误: {str(e)}"
        elif "connection" in str(e).lower():
            detail = f"数据库连接错误: {str(e)}"
        else:
            detail = f"获取会话列表失败: {str(e)}"
            
        raise HTTPException(status_code=500, detail=detail)

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

@router.get("/{session_id}/sync", response_model=Dict[str, Any])
async def sync_session(
    session_id: str = Path(..., description="会话ID"),
    direction: str = Query("both", description="同步方向: to_redis, to_mongodb, both"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_or_none)
):
    """
    在Redis和MongoDB之间同步会话数据
    
    参数:
        session_id: 会话ID
        direction: 同步方向 (to_redis, to_mongodb, both)
    """
    try:
        # 获取会话信息，确保存在
        session = await CustomSessionService.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")
            
        # 获取用户ID
        user_id = session.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="会话缺少用户ID")
            
        from app.models.custom_session import CustomSession
        
        # 执行同步
        results = {"success": False, "operations": []}
        
        if direction in ["to_redis", "both"]:
            # 从MongoDB同步到Redis
            redis_sync = await CustomSession.sync_session_to_redis(session_id)
            results["operations"].append({
                "direction": "to_redis",
                "success": redis_sync
            })
            
        if direction in ["to_mongodb", "both"]:
            # 从Redis同步到MongoDB
            mongodb_sync = await CustomSession.sync_session_to_mongodb(session_id, user_id)
            results["operations"].append({
                "direction": "to_mongodb",
                "success": mongodb_sync
            })
            
        # 检查同步结果
        results["success"] = any(op["success"] for op in results["operations"])
        
        if not results["success"]:
            logger.error(f"会话同步失败: {session_id}, 方向: {direction}")
            return {"success": False, "message": "同步失败，请查看日志"}
            
        return results
    except Exception as e:
        logger.error(f"会话同步失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"会话同步失败: {str(e)}")

@router.get("/inconsistencies", response_model=Dict[str, Any])
async def get_inconsistencies(
    limit: int = Query(100, description="最多检查的会话数量"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_or_none)
):
    """
    获取Redis和MongoDB中的会话数据不一致情况
    
    参数:
        limit: 最多检查的会话数量
    """
    try:
        from app.models.custom_session import CustomSession
        
        # 检测不一致情况
        inconsistencies = await CustomSession.get_session_inconsistencies(limit)
        
        return {
            "success": True,
            "count": len(inconsistencies),
            "inconsistencies": inconsistencies
        }
    except Exception as e:
        logger.error(f"检测会话不一致失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"检测会话不一致失败: {str(e)}")

@router.delete("/{session_id}", response_model=Dict[str, bool])
async def delete_session(
    session_id: str = Path(..., description="会话ID"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_or_none)
):
    """
    删除指定ID的会话
    
    参数:
        session_id: 要删除的会话ID
    """
    logger.info(f"请求删除会话: {session_id}")
    
    try:
        # 验证会话是否存在
        session_exists = await CustomSessionService.check_session_exists(session_id)
        if not session_exists:
            logger.warning(f"尝试删除不存在的会话: {session_id}")
            raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")
        
        # 执行删除操作
        success = await CustomSessionService.delete_session(session_id)
        
        if success:
            logger.info(f"会话删除成功: {session_id}")
            return {"success": True}
        else:
            logger.error(f"会话删除失败: {session_id}")
            raise HTTPException(status_code=500, detail="删除会话失败")
            
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        error_detail = f"删除会话时发生错误: {str(e)}"
        logger.error(error_detail)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_detail) 