"""
记忆模块的API端点
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from app.auth.auth_handler import get_current_user, get_current_user_optional
from app.memory.memory_manager import get_memory_manager
from app.memory.schemas import SessionResponse, MemoryContext
from typing import Dict, List, Optional
import logging

router = APIRouter(prefix="/api/memory", tags=["memory"])

logger = logging.getLogger(__name__)

@router.post("/session/start")
async def start_session(current_user: Dict = Depends(get_current_user_optional)):
    """
    开始一个新的会话
    
    Returns:
        会话信息，包含会话ID
    """
    try:
        user_id = current_user.get("id", "anonymous_user")
        memory_manager = await get_memory_manager()
        session_id = await memory_manager.start_new_session(user_id)
        return {"success": True, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建会话失败: {str(e)}")

@router.post("/session/{session_id}/end")
async def end_session(
    session_id: str,
    current_user: Dict = Depends(get_current_user_optional)
):
    """
    结束会话并生成摘要
    
    Args:
        session_id: 会话ID
        
    Returns:
        会话结果，包含摘要
    """
    try:
        user_id = current_user.get("id", "anonymous_user")
        memory_manager = await get_memory_manager()
        result = await memory_manager.end_session(session_id, user_id)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"结束会话失败: {str(e)}")

@router.post("/session/{session_id}/message")
async def add_message(
    session_id: str,
    role: str,
    content: str,
    roleid: Optional[str] = None,
    current_user: Dict = Depends(get_current_user_optional)
):
    """
    添加消息到会话
    
    Args:
        session_id: 会话ID
        role: 角色 (user, assistant, system)
        content: 消息内容
        roleid: 角色ID，对应MongoDB roles表中的_id字段
        
    Returns:
        操作结果
    """
    try:
        user_id = current_user.get("id", "anonymous_user")
        
        # 记录重要参数
        logger.info(f"添加消息参数: session_id={session_id}, role={role}, roleid={roleid}, content前20字符={content[:20] if content else ''}")
        
        # 确保roleid是有效值
        if roleid == "null" or roleid == "":
            roleid = None
            
        memory_manager = await get_memory_manager()
        try:
            success = await memory_manager.add_message(
                session_id, 
                user_id,
                role, 
                content,
                role_id=roleid
            )
            
            if not success:
                raise HTTPException(status_code=400, detail="添加消息失败")
                
            return {"success": True}
        except ValueError as e:
            # 返回400错误，提示用户选择名称
            logger.warning(f"用户需要选择名称: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加消息失败: {str(e)}")

@router.get("/session/{session_id}/context")
async def get_context(
    session_id: str,
    current_message: Optional[str] = None,
    current_user: Dict = Depends(get_current_user_optional)
):
    """
    获取会话上下文
    
    Args:
        session_id: 会话ID
        current_message: 当前消息（可选）
        
    Returns:
        会话上下文
    """
    try:
        user_id = current_user.get("id", "anonymous_user")
        memory_manager = await get_memory_manager()
        context = await memory_manager.build_context(
            session_id, 
            user_id,
            current_message
        )
        
        return context
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取上下文失败: {str(e)}")

@router.get("/sessions")
async def get_sessions(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: Dict = Depends(get_current_user_optional)
):
    """
    获取用户的会话列表
    
    Args:
        page: 页码
        limit: 每页数量
        
    Returns:
        会话列表
    """
    try:
        user_id = current_user.get("id", "anonymous_user")
        memory_manager = await get_memory_manager()
        skip = (page - 1) * limit
        sessions = await memory_manager.get_user_sessions(
            user_id,
            limit=limit,
            skip=skip
        )
        
        return {"success": True, "sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")

@router.get("/session/{session_id}")
async def get_session(
    session_id: str,
    current_user: Dict = Depends(get_current_user_optional)
):
    """
    获取会话详情
    
    Args:
        session_id: 会话ID
        
    Returns:
        会话详情
    """
    try:
        user_id = current_user.get("id", "anonymous_user")
        memory_manager = await get_memory_manager()
        session = await memory_manager.get_session_detail(
            session_id,
            user_id
        )
        
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
            
        return {"success": True, "session": session}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话详情失败: {str(e)}")

# 为现有聊天API添加上下文注入功能
@router.get("/inject-context/{session_id}")
async def inject_context(
    session_id: str,
    current_message: str,
    current_user: Dict = Depends(get_current_user_optional)
):
    """
    生成用于注入到聊天中的上下文
    
    Args:
        session_id: 会话ID
        current_message: 当前消息
        
    Returns:
        系统提示词
    """
    try:
        user_id = current_user.get("id", "anonymous_user")
        memory_manager = await get_memory_manager()
        context = await memory_manager.build_context(
            session_id, 
            user_id,
            current_message
        )
        
        # 如果有相关摘要，构建系统提示词
        if context["related_summaries"]:
            summaries_text = "\n\n".join(context["related_summaries"])
            
            system_prompt = f"""以下是与当前对话相关的历史对话摘要，请在回答用户问题时参考这些信息：

{summaries_text}

请将上述信息与用户当前的问题结合起来，提供连贯、相关的回答。不要直接告诉用户你在使用他们的历史对话内容。"""

            return {"success": True, "system_prompt": system_prompt}
        
        # 没有相关摘要
        return {"success": True, "system_prompt": None}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成上下文失败: {str(e)}")

@router.post("/session/{session_id}/select-user")
async def select_session_user(
    session_id: str,
    username: str,
    current_user: Dict = Depends(get_current_user_optional)
):
    """
    为匿名会话选择一个用户名
    
    Args:
        session_id: 会话ID
        username: 选择的用户名
        
    Returns:
        操作结果
    """
    try:
        user_id = current_user.get("id", "anonymous_user")
        
        # 如果用户已登录，不需要选择用户名
        if user_id != "anonymous_user":
            return {"success": True, "message": "已登录用户无需选择用户名"}
            
        if not username or username.strip() == "":
            raise HTTPException(status_code=400, detail="用户名不能为空")
            
        # 获取memory manager
        memory_manager = await get_memory_manager()
        
        # 确保会话存在
        actual_session_id = await memory_manager.ensure_session_exists(user_id, session_id)
        
        # 检查内存管理器和Redis可用性
        if not memory_manager.short_term_memory or not memory_manager.short_term_memory.redis:
            raise HTTPException(status_code=500, detail="内存管理器或Redis不可用")
            
        # 设置选定的用户名
        session_key = f"session:{user_id}:{actual_session_id}"
        await memory_manager.short_term_memory.redis.hset(session_key, "selected_username", username)
        logger.info(f"已为会话 {actual_session_id} 设置选中的用户名: {username}")
        
        return {"success": True, "message": f"已选择用户名: {username}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"选择用户名失败: {str(e)}") 

@router.get("/session/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    current_user: Dict = Depends(get_current_user_optional)
):
    """
    获取会话消息列表
    
    Args:
        session_id: 会话ID
        
    Returns:
        会话消息列表
    """
    try:
        user_id = current_user.get("id", "anonymous_user")
        memory_manager = await get_memory_manager()
        
        # 获取会话消息
        messages = await memory_manager.short_term_memory.get_session_messages(session_id, user_id)
        
        if messages is None:
            messages = []
            
        # 转换消息格式
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.get("role"),
                "content": msg.get("content"),
                "timestamp": msg.get("timestamp"),
                "sender": msg.get("sender"),
                "user_name": msg.get("user_name"),
                "assistant_name": msg.get("assistant_name")
            })
            
        return {"success": True, "messages": formatted_messages}
    except Exception as e:
        logger.error(f"获取会话消息失败: {str(e)}", exc_info=True)
        return {"success": False, "error": f"获取会话消息失败: {str(e)}", "messages": []} 