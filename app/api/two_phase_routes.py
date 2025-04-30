"""
Two-Phase API Routes - 实现基于RAG增强的两阶段聊天API

第一阶段：基于用户输入生成响应并选择合适的角色
第二阶段：处理对生成响应的反馈并生成改进的回复
"""

from fastapi import APIRouter, HTTPException, Body, Depends, Query, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, Any, Optional, List, AsyncGenerator
import logging
import uuid
import json
import time
import traceback
from datetime import datetime
import os

from app.services.redis_manager import get_redis_client
from app.services.custom_session_service import CustomSessionService
from app.services.rag_enhanced_service import RAGEnhancedService
from app.utils.redis_lock import RedisLock, obtain_lock
from app.memory.memory_manager import get_memory_manager
from app.services.llm_service import LLMService
from app.services.role_service import RoleService

router = APIRouter(
    prefix="/api/two-phase-streamrag",
    tags=["two-phase-streamrag"]
)

logger = logging.getLogger(__name__)

# 创建RAGEnhancedService依赖
def get_rag_service() -> RAGEnhancedService:
    # 返回全局初始化服务实例，保证对象已初始化
    global _rag_service
    if not _rag_service:
        # 创建新实例，但不要在同步上下文中进行初始化
        # 初始化会在 get_initialized_rag_service 中完成
        _rag_service = RAGEnhancedService()
    return _rag_service

# 初始化服务实例 - 将懒加载
_rag_service = None

async def get_initialized_rag_service(rag_interface: str = None) -> RAGEnhancedService:
    """获取已初始化的RAG服务实例"""
    # 添加日志记录
    logger.info(f"get_initialized_rag_service调用: rag_interface={rag_interface}")
    
    global _rag_service
    if _rag_service is None:
        logger.info("创建新的RAG服务实例")
        _rag_service = RAGEnhancedService()
        await _rag_service.initialize()
        logger.info("RAG服务实例初始化完成")
    elif not _rag_service._initialized:
        # 确保服务已初始化
        logger.info("确保已有的RAG服务实例已初始化")
        await _rag_service.initialize()
        logger.info("RAG服务实例初始化完成")
    else:
        logger.info("使用已存在的已初始化RAG服务实例")
        
    # 记录环境变量状态
    logger.info(f"环境变量状态: DEFAULT_DATASET_ID={os.getenv('DEFAULT_DATASET_ID')}, SECONDARY_RAG_DATASET_IDS={os.getenv('SECONDARY_RAG_DATASET_IDS')}")
    
    # 检查服务的rag_interfaces状态
    if hasattr(_rag_service.app_config, 'rag_interfaces'):
        for name, interface in _rag_service.app_config.rag_interfaces.items():
            logger.info(f"RAG接口配置: name={name}, dataset_ids={interface.dataset_ids}")
    
    return _rag_service

# 添加GET方法支持的生成响应端点
@router.get("/generate", response_model=Dict[str, Any])
async def generate_response_get(
    session_id: str = Query(..., description="会话标识符"),
    message: str = Query(..., description="用户消息内容"),
    user_id: str = Query("anonymous", description="用户ID"),
    enable_rag: bool = Query(True, description="是否启用RAG增强"),
    auto_role_match: bool = Query(True, description="是否启用自动角色匹配"),
    role_id: Optional[str] = Query(None, description="指定角色ID"),
    stream: bool = Query(False, description="是否使用流式响应"),
    show_thinking: bool = Query(True, description="是否显示思考过程"),
    rag_interface: Optional[str] = Query(None, description="使用的RAG接口名称")
):
    """
    第一阶段API（GET方法）：基于用户消息生成响应
    
    参数:
        session_id: 会话标识符
        message: 用户消息内容
        user_id: 用户ID（可选）
        enable_rag: 是否启用RAG增强（可选，默认为True）
        auto_role_match: 是否启用自动角色匹配（可选，默认为True）
        role_id: 指定角色ID（可选）
        stream: 是否使用流式响应（可选，默认为False）
        show_thinking: 是否显示思考过程（可选，默认为True）
        rag_interface: 使用的RAG接口名称（可选）
    
    返回:
        如果stream=False:
            message_id: 此消息的唯一标识符
            response: 生成的响应
            selected_role: 回复的角色名称
            match_score: 匹配分数
            match_reason: 匹配原因
            references: 如果启用RAG，返回的相关参考资料
            thinking_process: 如果show_thinking=True，返回思考过程
        如果stream=True:
            返回SSE格式的数据流
    """
    # 如果请求流式响应，使用StreamingResponse
    if stream:
        return StreamingResponse(
            streaming_response_generator(
                session_id=session_id,
                message=message,
                user_id=user_id,
                enable_rag=enable_rag,
                auto_role_match=auto_role_match,
                role_id=role_id,
                show_thinking=show_thinking,
                rag_interface=rag_interface
            ),
            media_type="text/event-stream"
        )
    
    # 非流式处理，将查询参数转换为字典，复用POST方法处理逻辑
    data = {
        "session_id": session_id,
        "message": message,
        "user_id": user_id,
        "enable_rag": enable_rag,
        "auto_role_match": auto_role_match,
        "show_thinking": show_thinking
    }
    
    # 如果提供了role_id，添加到数据中
    if role_id:
        data["role_id"] = role_id
        
    # 如果提供了rag_interface，添加到数据中
    if rag_interface:
        data["rag_interface"] = rag_interface
        
    # 调用POST方法处理函数
    return await generate_response(data)

@router.post("/generate", response_model=Dict[str, Any])
async def generate_response(
    data: Dict[str, Any] = Body(...),
    stream: bool = Query(False, description="是否使用流式响应"),
    show_thinking: bool = Query(True, description="是否显示思考过程")
):
    """
    第一阶段API：基于用户消息生成响应
    
    参数:
        session_id: 会话标识符
        message: 用户消息内容
        user_id: 用户ID
        enable_rag: 是否启用RAG增强（可选，默认为True）
        auto_role_match: 是否启用自动角色匹配（可选，默认为True）
        show_thinking:
            是否显示思考过程（可选，默认为True）
        stream: 是否使用流式响应（查询参数，默认为False）
        rag_interface: 使用的RAG接口名称（可选）
    
    返回:
        如果stream=False:
            message_id: 此消息的唯一标识符
            response: 生成的响应
            selected_role: 回复的角色名称
            match_score: 匹配分数
            match_reason: 匹配原因
            references: 如果启用RAG，返回的相关参考资料
            thinking_process: 如果show_thinking=True，返回思考过程
        如果stream=True:
            返回SSE格式的数据流
    """
    # 从查询参数或请求体获取show_thinking
    show_thinking_value = show_thinking
    if "show_thinking" in data:
        show_thinking_value = data.get("show_thinking")
    
    # 如果请求流式响应
    if stream:
        return StreamingResponse(
            streaming_response_generator(
                session_id=data.get("session_id"),
                message=data.get("message"),
                user_id=data.get("user_id", "anonymous"),
                enable_rag=data.get("enable_rag", True),
                auto_role_match=data.get("auto_role_match", True),
                role_id=data.get("role_id"),
                show_thinking=show_thinking_value,
                rag_interface=data.get("rag_interface")
            ),
            media_type="text/event-stream"
        )
    
    session_id = data.get("session_id")
    message = data.get("message")
    user_id = data.get("user_id", "anonymous")
    enable_rag = data.get("enable_rag", True)
    auto_role_match = data.get("auto_role_match", True)
    role_id = data.get("role_id")  # 可选，指定角色ID
    rag_interface = data.get("rag_interface")  # 可选，指定RAG接口
    
    if not session_id or not message:
        raise HTTPException(status_code=400, detail="缺少必须的参数")
    
    try:
        # 1. 获取RAG服务和内存管理器
        rag_service = await get_initialized_rag_service(rag_interface)
        memory_manager = await get_memory_manager()
        
        # 2. 获取会话锁，防止并发处理
        redis_client = await get_redis_client()
        lock_name = f"two_phase:lock:{session_id}"
        lock = await obtain_lock(redis_client, lock_name, expire_seconds=60)
        
        if lock is None:
            raise HTTPException(status_code=429, detail="会话正忙，请稍后再试")
        
        try:
            # 3. 生成唯一消息ID
            message_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()
            
            # 4. 将用户消息添加到短期记忆
            await memory_manager.add_message(
                session_id=session_id,
                user_id=user_id,
                role="user",
                content=message,
                message_id=f"user-{message_id}"
            )
            
            # 5. 存储请求信息到Redis
            request_key = f"two_phase:request:{message_id}"
            
            await redis_client.hset(request_key, mapping={
                "session_id": session_id,
                "user_id": user_id,
                "message": message,
                "status": "pending",
                "created_at": timestamp,
                "enable_rag": str(enable_rag),
                "auto_role_match": str(auto_role_match),
                "show_thinking": str(show_thinking),
                "rag_interface": str(rag_interface) if rag_interface else ""
            })
            await redis_client.expire(request_key, 3600)  # 1小时过期
            
            # 6. 获取会话信息和会话绑定的角色
            session = await CustomSessionService.get_session(session_id)
            
            if not session or "roles" not in session:
                logger.error(f"未找到会话或会话中没有角色: {session_id}")
                raise HTTPException(status_code=404, detail="未找到会话或会话中没有角色")
            
            available_roles = session.get("roles", [])
            if not available_roles:
                logger.error(f"会话中没有可用角色: {session_id}")
                raise HTTPException(status_code=404, detail="会话中没有可用角色")
            
            # 7. 准备消息
            messages = [{"role": "user", "content": message}]
            
            # 8. 根据参数决定是否进行角色匹配
            role_info = None
            match_score = 0.0
            match_reason = ""
            
            try:
                # 如果有指定角色ID，直接使用
                if not auto_role_match and role_id:
                    logger.info(f"使用指定角色ID: {role_id}")
                else:
                    # 从可用角色中获取第一个作为备用
                    fallback_role = available_roles[0]
                    fallback_role_id = fallback_role.get("role_id") or fallback_role.get("id") or str(fallback_role.get("_id"))
                    
                    if auto_role_match:
                        match_result = await (await get_initialized_rag_service(rag_interface)).match_role_for_chat(
                            messages=messages,
                            session_id=session_id,
                            user_id=user_id
                        )
                        
                        # 从匹配结果中提取角色信息
                        if match_result and match_result.get("success"):
                            # 直接使用返回的完整角色对象，而不仅提取ID
                            matched_role = match_result.get("role", {})
                            
                            if matched_role:
                                # 直接使用完整的角色对象
                                role_info = matched_role
                                role_id = matched_role.get("id") or matched_role.get("role_id") or matched_role.get("_id")
                                match_score = match_result.get("match_score", 0.0)
                                match_reason = match_result.get("match_reason", "")
                                
                                role_name = matched_role.get("name") or matched_role.get("role_name", "未知角色")
                                logger.info(f"自动匹配到角色: {role_id}, 角色名: {role_name}")
            except Exception as e:
                logger.warning(f"角色匹配过程出错: {str(e)}，将使用默认角色")
            
            # 9. 如果没有匹配到角色，使用第一个可用角色
            if not role_id:
                first_role = available_roles[0]
                role_id = first_role.get("role_id") or first_role.get("id") or first_role.get("_id")
                role_info = first_role  # 直接使用available_roles中的完整角色对象
                logger.info(f"使用默认角色: {role_id}")
            
            # 查找完会话中的角色后，如果仍然没找到角色信息，直接返回错误
            if not role_info:
                raise HTTPException(
                    status_code=400, 
                    detail=f"在会话中未找到指定角色: {role_id}"
                )
            
            # 11. 生成响应
            full_response = ""
            references = []
            
            try:
                async for chunk in (await get_initialized_rag_service(rag_interface)).process_chat(
                    messages=messages,
                    model=role_info.get("model") if role_info else None,
                    session_id=session_id,
                    user_id=user_id,
                    enable_rag=enable_rag,
                    stream=False,
                    role_id=role_id,
                    auto_role_match=False,  # 已经完成角色匹配
                    show_thinking=show_thinking,
                    rag_interface=rag_interface  # 传递RAG接口参数
                ):
                    if isinstance(chunk, dict) and "references" in chunk:
                        references = chunk.get("references", [])
                    elif isinstance(chunk, str):
                        full_response += chunk
            except Exception as e:
                logger.error(f"生成响应时出错: {str(e)}")
                # 提供一个默认响应
                full_response = f"非常抱歉，我在处理您的请求时遇到了问题。错误信息: {str(e)[:100]}..."
            
            # 12. 将响应添加到短期记忆
            await memory_manager.add_message(
                session_id=session_id,
                user_id=user_id,
                role="assistant",
                content=full_response,
                role_id=role_id,
                message_id=f"assistant-{message_id}"
            )
            
            # 13. 存储完整响应到Redis
            match_score_str = str(match_score)
            match_reason_str = match_reason
            
            await redis_client.hset(request_key, mapping={
                "response": full_response,
                "selected_role": json.dumps(role_info),
                "role_id": role_id,
                "match_score": match_score_str,
                "match_reason": match_reason_str,
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat()
            })
            
            # 14. 添加到会话历史
            history_key = f"two_phase:history:{session_id}"
            await redis_client.rpush(history_key, message_id)
            await redis_client.expire(history_key, 86400 * 7)  # 7天过期
            
            # 15. 返回响应
            return {
                "message_id": message_id,
                "response": full_response,
                "selected_role": role_info.get("name") or role_info.get("role_name", "Unknown"),
                "match_score": match_score,
                "match_reason": match_reason,
                "references": references if enable_rag and references else [],
                "thinking_process": show_thinking_value and bool(full_response)
            }
        finally:
            # 释放锁
            if lock:
                await lock.release()
            
    except HTTPException:
        raise
    except Exception as e:
        error_detail = f"生成响应时出错: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

# 添加GET方法支持的反馈端点
@router.get("/feedback", response_model=Dict[str, Any])
async def provide_feedback_get(
    session_id: str = Query(..., description="会话标识符"),
    message_id: str = Query(..., description="第一阶段生成的消息ID"),
    is_accepted: bool = Query(False, description="响应是否被接受"),
    user_id: str = Query("anonymous", description="用户ID"),
    stream: bool = Query(False, description="是否使用流式响应")
):
    """
    第二阶段API（GET方法）：对生成的响应提供反馈
    
    参数:
        session_id: 会话标识符
        message_id: 第一阶段生成的消息ID
        is_accepted: 响应是否被接受
        user_id: 用户ID（可选）
        stream: 是否使用流式响应（可选，默认为False）
    
    返回:
        如果stream=False:
            success: 反馈是否处理成功
            is_accepted: 提供的反馈值
            improved_response: 如果拒绝，返回改进的响应
        如果stream=True:
            返回SSE格式的改进响应数据流
    """
    # 如果请求流式响应，使用StreamingResponse
    if stream:
        return StreamingResponse(
            streaming_feedback_generator(
                session_id=session_id,
                message_id=message_id,
                accepted=is_accepted,
                feedback=None,
                user_id=user_id,
                enable_rag=True,
                show_thinking=True,
                rag_interface=None
            ),
            media_type="text/event-stream"
        )
    
    # 非流式处理，将查询参数转换为字典，复用POST方法处理逻辑
    data = {
        "session_id": session_id,
        "message_id": message_id,
        "accepted": is_accepted,
        "user_id": user_id
    }
    
    # 调用POST方法处理函数
    return await provide_feedback(data)

@router.post("/feedback")
async def feedback_response(
    request: Request,
    background_tasks: BackgroundTasks
):
    """反馈处理接口"""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        message_id = data.get("message_id")
        accepted = data.get("accepted", False)
        feedback = data.get("feedback", "")
        user_id = data.get("user_id", "anonymous")
        enable_rag = data.get("enable_rag", True)
        show_thinking = data.get("show_thinking", True)
        rag_interface = data.get("rag_interface")
        
        if not session_id or not message_id:
            return JSONResponse(
                status_code=400,
                content={"message": "请求缺少必要的参数"}
            )
            
        return StreamingResponse(
            streaming_feedback_generator(
                session_id=session_id,
                message_id=message_id,
                accepted=accepted,
                feedback=feedback,
                user_id=user_id,
                enable_rag=enable_rag,
                show_thinking=show_thinking,
                rag_interface=rag_interface
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"处理反馈请求时出错: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"message": f"服务器错误: {str(e)}"}
        )

@router.get("/debug/check")
async def debug_check():
    """
    调试端点，检查两阶段API是否正常工作
    
    返回:
        status: 始终为"ok"
        timestamp: 当前时间
    """
    # 检查Redis连接
    redis_status = "unknown"
    try:
        redis_client = await get_redis_client()
        await redis_client.set("two_phase:debug:check", "ok", ex=60)
        test_value = await redis_client.get("two_phase:debug:check")
        redis_status = "ok" if test_value == "ok" else f"error: got {test_value}"
    except Exception as e:
        redis_status = f"error: {str(e)}"
    
    # 检查RAG服务
    rag_status = "unknown"
    try:
        rag_service = await get_initialized_rag_service()
        rag_status = "ok" if rag_service._initialized else "error: not initialized"
    except Exception as e:
        rag_status = f"error: {str(e)}"
    
    # 检查记忆管理器
    memory_status = "unknown"
    try:
        memory_manager = await get_memory_manager()
        memory_status = "ok" if memory_manager else "error: not initialized"
    except Exception as e:
        memory_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "redis_status": redis_status,
        "rag_status": rag_status,
        "memory_status": memory_status,
        "api": "two-phase"
    }

async def streaming_response_generator(
    session_id: str,
    message: str,
    user_id: str,
    enable_rag: bool,
    auto_role_match: bool,
    role_id: Optional[str],
    show_thinking: bool = True,  # 是否显示思考过程
    rag_interface: Optional[str] = None  # 使用的RAG接口名称
) -> AsyncGenerator[str, None]:
    """
    生成流式响应
    
    参数:
        session_id: 会话ID
        message: 用户消息
        user_id: 用户ID
        enable_rag: 是否启用RAG增强
        auto_role_match: 是否启用自动角色匹配
        role_id: 角色ID
        show_thinking: 是否显示思考过程
        rag_interface: 使用的RAG接口名称
        
    生成:
        SSE格式的事件数据
    """
    lock = None
    try:
        # 1. 获取Redis客户端和记忆管理器
        redis_client = await get_redis_client()
        memory_manager = await get_memory_manager()
        
        # 2. 创建消息ID和请求键
        message_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        request_key = f"two_phase:request:{message_id}"
        
        # 3. 在Redis中存储请求信息
        await redis_client.hset(request_key, mapping={
            "session_id": session_id,
            "user_id": user_id,
            "message": message,
            "enable_rag": str(enable_rag),
            "auto_role_match": str(auto_role_match),
            "role_id": role_id or "",
            "show_thinking": str(show_thinking),
            "rag_interface": rag_interface or "",
            "created_at": datetime.utcnow().isoformat(),
            "status": "processing"
        })
        await redis_client.expire(request_key, 3600)  # 1小时过期
        
        # 4. 获取会话锁，防止并发处理
        lock_name = f"two_phase:lock:{session_id}"
        
        # 先尝试删除可能存在的旧锁
        try:
            await redis_client.delete(lock_name)
            logger.info(f"删除可能存在的旧锁: {lock_name}")
        except Exception as e:
            logger.warning(f"尝试删除旧锁时出错: {str(e)}")
        
        # 获取新锁，减少锁的过期时间以避免长时间锁定
        lock = await obtain_lock(redis_client, lock_name, expire_seconds=10)
        
        if lock is None:
            error_data = {
                "type": "error",
                "message": "会话正在处理中，请稍后再试"
            }
            yield format_sse_event(
                data=json.dumps(error_data)
            )
            return
            
        try:
            # 获取会话信息和会话绑定的角色
            session = await CustomSessionService.get_session(session_id)
            
            if not session or "roles" not in session:
                error_data = {
                    "type": "error",
                    "message": "未找到会话或会话中没有角色"
                }
                yield format_sse_event(
                    data=json.dumps(error_data)
                )
                return
            
            # 获取会话中的角色列表
            available_roles = session.get("roles", [])
            if not available_roles:
                error_data = {
                    "type": "error",
                    "message": "会话中没有可用角色"
                }
                yield format_sse_event(
                    data=json.dumps(error_data)
                )
                return
            
            # 获取RAG服务并准备消息
            messages = [{"role": "user", "content": message}]
            
            # 角色匹配
            match_result = None
            role_info = None
            match_score = 0.0
            match_reason = ""
            
            try:
                # 如果有指定角色ID，直接使用
                if not auto_role_match and role_id:
                    logger.info(f"使用指定角色ID: {role_id}")
                else:
                    # 从可用角色中获取第一个作为备用
                    fallback_role = available_roles[0]
                    fallback_role_id = fallback_role.get("role_id") or fallback_role.get("id") or str(fallback_role.get("_id"))
                    
                    if auto_role_match:
                        match_result = await (await get_initialized_rag_service(rag_interface)).match_role_for_chat(
                            messages=messages,
                            session_id=session_id,
                            user_id=user_id
                        )
                        
                        # 从匹配结果中提取角色信息
                        if match_result and match_result.get("success"):
                            # 直接使用返回的完整角色对象，而不仅提取ID
                            matched_role = match_result.get("role", {})
                            
                            if matched_role:
                                # 直接使用完整的角色对象
                                role_info = matched_role
                                role_id = matched_role.get("id") or matched_role.get("role_id") or matched_role.get("_id")
                                match_score = match_result.get("match_score", 0.0)
                                match_reason = match_result.get("match_reason", "")
                                
                                role_name = matched_role.get("name") or matched_role.get("role_name", "未知角色")
                                logger.info(f"自动匹配到角色: {role_id}, 角色名: {role_name}")
            except Exception as e:
                logger.warning(f"角色匹配过程出错: {str(e)}，将使用默认角色")
            
            # 如果没有匹配到角色，使用第一个可用角色
            if not role_id:
                first_role = available_roles[0]
                role_id = first_role.get("role_id") or first_role.get("id") or first_role.get("_id")
                role_info = first_role  # 直接使用available_roles中的完整角色对象
                logger.info(f"使用默认角色: {role_id}")
            
            # 10. 如果上面的步骤没有获取到完整角色信息，则在available_roles中查找
            if not role_info:
                for role in available_roles:
                    role_id_in_list = role.get("role_id") or role.get("id") or str(role.get("_id"))
                    if role_id_in_list == role_id:
                        role_info = role
                        logger.info(f"在会话角色中找到匹配的角色信息: {role_id}")
                        break
            
            # 查找完会话中的角色后，如果仍然没找到角色信息，直接返回错误
            if not role_info:
                raise HTTPException(
                    status_code=400, 
                    detail=f"在会话中未找到指定角色: {role_id}"
                )
            
            # 11. 发送初始匹配信息
            role_name = role_info.get("name") or role_info.get("role_name", "未知角色")
            
            initial_data = {
                "type": "match_info",
                "message_id": message_id,
                "selected_role": role_name,
                "match_score": match_score,
                "match_reason": match_reason
            }
            yield format_sse_event(
                data=json.dumps(initial_data)
            )
            
            # 12. 设置缓冲区
            response_buffer = ""  # 存储最终响应内容
            thinking_buffer = ""  # 存储思考过程内容
            references = []       # 存储参考文档
            
            # 13. 生成响应（流式）
            try:
                rag_service = await get_initialized_rag_service(rag_interface)
                async for chunk in rag_service.process_chat(
                    messages=messages,
                    model=role_info.get("model") if role_info else None,
                    session_id=session_id,
                    user_id=user_id,
                    enable_rag=enable_rag,
                    stream=True,
                    role_id=role_id,
                    auto_role_match=False,  # 已经完成角色匹配
                    show_thinking=show_thinking,
                    rag_interface=rag_interface  # 传递RAG接口参数
                ):
                    if isinstance(chunk, dict):
                        # 处理不同类型的事件
                        chunk_type = chunk.get("type", "")
                        
                        # 思考过程相关事件
                        if chunk_type in ["thinking_mode", "thinking_start", "thinking_content", 
                                         "thinking_reference", "thinking_end", "thinking_error"]:
                            # 直接传递思考过程事件
                            yield format_sse_event(json.dumps({"type": "thinking", "data": chunk}))
                            
                            # 记录思考内容到缓冲区
                            if chunk_type == "thinking_content" and "content" in chunk:
                                thinking_buffer += chunk["content"] + "\n"
                                # 定期更新Redis中的思考过程记录
                                if len(thinking_buffer) % 500 == 0:
                                    await redis_client.hset(
                                        request_key,
                                        "thinking_process", 
                                        thinking_buffer
                                    )
                            
                            # 记录参考文档
                            if chunk_type == "thinking_reference":
                                references.append({
                                    "title": chunk.get("title", ""),
                                    "content": chunk.get("content", ""),
                                    "relevance": chunk.get("relevance", 0)
                                })
                        
                        # 内容片段事件
                        elif chunk_type == "content":
                            content = chunk.get("content", "")
                            content_data = {"type": "content", "content": content}
                            yield format_sse_event(json.dumps(content_data))
                            response_buffer += content
                            
                            # 定期更新Redis中的响应记录
                            if len(response_buffer) % 100 == 0:
                                await redis_client.hset(
                                    request_key,
                                    "partial_response", 
                                    response_buffer
                                )
                        
                        # 引用事件
                        elif chunk_type == "references":
                            references = chunk.get("references", [])
                            yield format_sse_event(json.dumps({"type": "references", "data": references}))
                        
                        # 其他类型事件直接传递
                        else:
                            yield format_sse_event(json.dumps(chunk))
                    
                    elif isinstance(chunk, str):
                        # 向后兼容：字符串类型直接作为内容发送
                        content_data = {"type": "content", "content": chunk}
                        yield format_sse_event(json.dumps(content_data))
                        response_buffer += chunk
            except Exception as e:
                logger.error(f"生成响应时出错: {str(e)}")
                # 发送错误信息
                error_data = {
                    "type": "error",
                    "message": f"生成响应时出错: {str(e)[:100]}..."
                }
                yield format_sse_event(json.dumps(error_data))
                
                # 如果出错但已有部分响应，仍然使用它
                if not response_buffer:
                    response_buffer = f"非常抱歉，我在处理您的请求时遇到了问题。错误信息: {str(e)[:100]}..."
            
            # 14. 将响应添加到短期记忆
            await memory_manager.add_message(
                session_id=session_id,
                user_id=user_id,
                role="assistant",
                content=response_buffer,
                role_id=role_id,
                message_id=f"assistant-{message_id}"
            )
            
            # 15. 存储完整数据到Redis
            await redis_client.hset(request_key, mapping={
                "response": response_buffer,
                "thinking_process": thinking_buffer if show_thinking else "",
                "references": json.dumps(references),
                "selected_role": json.dumps(role_info),
                "role_id": role_id,
                "match_score": str(match_score),
                "match_reason": match_reason,
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat()
            })
            
            # 16. 添加到会话历史
            history_key = f"two_phase:history:{session_id}"
            await redis_client.rpush(history_key, message_id)
            await redis_client.expire(history_key, 86400 * 7)  # 7天过期
            
            # 17. 发送完成信号
            complete_data = {
                "type": "complete",
                "message_id": message_id,
                "has_thinking": show_thinking and bool(thinking_buffer)
            }
            yield format_sse_event(json.dumps(complete_data))
            
        finally:
            # 释放锁
            if lock:
                try:
                    await lock.release()
                    logger.info(f"成功释放锁: {lock_name}")
                except Exception as e:
                    logger.error(f"释放锁时出错: {str(e)}")
                    # 强制删除锁
                    if redis_client and lock_name:
                        try:
                            await redis_client.delete(lock_name)
                            logger.info(f"强制删除锁: {lock_name}")
                        except Exception as ex:
                            logger.error(f"强制删除锁时出错: {str(ex)}")
            
    except Exception as e:
        error_detail = f"生成响应时出错: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        
        error_data = {
            "type": "error",
            "message": error_detail[:200] + "..." if len(error_detail) > 200 else error_detail
        }
        yield format_sse_event(json.dumps(error_data))

async def streaming_feedback_generator(
    session_id: str,
    message_id: str,
    accepted: bool,
    feedback: Optional[str] = None,
    user_id: str = "anonymous",
    enable_rag: bool = True,
    show_thinking: bool = True,
    rag_interface: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """流式生成反馈响应"""
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] 开始处理反馈: session_id={session_id}, message_id={message_id}, accepted={accepted}")
    
    # 初始化变量
    redis_client = None
    lock = None
    lock_name = None
    
    try:
        # 1. 获取Redis客户端和记忆管理器
        redis_client = await get_redis_client()
        memory_manager = await get_memory_manager()
        
        # 2. 获取存储的请求/响应数据
        request_key = f"two_phase:request:{message_id}"
        stored_data = await redis_client.hgetall(request_key)
        
        if not stored_data:
            yield format_sse_event(json.dumps({"type": "error", "message": f"未找到请求: {message_id}"}))
            return
        
        # 3. 验证消息属于指定会话
        if stored_data.get("session_id") != session_id:
            yield format_sse_event(json.dumps({"type": "error", "message": "会话ID不匹配"}))
            return
        
        # 4. 获取会话锁，防止并发处理
        lock_name = f"two_phase:lock:{message_id}"
        lock = await obtain_lock(redis_client, lock_name, expire_seconds=30)
        
        if lock is None:
            yield format_sse_event(json.dumps({"type": "error", "message": "消息正在处理中，请稍后再试"}))
            return
        
        # 5. 更新反馈状态
        await redis_client.hset(request_key, mapping={
            "feedback": "accepted" if accepted else "rejected",
            "feedback_time": datetime.utcnow().isoformat()
        })
        
        # 发送初始状态
        yield format_sse_event(json.dumps({
            "type": "status",
            "status": "feedback_received",
            "accepted": accepted
        }))
        
        # 如果接受，直接返回完成事件
        if accepted:
            yield format_sse_event(json.dumps({"type": "completion", "message": "反馈已接受"}))
            return
        
        # 6. 如果拒绝，生成改进的响应
        # 获取角色信息
        role_id = stored_data.get("role_id")
        logger.info(f"[{request_id}] 获取到的role_id: {role_id}, 类型: {type(role_id)}")

        # 检查role_id的有效性
        if role_id is None or role_id == "":
            logger.warning(f"[{request_id}] role_id为空或无效: '{role_id}'")
            role_id = None
        elif isinstance(role_id, str) and role_id.lower() in ["none", "null", "undefined", "unknown"]:
            logger.warning(f"[{request_id}] role_id为特殊值: '{role_id}'，将设置为None")
            role_id = None
        else:
            try:
                # 检查是否是有效的ObjectId（如果使用MongoDB）
                from bson import ObjectId
                if len(role_id) == 24:  # ObjectId应该是24位十六进制字符串
                    try:
                        ObjectId(role_id)
                        logger.info(f"[{request_id}] role_id是有效的ObjectId: {role_id}")
                    except Exception as e:
                        logger.warning(f"[{request_id}] role_id不是有效的ObjectId: {str(e)}")
            except ImportError:
                logger.info(f"[{request_id}] 未导入bson库，跳过ObjectId验证")

        # 显示完整的stored_data内容以帮助调试
        logger.info(f"[{request_id}] stored_data内容: {stored_data}")
        
        # 获取RAG服务和角色配置
        if rag_interface:
            rag_service = await get_initialized_rag_service(rag_interface)
        else:
            rag_service = await get_initialized_rag_service()
            
        # 使用RoleService替代get_role_config
        role_info = None
        if role_id:
            try:
                role_info = await RoleService.get_role_by_id(role_id)
                logger.info(f"[{request_id}] 获取到的role_info: {type(role_info)}, 值: {role_info}")
            except Exception as e:
                logger.error(f"[{request_id}] 调用RoleService.get_role_by_id出错: {str(e)}")
                role_info = None
        
        # 检查role_info是否为字典类型
        if role_info is not None and not isinstance(role_info, dict):
            logger.error(f"[{request_id}] role_info不是字典类型! 实际类型: {type(role_info)}, 值: {role_info}")
            # 如果不是字典，尝试转换或创建一个空字典
            try:
                if isinstance(role_info, str):
                    # 尝试将字符串解析为JSON
                    import json
                    role_info = json.loads(role_info)
                    logger.info(f"[{request_id}] 成功将role_info字符串转换为字典: {role_info}")
                else:
                    # 如果无法转换，使用空字典
                    logger.warning(f"[{request_id}] 无法处理role_info，使用空字典替代")
                    role_info = {}
            except Exception as e:
                logger.error(f"[{request_id}] 转换role_info时出错: {str(e)}")
                role_info = {}
        
        # 获取原始消息和响应
        original_message = stored_data.get("message", "")
        original_response = stored_data.get("response", "")
        
        # 构建改进提示
        improvement_prompt = f"""你之前的回复被用户拒绝，需要改进。

原始用户消息: "{original_message}"

你之前的回复: "{original_response}"

{feedback if feedback else ""}

请根据角色设定提供一个更好的回复。特别注意:
1. 确保回复与角色人设一致
2. 提供更有帮助、更详细的信息
3. 请考虑历史对话中的所有相关信息
4. 避免重复之前回复中的问题

请直接给出改进后的回复，不要解释你做了什么改变。"""

        # 构建消息列表
        messages = [
            {"role": "user", "content": original_message},
            {"role": "assistant", "content": original_response},
            {"role": "user", "content": improvement_prompt}
        ]
        
        # 生成改进响应
        improved_response_chunks = []
        improved_response = ""
        
        try:
            # 使用RAG服务生成改进响应
            yield format_sse_event(json.dumps({"type": "status", "status": "generating"}))
            
            # 记录调用generate_response前的参数信息
            logger.info(f"[{request_id}] 调用generate_response前的参数: session_id={session_id}, user_id={user_id}, role_id={role_id}")
            logger.info(f"[{request_id}] role_info的类型: {type(role_info)}, 值: {role_info}")
            
            async for chunk in rag_service.generate_response(
                messages=messages,
                model="default",
                session_id=session_id,
                user_id=user_id,
                role_id=role_id,
                role_info=role_info,
                stream=True,
                enable_rag=enable_rag,
                show_thinking=show_thinking
            ):
                # 处理不同类型的事件
                if isinstance(chunk, dict):
                    logger.debug(f"[{request_id}] 收到字典类型的chunk: {chunk}")
                    event_type = chunk.get("event")
                    event_data = chunk.get("data", {})
                    
                    if event_type == "thinking":
                        # 只有在启用思考显示时才发送
                        if show_thinking:
                            yield format_sse_event(json.dumps({"type": "thinking", "data": event_data}))
                    elif event_type == "token":
                        token = event_data.get("token", "")
                        improved_response_chunks.append(token)
                        yield format_sse_event(json.dumps({"type": "token", "token": token}))
                    elif event_type == "reference":
                        # 引用信息直接传递
                        yield format_sse_event(json.dumps({"type": "reference", "data": event_data}))
                elif isinstance(chunk, str):
                    logger.debug(f"[{request_id}] 收到字符串类型的chunk: {chunk[:50]}...")
                    improved_response_chunks.append(chunk)
                    yield format_sse_event(json.dumps({"type": "token", "token": chunk}))
                else:
                    logger.warning(f"[{request_id}] 收到未知类型的chunk: {type(chunk)}")
            
            improved_response = "".join(improved_response_chunks)
            
            # 存储改进的响应
            await redis_client.hset(request_key, "improved_response", improved_response)
            
            # 将改进的响应添加到短期记忆
            await memory_manager.add_message(
                session_id=session_id,
                user_id=user_id,
                role="assistant",
                content=improved_response,
                role_id=role_id,
                message_id=f"improved-{message_id}"
            )
            
            # 添加系统消息，标记这是改进后的回复
            await memory_manager.add_message(
                session_id=session_id,
                user_id=user_id,
                role="system",
                content="上面是改进后的回复",
                message_id=f"system-{message_id}"
            )
            
            # 发送完成事件
            yield format_sse_event(json.dumps({"type": "completion", "message": "已生成改进的回复", "improved_response": improved_response}))
            
        except Exception as e:
            error_msg = f"生成改进响应时出错: {str(e)}"
            error_trace = traceback.format_exc()
            logger.error(f"[{request_id}] {error_msg}\n{error_trace}")
            yield format_sse_event(json.dumps({"type": "error", "message": error_msg}))
        
    except Exception as e:
        error_msg = f"处理反馈时出错: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"[{request_id}] {error_msg}")
        yield format_sse_event(json.dumps({"type": "error", "message": f"服务器错误: {str(e)}"}))
    
    finally:
        # 确保锁被释放，即使在错误发生时
        if lock:
            try:
                await lock.release()
                logger.debug(f"[{request_id}] 成功释放锁: {lock_name}")
            except Exception as e:
                logger.error(f"[{request_id}] 释放锁失败: {str(e)}")
                # 尝试强制删除锁
                if redis_client and lock_name:
                    try:
                        await redis_client.delete(lock_name)
                        logger.debug(f"[{request_id}] 成功强制删除锁: {lock_name}")
                    except Exception as e2:
                        logger.error(f"[{request_id}] 强制删除锁失败: {str(e2)}")
        
        logger.info(f"[{request_id}] 完成反馈处理: accepted={accepted}")

def format_sse_event(data):
    """格式化服务器发送事件(SSE)数据"""
    return f"data: {data}\n\n" 