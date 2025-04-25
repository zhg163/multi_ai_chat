"""
Two-Phase API Routes - 实现基于RAG增强的两阶段聊天API

第一阶段：基于用户输入生成响应并选择合适的角色
第二阶段：处理对生成响应的反馈并生成改进的回复
"""

from fastapi import APIRouter, HTTPException, Body, Depends, Query
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional, List, AsyncGenerator
import logging
import uuid
import json
import time
import traceback
from datetime import datetime

from app.services.redis_manager import get_redis_client
from app.services.custom_session_service import CustomSessionService
from app.services.rag_enhanced_service import RAGEnhancedService
from app.utils.redis_lock import RedisLock, obtain_lock
from app.memory.memory_manager import get_memory_manager
from app.services.llm_service import LLMService

router = APIRouter(
    prefix="/api/two-phase",
    tags=["two-phase"]
)

logger = logging.getLogger(__name__)

# 创建RAGEnhancedService依赖
def get_rag_service() -> RAGEnhancedService:
    return RAGEnhancedService()

# 初始化服务实例 - 将懒加载
_rag_service = None

async def get_initialized_rag_service() -> RAGEnhancedService:
    """获取已初始化的RAG服务实例"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGEnhancedService()
        await _rag_service.initialize()
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
    stream: bool = Query(False, description="是否使用流式响应")
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
    
    返回:
        如果stream=False:
            message_id: 此消息的唯一标识符
            response: 生成的响应
            selected_role: 回复的角色名称
            match_score: 匹配分数
            match_reason: 匹配原因
            references: 如果启用RAG，返回的相关参考资料
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
                role_id=role_id
            ),
            media_type="text/event-stream"
        )
    
    # 非流式处理，将查询参数转换为字典，复用POST方法处理逻辑
    data = {
        "session_id": session_id,
        "message": message,
        "user_id": user_id,
        "enable_rag": enable_rag,
        "auto_role_match": auto_role_match
    }
    
    # 如果提供了role_id，添加到数据中
    if role_id:
        data["role_id"] = role_id
        
    # 调用POST方法处理函数
    return await generate_response(data)

@router.post("/generate", response_model=Dict[str, Any])
async def generate_response(
    data: Dict[str, Any] = Body(...),
    stream: bool = Query(False, description="是否使用流式响应")
):
    """
    第一阶段API：基于用户消息生成响应
    
    参数:
        session_id: 会话标识符
        message: 用户消息内容
        user_id: 用户ID（可选）
        enable_rag: 是否启用RAG增强（可选，默认为True）
        auto_role_match: 是否启用自动角色匹配（可选，默认为True）
        stream: 是否使用流式响应（查询参数，默认为False）
    
    返回:
        如果stream=False:
            message_id: 此消息的唯一标识符
            response: 生成的响应
            selected_role: 回复的角色名称
            match_score: 匹配分数
            match_reason: 匹配原因
            references: 如果启用RAG，返回的相关参考资料
        如果stream=True:
            返回SSE格式的数据流
    """
    # 如果请求流式响应
    if stream:
        return StreamingResponse(
            streaming_response_generator(
                session_id=data.get("session_id"),
                message=data.get("message"),
                user_id=data.get("user_id", "anonymous"),
                enable_rag=data.get("enable_rag", True),
                auto_role_match=data.get("auto_role_match", True),
                role_id=data.get("role_id")
            ),
            media_type="text/event-stream"
        )
    
    session_id = data.get("session_id")
    message = data.get("message")
    user_id = data.get("user_id", "anonymous")
    enable_rag = data.get("enable_rag", True)
    auto_role_match = data.get("auto_role_match", True)
    role_id = data.get("role_id")  # 可选，指定角色ID
    
    if not session_id or not message:
        raise HTTPException(status_code=400, detail="缺少必须的参数")
    
    try:
        # 1. 生成唯一消息ID
        message_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # 2. 获取Redis客户端
        redis_client = await get_redis_client()
        
        # 3. 获取记忆管理器
        memory_manager = await get_memory_manager()
        
        # 4. 将用户消息添加到短期记忆
        await memory_manager.add_message(
            session_id=session_id,
            user_id=user_id,
            role="user",
            content=message,
            message_id=f"user-{message_id}"
        )
        
        # 5. 获取会话锁，防止并发处理
        lock_name = f"two_phase:lock:{session_id}"
        lock = await obtain_lock(redis_client, lock_name, expire_seconds=60)
        
        if lock is None:
            raise HTTPException(
                status_code=409, 
                detail="会话正忙，请稍后再试"
            )
        
        try:
            # 6. 存储请求信息到Redis
            request_key = f"two_phase:request:{message_id}"
            
            await redis_client.hset(request_key, mapping={
                "session_id": session_id,
                "user_id": user_id,
                "message": message,
                "status": "pending",
                "created_at": timestamp,
                "enable_rag": str(enable_rag),
                "auto_role_match": str(auto_role_match)
            })
            await redis_client.expire(request_key, 3600)  # 1小时过期
            
            # 7. 获取会话信息
            session = await CustomSessionService.get_session(session_id)
            
            if not session or "roles" not in session:
                raise HTTPException(status_code=404, detail="未找到会话或会话中没有角色")
            
            # 获取会话中的角色列表
            available_roles = session.get("roles", [])
            if not available_roles:
                raise HTTPException(status_code=400, detail="会话中没有可用角色")
            
            # 8. 获取RAG服务并准备消息
            rag_service = await get_initialized_rag_service()
            messages = [{"role": "user", "content": message}]
            
            # 9. 角色匹配
            match_result = None
            role_info = None  # 用于存储角色完整信息
            match_score = 0.0
            match_reason = ""
            
            try:
                if auto_role_match:
                    match_result = await rag_service.match_role_for_chat(
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
            
            # 10. 如果没有匹配到角色，使用第一个可用角色
            if not role_id:
                first_role = available_roles[0]
                role_id = first_role.get("role_id") or first_role.get("id") or first_role.get("_id")
                role_info = first_role  # 直接使用available_roles中的完整角色对象
                logger.info(f"使用默认角色: {role_id}")
            
            # 11. 如果上面的步骤没有获取到完整角色信息，则在available_roles中查找
            if not role_info:
                for role in available_roles:
                    role_id_in_list = role.get("role_id") or role.get("id") or str(role.get("_id"))
                    if role_id_in_list == role_id:
                        role_info = role
                        logger.info(f"在会话角色中找到匹配的角色信息: {role_id}")
                        break
            
            # 只有在前面的步骤都没有获取到角色信息时，才查询数据库
            if not role_info:
                try:
                    role_info = await rag_service.get_role_info(role_id)
                    if not role_info:
                        # 如果找不到角色，尝试使用第一个可用角色
                        logger.warning(f"未找到角色: {role_id}，尝试使用替代角色")
                        if available_roles:
                            alternative_role = available_roles[0]
                            alternative_role_id = alternative_role.get("role_id") or alternative_role.get("id") or alternative_role.get("_id")
                            if alternative_role_id != role_id:  # 确保不是同一个角色
                                role_id = alternative_role_id
                                role_info = await rag_service.get_role_info(role_id)
                                logger.info(f"使用替代角色: {role_id}")
                            # 如果替代角色也查询不到，但有原始角色对象，直接使用
                            elif not role_info and "name" in alternative_role:
                                role_info = alternative_role
                                logger.info(f"使用原始角色对象: {role_id}")
                    
                    # 如果仍然找不到角色信息，创建一个默认角色信息
                    if not role_info:
                        logger.warning(f"未能找到有效角色，使用默认角色信息")
                        role_info = {
                            "id": role_id,
                            "name": "未知角色",
                            "description": "此角色信息不可用",
                            "system_prompt": "你是一个助手，请根据用户的问题提供有用的回答。"
                        }
                except Exception as e:
                    logger.error(f"获取角色信息时出错: {str(e)}")
                    # 创建一个默认角色信息
                    role_info = {
                        "id": role_id or "default",
                        "name": "默认助手",
                        "description": "无法获取角色信息时的默认助手",
                        "system_prompt": "你是一个助手，请根据用户的问题提供有用的回答。"
                    }
            
            # 12. 生成响应
            full_response = ""
            references = []
            
            try:
                async for chunk in rag_service.generate_response(
                    messages=messages,
                    model="default",
                    session_id=session_id,
                    user_id=user_id,
                    role_id=role_id,
                    role_info=role_info,
                    stream=False
                ):
                    if isinstance(chunk, dict) and "references" in chunk:
                        references = chunk.get("references", [])
                    elif isinstance(chunk, str):
                        full_response += chunk
            except Exception as e:
                logger.error(f"生成响应时出错: {str(e)}")
                # 提供一个默认响应
                full_response = f"非常抱歉，我在处理您的请求时遇到了问题。错误信息: {str(e)[:100]}..."
            
            # 13. 将响应添加到短期记忆
            await memory_manager.add_message(
                session_id=session_id,
                user_id=user_id,
                role="assistant",
                content=full_response,
                role_id=role_id,
                message_id=f"assistant-{message_id}"
            )
            
            # 14. 存储完整响应到Redis
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
            
            # 15. 添加到会话历史
            history_key = f"two_phase:history:{session_id}"
            await redis_client.rpush(history_key, message_id)
            await redis_client.expire(history_key, 86400 * 7)  # 7天过期
            
            # 16. 返回响应
            return {
                "message_id": message_id,
                "response": full_response,
                "selected_role": role_info.get("name") or role_info.get("role_name", "Unknown"),
                "match_score": match_score,
                "match_reason": match_reason,
                "references": references if enable_rag and references else []
            }
        finally:
            # 释放锁
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
                is_accepted=is_accepted,
                user_id=user_id
            ),
            media_type="text/event-stream"
        )
    
    # 非流式处理，将查询参数转换为字典，复用POST方法处理逻辑
    data = {
        "session_id": session_id,
        "message_id": message_id,
        "is_accepted": is_accepted,
        "user_id": user_id
    }
    
    # 调用POST方法处理函数
    return await provide_feedback(data)

@router.post("/feedback", response_model=Dict[str, Any])
async def provide_feedback(
    data: Dict[str, Any] = Body(...),
    stream: bool = Query(False, description="是否使用流式响应")
):
    """
    第二阶段API：对生成的响应提供反馈
    
    参数:
        session_id: 会话标识符
        message_id: 第一阶段生成的消息ID
        is_accepted: 响应是否被接受
        user_id: 用户ID（可选）
        stream: 是否使用流式响应（查询参数，默认为False）
    
    返回:
        如果stream=False:
            success: 反馈是否处理成功
            is_accepted: 提供的反馈值
            improved_response: 如果拒绝，返回改进的响应
        如果stream=True:
            返回SSE格式的改进响应数据流
    """
    # 如果请求流式响应
    if stream:
        return StreamingResponse(
            streaming_feedback_generator(
                session_id=data.get("session_id"),
                message_id=data.get("message_id"),
                is_accepted=data.get("is_accepted", False),
                user_id=data.get("user_id", "anonymous")
            ),
            media_type="text/event-stream"
        )
    
    session_id = data.get("session_id")
    message_id = data.get("message_id")
    is_accepted = data.get("is_accepted", False)
    user_id = data.get("user_id", "anonymous")
    
    if not session_id or not message_id:
        raise HTTPException(status_code=400, detail="缺少必须的参数")
    
    try:
        # 1. 获取Redis客户端和记忆管理器
        redis_client = await get_redis_client()
        memory_manager = await get_memory_manager()
        
        # 2. 获取存储的请求/响应数据
        request_key = f"two_phase:request:{message_id}"
        
        stored_data = await redis_client.hgetall(request_key)
        if not stored_data:
            raise HTTPException(status_code=404, detail=f"未找到请求: {message_id}")
        
        # 3. 验证消息属于指定会话
        if stored_data.get("session_id") != session_id:
            raise HTTPException(status_code=403, detail="会话ID不匹配")
        
        # 4. 获取会话锁，防止并发处理
        lock_name = f"two_phase:lock:{message_id}"
        lock = await obtain_lock(redis_client, lock_name, expire_seconds=30)
        
        if lock is None:
            raise HTTPException(
                status_code=409, 
                detail="消息正在处理中，请稍后再试"
            )
        
        try:
            # 5. 更新反馈状态
            await redis_client.hset(request_key, mapping={
                "feedback": "accepted" if is_accepted else "rejected",
                "feedback_time": datetime.utcnow().isoformat()
            })
            
            improved_response = None
            
            # 6. 如果拒绝，生成改进的响应
            if not is_accepted:
                # 获取角色信息
                role_id = stored_data.get("role_id")
                
                # 获取原始消息和响应
                original_message = stored_data.get("message", "")
                original_response = stored_data.get("response", "")
                
                # 获取RAG服务
                rag_service = await get_initialized_rag_service()
                
                # 是否启用RAG
                enable_rag = stored_data.get("enable_rag", "True").lower() == "true"
                
                # 构建改进提示
                improvement_prompt = f"""你之前的回复被用户拒绝，需要改进。

原始用户消息: "{original_message}"

你之前的回复: "{original_response}"

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
                
                try:
                    # 使用RAG服务生成改进响应
                    async for chunk in rag_service.generate_response(
                        messages=messages,
                        model="default",
                        session_id=session_id,
                        user_id=user_id,
                        role_id=role_id,
                        role_info=role_info,
                        stream=False
                    ):
                        if isinstance(chunk, str):
                            improved_response_chunks.append(chunk)
                    
                    improved_response = "".join(improved_response_chunks)
                except Exception as e:
                    logger.error(f"生成改进响应时出错: {str(e)}")
                    improved_response = "抱歉，我无法生成改进的回复。请稍后再试。"
                
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
            
            return {
                "success": True,
                "is_accepted": is_accepted,
                "improved_response": improved_response
            }
        finally:
            # 释放锁
            await lock.release()
            
    except HTTPException:
        raise
    except Exception as e:
        error_detail = f"处理反馈时出错: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

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
    role_id: Optional[str]
) -> AsyncGenerator[str, None]:
    """生成流式响应
    
    生成符合SSE规范的数据流，分阶段返回匹配信息和生成内容
    
    Args:
        session_id: 会话ID
        message: 用户消息
        user_id: 用户ID
        enable_rag: 是否启用RAG
        auto_role_match: 是否自动匹配角色
        role_id: 指定角色ID（可选）
        
    Yields:
        SSE格式的数据流
    """
    try:
        # 1. 生成唯一消息ID
        message_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # 2. 获取Redis客户端和记忆管理器
        redis_client = await get_redis_client()
        memory_manager = await get_memory_manager()
        
        # 3. 将用户消息添加到短期记忆
        await memory_manager.add_message(
            session_id=session_id,
            user_id=user_id,
            role="user",
            content=message,
            message_id=f"user-{message_id}"
        )
        
        # 4. 获取会话锁，防止并发处理
        lock_name = f"two_phase:lock:{session_id}"
        lock = await obtain_lock(redis_client, lock_name, expire_seconds=60)
        
        if lock is None:
            error_data = {
                "type": "error",
                "message": "会话正忙，请稍后再试"
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            return
        
        try:
            # 5. 存储请求信息到Redis
            request_key = f"two_phase:request:{message_id}"
            
            await redis_client.hset(request_key, mapping={
                "session_id": session_id,
                "user_id": user_id,
                "message": message,
                "status": "pending",
                "created_at": timestamp,
                "enable_rag": str(enable_rag),
                "auto_role_match": str(auto_role_match)
            })
            await redis_client.expire(request_key, 3600)  # 1小时过期
            
            # 6. 获取会话信息
            session = await CustomSessionService.get_session(session_id)
            
            if not session or "roles" not in session:
                error_data = {
                    "type": "error",
                    "message": "未找到会话或会话中没有角色"
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                return
            
            # 获取会话中的角色列表
            available_roles = session.get("roles", [])
            if not available_roles:
                error_data = {
                    "type": "error",
                    "message": "会话中没有可用角色"
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                return
            
            # 7. 获取RAG服务并准备消息
            rag_service = await get_initialized_rag_service()
            messages = [{"role": "user", "content": message}]
            
            # 8. 角色匹配
            match_result = None
            role_info = None
            match_score = 0.0
            match_reason = ""
            
            try:
                if auto_role_match:
                    match_result = await rag_service.match_role_for_chat(
                        messages=messages,
                        session_id=session_id,
                        user_id=user_id
                    )
                    
                    # 从匹配结果中提取角色信息
                    if match_result and match_result.get("success"):
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
            
            # 10. 如果上面的步骤没有获取到完整角色信息，则在available_roles中查找
            if not role_info:
                for role in available_roles:
                    role_id_in_list = role.get("role_id") or role.get("id") or str(role.get("_id"))
                    if role_id_in_list == role_id:
                        role_info = role
                        logger.info(f"在会话角色中找到匹配的角色信息: {role_id}")
                        break
            
            # 只有在前面的步骤都没有获取到角色信息时，才查询数据库
            if not role_info:
                try:
                    role_info = await rag_service.get_role_info(role_id)
                    if not role_info:
                        # 如果找不到角色，尝试使用第一个可用角色
                        logger.warning(f"未找到角色: {role_id}，尝试使用替代角色")
                        if available_roles:
                            alternative_role = available_roles[0]
                            alternative_role_id = alternative_role.get("role_id") or alternative_role.get("id") or alternative_role.get("_id")
                            if alternative_role_id != role_id:  # 确保不是同一个角色
                                role_id = alternative_role_id
                                role_info = await rag_service.get_role_info(role_id)
                                logger.info(f"使用替代角色: {role_id}")
                            # 如果替代角色也查询不到，但有原始角色对象，直接使用
                            elif not role_info and "name" in alternative_role:
                                role_info = alternative_role
                                logger.info(f"使用原始角色对象: {role_id}")
                    
                    # 如果仍然找不到角色信息，创建一个默认角色信息
                    if not role_info:
                        logger.warning(f"未能找到有效角色，使用默认角色信息")
                        role_info = {
                            "id": role_id,
                            "name": "未知角色",
                            "description": "此角色信息不可用",
                            "system_prompt": "你是一个助手，请根据用户的问题提供有用的回答。"
                        }
                except Exception as e:
                    logger.error(f"获取角色信息时出错: {str(e)}")
                    # 创建一个默认角色信息
                    role_info = {
                        "id": role_id or "default",
                        "name": "默认助手",
                        "description": "无法获取角色信息时的默认助手",
                        "system_prompt": "你是一个助手，请根据用户的问题提供有用的回答。"
                    }
            
            # 11. 发送初始匹配信息
            role_name = role_info.get("name") or role_info.get("role_name", "未知角色")
            
            initial_data = {
                "type": "match_info",
                "message_id": message_id,
                "selected_role": role_name,
                "match_score": match_score,
                "match_reason": match_reason
            }
            yield f"data: {json.dumps(initial_data)}\n\n"
            
            # 12. 生成响应（流式）
            response_buffer = ""
            references = []
            
            try:
                async for chunk in rag_service.generate_response(
                    messages=messages,
                    model="default",
                    session_id=session_id,
                    user_id=user_id,
                    role_id=role_id,
                    role_info=role_info,
                    stream=True  # 确保使用流式模式
                ):
                    if isinstance(chunk, dict) and "references" in chunk:
                        references = chunk.get("references", [])
                        # 发送引用信息
                        ref_data = {
                            "type": "references",
                            "references": references
                        }
                        yield f"data: {json.dumps(ref_data)}\n\n"
                    elif isinstance(chunk, str):
                        # 将响应片段添加到缓冲区
                        response_buffer += chunk
                        
                        # 发送响应片段
                        chunk_data = {
                            "type": "content",
                            "content": chunk
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                        
                        # 每积累一定长度就更新Redis中的响应
                        if len(response_buffer) % 100 == 0:
                            await redis_client.hset(
                                request_key,
                                "partial_response", 
                                response_buffer
                            )
            except Exception as e:
                logger.error(f"生成响应时出错: {str(e)}")
                # 发送错误信息
                error_data = {
                    "type": "error",
                    "message": f"生成响应时出错: {str(e)[:100]}..."
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                
                # 如果出错但已有部分响应，仍然使用它
                if not response_buffer:
                    response_buffer = f"非常抱歉，我在处理您的请求时遇到了问题。错误信息: {str(e)[:100]}..."
            
            # 13. 将响应添加到短期记忆
            await memory_manager.add_message(
                session_id=session_id,
                user_id=user_id,
                role="assistant",
                content=response_buffer,
                role_id=role_id,
                message_id=f"assistant-{message_id}"
            )
            
            # 14. 存储完整响应到Redis
            match_score_str = str(match_score)
            match_reason_str = match_reason
            
            await redis_client.hset(request_key, mapping={
                "response": response_buffer,
                "selected_role": json.dumps(role_info),
                "role_id": role_id,
                "match_score": match_score_str,
                "match_reason": match_reason_str,
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat()
            })
            
            # 15. 添加到会话历史
            history_key = f"two_phase:history:{session_id}"
            await redis_client.rpush(history_key, message_id)
            await redis_client.expire(history_key, 86400 * 7)  # 7天过期
            
            # 16. 发送完成信号
            complete_data = {
                "type": "complete",
                "message_id": message_id
            }
            yield f"data: {json.dumps(complete_data)}\n\n"
            
        finally:
            # 释放锁
            await lock.release()
            
    except Exception as e:
        error_detail = f"生成响应时出错: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        
        error_data = {
            "type": "error",
            "message": error_detail[:200] + "..." if len(error_detail) > 200 else error_detail
        }
        yield f"data: {json.dumps(error_data)}\n\n"

async def streaming_feedback_generator(
    session_id: str,
    message_id: str,
    is_accepted: bool,
    user_id: str
) -> AsyncGenerator[str, None]:
    """生成流式反馈响应
    
    生成符合SSE规范的改进响应数据流
    
    Args:
        session_id: 会话ID
        message_id: 消息ID
        is_accepted: 是否接受原始响应
        user_id: 用户ID
        
    Yields:
        SSE格式的数据流
    """
    try:
        # 1. 获取Redis客户端和记忆管理器
        redis_client = await get_redis_client()
        memory_manager = await get_memory_manager()
        
        # 2. 获取存储的请求/响应数据
        request_key = f"two_phase:request:{message_id}"
        
        stored_data = await redis_client.hgetall(request_key)
        if not stored_data:
            error_data = {
                "type": "error",
                "message": f"未找到请求: {message_id}"
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            return
        
        # 3. 验证消息属于指定会话
        if stored_data.get("session_id") != session_id:
            error_data = {
                "type": "error",
                "message": "会话ID不匹配"
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            return
        
        # 4. 获取会话锁，防止并发处理
        lock_name = f"two_phase:lock:{message_id}"
        lock = await obtain_lock(redis_client, lock_name, expire_seconds=30)
        
        if lock is None:
            error_data = {
                "type": "error",
                "message": "消息正在处理中，请稍后再试"
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            return
        
        try:
            # 5. 更新反馈状态
            await redis_client.hset(request_key, mapping={
                "feedback": "accepted" if is_accepted else "rejected",
                "feedback_time": datetime.utcnow().isoformat()
            })
            
            # 6. 如果接受，发送简单的确认信息并结束
            if is_accepted:
                accepted_data = {
                    "type": "feedback_info",
                    "message_id": message_id,
                    "status": "accepted"
                }
                yield f"data: {json.dumps(accepted_data)}\n\n"
                
                complete_data = {
                    "type": "complete",
                    "message_id": message_id
                }
                yield f"data: {json.dumps(complete_data)}\n\n"
                return
            
            # 7. 如果拒绝，生成改进的响应
            # 获取角色信息
            role_id = stored_data.get("role_id")
            selected_role_json = stored_data.get("selected_role", "{}")
            
            try:
                selected_role = json.loads(selected_role_json)
            except json.JSONDecodeError:
                selected_role = {"name": "未知角色", "system_prompt": "你是一个助手，请回答用户的问题。"}
            
            # 获取原始消息和响应
            original_message = stored_data.get("message", "")
            original_response = stored_data.get("response", "")
            
            # 发送初始反馈信息
            initial_data = {
                "type": "feedback_info",
                "message_id": message_id,
                "status": "improving"
            }
            yield f"data: {json.dumps(initial_data)}\n\n"
            
            # 获取RAG服务
            rag_service = await get_initialized_rag_service()
            
            # 构建改进提示
            improvement_prompt = f"""你之前的回复被用户拒绝，需要改进。

原始用户消息: "{original_message}"

你之前的回复: "{original_response}"

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
            improved_buffer = ""
            
            try:
                # 使用RAG服务生成改进响应（流式）
                async for chunk in rag_service.generate_response(
                    messages=messages,
                    model="default",
                    session_id=session_id,
                    user_id=user_id,
                    role_id=role_id,
                    role_info=selected_role,
                    stream=True
                ):
                    if isinstance(chunk, str):
                        # 将响应片段添加到缓冲区
                        improved_buffer += chunk
                        
                        # 发送响应片段
                        chunk_data = {
                            "type": "content",
                            "content": chunk
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                        
                        # 每积累一定长度就更新Redis中的响应
                        if len(improved_buffer) % 100 == 0:
                            await redis_client.hset(
                                request_key,
                                "partial_improved_response", 
                                improved_buffer
                            )
            except Exception as e:
                logger.error(f"生成改进响应时出错: {str(e)}")
                error_data = {
                    "type": "error",
                    "message": f"生成改进响应时出错: {str(e)}"
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                
                # 如果出错但还没有响应，提供默认响应
                if not improved_buffer:
                    improved_buffer = "抱歉，我无法生成改进的回复。请稍后再试。"
            
            # 存储改进的响应
            await redis_client.hset(request_key, "improved_response", improved_buffer)
            
            # 将改进的响应添加到短期记忆
            await memory_manager.add_message(
                session_id=session_id,
                user_id=user_id,
                role="assistant",
                content=improved_buffer,
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
            
            # 发送完成信号
            complete_data = {
                "type": "complete",
                "message_id": message_id
            }
            yield f"data: {json.dumps(complete_data)}\n\n"
            
        finally:
            # 释放锁
            await lock.release()
            
    except Exception as e:
        error_detail = f"处理反馈时出错: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        
        error_data = {
            "type": "error",
            "message": error_detail[:200] + "..." if len(error_detail) > 200 else error_detail
        }
        yield f"data: {json.dumps(error_data)}\n\n" 