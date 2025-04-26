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
import httpx
import os  # 导入os模块

from app.services.redis_manager import get_redis_client
from app.services.custom_session_service import CustomSessionService
from app.services.rag_enhanced_service import RAGEnhancedService
from app.utils.redis_lock import RedisLock, obtain_lock
from app.memory.memory_manager import get_memory_manager
from app.services.llm_service import LLMService

router = APIRouter(
    prefix="/api/two-phase-streamrag",
    tags=["two-phase-streamrag"]
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
    stream: bool = Query(False, description="是否使用流式响应"),
    show_thinking: bool = Query(True, description="是否显示思考过程")
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
                show_thinking=show_thinking
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
        user_id: 用户ID（可选）
        enable_rag: 是否启用RAG增强（可选，默认为True）
        auto_role_match: 是否启用自动角色匹配（可选，默认为True）
        show_thinking: 是否显示思考过程（可选，默认为True）
        stream: 是否使用流式响应（查询参数，默认为False）
    
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
                show_thinking=show_thinking_value
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
        # 1. 获取RAG服务和内存管理器
        rag_service = await get_initialized_rag_service()
        memory_manager = await get_memory_manager()
        
        # 2. 获取会话锁，防止并发处理同一会话
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
                "enable_rag": str(enable_rag)
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
                            alternative_role_id = alternative_role.get("role_id") or alternative_role.get("id") or str(alternative_role.get("_id"))
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
            
            # 11. 生成响应
            full_response = ""
            references = []
            thinking_content = []
            
            try:
                # 使用异步for循环正确处理generate_response返回的异步生成器
                response_gen = rag_service.generate_response(
                    messages=messages,
                    model="default",
                    session_id=session_id,
                    user_id=user_id,
                    role_id=role_id,
                    role_info=role_info,
                    stream=False
                )
                
                async for chunk in response_gen:
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
                "thinking_process": []  # generate_response不提供思考内容
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
                
                # 获取存储的角色信息
                try:
                    selected_role_json = stored_data.get("selected_role", "{}")
                    selected_role = json.loads(selected_role_json)
                except json.JSONDecodeError:
                    # 如果解析失败，使用默认角色信息
                    selected_role = {
                        "id": role_id,
                        "name": "未知角色",
                        "description": "无法获取角色信息",
                        "system_prompt": "你是一个助手，请回答用户的问题。"
                    }
                
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
                        role_info=selected_role,
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
    role_id: Optional[str],
    show_thinking: bool = True  # 新增参数控制是否显示思考过程
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
        show_thinking: 是否显示思考过程（可选，默认为True）
        
    Yields:
        SSE格式的数据流
    """
    try:
        # 获取API密钥和端点
        retrieval_api_key = os.getenv("RETRIEVAL_API_KEY", "")
        retrieval_endpoint = os.getenv("RETRIEVAL_ENDPOINT", "http://localhost:9222/api/v1/chats_openai/ragflow-default/chat/completions")
        
        # 获取默认提供商和模型名称
        provider = os.getenv("LLM_PROVIDER", "deepseek")
        model_name = os.getenv("LLM_MODEL", "deepseek-chat") 
        model = model_name  # 设置model变量与model_name一致
        
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
                "auto_role_match": str(auto_role_match),
                "show_thinking": str(show_thinking)
            })
            await redis_client.expire(request_key, 3600)  # 1小时过期
            
            # 6. 获取会话信息和会话绑定的角色
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
                # 如果有指定角色ID，直接使用
                if not auto_role_match and role_id:
                    logger.info(f"使用指定角色ID: {role_id}")
                else:
                    # 从可用角色中获取第一个作为备用
                    fallback_role = available_roles[0]
                    fallback_role_id = fallback_role.get("role_id") or fallback_role.get("id") or str(fallback_role.get("_id"))
                    
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
                            alternative_role_id = alternative_role.get("role_id") or alternative_role.get("id") or str(alternative_role.get("_id"))
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
            
            # 12. 设置缓冲区
            response_buffer = ""  # 存储最终响应内容
            thinking_buffer = ""  # 存储思考过程内容
            references = []       # 存储参考文档
            
            # 13. 生成响应（流式）
            try:
                # 构造请求头
                headers = {}
                if retrieval_api_key.startswith("sk"):
                    headers["Authorization"] = f"Bearer {retrieval_api_key}"
                else:
                    headers["Api-Key"] = retrieval_api_key
                
                # 打印API KEY前缀用于调试
                mask_key = retrieval_api_key[:8] + "..." if len(retrieval_api_key) > 10 else "未设置密钥"
                logger.debug(f"使用RAG API密钥前缀: {mask_key}")
                logger.debug(f"RAG请求URL: {retrieval_endpoint}")
                
                # 构造请求体
                payload = {
                    "messages": [{"role": "user", "content": message}],
                    "stream": False
                }
                
                # 发送请求
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(retrieval_endpoint, json=payload, headers=headers)
                    
                    # 使用辅助函数处理响应
                    rag_data = await handle_rag_response(response, headers, role_id, role_name)
                    
                    # 检查处理结果
                    if rag_data is None:
                        # 处理失败，降级为直接LLM调用
                        logger.warning("RAG服务响应处理失败，降级为直接LLM调用")
                        
                        # 通知前端RAG调用失败
                        yield f"data: {json.dumps({
                            'type': 'thinking_error',
                            'error': 'RAG服务调用失败，将使用模型直接回答'
                        })}\n\n"
                        
                        # 执行直接生成
                        async for chunk in rag_service.generate_response(
                            messages=messages,
                            model=model,
                            session_id=session_id,
                            role_id=role_id,
                            stream=True,
                            provider=provider,
                            model_name=model_name,
                            role_info=role_info
                        ):
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
                    else:
                        # RAG响应处理成功，正常处理响应数据
                        # ... 原有代码处理rag_data ...
                        pass
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
            
            # 获取存储的角色信息
            try:
                selected_role_json = stored_data.get("selected_role", "{}")
                selected_role = json.loads(selected_role_json)
            except json.JSONDecodeError:
                # 如果解析失败，使用默认角色信息
                selected_role = {
                    "id": role_id,
                    "name": "未知角色",
                    "description": "无法获取角色信息",
                    "system_prompt": "你是一个助手，请回答用户的问题。"
                }
            
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
                generate_response_gen = rag_service.generate_response(
                    messages=messages,
                    model="default",
                    session_id=session_id,
                    user_id=user_id,
                    role_id=role_id,
                    role_info=selected_role,
                    stream=True
                )
                
                # 使用异步for循环正确处理generate_response返回的异步生成器
                async for chunk in generate_response_gen:
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

async def handle_rag_response(response, headers, role_id, role_name):
    """处理RAG服务响应，增加错误处理"""
    try:
        # 读取并解析响应
        response_data = await response.json()
        
        # 打印响应数据用于调试
        logger.debug(f"RAG服务响应: {response_data}")
        
        # 检查响应数据结构
        if not response_data:
            logger.error("RAG服务返回空响应")
            return None
        
        # 检查错误码
        if 'code' in response_data and response_data['code'] != 0:
            error_msg = response_data.get('message', '未知错误')
            logger.error(f"RAG服务返回错误: 代码={response_data['code']}, 消息={error_msg}")
            return None
        
        # 安全获取data字段
        data = response_data.get('data')
        if data is None:
            logger.error("RAG响应中data字段为null")
            return None
        
        # 处理响应数据...
        return data
    except json.JSONDecodeError as e:
        logger.error(f"解析RAG响应JSON出错: {e}")
        return None
    except Exception as e:
        logger.error(f"处理RAG响应出错: {e}")
        return None 