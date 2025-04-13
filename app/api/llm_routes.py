"""
LLM API路由模块

提供LLM服务的API接口，包括聊天和模型列表
"""

import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks, Query, Form
from fastapi.responses import StreamingResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
import asyncio
import json
import time

from pydantic import BaseModel, Field

from app.services.llm_service import (
    LLMService, LLMProvider, LLMConfig, 
    Message, MessageRole, PromptTemplate, LLMResponse, StreamResponse
)
from app.memory.memory_manager import get_memory_manager
from app.auth.auth_handler import get_current_user, get_current_user_optional
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

# 创建路由器，设置前缀为/api/llm
router = APIRouter(prefix="/api/llm", tags=["llm"])

# AI服务获取函数
def get_ai_service(provider=None):
    """
    获取AI服务实例，如果provider不为空，则使用特定的提供商
    否则使用默认服务
    """
    # 这里可以根据provider返回不同的AI服务实例
    # 目前简单返回llm_service
    return llm_service

async def prepare_chat_context(session_id: str, user_id: str, role_id: str = None) -> tuple:
    """
    准备聊天上下文，包括系统提示词和历史消息
    
    Args:
        session_id: 会话ID
        user_id: 用户ID
        role_id: 角色ID
        
    Returns:
        (prompt, history): 系统提示词和历史消息
    """
    try:
        # 获取记忆管理器
        memory_manager = await get_memory_manager()
        
        # 获取会话消息
        messages = memory_manager.short_term.get_session_messages(session_id, user_id)
        if not messages:
            logger.warning(f"会话为空: {session_id}")
            return "", []
            
        # 构建历史消息
        history = []
        for msg in messages:
            if msg["role"] == "system":
                continue  # 系统消息不包含在历史中
                
            history.append({
                "role": "user" if msg["role"] in ["user", "human"] else "assistant",
                "content": msg["content"]
            })
        
        # 如果有角色ID，获取角色系统提示词
        prompt = ""
        if role_id:
            try:
                from app.database.connection import get_database
                from bson.objectid import ObjectId
                
                db = await get_database()
                if db is not None:
                    role_info = await db.roles.find_one({"_id": ObjectId(role_id)})
                    if role_info and "system_prompt" in role_info:
                        prompt = role_info["system_prompt"]
                        logger.info(f"获取到角色系统提示词: {prompt[:50]}...")
            except Exception as e:
                logger.error(f"获取角色系统提示词失败: {str(e)}")
        
        return prompt, history
    except Exception as e:
        logger.error(f"准备聊天上下文失败: {str(e)}")
        return "", []

async def stream_response(session_id, user_id, message, role_id, provider, model, temperature, max_tokens):
    """
    生成流式响应
    
    Args:
        session_id: 会话ID
        user_id: 用户ID
        message: 用户消息
        role_id: 角色ID
        provider: 提供商
        model: 模型
        temperature: 温度
        max_tokens: 最大token数
        
    Yields:
        流式响应内容
    """
    try:
        # 获取角色系统提示词和对话历史
        prompt, history = await prepare_chat_context(session_id, user_id, role_id)
        
        # 调用AI服务生成流式回复
        ai_service = get_ai_service(provider)
        full_response = ""
        
        logger.info(f"开始生成流式回复: session_id={session_id}, role_id={role_id}")
        
        # 适配不同的流式生成方法
        if hasattr(ai_service, "generate_chat_stream"):
            # 如果有专门针对聊天的流式生成方法
            stream_generator = ai_service.generate_chat_stream(
                prompt=prompt,
                history=history,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif hasattr(ai_service, "generate_stream"):
            # 适配通用的流式生成方法，转换参数格式
            # 构建消息格式
            messages = []
            # 添加系统提示词
            if prompt:
                messages.append({"role": "system", "content": prompt})
            # 添加历史消息
            messages.extend(history)
            # 添加当前消息
            if message and not any(msg.get("content") == message and msg.get("role") == "user" for msg in messages):
                messages.append({"role": "user", "content": message})
            
            # 构建配置
            from app.services.llm_service import LLMConfig, LLMProvider
            config = None
            if model or provider:
                # 获取API密钥
                from app.services.llm_service import get_api_key
                api_key = None
                
                if provider:
                    try:
                        provider_enum = LLMProvider(provider)
                        api_key = get_api_key(provider_enum)
                    except (ValueError, ImportError):
                        logger.warning(f"无法获取提供商 {provider} 的API密钥")
                
                config = LLMConfig(
                    provider=provider_enum if provider else None,
                    model_name=model,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
            logger.info(f"调用generate_stream: messages={len(messages)}条, config={config is not None}")
            stream_generator = ai_service.generate_stream(
                messages=messages,
                config=config
            )
        else:
            # 如果没有流式生成方法，则抛出错误
            raise AttributeError("AI服务不支持流式生成，既没有generate_chat_stream也没有generate_stream方法")
        
        # 处理流式生成结果
        async for chunk in stream_generator:
            chunk_content = None
            
            # 处理不同类型的响应
            if isinstance(chunk, str):
                # 如果直接返回字符串
                chunk_content = chunk
            elif hasattr(chunk, 'content') and chunk.content:
                # 如果返回包含content属性的对象，确保content不是None
                chunk_content = chunk.content
            elif chunk and not (hasattr(chunk, 'is_start') or hasattr(chunk, 'is_end')):
                # 注意：只有在不是控制消息(is_start/is_end)的情况下才尝试字符串化
                # 避免将控制消息转换为字符串，如"StreamResponse(is_end=True)"
                try:
                    chunk_content = str(chunk)
                    # 检查并过滤掉包含元数据的内容
                    if any(x in chunk_content for x in ["content=None", "model=", "provider=", "tokens_used=", "finish_reason=", "is_start=", "is_end="]):
                        logger.warning(f"过滤掉包含元数据的内容: {chunk_content}")
                        chunk_content = None
                except:
                    chunk_content = None
            
            # 过滤掉[DONE]标记，避免它出现在响应内容中
            if chunk_content == "[DONE]":
                logger.info("跳过[DONE]标记，不添加到响应")
                chunk_content = None
            
            # 处理内容输出
            if chunk_content:
                # 追加到完整响应中
                full_response += chunk_content
                # 正确格式化为SSE事件标准格式：data: 内容\n\n
                yield f"data: {chunk_content}\n\n".encode('utf-8')
            # 处理控制消息
            elif hasattr(chunk, 'is_start') and chunk.is_start:
                # 开始事件，可以发送一个空的data事件表示开始
                yield b"data: \n\n"
            elif hasattr(chunk, 'is_end') and chunk.is_end:
                # 结束事件，使用特殊的事件类型而不是在数据中包含[DONE]标记
                # 这样前端可以识别流结束，但不会在聊天内容中显示[DONE]
                yield b"event: done\ndata: \n\n"
        
        # 保存完整响应到记忆
        if full_response:
            # 清理响应中可能存在的元数据痕迹
            # 移除任何包含元数据的部分
            import re
            cleaned_response = re.sub(r'content=None model=.*? is_end=(True|False)', '', full_response)
            cleaned_response = re.sub(r'model=.*? provider=.*? tokens_used=.*? finish_reason=.*?', '', cleaned_response)
            # 确保移除[DONE]标记
            cleaned_response = cleaned_response.replace('[DONE]', '').strip()
            
            memory_manager = await get_memory_manager()
            await memory_manager.add_message(session_id, user_id, "assistant", cleaned_response, role_id)
            logger.info(f"保存流式回复完成: session_id={session_id}, 长度={len(cleaned_response)}")
    except Exception as e:
        logger.error(f"生成流式回复失败: {str(e)}", exc_info=True)
        error_message = f"生成回复失败: {str(e)}"
        # 确保错误消息也按SSE格式返回
        yield f"data: {error_message}\n\n".encode('utf-8')
        # 表示流结束
        yield b"event: done\ndata: \n\n"

# 确保有LLM API密钥
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY", "")

if not DEEPSEEK_API_KEY and not ZHIPU_API_KEY:
    logger.warning("未配置LLM API密钥，服务可能不可用")

# 初始化LLM服务实例
try:
    # 尝试使用默认配置初始化服务
    llm_service = LLMService()
    
    # 验证服务是否具有generate_stream方法
    if not hasattr(llm_service, 'generate_stream'):
        logger.error("LLMService实例缺少generate_stream方法，正在重新加载")
        # 如果缺少方法，则重新导入模块并创建实例
        import importlib
        import sys
        # 确保重新加载所有依赖模块
        for module_name in list(sys.modules.keys()):
            if module_name.startswith('app.services.llm_service'):
                del sys.modules[module_name]
        
        try:
            # 重新导入模块
            llm_module = importlib.import_module('app.services.llm_service')
            # 确保导入了最新的代码
            importlib.reload(llm_module)
            # 重新创建实例
            llm_service = llm_module.LLMService()
            
            # 再次验证
            if not hasattr(llm_service, 'generate_stream'):
                raise AttributeError("重载后的LLMService实例仍然缺少generate_stream方法")
        except Exception as reload_error:
            logger.error(f"重新加载LLMService模块失败: {str(reload_error)}")
            # 直接添加临时实现
            from typing import List, Dict, Union, AsyncGenerator, Optional
            async def temp_generate_stream(messages, config=None):
                """临时实现的generate_stream方法"""
                # 确保我们使用正确的StreamResponse类
                from app.services.llm_service import StreamResponse
                
                yield StreamResponse(
                    is_start=True,
                    model="temp-model",
                    provider="temp-provider"
                )
                
                yield StreamResponse(
                    content="LLM服务正在重新配置中，请稍后再试。管理员已收到相关错误通知。",
                    model="temp-model",
                    provider="temp-provider"
                )
                
                yield StreamResponse(
                    is_end=True,
                    model="temp-model",
                    provider="temp-provider"
                )
            
            # 添加临时方法到实例
            import types
            llm_service.generate_stream = types.MethodType(temp_generate_stream, llm_service)
            logger.info("已添加临时generate_stream方法")
    
    logger.info("LLM服务初始化成功")
except Exception as e:
    logger.error(f"LLM服务初始化失败: {str(e)}")
    # 创建一个基本实例，后续将检查其可用性
    llm_service = None
    
    # 遇到错误时创建一个临时服务
    try:
        from app.services.llm_service import LLMService, StreamResponse
        llm_service = LLMService()
        
        # 定义临时方法
        from typing import List, Dict, Union, AsyncGenerator, Optional
        async def emergency_generate_stream(self, messages, config=None):
            """应急实现的generate_stream方法"""            
            yield StreamResponse(
                is_start=True,
                model="emergency-model",
                provider="emergency-provider"
            )
            
            yield StreamResponse(
                content="系统暂时无法连接到LLM服务，请稍后再试。",
                model="emergency-model",
                provider="emergency-provider"
            )
            
            yield StreamResponse(
                is_end=True,
                model="emergency-model",
                provider="emergency-provider"
            )
        
        # 动态添加方法
        import types
        llm_service.generate_stream = types.MethodType(emergency_generate_stream, llm_service)
        logger.info("已添加应急generate_stream方法")
    except Exception as fallback_error:
        logger.critical(f"创建应急LLM服务失败: {str(fallback_error)}")
        llm_service = None

# 请求模型
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    stream: bool = False
    model: Optional[str] = None
    provider: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    roleid: Optional[str] = None  # 添加roleid字段，用于指定角色ID

# 响应模型
class ChatResponse(BaseModel):
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None

class ModelInfo(BaseModel):
    name: str
    provider: str
    description: Optional[str] = None

class ModelsResponse(BaseModel):
    models: List[ModelInfo]

# 添加用于调试的响应模型
class LLMServiceInfoResponse(BaseModel):
    initialized: bool
    available_methods: List[str]
    available_attributes: List[str]
    service_class: str
    class_hierarchy: List[str]
    has_generate_stream: bool
    stream_method_source: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# 路由处理函数
@router.post("/chat")
async def chat(
    request: Request,
    background_tasks: BackgroundTasks,
    chat_request: Optional[ChatRequest] = None,
    provider: Optional[str] = Form(None, include_in_schema=False),
    model: Optional[str] = Form(None, include_in_schema=False),
    message: Optional[str] = Form(None, include_in_schema=False),
    roleid: Optional[str] = Form(None, include_in_schema=False),
    temperature: Optional[float] = Form(0.7, include_in_schema=False),
    max_tokens: Optional[int] = Form(None, include_in_schema=False),
    session_id: Optional[str] = Form(None, include_in_schema=False),
    stream: Optional[bool] = Form(False, include_in_schema=False),
    selected_username: Optional[str] = Form(None, include_in_schema=False),
    current_user: Optional[Dict] = Depends(get_current_user_optional)
):
    """聊天接口，返回AI回复，支持JSON和表单数据"""
    try:
        # 判断请求是JSON还是表单数据
        content_type = request.headers.get("content-type", "")
        
        # 记录请求信息
        logger.info(f"收到聊天请求: content-type={content_type}")
        
        # 处理JSON请求
        if "application/json" in content_type:
            if chat_request is None:
                # 尝试手动读取和解析请求体
                try:
                    body = await request.json()
                    provider = body.get("provider")
                    model = body.get("model")
                    
                    # 获取用户消息
                    messages = body.get("messages", [])
                    message = None
                    if messages:
                        for msg in reversed(messages):
                            if msg.get("role") == "user":
                                message = msg.get("content")
                                break
                    
                    roleid = body.get("roleid")
                    temperature = body.get("temperature", 0.7)
                    max_tokens = body.get("max_tokens")
                    session_id = body.get("session_id")
                    stream = body.get("stream", False)
                    selected_username = body.get("selected_username")
                    
                    # 获取用户ID - 优先使用current_user
                    user_id = current_user.get("id", "anonymous_user") if current_user else "anonymous_user"
                    
                    logger.info(f"手动解析JSON: provider={provider}, model={model}, roleid={roleid}, stream={stream}, user_id={user_id}")
                except Exception as parse_error:
                    logger.error(f"解析JSON请求失败: {str(parse_error)}")
                    raise HTTPException(status_code=400, detail=f"无效的JSON请求: {str(parse_error)}")
            else:
                # 使用Pydantic模型
                provider = chat_request.provider
                model = chat_request.model
                
                # 从messages中获取用户消息
                message = None
                for msg in reversed(chat_request.messages):
                    if msg.get("role") == "user":
                        message = msg.get("content")
                        break
                
                roleid = chat_request.roleid
                temperature = chat_request.temperature
                max_tokens = chat_request.max_tokens
                session_id = None  # JSON请求通常不直接提供session_id
                stream = chat_request.stream
                selected_username = None  # JSON请求可能未提供selected_username
                
                # 获取用户ID - 优先使用current_user
                user_id = current_user.get("id", "anonymous_user") if current_user else "anonymous_user"
                
                logger.info(f"使用Pydantic模型: provider={provider}, model={model}, roleid={roleid}, stream={stream}, user_id={user_id}")
        else:
            # 从current_user获取用户ID
            user_id = current_user.get("id", "anonymous_user") if current_user else "anonymous_user"
        
        # 确保有用户消息
        if message is None:
            logger.error("请求中没有用户消息")
            raise HTTPException(status_code=400, detail="请求中必须包含用户消息")
            
        logger.info(f"处理聊天请求: provider={provider}, model={model}, roleid={roleid}, session_id={session_id}, stream={stream}, selected_username={selected_username}, user_id={user_id}")
        
        # 获取记忆管理器
        memory_manager = await get_memory_manager()
        
        # 如果没有会话ID，创建新会话
        if not session_id:
            # 创建新会话，传递选中的用户名称
            session_id = await memory_manager.start_new_session(user_id, selected_username)
            logger.info(f"创建新会话: {session_id}, 选中用户名称: {selected_username}")
            
            # 创建新的响应流如果是流式请求
            if stream:
                response = StreamingResponse(
                    stream_response(session_id, user_id, message, roleid, provider, model, temperature, max_tokens),
                    media_type="text/event-stream"
                )
                logger.info(f"流式响应创建新会话: {session_id}")
                return response
        
        # 保存用户消息到记忆
        logger.info(f"保存用户消息: user_id={user_id}, session_id={session_id}, role=user, roleid={roleid}")
        await memory_manager.add_message(session_id, user_id, "user", message, roleid)
        
        if stream:
            # 返回流式响应
            response = StreamingResponse(
                stream_response(session_id, user_id, message, roleid, provider, model, temperature, max_tokens),
                media_type="text/event-stream"
            )
            return response
        
        # 获取角色系统提示词和对话历史
        prompt, history = await prepare_chat_context(session_id, user_id, roleid)
        
        # 调用AI服务生成回复
        ai_service = get_ai_service(provider)
        response_text = await ai_service.generate_chat_response(
            prompt=prompt,
            history=history,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # 保存AI回复到记忆
        logger.info(f"保存AI回复: user_id={user_id}, session_id={session_id}, role=assistant, roleid={roleid}")
        await memory_manager.add_message(session_id, user_id, "assistant", response_text, roleid)
        
        # 返回完整响应
        return {
            "session_id": session_id,
            "message": response_text,
            "created": int(time.time())
        }
    except Exception as e:
        logger.error(f"聊天API错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI服务错误: {str(e)}")

# 添加对GET请求的支持
@router.get("/chat")
async def chat_stream(
    stream: bool = Query(True),  # 默认启用流式响应
    content: str = Query(None),  # 用户消息内容
    model: str = Query(None),    # 可选模型
    provider: str = Query(None), # 可选提供商
    current_user: Dict = Depends(get_current_user_optional)  # 添加用户认证
):
    """
    处理流式聊天的GET请求 (用于EventSource)
    
    Args:
        stream: 是否启用流式响应
        content: 用户消息内容
        model: 可选模型名称
        provider: 可选提供商
        current_user: 当前用户信息（可选）
        
    Returns:
        流式响应
    """
    # 如果未提供内容，返回示例响应
    if not content:
        return EventSourceResponse(
            [{"data": "请提供消息内容 (content 参数)"}]
        )
    
    # 构建请求对象
    request = ChatRequest(
        messages=[{"role": "user", "content": content}],
        stream=True,
        model=model,
        provider=provider
    )
    
    # 如果指定了提供商，创建相应的配置
    config = None
    if provider:
        try:
            provider_enum = LLMProvider(provider)
            if provider_enum == LLMProvider.DEEPSEEK and not DEEPSEEK_API_KEY:
                return EventSourceResponse(
                    [{"event": "error", "data": "DeepSeek API密钥未配置"}]
                )
            elif provider_enum == LLMProvider.ZHIPU and not ZHIPU_API_KEY:
                return EventSourceResponse(
                    [{"event": "error", "data": "智谱AI API密钥未配置"}]
                )
            
            api_key = DEEPSEEK_API_KEY if provider_enum == LLMProvider.DEEPSEEK else ZHIPU_API_KEY
            
            config = LLMConfig(
                provider=provider_enum,
                model_name=model or (
                    "deepseek-chat" if provider_enum == LLMProvider.DEEPSEEK else "glm-4"
                ),
                api_key=api_key
            )
        except ValueError:
            return EventSourceResponse(
                [{"event": "error", "data": f"不支持的LLM提供商: {provider}"}]
            )
    
    # 处理流式响应，传递用户信息
    return await handle_stream_chat(request, config, current_user)

@router.post("/stream", response_model=None, response_class=EventSourceResponse)
async def stream_chat(request: ChatRequest, current_user: Dict = Depends(get_current_user_optional)):
    """
    流式生成AI回复
    
    Args:
        request: 请求体，包含消息列表
        current_user: 当前用户信息（可选）
        
    Returns:
        流式响应
    """
    # 验证请求必须包含消息
    if not request.messages:
        async def empty_messages_generator():
            yield {"event": "error", "data": "请求必须包含至少一条消息"}
            yield {"event": "end", "data": json.dumps({"type": "end"})}
        return EventSourceResponse(empty_messages_generator())
    
    # 构建LLM配置
    config = None
    if request.model or request.provider:
        try:
            # 这里获取API密钥
            api_key = None
            if request.provider == LLMProvider.DEEPSEEK:
                api_key = DEEPSEEK_API_KEY
            elif request.provider == LLMProvider.ZHIPU:
                api_key = ZHIPU_API_KEY
                
            if api_key:
                config = LLMConfig(
                    provider=request.provider or llm_service.default_config.provider,
                    model_name=request.model or llm_service.default_config.model_name,
                    api_key=api_key,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
            else:
                logger.warning(f"未找到所请求提供商的API密钥: {request.provider}")
        except Exception as e:
            logger.error(f"构建LLM配置失败: {str(e)}")
    
    return await handle_stream_chat(request, config, current_user)

async def handle_stream_chat(request: ChatRequest, config: Optional[LLMConfig] = None, 
                        current_user: Dict = None):
    """
    处理流式聊天请求
    
    Args:
        request: 聊天请求
        config: LLM配置（可选）
        current_user: 当前用户信息（可选）
        
    Returns:
        流式响应
    """
    
    # 确保LLM服务可用且有generate_stream方法
    global llm_service
    
    # 检查LLM服务是否已初始化
    if llm_service is None:
        # 直接返回错误事件流
        async def error_generator():
            yield {"event": "error", "data": "LLM服务未初始化，请稍后再试"}
            yield {"event": "end", "data": json.dumps({"type": "end"})}
        return EventSourceResponse(error_generator())
    
    # 检查generate_stream方法是否存在
    if not hasattr(llm_service, 'generate_stream'):
        logger.error("LLM服务缺少generate_stream方法，尝试动态添加")
        
        try:
            # 尝试重新加载LLM服务
            import importlib
            import sys
            import types
            from app.services.llm_service import StreamResponse
            
            # 如果新实例仍然没有方法，添加临时方法
            async def emergency_generate_stream(self, messages, config=None):
                """应急实现的generate_stream方法"""
                logger.warning("使用应急generate_stream方法")
                yield StreamResponse(
                    is_start=True,
                    model="emergency-model",
                    provider="emergency-provider"
                )
                
                try:
                    # 尝试使用generate_stream_response方法
                    if hasattr(self, 'generate_stream_response'):
                        content_buffer = ""
                        async for content in self.generate_stream_response(messages, config):
                            content_buffer += content
                            yield StreamResponse(
                                content=content,
                                model="emergency-model",
                                provider="emergency-provider"
                            )
                        
                        if not content_buffer:
                            yield StreamResponse(
                                content="无法生成回复内容，请稍后再试。",
                                model="emergency-model",
                                provider="emergency-provider"
                            )
                    else:
                        yield StreamResponse(
                            content="LLM服务不完整，缺少generate_stream_response方法。",
                            model="emergency-model",
                            provider="emergency-provider"
                        )
                except Exception as e:
                    logger.error(f"应急generate_stream发生错误: {str(e)}")
                    yield StreamResponse(
                        content=f"生成回复时出错: {str(e)}",
                        model="emergency-model",
                        provider="emergency-provider"
                    )
                
                yield StreamResponse(
                    is_end=True,
                    model="emergency-model",
                    provider="emergency-provider"
                )
            
            import types
            llm_service.generate_stream = types.MethodType(emergency_generate_stream, llm_service)
            logger.info("已添加应急generate_stream方法")
        except Exception as reload_error:
            logger.error(f"在handle_stream_chat中重新加载LLM服务失败: {str(reload_error)}")
            # 返回错误事件流
            async def reload_error_generator():
                yield {"event": "error", "data": f"重新加载LLM服务失败: {str(reload_error)}"}
                yield {"event": "end", "data": json.dumps({"type": "end"})}
            return EventSourceResponse(reload_error_generator())

    async def event_generator():
        try:
            # 初始化响应内容，用于记忆存储
            full_content = ""
            session_id = None
            
            # 从请求中提取session_id (如果有)
            if current_user and len(request.messages) > 0:
                session_id = next((msg.get("session_id") for msg in request.messages if "session_id" in msg), None)
                
                # 如果没有会话ID，创建一个新的
                if not session_id:
                    try:
                        memory_manager = await get_memory_manager()
                        session_id = await memory_manager.start_new_session(current_user["id"])
                        logger.info(f"流式响应创建新会话: {session_id}")
                    except Exception as e:
                        logger.error(f"创建会话失败: {str(e)}")
                
                # 保存用户消息 (使用最后一条用户消息)
                try:
                    user_message = next((msg["content"] for msg in reversed(request.messages) 
                                        if msg.get("role") == "user"), None)
                    if user_message:
                        # 尝试获取角色ID
                        user_msg = next((msg for msg in reversed(request.messages) 
                                        if msg.get("role") == "user"), None)
                        role_id = user_msg.get("roleid") if user_msg else None
                        
                        # 如果请求中有roleid，优先使用请求中的roleid
                        role_id = request.roleid or role_id
                        
                        memory_manager = await get_memory_manager()
                        await memory_manager.add_message(
                            session_id=session_id,
                            user_id=current_user["id"],
                            role="user",
                            content=user_message,
                            role_id=role_id
                        )
                        logger.info(f"已保存用户消息到流式会话 {session_id}，roleid: {role_id}")
                except Exception as e:
                    logger.error(f"保存用户消息失败: {str(e)}")
            
            # 开始流式生成
            try:
                async for chunk in llm_service.generate_stream(
                    messages=request.messages,
                    config=config
                ):
                    # 适当处理不同类型的返回
                    if chunk.is_start:
                        data = json.dumps({
                            "type": "start",
                            "model": chunk.model,
                            "provider": chunk.provider
                        })
                        yield {"event": "start", "data": data}
                    elif chunk.is_end:
                        data = json.dumps({
                            "type": "end", 
                            "model": chunk.model,
                            "provider": chunk.provider,
                            "tokens_used": chunk.tokens_used
                        })
                        
                        # 在流结束时保存完整回复到记忆模块
                        if current_user and session_id and full_content:
                            try:
                                # 清理响应中可能存在的元数据痕迹
                                import re
                                cleaned_content = re.sub(r'content=None model=.*? is_end=(True|False)', '', full_content)
                                cleaned_content = re.sub(r'model=.*? provider=.*? tokens_used=.*? finish_reason=.*?', '', cleaned_content)
                                
                                # 尝试获取AI角色ID
                                assistant_role_id = next((msg.get("roleid") for msg in request.messages 
                                                        if msg.get("role") == "assistant"), None)
                                
                                # 如果请求中有roleid，优先使用请求中的roleid
                                assistant_role_id = request.roleid or assistant_role_id
                                
                                memory_manager = await get_memory_manager()
                                await memory_manager.add_message(
                                    session_id=session_id,
                                    user_id=current_user["id"],
                                    role="assistant",
                                    content=cleaned_content,
                                    role_id=assistant_role_id
                                )
                                logger.info(f"已保存完整AI回复到流式会话 {session_id}，roleid: {assistant_role_id}")
                            except Exception as e:
                                logger.error(f"保存AI回复失败: {str(e)}")
                        
                        yield {"event": "end", "data": data}
                    elif hasattr(chunk, 'content') and chunk.content:
                        # 累积响应内容，过滤掉可能的元数据
                        content = chunk.content
                        # 检查内容是否包含元数据，如果包含则跳过
                        if any(x in content for x in ["content=None", "model=", "provider=", "tokens_used=", "finish_reason=", "is_start=", "is_end="]):
                            logger.warning(f"跳过包含元数据的内容: {content}")
                            continue
                            
                        full_content += content
                        yield {"event": "message", "data": content}
                    elif isinstance(chunk, str):
                        # 直接返回字符串内容
                        full_content += chunk
                        yield {"event": "message", "data": chunk}
            except AttributeError as e:
                error_msg = f"LLM服务不支持流式响应: {str(e)}"
                logger.error(error_msg)
                yield {"event": "error", "data": error_msg}
                yield {"event": "end", "data": json.dumps({"type": "end"})}
                
        except Exception as e:
            error_msg = f"生成流式回复时出错: {str(e)}"
            logger.error(error_msg)
            yield {"event": "error", "data": error_msg}
            yield {"event": "end", "data": json.dumps({"type": "end"})}
    
    # 返回事件流响应
    return EventSourceResponse(event_generator())

@router.get("/models", response_model=ModelsResponse)
async def list_models():
    """
    获取可用模型列表
    
    Returns:
        可用模型列表
    """
    try:
        models = []
        
        # 添加DeepSeek模型
        if DEEPSEEK_API_KEY:
            deepseek_models = await llm_service.get_available_models(LLMProvider.DEEPSEEK)
            for model in deepseek_models:
                models.append(ModelInfo(
                    name=model,
                    provider=LLMProvider.DEEPSEEK,
                    description=get_model_description(model, LLMProvider.DEEPSEEK)
                ))
        
        # 添加智谱AI模型
        if ZHIPU_API_KEY:
            zhipu_models = await llm_service.get_available_models(LLMProvider.ZHIPU)
            for model in zhipu_models:
                models.append(ModelInfo(
                    name=model,
                    provider=LLMProvider.ZHIPU,
                    description=get_model_description(model, LLMProvider.ZHIPU)
                ))
        
        return ModelsResponse(models=models)
        
    except Exception as e:
        logger.error(f"获取模型列表出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取模型列表时出错: {str(e)}")

def get_model_description(model_name: str, provider: LLMProvider) -> str:
    """
    获取模型描述
    
    Args:
        model_name: 模型名称
        provider: 提供商
        
    Returns:
        模型描述
    """
    descriptions = {
        LLMProvider.DEEPSEEK: {
            "deepseek-chat": "DeepSeek Chat是一个通用对话模型，适用于各种对话场景",
            "deepseek-coder": "DeepSeek Coder是专门用于代码生成和理解的模型",
            "deepseek-lite": "DeepSeek Lite是轻量级模型，响应速度更快"
        },
        LLMProvider.ZHIPU: {
            "glm-3-turbo": "智谱GLM-3-Turbo是一个高效的对话模型，平衡了速度和性能",
            "glm-4": "智谱GLM-4是最强大的对话模型，具有出色的理解能力和创造力",
            "glm-4v": "智谱GLM-4V是具有视觉能力的多模态大模型，可以理解图像内容"
        }
    }
    
    return descriptions.get(provider, {}).get(model_name, f"{provider}提供的{model_name}模型")

@router.get("/debug/service-info", response_model=LLMServiceInfoResponse)
async def get_llm_service_info():
    """
    返回LLM服务的调试信息
    
    Returns:
        服务信息，包括初始化状态、可用方法等
    """
    try:
        if llm_service is None:
            return LLMServiceInfoResponse(
                initialized=False,
                available_methods=[],
                has_generate_stream=False,
                error="LLM服务未初始化"
            )
        
        # 获取实例的所有方法
        methods = [method for method in dir(llm_service) 
                  if callable(getattr(llm_service, method)) and not method.startswith('_')]
        
        # 获取实例的所有属性（排除方法和私有属性）
        attributes = [attr for attr in dir(llm_service)
                     if not callable(getattr(llm_service, attr)) and not attr.startswith('_')]
        
        # 检查实例的类型和MRO (方法解析顺序)
        service_class = llm_service.__class__.__name__
        class_mro = [c.__name__ for c in llm_service.__class__.__mro__]
        
        # 检查是否有generate_stream方法
        has_generate_stream = 'generate_stream' in methods
        
        # 检查generate_stream方法的来源（如果存在）
        stream_method_source = None
        if has_generate_stream:
            method = getattr(llm_service, 'generate_stream')
            method_owner = method.__self__.__class__.__name__ if hasattr(method, '__self__') else "Unknown"
            is_bound_method = hasattr(method, '__self__')
            is_patched = method_owner != service_class
            
            stream_method_source = {
                "owner_class": method_owner,
                "is_bound_method": is_bound_method,
                "is_dynamically_patched": is_patched
            }
        
        return LLMServiceInfoResponse(
            initialized=True,
            available_methods=methods,
            available_attributes=attributes,
            service_class=service_class,
            class_hierarchy=class_mro,
            has_generate_stream=has_generate_stream,
            stream_method_source=stream_method_source
        )
    except Exception as e:
        logger.error(f"获取服务信息时出错: {str(e)}")
        return LLMServiceInfoResponse(
            initialized=False,
            available_methods=[],
            has_generate_stream=False,
            error=f"获取服务信息时出错: {str(e)}"
        )

@router.get("/fix_llm_service")
async def fix_llm_service():
    """
    尝试修复LLM服务中的问题，特别是缺少generate_stream方法
    
    这是一个诊断和修复工具路由，用于手动解决问题
    """
    global llm_service
    
    result = {
        "status": "unknown",
        "before": {
            "has_llm_service": llm_service is not None,
            "has_generate_stream": False,
            "has_generate_stream_response": False,
            "methods": []
        },
        "actions": [],
        "after": {
            "has_llm_service": False,
            "has_generate_stream": False,
            "has_generate_stream_response": False,
            "methods": []
        },
        "errors": []
    }
    
    # 检查初始状态
    if llm_service:
        result["before"]["has_generate_stream"] = hasattr(llm_service, 'generate_stream')
        result["before"]["has_generate_stream_response"] = hasattr(llm_service, 'generate_stream_response')
        result["before"]["methods"] = [method for method in dir(llm_service) if not method.startswith('_') and callable(getattr(llm_service, method))]
    
    try:
        # 尝试重新加载LLM服务模块
        import importlib
        import sys
        import types
        from app.services.llm_service import StreamResponse, LLMService
        
        # 记录动作
        result["actions"].append("清除模块缓存")
        
        # 清除相关模块缓存
        for module_name in list(sys.modules.keys()):
            if module_name.startswith('app.services.llm_service'):
                del sys.modules[module_name]
        
        # 重新导入
        result["actions"].append("重新导入模块")
        llm_module = importlib.import_module('app.services.llm_service')
        llm_module = importlib.reload(llm_module)
        
        # 检查模块中是否有所需方法
        has_module_stream = hasattr(llm_module.LLMService, 'generate_stream')
        has_module_stream_response = hasattr(llm_module.LLMService, 'generate_stream_response')
        
        result["actions"].append(f"检查模块中的方法: generate_stream={has_module_stream}, generate_stream_response={has_module_stream_response}")
        
        # 创建新实例
        result["actions"].append("创建新LLM服务实例")
        new_service = llm_module.LLMService()
        
        # 检查新实例是否有所需方法
        has_instance_stream = hasattr(new_service, 'generate_stream')
        has_instance_stream_response = hasattr(new_service, 'generate_stream_response')
        
        result["actions"].append(f"检查实例方法: generate_stream={has_instance_stream}, generate_stream_response={has_instance_stream_response}")
        
        # 如果新实例没有generate_stream方法，但有generate_stream_response方法，则添加应急实现
        if not has_instance_stream and has_instance_stream_response:
            result["actions"].append("添加应急generate_stream方法")
            
            async def emergency_generate_stream(self, messages, config=None):
                """应急实现的generate_stream方法"""
                logger.warning("使用应急generate_stream方法")
                yield StreamResponse(
                    is_start=True,
                    model="emergency-model",
                    provider="emergency-provider"
                )
                
                try:
                    # 使用generate_stream_response方法
                    content_buffer = ""
                    async for content in self.generate_stream_response(messages, config):
                        content_buffer += content
                        yield StreamResponse(
                            content=content,
                            model="emergency-model",
                            provider="emergency-provider"
                        )
                    
                    if not content_buffer:
                        yield StreamResponse(
                            content="无法生成回复内容，请稍后再试。",
                            model="emergency-model",
                            provider="emergency-provider"
                        )
                except Exception as e:
                    logger.error(f"应急generate_stream发生错误: {str(e)}")
                    yield StreamResponse(
                        content=f"生成回复时出错: {str(e)}",
                        model="emergency-model",
                        provider="emergency-provider"
                    )
                
                yield StreamResponse(
                    is_end=True,
                    model="emergency-model",
                    provider="emergency-provider"
                )
            
            new_service.generate_stream = types.MethodType(emergency_generate_stream, new_service)
            has_instance_stream = True
            result["actions"].append("应急方法添加成功")
        
        # 如果新实例有所需方法，则替换全局实例
        if has_instance_stream:
            llm_service = new_service
            result["actions"].append("全局LLM服务实例已更新")
            result["status"] = "success"
        else:
            result["status"] = "failed"
            result["errors"].append("新实例缺少必要的generate_stream方法")
        
        # 检查最终状态
        result["after"]["has_llm_service"] = llm_service is not None
        if llm_service:
            result["after"]["has_generate_stream"] = hasattr(llm_service, 'generate_stream')
            result["after"]["has_generate_stream_response"] = hasattr(llm_service, 'generate_stream_response')
            result["after"]["methods"] = [method for method in dir(llm_service) if not method.startswith('_') and callable(getattr(llm_service, method))]
    
    except Exception as e:
        result["status"] = "error"
        result["errors"].append(str(e))
    
    return result 