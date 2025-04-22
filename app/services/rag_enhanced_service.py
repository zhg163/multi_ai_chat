from typing import Dict, Any, List, Optional, AsyncGenerator, Union, Tuple
import logging
import httpx
import re
import json
import uuid
from datetime import datetime
import traceback
import os
import sys
import time
import asyncio
from fastapi import Depends, HTTPException
from urllib.parse import urlparse

# 导入配置
from app.config import settings

from ..services.llm_service import LLMService
from ..services.message_service import MessageService
from ..services.session_service import SessionService
from ..services.role_service import RoleService
from ..services.session_role_manager import SessionRoleManager
from ..services.role_matching_service import role_matching_service, RoleMatchingService
from ..utils import role_utils
from ..config import DEFAULT_SYSTEM_PROMPT
from ..services.embedding_service import EmbeddingService
from ..memory.memory_manager import MemoryManager
from ..models.session_role import SessionRole
from ..models.custom_session import CustomSession

# 创建日志记录器
logger = logging.getLogger("rag_enhanced")

# 功能开关
ENABLE_ROLE_BASED_CHAT = getattr(settings, "ENABLE_ROLE_BASED_CHAT", True)
DEFAULT_SYSTEM_PROMPT = getattr(settings, "DEFAULT_SYSTEM_PROMPT", 
                               "You are a knowledgeable assistant. Please answer based on the user's question, and refer to the provided materials if available.")

class RAGEnhancedService:
    """RAG enhanced service - provides automatic determination of whether RAG is needed, and optimized RAG retrieval and generation functionality"""
    
    def __init__(
        self, 
        llm_service: Optional[LLMService] = None,
        message_service: Optional[MessageService] = None,
        session_service: Optional[SessionService] = None,
        role_service: Optional[RoleService] = None
    ):
        self.logger = logging.getLogger("rag_enhanced")
        self.llm_service = llm_service or LLMService()
        self.message_service = message_service or MessageService()
        self.session_service = session_service or SessionService()
        self.role_service = role_service or RoleService()
        self.retrieval_url = settings.RETRIEVAL_SERVICE_URL
        self.api_key = settings.RETRIEVAL_API_KEY
        self.ragflow_chat_id = settings.RAGFLOW_CHAT_ID
        self.stop_generation = {}  # Store message IDs that need to stop generation
        self._initialized = False
    
    async def initialize(self):
        """Asynchronous initialization of the service"""
        if self._initialized:
            return
            
        # Initialize message service
        if hasattr(self.message_service, 'initialize'):
            await self.message_service.initialize()
            
        # Initialize session service
        if hasattr(self.session_service, 'initialize'):
            await self.session_service.initialize()
            
        # Initialize role service
        if hasattr(self.role_service, 'initialize'):
            await self.role_service.initialize()
            
        # Initialize Redis connection
        from app.services.redis_service import redis_service
        self.redis = await redis_service.get_redis()
        self.logger.info("Redis connection initialized successfully")
        
        # Initialize memory manager
        from app.memory.memory_manager import get_memory_manager
        self.memory_manager = await get_memory_manager()
        self.logger.info("Memory manager initialized successfully")
        
        # Initialize LLM client configuration
        from app.config import settings
        self.default_provider = getattr(settings, "DEFAULT_LLM_PROVIDER", "openai")
        self.default_model = getattr(settings, "DEFAULT_LLM_MODEL", "gpt-3.5-turbo")
        self.api_keys = {
            "openai": getattr(settings, "OPENAI_API_KEY", ""),
            "anthropic": getattr(settings, "ANTHROPIC_API_KEY", ""),
            "google": getattr(settings, "GOOGLE_API_KEY", ""),
        }
        self.logger.info(f"LLM default provider: {self.default_provider}, default model: {self.default_model}")
            
        self._initialized = True
        self.logger.info("RAGEnhancedService initialized successfully")
        
    async def _ensure_initialized(self):
        """Ensure the service is initialized"""
        if not self._initialized:
            await self.initialize()
    
    async def analyze_question(self, question: str, model: str) -> Tuple[bool, str]:
        """
        Analyze the question to determine whether external knowledge base information is needed
        
        Args:
            question: User question
            model: LLM model used (ignored, forced to use deepseek-chat)
            
        Returns:
            (need_rag, thinking): Whether RAG is needed, analysis thinking process
        """
        # Ensure the service is initialized
        await self._ensure_initialized()
        
        self.logger.info(f"Analyzing whether RAG is needed for the question: {question[:50]}...")
        
        # Force using deepseek-chat model
        model = "deepseek-chat"
        
        # Build analysis prompt
        prompt = f"""Please analyze the following question and determine whether you need to retrieve information from the external knowledge base to provide an accurate answer.

Question: {question}

Please analyze through the following steps:
1. Question type analysis: What type of question is this? Is it general knowledge, specific domain knowledge, or real-time information?
2. Necessary knowledge assessment: What specific knowledge is needed to answer this question?
3. Knowledge coverage analysis: Do you have all the necessary knowledge to answer the question? Are there any information gaps?
4. RAG judgment: Whether you need to retrieve information from the external knowledge base to supplement missing information?

Finally, please clearly give the judgment result in the format:
【Need to retrieve】: Yes/No
【Analysis reason】: Briefly explain your judgment basis

Please keep the analysis concise and logical, focusing on the relevance to the question. """

        # Call LLM for analysis
        try:
            response = await self.llm_service.chat_completion(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a responsible assistant for determining whether a question needs external knowledge. Please analyze the given question and determine whether you need to retrieve information from the external knowledge base."},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                temperature=0.1,
                max_tokens=500
            )
            
            analysis = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            self.logger.info(f"Question analysis result: {analysis[:100]}...")
            
            # Extract whether RAG is needed
            need_rag = False
            if "【Need to retrieve】: Yes" in analysis or "【Need to retrieve】:Yes" in analysis:
                need_rag = True
            
            return need_rag, analysis
        except Exception as e:
            self.logger.error(f"Failed to analyze question: {str(e)}")
            # Default to RAG when an error occurs
            return True, f"An error occurred during analysis, and RAG will be used for safety. Error information: {str(e)}"
    
    async def retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve related documents from the knowledge base
        
        Args:
            query: Query content
            
        Returns:
            Retrieved document list
        """
        # Ensure the service is initialized
        await self._ensure_initialized()
        
        self.logger.info(f"Retrieving content from the knowledge base: {query[:50]}...")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    "query": query,
                    "chat_id": self.ragflow_chat_id
                }
                
                headers = {
                    "Content-Type": "application/json",
                }
                
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                response = await client.post(
                    self.retrieval_url,
                    json=payload,
                    headers=headers
                )
                
                if response.status_code != 200:
                    self.logger.error(f"Failed to retrieve from the knowledge base: {response.status_code}, {response.text}")
                    return []
                
                result = response.json()
                documents = result.get("documents", [])
                
                if documents:
                    self.logger.info(f"Retrieved {len(documents)} related documents")
                else:
                    self.logger.warning("No related documents retrieved")
                
                return documents
        except Exception as e:
            self.logger.error(f"An error occurred when retrieving documents: {str(e)}")
            return []
    
    def format_retrieved_documents(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents for use in prompts
        
        Args:
            documents: Document list
            
        Returns:
            Formatted document string
        """
        # Note: This method does not require initialization because it does not access the database
        
        if not documents:
            return "No related reference materials found."
        
        formatted_text = "The following are related reference materials:\n\n"
        
        for i, doc in enumerate(documents, 1):
            title = doc.get("title", "Unknown document")
            content = doc.get("content", "")
            source = doc.get("source", "Unknown source")
            score = doc.get("score", 0)
            
            formatted_text += f"[{i}] {title}\n"
            formatted_text += f"Source: {source}\n"
            formatted_text += f"Relevance: {score:.2f}\n"
            formatted_text += f"Content: {content}\n\n"
        
        return formatted_text
    
    async def verify_session_exists(self, session_id: str, user_id: str) -> bool:
        """
        验证会话是否存在
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            会话是否存在
        """
        if not session_id or not user_id:
            return False
            
        try:
            # 先检查MongoDB中是否存在
            session_data = await CustomSession.get_session_by_id(session_id)
            if session_data:
                return True
                
            # 如果MongoDB中不存在，检查Redis中是否存在
            from app.memory.memory_manager import get_memory_manager
            memory_manager = await get_memory_manager()
            
            if memory_manager and memory_manager.short_term_memory and memory_manager.short_term_memory.redis:
                redis_client = memory_manager.short_term_memory.redis
                redis_key = f"session:{user_id}:{session_id}"
                
                exists = await redis_client.exists(redis_key)
                return exists
                
            return False
        except Exception as e:
            self.logger.error(f"验证会话存在性时出错: {str(e)}")
            return False
    
    async def process_chat(
        self,
        messages: List[Dict],
        model: str = None,
        session_id: str = None,
        user_id: str = None,
        enable_rag: bool = True,
        lang: str = "zh",
        provider: str = None,
        model_name: str = None,
        api_key: str = None,
        stream: bool = True,
        role_id: str = None,
        auto_role_match: bool = False
    ) -> AsyncGenerator[Union[Dict, str], None]:
        """
        Process chat with RAG enhancement
        
        Args:
            messages: message list
            model: model name (optional)
            session_id: session ID for conversation history retrieval
            user_id: user ID for conversation history retrieval
            enable_rag: whether enable RAG
            lang: language, affects generated prompt, default zh
            provider: LLM provider name
            model_name: specific model name for this provider
            api_key: API key
            stream: whether to use streaming response
            role_id: specific role ID to use
            auto_role_match: whether to use automatic role matching
            
        Yields:
            Streaming response or error message
        """
        # 确保服务已初始化
        await self._ensure_initialized()
        
        # 为本次请求生成唯一的消息ID
        message_id = str(uuid.uuid4())
        self.stop_generation[message_id] = False
        
        # 在返回的第一个数据包中包含message_id，供前端保存和停止生成使用
        yield {"message_id": message_id}
        
        # 验证会话是否存在
        if session_id and user_id:
            session_exists = await self.verify_session_exists(session_id, user_id)
            if not session_exists:
                self.logger.warning(f"会话 {session_id} 不存在，这可能是新会话或ID错误")
                # 我们可以选择继续处理，但不会尝试加载历史消息
        
        try:
            # 验证消息列表
            if not messages or not isinstance(messages, list):
                yield {"error": "消息列表为空或格式不正确"}
                return
            
            # 获取最后一条用户消息
            last_user_message = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user_message = msg.get("content", "").strip()
                    break
            
            if not last_user_message:
                yield {"error": "找不到用户消息"}
                return
            
            # 确定要使用的模型
            model_config = self._determine_model(model, provider, model_name, api_key)
            if isinstance(model_config, str):
                # 如果返回的是错误消息而不是模型配置
                yield {"error": model_config}
                return
            
            # 角色选择逻辑
            role_info = {}
            sys_msg = None
            
            # 优先自动角色匹配
            if auto_role_match and not role_id:
                logger.info("开始自动角色匹配")
                try:
                    # 传入session_id参数，限制在会话角色范围内
                    match_result = await role_matching_service.find_matching_role(last_user_message, session_id)
                    if match_result:
                        role_id = match_result.get("role_id")
                        
                        # 使用匹配的角色信息
                        if role_id:
                            role_info = {
                                "id": role_id,
                                "name": match_result.get("name", ""),
                                "match_reason": match_result.get("match_reason", ""),
                                "score": match_result.get("score", 0)
                            }
                            
                            # 获取角色系统消息
                            session_role_manager = SessionRoleManager()
                            role_data = await session_role_manager.get_role(role_id)
                            if role_data and "system_prompt" in role_data:
                                sys_msg = role_data["system_prompt"].strip()
                                
                                # 更新角色使用计数
                                if session_id:
                                    await session_role_manager.update_role_usage_count(session_id, role_id)
                            
                            # 输出角色匹配信息
                            yield {
                                "role_match": {
                                    "success": True,
                                    "role": role_info
                                }
                            }
                except Exception as e:
                    logger.error(f"自动角色匹配失败: {str(e)}")
                    yield {"error": f"自动角色匹配失败: {str(e)}"}
            
            # 如果有指定角色ID
            elif role_id:
                logger.info(f"使用指定角色: {role_id}")
                try:
                    from app.services.session_role_manager import SessionRoleManager
                    session_role_manager = SessionRoleManager()
                    role_data = await session_role_manager.get_role(role_id)
                    if role_data:
                        role_info = {
                            "id": role_id,
                            "name": role_data.get("name", ""),
                            "description": role_data.get("description", "")
                        }
                        
                        # 获取角色系统消息
                        if "system_prompt" in role_data:
                            sys_msg = role_data["system_prompt"].strip()
                            
                            # 更新角色使用计数
                            if session_id:
                                await session_role_manager.update_role_usage_count(session_id, role_id)
                    else:
                        logger.warning(f"找不到角色: {role_id}")
                        yield {"error": f"找不到角色: {role_id}"}
                except Exception as e:
                    logger.error(f"获取角色信息失败: {str(e)}")
                    yield {"error": f"获取角色信息失败: {str(e)}"}
            
            # 如果没有系统消息，使用默认的
            if not sys_msg:
                sys_msg = DEFAULT_SYSTEM_PROMPT
            
            # 构建完整的消息列表
            complete_messages = []
            
            # 添加系统消息
            complete_messages.append({"role": "system", "content": sys_msg})
            
            # 获取历史消息（如果有会话ID和用户ID）
            if session_id and user_id:
                try:
                    # 检查会话是否存在
                    session_exists = await self.verify_session_exists(session_id, user_id)
                    if not session_exists:
                        logger.warning(f"会话 {session_id} 不存在，跳过历史消息加载")
                    else:
                        logger.info(f"开始获取会话 {session_id} 的历史消息")
                        
                        # 使用 CustomSession 获取会话信息
                        session_data = await CustomSession.get_session_by_id(session_id)
                        logger.info(f"获取会话数据: {session_data is not None}")
                        
                        if session_data:
                            # 构建Redis消息键
                            redis_key = f"messages:{user_id}:{session_id}"
                            logger.info(f"准备从Redis获取消息，键名: {redis_key}")
                            
                            # 从内存管理器获取Redis客户端
                            from app.memory.memory_manager import get_memory_manager
                            memory_manager = await get_memory_manager()
                            
                            if memory_manager and memory_manager.short_term_memory and memory_manager.short_term_memory.redis:
                                redis_client = memory_manager.short_term_memory.redis
                                
                                # 检查消息键是否存在
                                key_exists = await redis_client.exists(redis_key)
                                logger.info(f"Redis消息键 {redis_key} 存在: {key_exists}")
                                
                                if key_exists:
                                    # 获取消息记录
                                    messages_data = await redis_client.lrange(redis_key, 0, -1)
                                    logger.info(f"从Redis获取到 {len(messages_data)} 条消息")
                                    
                                    history = []
                                    
                                    # 解析消息记录
                                    for msg_data in messages_data:
                                        try:
                                            msg = json.loads(msg_data)
                                            if msg.get("role") in ["user", "assistant", "system"]:
                                                history.append(msg)
                                        except json.JSONDecodeError:
                                            logger.warning(f"解析消息失败: {msg_data[:100]}")
                                            continue
                                    
                                    # 将历史消息添加到完整消息列表中
                                    logger.info(f"成功解析 {len(history)} 条有效历史消息")
                                    for msg in history:
                                        complete_messages.append({
                                            "role": msg.get("role", "user"),
                                            "content": msg.get("content", "")
                                        })
                                else:
                                    logger.warning(f"Redis中不存在消息键 {redis_key}")
                            else:
                                logger.warning("Redis客户端不可用，无法获取历史消息")
                        else:
                            logger.warning(f"在MongoDB中找不到会话 {session_id}")
                except Exception as e:
                    logger.error(f"获取历史消息出错: {str(e)}", exc_info=True)
                    # 如果获取历史消息失败，继续处理但记录错误
            
            # 添加当前消息
            for msg in messages:
                complete_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
            
            # 分析用户问题，确定是否需要搜索
            search_needed = False
            if enable_rag:
                # 使用分析器检查问题是否需要搜索相关文档
                search_needed = await self.analyze_question(last_user_message, model)
                logger.info(f"问题分析结果: 【需要检索】: {'是' if search_needed else '否'}")
            
            # 如果需要搜索，则检索文档并增强查询
            if search_needed:
                logger.info(f"从知识库检索内容: {last_user_message[:50]}...")
                
                try:
                    # 检索相关文档
                    retrieved_docs = await self.retrieve_documents(last_user_message)
                    
                    # 如果没有找到相关文档，发出警告并直接使用原始消息调用LLM
                    if not retrieved_docs:
                        logger.warning("未检索到相关文档")
                        async for response in self.llm_service.generate_stream(
                            complete_messages,
                            config=model_config,
                            message_id=message_id,
                            stop_generation=self.stop_generation
                        ):
                            yield response
                        return
                    
                    # 构建格式化的上下文
                    context_str = self.format_retrieved_documents(retrieved_docs)
                    
                    # 构建增强的提示
                    augmented_msg = self._build_augmented_prompt(
                        last_user_message, context_str, lang=lang
                    )
                    
                    # 用增强的提示替换原始用户消息
                    enhanced_messages = complete_messages.copy()
                    enhanced_messages[-1]["content"] = augmented_msg
                    
                    # 调用LLM服务生成回复
                    async for response in self.llm_service.generate_stream(
                        enhanced_messages,
                        config=model_config,
                        message_id=message_id,
                        stop_generation=self.stop_generation
                    ):
                        yield response
                    
                except Exception as e:
                    logger.error(f"处理RAG增强查询时出错: {str(e)}", exc_info=True)
                    yield {"error": f"处理增强查询失败: {str(e)}"}
            else:
                # 不需要搜索，直接使用原始消息调用LLM
                try:
                    async for response in self.llm_service.generate_stream(
                        complete_messages,
                        config=model_config,
                        message_id=message_id,
                        stop_generation=self.stop_generation
                    ):
                        yield response
                except Exception as e:
                    logger.error(f"生成回复时出错: {str(e)}", exc_info=True)
                    yield {"error": f"生成回复失败: {str(e)}"}
                    
        except Exception as e:
            # 捕获整个处理过程中的所有其他错误
            error_msg = f"处理聊天请求时出现意外错误: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield {"error": error_msg}
        finally:
            # 清理停止生成字典中的当前消息ID
            if message_id in self.stop_generation:
                del self.stop_generation[message_id]
    
    async def save_to_memory(self, session_id: str, user_id: str, last_question: str, 
                        ai_response: str, role_id: Optional[str] = None):
        """
        Save user question and AI reply to short-term memory (MongoDB and Redis)
        
        Args:
            session_id: Session ID
            user_id: User ID
            last_question: User's last question
            ai_response: AI's reply
            role_id: Used role ID
        """
        # Ensure the service is initialized
        await self._ensure_initialized()
        
        if not session_id or not user_id:
            self.logger.error("Failed to save to memory: Session ID or user ID is empty")
            return
            
        try:
            # 使用 message_service 创建消息记录到 MongoDB
            user_message = await self.message_service.create_message(
                content=last_question,
                message_type="USER",
                user_id=user_id,
                session_id=session_id,
                metadata={"role_id": role_id} if role_id else None
            )
            
            ai_message = await self.message_service.create_message(
                content=ai_response,
                message_type="AI",
                user_id=user_id,
                session_id=session_id,
                metadata={"role_id": role_id} if role_id else None
            )
            
            # 直接将消息保存到 Redis
            try:
                # 获取 Redis 客户端
                from app.memory.memory_manager import get_memory_manager
                memory_manager = await get_memory_manager()
                
                if memory_manager and memory_manager.short_term_memory and memory_manager.short_term_memory.redis:
                    redis_client = memory_manager.short_term_memory.redis
                    redis_key = f"messages:{user_id}:{session_id}"
                    
                    # 准备用户消息
                    user_msg = {
                        "role": "user",
                        "content": last_question,
                        "timestamp": datetime.utcnow().isoformat(),
                        "message_id": str(user_message.id) if hasattr(user_message, 'id') else str(uuid.uuid4()),
                        "metadata": {"role_id": role_id} if role_id else {}
                    }
                    
                    # 准备AI消息
                    ai_msg = {
                        "role": "assistant",
                        "content": ai_response,
                        "timestamp": datetime.utcnow().isoformat(),
                        "message_id": str(ai_message.id) if hasattr(ai_message, 'id') else str(uuid.uuid4()),
                        "metadata": {"role_id": role_id} if role_id else {}
                    }
                    
                    # 将消息添加到 Redis 列表
                    await redis_client.rpush(redis_key, json.dumps(user_msg))
                    await redis_client.rpush(redis_key, json.dumps(ai_msg))
                    
                    # 设置过期时间 (7天)
                    await redis_client.expire(redis_key, 7 * 24 * 60 * 60)
                    
                    self.logger.info(f"Saved conversation to Redis, key: {redis_key}")
            except Exception as e:
                self.logger.error(f"Failed to save to Redis: {str(e)}")
            
            self.logger.info(f"Saved conversation to MongoDB, user message ID: {user_message.id}, AI message ID: {ai_message.id}")
            
            # If there is a role ID, update role usage count
            if role_id and ENABLE_ROLE_BASED_CHAT:
                try:
                    # Initialize SessionRoleManager
                    from app.memory.memory_manager import get_memory_manager
                    memory_manager = await get_memory_manager()
                    
                    if memory_manager and memory_manager.short_term_memory and memory_manager.short_term_memory.redis:
                        redis_client = memory_manager.short_term_memory.redis
                        session_role_manager = SessionRoleManager(redis_client)
                        
                        # Update role usage count
                        await session_role_manager.update_role_usage_count(session_id, user_id, role_id)
                        self.logger.info(f"Updated role {role_id} usage count")
                except Exception as e:
                    self.logger.error(f"Failed to update role usage count: {str(e)}")
            
            return user_message.id, ai_message.id
        except Exception as e:
            self.logger.error(f"Failed to save to memory: {str(e)}")
            return None, None
    
    async def stop_message_generation(self, message_id: str) -> bool:
        """
        Stop generation of specified message
        
        Args:
            message_id: Message ID
            
        Returns:
            Whether generation stopped successfully
        """
        # Ensure the service is initialized
        await self._ensure_initialized()
        
        if message_id not in self.stop_generation:
            return False
        
        self.stop_generation[message_id] = True
        return True
    
    def _get_llm_client(self, provider: str, model_name: str, api_key: Optional[str] = None):
        """Get LLM client"""
        try:
            if provider == "anthropic":
                from app.services.llm.anthropic_client import AnthropicClient
                return AnthropicClient(api_key=api_key, model_name=model_name)
            elif provider == "openai":
                from app.services.llm.openai_client import OpenAIClient
                return OpenAIClient(api_key=api_key, model_name=model_name)
            elif provider == "google":
                from app.services.llm.google_client import GoogleClient
                return GoogleClient(api_key=api_key, model_name=model_name)
            elif provider == "local":
                from app.services.llm.local_client import LocalClient
                return LocalClient(model_name=model_name)
            else:
                logger.warning(f"Unknown LLM provider: {provider}, using default provider OpenAI")
                from app.services.llm.openai_client import OpenAIClient
                return OpenAIClient(api_key=api_key, model_name=model_name or "gpt-3.5-turbo")
        except Exception as e:
            logger.error(f"Failed to create LLM client: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _get_llm_params(self, provider: str) -> Dict[str, Any]:
        """Get LLM parameters"""
        common_params = {
            "temperature": 0.7,
            "max_tokens": None
        }
        
        # Return specific parameters based on provider
        if provider == "anthropic":
            return {
                **common_params,
                "max_tokens": 4000
            }
        elif provider == "openai":
            return {
                **common_params,
                "max_tokens": 2000,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }
        elif provider == "google":
            return {
                **common_params,
                "max_output_tokens": 2048,
                "top_p": 0.95,
                "top_k": 40
            }
        else:
            return common_params 
    
    def _determine_model(self, model: str, provider: str, model_name: str, api_key: str) -> Union[Dict[str, Any], str]:
        """
        确定要使用的模型参数
        
        Args:
            model: 模型标识（旧版参数，将被转换为 provider 和 model_name）
            provider: 提供商
            model_name: 模型名称
            api_key: API密钥
            
        Returns:
            模型配置字典，或错误消息字符串
        """
        try:
            # 如果提供了provider和model_name，优先使用这些
            if provider and model_name:
                logger.info(f"使用指定的提供商 {provider} 和模型 {model_name}")
                
                # 获取基础参数
                params = self._get_llm_params(provider)
                
                # 构建完整的配置
                return {
                    "provider": provider,
                    "model_name": model_name,
                    "api_key": api_key,
                    **params
                }
            
            # 兼容旧的model参数
            elif model:
                # 假设model格式为"provider/model_name"或直接为model_name
                if "/" in model:
                    provider_part, model_part = model.split("/", 1)
                    logger.info(f"从旧格式解析: 提供商 {provider_part}, 模型 {model_part}")
                    
                    # 获取基础参数
                    params = self._get_llm_params(provider_part)
                    
                    # 构建完整的配置
                    return {
                        "provider": provider_part,
                        "model_name": model_part,
                        "api_key": api_key,
                        **params
                    }
                else:
                    # 默认使用deepseek作为提供商
                    logger.info(f"只提供了模型名称 {model}，使用默认提供商 deepseek")
                    
                    # 获取基础参数
                    params = self._get_llm_params("deepseek")
                    
                    # 构建完整的配置
                    return {
                        "provider": "deepseek",
                        "model_name": model,
                        "api_key": api_key,
                        **params
                    }
            
            # 使用默认配置
            else:
                logger.info("使用默认提供商 deepseek 和模型 deepseek-chat")
                
                # 获取基础参数
                params = self._get_llm_params("deepseek")
                
                # 构建完整的配置
                return {
                    "provider": "deepseek",
                    "model_name": "deepseek-chat",
                    "api_key": api_key,
                    **params
                }
                
        except Exception as e:
            error_msg = f"确定模型参数时出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
    
    async def close(self):
        """关闭服务和相关资源"""
        self.logger.info("正在关闭RAG增强服务...")
        
        # 关闭LLM服务
        if self.llm_service and hasattr(self.llm_service, 'close'):
            await self.llm_service.close()
            self.logger.info("LLM服务已关闭")
            
        # 关闭其他可能的资源
        # ...
        
        self.logger.info("RAG增强服务关闭完成") 