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
        
        # 新增：初始化session_role_manager (将在initialize方法中创建)
        self.session_role_manager = None
        
        # 获取检索API密钥并确保正确格式
        retrieval_key = settings.RETRIEVAL_API_KEY
        if retrieval_key and not retrieval_key.startswith("skragflow-"):
            retrieval_key = f"skragflow-{retrieval_key.strip()}"
            
        self.retrieval_url = settings.RETRIEVAL_SERVICE_URL
        self.api_key = retrieval_key
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
        
        self.default_provider = getattr(settings, "DEFAULT_LLM_PROVIDER", "deepseek")
        self.default_model = getattr(settings, "DEFAULT_LLM_MODEL", "deepseek-chat")
        
        # 直接从环境变量获取API密钥
        self.api_keys = {
            "openai": getattr(settings, "OPENAI_API_KEY", ""),
            "anthropic": getattr(settings, "ANTHROPIC_API_KEY", ""),
            "google": getattr(settings, "GOOGLE_API_KEY", ""),
            "deepseek": os.environ.get("DEEPSEEK_API_KEY", "")  # 直接从环境变量获取
        }
        self.logger.info(f"LLM default provider: {self.default_provider}, default model: {self.default_model}")
        
        # 验证API密钥
        self._validate_api_keys()
            
        # 初始化SessionRoleManager
        from app.services.session_role_manager import SessionRoleManager
        self.session_role_manager = SessionRoleManager(redis_client=self.redis)
        logger.info("Session role manager initialized successfully")
        
        self._initialized = True
        self.logger.info("RAGEnhancedService initialized successfully")
        
    def _validate_api_keys(self):
        """验证API密钥的有效性和格式"""
        import os
        
        # 检查知识库检索API密钥，仅当启用RAG时才警告
        if not self.api_key or not self.api_key.strip():
            self.logger.warning("未配置知识库检索API密钥，检索功能可能不可用")
        else:
            # 确保API密钥格式正确
            api_key = self.api_key.strip()
            if not api_key.startswith("skragflow-"):
                self.logger.warning("知识库检索API密钥不是以skragflow-开头，自动添加前缀")
                self.api_key = f"skragflow-{api_key}"
        
        # 只检查当前使用的提供商API密钥
        current_provider = self.default_provider.lower()
        if current_provider in self.api_keys:
            key = self.api_keys[current_provider]
            
            # 如果是deepseek且密钥为空，尝试再次从环境变量获取
            if current_provider == "deepseek" and (not key or not key.strip()):
                env_key = os.environ.get("DEEPSEEK_API_KEY", "")
                if env_key and env_key.strip():
                    self.logger.info("从环境变量直接获取DEEPSEEK API密钥")
                    self.api_keys["deepseek"] = env_key.strip()
                    key = env_key.strip()
            
            if not key or not key.strip():
                self.logger.warning(f"未配置{current_provider.upper()}的API密钥，当前服务可能不可用")
            elif current_provider == "deepseek" and not key.startswith("sk-"):
                self.logger.warning(f"{current_provider.upper()}的API密钥格式不正确，应以sk-开头")
                self.api_keys[current_provider] = f"sk-{key.strip()}"
                self.logger.info(f"已修正{current_provider.upper()}的API密钥格式")
        else:
            self.logger.warning(f"未知的提供商: {current_provider}，无法验证API密钥")
        
        # 其他提供商的API密钥以调试级别记录
        for provider, key in self.api_keys.items():
            if provider != current_provider:
                if not key or not key.strip():
                    self.logger.debug(f"未配置{provider.upper()}的API密钥，该服务将不可用")
        
    async def _ensure_initialized(self):
        """Ensure the service is initialized"""
        if self._initialized:
            return
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        logger.debug(f"[{request_id}] 开始RAG服务初始化")
        try:
            # 初始化LLM服务
            llm_init_start = time.time()
            logger.debug(f"[{request_id}] 开始初始化LLM服务")
            if not self.llm_service:
                from ..services.llm_service import llm_service
                self.llm_service = llm_service
            
            # 初始化嵌入服务
            embed_init_start = time.time()
            logger.debug(f"[{request_id}] 开始初始化嵌入服务")
            if not self.embedding_service:
                from ..services.embedding_service import embedding_service
                self.embedding_service = embedding_service
                if not self.embedding_service.initialized:
                    logger.debug(f"[{request_id}] 嵌入服务需要显式初始化")
                    await self.embedding_service.initialize()
            embed_init_time = time.time() - embed_init_start
            logger.debug(f"[{request_id}] 嵌入服务初始化完成，耗时: {embed_init_time:.4f}秒")
            
            llm_init_time = time.time() - llm_init_start
            logger.debug(f"[{request_id}] LLM服务初始化完成，耗时: {llm_init_time:.4f}秒")
            
            # 获取Redis客户端
            redis_init_start = time.time()
            logger.debug(f"[{request_id}] 开始初始化Redis客户端")
            if not self.redis:
                from ..common.redis_client import get_redis_client
                self.redis = await asyncio.wait_for(get_redis_client(), timeout=5.0)
            redis_init_time = time.time() - redis_init_start
            logger.debug(f"[{request_id}] Redis客户端初始化完成，耗时: {redis_init_time:.4f}秒")
            
            # 初始化其他服务
            other_init_start = time.time()
            logger.debug(f"[{request_id}] 开始初始化其他服务")
            if not self.message_service:
                from ..services.message_service import MessageService
                self.message_service = MessageService()
                
            if not self.session_service:
                from ..services.session_service import SessionService
                self.session_service = SessionService()
                
            if not self.role_service:
                from ..services.role_service import RoleService
                self.role_service = RoleService()
            other_init_time = time.time() - other_init_start
            logger.debug(f"[{request_id}] 其他服务初始化完成，耗时: {other_init_time:.4f}秒")
            
            # 验证API密钥
            key_validate_start = time.time()
            logger.debug(f"[{request_id}] 开始验证API密钥")
            self._validate_api_keys()
            key_validate_time = time.time() - key_validate_start
            logger.debug(f"[{request_id}] API密钥验证完成，耗时: {key_validate_time:.4f}秒")
            
            self._initialized = True
            total_time = time.time() - start_time
            logger.info(f"[{request_id}] RAG服务初始化完成，总耗时: {total_time:.4f}秒")
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"[{request_id}] RAG服务初始化失败，耗时: {total_time:.4f}秒, 错误: {str(e)}")
            raise
        
        return True
    
    def should_skip_rag_analysis(self, question: str) -> tuple[bool, str]:
        """
        快速判断是否需要跳过RAG分析流程
        
        Args:
            question: 用户问题
        
        Returns:
            (是否跳过, 跳过原因)
        """
        # 清理并准备问题文本
        cleaned_question = question.strip()
        
        # 规则1: 纯数字问题
        if cleaned_question.isdigit():
            return True, "纯数字问题无需RAG分析"
        
        # 规则2: 极短问题
        if len(cleaned_question) < 6:
            return True, "问题过短，无需RAG分析"
        
        # 规则3: 常见问候语
        greetings = ["你好", "hello", "hi", "嗨", "早上好", "晚上好", "下午好"]
        if cleaned_question.lower() in greetings or any(g in cleaned_question.lower() for g in greetings):
            return True, "简单问候语无需RAG分析"
        
        # 规则4: 无实质内容
        if set(cleaned_question).issubset(set("!！?？.,，。;；:：""''\"' ")):
            return True, "问题仅包含标点符号，无需RAG分析"
            
        # 规则5: 简单指令
        simple_commands = ["停止", "退出", "取消", "stop", "quit", "cancel", "谢谢", "谢谢你", "thanks", "thank you"]
        if cleaned_question.lower() in simple_commands:
            return True, "简单指令无需RAG分析"
        
        # 默认不跳过
        return False, ""

    async def analyze_question(self, question: str, model: str = None) -> dict:
        """
        Analyze the question to determine whether external knowledge base information is needed
        
        Args:
            question: User question
            model: LLM model used (ignored, forced to use deepseek-chat)
            
        Returns:
            Dictionary containing analysis results
        """
        # Ensure the service is initialized
        await self._ensure_initialized()
        
        # 添加快速跳过规则检查
        should_skip, skip_reason = self.should_skip_rag_analysis(question)
        if should_skip:
            self.logger.info(f"快速规则触发: {skip_reason}, 跳过RAG分析")
            return {
                "need_rag": False,
                "reason": skip_reason,
                "analysis": skip_reason,
                "skip_full_analysis": True
            }
        
        self.logger.info(f"Analyzing whether RAG is needed for the question: {question[:50]}...")
        
        # If the question is too short or likely a greeting, don't use RAG
        if len(question.strip()) < 5:
            self.logger.info("Question too short, no RAG needed")
            return {
                "need_rag": False, 
                "reason": "Question too short, no RAG needed",
                "analysis": "Question too short, no RAG needed"
            }
        
        # Simple heuristic: these types of questions rarely need external knowledge
        simple_patterns = [
            r'^(你好|hello|hi)[!！.,？?]*$',  # Greetings
            r'^(谢谢|thank)[!！.,？?]*$',     # Thanks
            r'^(是的|yes|no|不是)[!！.,？?]*$',  # Simple yes/no
        ]
        
        for pattern in simple_patterns:
            if re.match(pattern, question.strip(), re.IGNORECASE):
                reason = f"Question matches simple pattern, no RAG needed: {pattern}"
                self.logger.info(reason)
                return {
                    "need_rag": False, 
                    "reason": reason,
                    "analysis": reason
                }
        
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
            max_retries = 2
            retry_count = 0
            
            while retry_count <= max_retries:
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
                    
                        return {
                            "need_rag": need_rag, 
                            "reason": "LLM analysis result", 
                            "analysis": analysis
                        }
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count <= max_retries:
                        self.logger.warning(f"Retry {retry_count}/{max_retries} for analyzing question after error: {str(e)}")
                        await asyncio.sleep(1)  # Wait before retrying
                    else:
                        raise
            
        except Exception as e:
            self.logger.error(f"Failed to analyze question: {str(e)}")
            # Default to RAG when an error occurs for safety
            error_message = f"An error occurred during analysis, and RAG will be used for safety. Error information: {str(e)}"
            return {
                "need_rag": True,
                "reason": "Error during analysis",
                "analysis": error_message
            }
    
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
                
                # 只有在API密钥存在时才添加Authorization头
                if self.api_key and self.api_key.strip():
                    # 确保API密钥格式正确
                    api_key = self.api_key.strip()
                    if not api_key.startswith("skragflow-"):
                        api_key = f"skragflow-{api_key}"
                    headers["Authorization"] = f"Bearer {api_key}"
                else:
                    self.logger.warning("未提供API密钥，可能导致检索服务授权失败")
                
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
    
    def _build_augmented_prompt(self, user_question: str, context: str, lang: str = "zh") -> str:
        """
        Build an augmented prompt with retrieved context for RAG
        
        Args:
            user_question: Original user question
            context: Retrieved context from knowledge base
            lang: Language for prompt templates (zh for Chinese, en for English)
            
        Returns:
            Augmented prompt for LLM
        """
        if lang.lower() == "en":
            return f"""I need you to answer the question based on both the provided reference materials and your knowledge.

Reference Materials:
{context}

Question: {user_question}

Instructions:
1. If the reference materials directly contain information relevant to the question, prioritize using this information.
2. Cite your sources by referring to the reference number: [1], [2], etc. whenever you use information from the references.
3. If the reference materials do not contain enough information, supplement with your own knowledge.
4. Provide a clear, concise, and accurate answer.
5. If you're adding information beyond what's in the references, clearly indicate this with phrases like "Beyond the provided references..." or "Based on my knowledge..."
6. If the references contradict each other, note this and explain which information is likely more reliable.
7. If the references are not relevant to the question, ignore them and answer based on your knowledge, making it clear you're doing so.

Answer:"""
        else:
            # Default to Chinese
            return f"""请根据提供的参考资料和你的知识来回答问题。

参考资料：
{context}

问题：{user_question}

回答要求：
1. 如果参考资料中直接包含与问题相关的信息，请优先使用这些信息。
2. 引用来源时请参考资料编号：[1]、[2]等，确保在使用参考资料信息时明确指出出处。
3. 如果参考资料中没有足够的信息，可以补充你自己的知识。
4. 提供清晰、简明、准确的回答。
5. 如果你添加了参考资料之外的信息，请明确指出，例如使用"除了提供的参考资料外..."或"根据我的知识..."等表述。
6. 如果参考资料之间存在矛盾，请注意指出并解释哪些信息可能更可靠。
7. 如果参考资料与问题无关，请忽略它们并根据你的知识回答，同时明确说明你这样做的原因。

回答："""
    
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
        
        # 用于收集完整的AI响应
        ai_response_parts = []
        user_question = ""
        
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
                error_msg = "消息列表为空或格式不正确"
                yield {"error": error_msg}
                return
            
            # 获取最后一条用户消息
            last_user_message = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user_message = msg.get("content", "").strip()
                    user_question = last_user_message
                    break
            
            if not last_user_message:
                error_msg = "找不到用户消息"
                yield {"error": error_msg}
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
                            from app.services.session_role_manager import SessionRoleManager
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
            analysis = ""
            if enable_rag:
                # 使用分析器检查问题是否需要搜索相关文档
                analysis_result = await self.analyze_question(last_user_message, model)
                search_needed = analysis_result.get("need_rag", False)
                analysis = analysis_result.get("analysis", "")
                reason = analysis_result.get("reason", "")
                
                # 检查是否使用了快速规则跳过
                if analysis_result.get("skip_full_analysis", False):
                    self.logger.info(f"使用快速规则评估: {reason}")
                else:
                    self.logger.info(f"问题分析结果: 【需要检索】: {'是' if search_needed else '否'}")
                    self.logger.debug(f"分析详情: {analysis[:200]}...")
            
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
                            stop_generation=self.stop_generation[message_id]
                        ):
                            # 收集响应以保存到内存
                            if isinstance(response, dict) and "content" in response:
                                ai_response_parts.append(response["content"])
                            yield response
                    else:
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
                                stop_generation=self.stop_generation[message_id]
                        ):
                            # 收集响应以保存到内存
                            if isinstance(response, dict) and "content" in response:
                                    ai_response_parts.append(response["content"])
                            yield response
                    
                except Exception as e:
                    error_msg = f"处理RAG增强查询时出错: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    yield {"error": error_msg}
                    return
            else:
                # 不需要搜索，直接使用原始消息调用LLM
                try:
                    async for response in self.llm_service.generate_stream(
                        complete_messages,
                        config=model_config,
                        message_id=message_id,
                        stop_generation=self.stop_generation[message_id]
                    ):
                        # 收集响应以保存到内存
                        if isinstance(response, dict) and "content" in response:
                            ai_response_parts.append(response["content"])
                        yield response
                except Exception as e:
                    error_msg = f"生成回复时出错: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    yield {"error": error_msg}
                    return
            
            # 保存对话到记忆（如果提供了会话ID和用户ID）
            if session_id and user_id and ai_response_parts:
                try:
                    # 组合完整的AI响应
                    full_ai_response = "".join(ai_response_parts)
                    logger.info(f"保存对话到记忆, 用户ID: {user_id}, 会话ID: {session_id}, AI响应长度: {len(full_ai_response)}")
                    
                    # 调用保存方法
                    user_msg_id, ai_msg_id = await self.save_to_memory(
                        session_id, user_id, user_question, full_ai_response, role_id
                    )
                    
                    if user_msg_id and ai_msg_id:
                        logger.info(f"对话已保存, 用户消息ID: {user_msg_id}, AI消息ID: {ai_msg_id}")
                    else:
                        logger.warning("保存对话失败")
                except Exception as e:
                    logger.error(f"保存对话到记忆时出错: {str(e)}", exc_info=True)
                    # 保存失败不影响响应返回
                    
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
                        
                        # Update role usage count - 修正传递的参数
                        await session_role_manager.update_role_usage_count(session_id, role_id)
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
        import os
        
        try:
            # 处理API密钥
            final_api_key = api_key
            
            # 如果提供了provider和model_name，优先使用这些
            if provider and model_name:
                logger.info(f"使用指定的提供商 {provider} 和模型 {model_name}")
                
                # 如果没有提供API密钥，尝试从配置中获取
                if not final_api_key and provider in self.api_keys:
                    final_api_key = self.api_keys.get(provider, "")
                    
                # 特别处理deepseek提供商
                if provider.lower() == "deepseek" and not final_api_key:
                    # 如果通过self.api_keys获取失败，尝试直接从环境变量获取
                    env_key = os.environ.get("DEEPSEEK_API_KEY", "")
                    if env_key and env_key.strip():
                        logger.info("从环境变量获取DEEPSEEK API密钥")
                        final_api_key = env_key.strip()
                    
                # 对deepseek提供商特殊处理
                if provider.lower() == "deepseek" and final_api_key and not final_api_key.startswith("sk-"):
                    logger.info("Deepseek API密钥格式调整，添加sk-前缀")
                    final_api_key = f"sk-{final_api_key}"
                
                # 获取基础参数
                params = self._get_llm_params(provider)
                
                # 构建完整的配置
                return {
                    "provider": provider,
                    "model_name": model_name,
                    "api_key": final_api_key,
                    **params
                }
            
            # 兼容旧的model参数
            elif model:
                # 假设model格式为"provider/model_name"或直接为model_name
                if "/" in model:
                    provider_part, model_part = model.split("/", 1)
                    logger.info(f"从旧格式解析: 提供商 {provider_part}, 模型 {model_part}")
                    
                    # 如果没有提供API密钥，尝试从配置中获取
                    if not final_api_key and provider_part in self.api_keys:
                        final_api_key = self.api_keys.get(provider_part, "")
                        
                    # 对deepseek提供商特殊处理
                    if provider_part.lower() == "deepseek" and final_api_key and not final_api_key.startswith("sk-"):
                        logger.info("Deepseek API密钥格式调整，添加sk-前缀")
                        final_api_key = f"sk-{final_api_key}"
                    
                    # 获取基础参数
                    params = self._get_llm_params(provider_part)
                    
                    # 构建完整的配置
                    return {
                        "provider": provider_part,
                        "model_name": model_part,
                        "api_key": final_api_key,
                        **params
                    }
                else:
                    # 默认使用deepseek作为提供商
                    logger.info(f"只提供了模型名称 {model}，使用默认提供商 deepseek")
                    
                    # 如果没有提供API密钥，尝试从配置中获取
                    if not final_api_key and "deepseek" in self.api_keys:
                        final_api_key = self.api_keys.get("deepseek", "")
                        
                    # 对deepseek提供商特殊处理
                    if final_api_key and not final_api_key.startswith("sk-"):
                        logger.info("Deepseek API密钥格式调整，添加sk-前缀")
                        final_api_key = f"sk-{final_api_key}"
                    
                    # 获取基础参数
                    params = self._get_llm_params("deepseek")
                    
                    # 构建完整的配置
                    return {
                        "provider": "deepseek",
                        "model_name": model,
                        "api_key": final_api_key,
                        **params
                    }
            
            # 使用默认配置
            else:
                logger.info("使用默认提供商 deepseek 和模型 deepseek-chat")
                
                # 如果没有提供API密钥，尝试从配置中获取
                if not final_api_key and "deepseek" in self.api_keys:
                    final_api_key = self.api_keys.get("deepseek", "")
                    
                # 对deepseek提供商特殊处理
                if final_api_key and not final_api_key.startswith("sk-"):
                    logger.info("Deepseek API密钥格式调整，添加sk-前缀")
                    final_api_key = f"sk-{final_api_key}"
                
                # 获取基础参数
                params = self._get_llm_params("deepseek")
                
                # 构建完整的配置
                return {
                    "provider": "deepseek",
                    "model_name": "deepseek-chat",
                    "api_key": final_api_key,
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

    # 在服务启动时验证关键配置
    def validate_api_keys(self):
        if not settings.DEEPSEEK_API_KEY:
            self.logger.warning("未配置DEEPSEEK_API_KEY，Deepseek服务将不可用")

        if self.api_key:
            if not self.api_key.startswith("sk-"):
                self.logger.warning("Deepseek API密钥不是以sk-开头，自动添加前缀")
                self.api_key = f"sk-{self.api_key}"
        else:
            self.logger.error("未提供Deepseek API密钥")
            raise ValueError("使用Deepseek服务需要提供有效的API密钥")

    async def match_role_for_chat(self, messages, session_id=None, user_id=None, lang="zh", 
                            provider=None, model_name=None, api_key=None):
        """匹配最适合的角色并返回结果"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        logger.info(f"[{request_id}] 开始角色匹配: 消息数={len(messages)}, 会话ID={session_id}")
        
        try:
            # 确保服务已初始化
            init_start = time.time()
            await self._ensure_initialized()
            init_time = time.time() - init_start
            logger.debug(f"[{request_id}] 服务初始化检查完成，耗时: {init_time:.4f}秒")
            
            # 新增逻辑：如果提供了session_id，先检查该会话是否只有一个角色
            if session_id and self.session_role_manager:
                session_roles_start = time.time()
                try:
                    # 从会话中获取角色列表 - 使用SessionRoleManager而不是SessionService
                    session_roles = await self.session_role_manager.get_session_roles(session_id)
                    session_roles_time = time.time() - session_roles_start
                    logger.debug(f"[{request_id}] 获取会话角色列表完成，耗时: {session_roles_time:.4f}秒，找到 {len(session_roles)} 个角色")
                    
                    # 如果会话中只有一个角色，直接返回该角色作为匹配结果
                    if len(session_roles) == 1:
                        role = session_roles[0]
                        logger.info(f"[{request_id}] 会话只有一个角色，直接使用: {role.get('role_name', '未知角色')}")
                        
                        # 构建返回结果
                        match_result = {
                            "success": True,
                            "role": role,
                            "match_reason": "会话中只有一个可用角色，直接使用",
                            "matched_keywords": []
                        }
                        
                        # 如果会话ID存在，保存匹配角色到Redis
                        if session_id and (role.get("role_id") or role.get("id") or role.get("_id")):
                            redis_start = time.time()
                            try:
                                role_id = role.get("role_id") or role.get("id") or str(role.get("_id"))
                                await asyncio.wait_for(
                                    self.redis.hset(f"chatrag:session:{session_id}:info", "matched_role_id", role_id),
                                    timeout=2.0
                                )
                                await asyncio.wait_for(
                                    self.redis.expire(f"chatrag:session:{session_id}:info", 86400),  # 1天过期
                                    timeout=1.0
                                )
                                redis_time = time.time() - redis_start
                                logger.debug(f"[{request_id}] 保存匹配角色到Redis完成，耗时: {redis_time:.4f}秒")
                            except Exception as e:
                                logger.warning(f"[{request_id}] 保存匹配角色到Redis失败: {str(e)}")
                        
                        total_time = time.time() - start_time
                        logger.info(f"[{request_id}] 角色匹配完成（单一角色直接使用），总耗时: {total_time:.4f}秒")
                        return match_result
                except Exception as e:
                    logger.warning(f"[{request_id}] 获取会话角色时出错，将继续常规匹配: {str(e)}")
                    # 继续执行正常的匹配流程
            
            # 获取用户最后一条消息内容
            msg_process_start = time.time()
            if not messages or len(messages) == 0:
                logger.error(f"[{request_id}] 消息列表为空")
                return {"success": False, "error": "消息列表为空"}
            
            # 找到最后一条用户消息
            last_user_message = None
            for msg in reversed(messages):
                if msg.get("role") == "user" and msg.get("content"):
                    last_user_message = msg.get("content")
                    break
            
            if not last_user_message:
                logger.error(f"[{request_id}] 找不到有效的用户消息")
                return {"success": False, "error": "找不到有效的用户消息"}
            
            msg_process_time = time.time() - msg_process_start
            logger.debug(f"[{request_id}] 消息处理完成，耗时: {msg_process_time:.4f}秒")
            
            # 调用角色服务进行匹配
            match_start = time.time()
            logger.debug(f"[{request_id}] 开始调用RoleService.match_role_for_message")
            
            try:
                match_result = await asyncio.wait_for(
                    self.role_service.match_role_for_message(last_user_message, session_id),
                    timeout=25.0  # 设置25秒超时
                )
                match_time = time.time() - match_start
                logger.info(f"[{request_id}] 角色匹配服务调用完成，耗时: {match_time:.4f}秒, 结果: {match_result['success']}")
            except asyncio.TimeoutError:
                logger.error(f"[{request_id}] 角色匹配服务调用超时")
                return {"success": False, "error": "角色匹配服务调用超时"}
            except Exception as e:
                logger.error(f"[{request_id}] 角色匹配服务调用失败: {str(e)}")
                return {"success": False, "error": f"角色匹配失败: {str(e)}"}
            
            # 如果会话ID存在且匹配成功，保存匹配角色到Redis
            if session_id and match_result.get("success") and match_result.get("role"):
                redis_start = time.time()
                try:
                    role_id = match_result["role"]["id"]
                    await asyncio.wait_for(
                        self.redis.hset(f"chatrag:session:{session_id}:info", "matched_role_id", role_id),
                        timeout=2.0
                    )
                    await asyncio.wait_for(
                        self.redis.expire(f"chatrag:session:{session_id}:info", 86400),  # 1天过期
                        timeout=1.0
                    )
                    redis_time = time.time() - redis_start
                    logger.debug(f"[{request_id}] 保存匹配角色到Redis完成，耗时: {redis_time:.4f}秒")
                except Exception as e:
                    logger.warning(f"[{request_id}] 保存匹配角色到Redis失败: {str(e)}")
            
            total_time = time.time() - start_time
            logger.info(f"[{request_id}] 角色匹配完成，总耗时: {total_time:.4f}秒")
            return match_result
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"[{request_id}] 角色匹配过程中出错，耗时: {total_time:.4f}秒, 错误: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def get_role_info(self, role_id: str) -> Dict:
        """
        获取角色信息
        
        Args:
            role_id: 角色ID
            
        Returns:
            Dict: 角色信息
        """
        if not self.role_service:
            logger.warning("角色服务未初始化")
            return None
        
        try:
            role_info = await self.role_service.get_role_by_id(role_id)
            if role_info:
                return role_info
            
            logger.warning(f"未找到角色: {role_id}")
            return None
        except Exception as e:
            logger.error(f"获取角色信息失败: {str(e)}")
            return None
    
    async def generate_response(self, messages: List[dict], model: str = "deepseek-chat", 
                            session_id: str = None, user_id: str = None, role_id: str = None,
                            role_info: Dict[str, Any] = None, stream: bool = True, 
                            provider: str = None, model_name: str = None, 
                            api_key: str = None) -> AsyncGenerator:
        """
        第二阶段：基于已选择的角色生成回复
        
        专门用于两阶段API，假设角色匹配已在第一阶段完成
        
        Args:
            messages: 消息列表
            model: 模型名称
            session_id: 会话ID
            user_id: 用户ID
            role_id: 角色ID
            role_info: 角色完整信息（如果提供则不再查询数据库）
            stream: 是否启用流式输出 
            provider: LLM提供商
            model_name: 模型名称（优先级高于model）
            api_key: API密钥
            
        Yields:
            生成内容或字典
        """
        # 确保服务初始化
        await self._ensure_initialized()
        
        if not messages:
            yield "错误：消息列表为空"
            return
        
        # 获取用户最新消息
        query = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                query = msg["content"]
                break
        
        # 检查是否需要执行RAG流程
        need_rag = True
        
        # 快速规则评估
        if len(query) < 10:
            need_rag = False
            logger.info(f"快速规则触发: 问题过短，无需RAG分析, 跳过RAG分析")
        
        # 重要：使用服务中现有的内部方法构建完整上下文
        # 如果提供了role_info，就不需要再查询角色信息
        if role_id and not role_info:
            role_info = await self.get_role_info(role_id)
        
        prepared_messages = await self._prepare_messages_with_role_info(messages, role_id, role_info)
        
        # 如果需要RAG，执行知识检索流程
        if need_rag:
            # 使用现有的RAG检索方法
            rag_results = await self._retrieve_knowledge(query)
            
            # 将RAG结果添加到消息中
            if rag_results:
                logger.info(f"添加{len(rag_results)}条检索结果到上下文")
                
                # 添加RAG结果到系统提示
                prepared_messages = await self._add_rag_to_context(prepared_messages, rag_results)
                
                # 可选：将结果以字典形式返回给前端（非文本内容）
                yield {
                    "references": rag_results,
                    "type": "references"
                }
        else:
            logger.info(f"使用快速规则评估: 问题过短，无需RAG分析")
        
        # 调用LLM生成回复
        try:
            model_to_use = model_name or model
            logger.info(f"调用LLM生成回复，提供商: {provider or '默认'}, 模型: {model_to_use}")
            
            # 确保provider不为None
            effective_provider = provider
            if effective_provider is None:
                effective_provider = self.llm_service.default_config.provider.value if hasattr(self.llm_service.default_config.provider, 'value') else 'deepseek'
                logger.debug(f"使用默认提供商: {effective_provider}")
            
            if stream:
                # 使用流式生成API
                async for content in self.llm_service.generate_stream(
                    messages=prepared_messages,
                    config={
                        "model": model_to_use,
                        "provider": effective_provider,
                        "api_key": api_key
                    }
                ):
                    if isinstance(content, dict) and "content" in content:
                        yield content["content"]
                    elif hasattr(content, "content"):
                        yield content.content
                    else:
                        yield content
            else:
                # 使用非流式生成API
                response = await self.llm_service.generate_response(
                    messages=prepared_messages,
                    config={
                        "model": model_to_use,
                        "provider": effective_provider,
                        "api_key": api_key
                    }
                )
                # 返回整个响应内容
                if hasattr(response, "content"):
                    yield response.content
                else:
                    yield response
                
            # 处理消息存储
            if session_id:
                try:
                    # 收集完整回复 - 在实际实现中这部分通常在API层完成
                    pass
                except Exception as storage_err:
                    logger.error(f"存储消息失败: {str(storage_err)}")
                
        except Exception as e:
            logger.error(f"LLM生成回复时出错: {str(e)}")
            yield f"生成回复时出错: {str(e)}"

    async def _retrieve_knowledge(self, query: str) -> List[Dict[str, Any]]:
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
                
                # 只有在API密钥存在时才添加Authorization头
                if self.api_key and self.api_key.strip():
                    # 确保API密钥格式正确
                    api_key = self.api_key.strip()
                    if not api_key.startswith("skragflow-"):
                        api_key = f"skragflow-{api_key}"
                    headers["Authorization"] = f"Bearer {api_key}"
                else:
                    self.logger.warning("未提供API密钥，可能导致检索服务授权失败")
                
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
    
    async def _add_rag_to_context(self, messages: List[dict], rag_results: List[Dict[str, Any]]) -> List[dict]:
        """
        添加检索结果到消息上下文中
        
        Args:
            messages: 原始消息列表
            rag_results: 检索结果
            
        Returns:
            添加了RAG结果的消息列表
        """
        # Ensure the service is initialized
        await self._ensure_initialized()
        
        # Format retrieved documents for use in prompts
        # No need for await here since format_retrieved_documents is synchronous
        context_str = self.format_retrieved_documents(rag_results)
        
        # Create a new messages list with the same content
        new_messages = messages.copy()
        
        # If the last message is from the user, add the RAG context to the system message
        if new_messages and new_messages[-1]["role"] == "user":
            # Find a system message if one exists
            system_idx = None
            for i, msg in enumerate(new_messages):
                if msg["role"] == "system":
                    system_idx = i
                    break
            
            # If there's a system message, add the context to it
            if system_idx is not None:
                new_messages[system_idx]["content"] = new_messages[system_idx]["content"] + "\n\n参考资料:\n" + context_str
            else:
                # Otherwise, add a new system message with the context
                new_messages.insert(0, {
                    "role": "system",
                    "content": "请基于以下参考资料回答用户的问题：\n\n" + context_str
                })
        
        return new_messages
    
    async def _prepare_messages_with_role_info(self, messages: List[dict], role_id: str, role_info: Dict[str, Any] = None) -> List[dict]:
        """
        使用预先获取的角色信息准备消息列表
        
        Args:
            messages: 消息列表
            role_id: 角色ID
            role_info: 预先获取的角色信息（如果为None则会查询数据库）
            
        Returns:
            更新后的消息列表
        """
        # 确保服务已初始化
        await self._ensure_initialized()
        
        # 如果没有提供角色信息且有角色ID，获取角色信息
        if not role_info and role_id:
            role_info = await self.get_role_info(role_id)
            
        if not role_info:
            logger.warning(f"未找到角色信息，使用原始消息: {role_id}")
            return messages
        
        # 构建完整的消息列表
        complete_messages = []
        
        # 添加系统消息
        complete_messages.append({"role": "system", "content": role_info.get("system_prompt", "你是一个助手，请根据用户的问题提供有用的回答。")})
        
        # 获取历史消息（如果有会话ID和用户ID）
        if role_id:
            try:
                # 检查会话是否存在
                session_exists = await self.verify_session_exists(role_id, role_id)
                if not session_exists:
                    logger.warning(f"会话 {role_id} 不存在，跳过历史消息加载")
                else:
                    logger.info(f"开始获取会话 {role_id} 的历史消息")
                    
                    # 使用 CustomSession 获取会话信息
                    session_data = await CustomSession.get_session_by_id(role_id)
                    logger.info(f"获取会话数据: {session_data is not None}")
                    
                    if session_data:
                        # 构建Redis消息键
                        redis_key = f"messages:{role_id}:{role_id}"
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
                        logger.warning(f"在MongoDB中找不到会话 {role_id}")
            except Exception as e:
                logger.error(f"获取历史消息出错: {str(e)}", exc_info=True)
                # 如果获取历史消息失败，继续处理但记录错误
        
        # 添加当前消息
        for msg in messages:
            complete_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        return complete_messages

    async def _prepare_messages(self, messages: List[dict], role_id: str) -> List[dict]:
        """
        Prepare messages for the LLM
        
        Args:
            messages: List of messages
            role_id: Used role ID
            
        Returns:
            Updated list of messages
        """
        # 为保持兼容性，调用新的实现
        return await self._prepare_messages_with_role_info(messages, role_id, None) 