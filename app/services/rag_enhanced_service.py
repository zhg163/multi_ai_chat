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
from app.config import config

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
from ..models.role import Role

# 创建日志记录器
logger = logging.getLogger("rag_enhanced")

# 功能开关
ENABLE_ROLE_BASED_CHAT = getattr(config, "ENABLE_ROLE_BASED_CHAT", True)
DEFAULT_SYSTEM_PROMPT = getattr(config, "DEFAULT_SYSTEM_PROMPT", 
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
        
        # 添加embedding_service初始化
        self.embedding_service = None
        
        # 新增：初始化session_role_manager (将在initialize方法中创建)
        self.session_role_manager = None
        
        # 保留兼容性的配置
        retrieval_key = config.RETRIEVAL_API_KEY
        if retrieval_key and not retrieval_key.startswith("ragflow-"):
            retrieval_key = f"ragflow-{retrieval_key.strip()}"
            
        self.retrieval_url = config.RETRIEVAL_SERVICE_URL
        self.api_key = retrieval_key
        self.ragflow_chat_id = config.RAGFLOW_CHAT_ID
        
        # 存储消息生成中止信号
        self.stop_generation = {}  # Store message IDs that need to stop generation
        self._initialized = False
        
        # 设置应用配置
        self.app_config = config
        
        # 调试模式配置
        self.debug_mode = os.environ.get("RAG_DEBUG_MODE", "False").lower() == "true"
        if self.debug_mode:
            self.logger.setLevel(logging.DEBUG)
            self.logger.info("RAG服务调试模式已启用")
            
        # 添加Redis客户端初始化为None
        self.redis = None

    @property
    def initialized(self):
        """提供对 _initialized 的访问"""
        return self._initialized
    
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
            if not api_key.startswith("ragflow-"):
                self.logger.warning("知识库检索API密钥不是以skragflow-开头，自动添加前缀")
                self.api_key = f"ragflow-{api_key}"
        
        # 确保 default_provider 属性存在，不存在则设置默认值
        if not hasattr(self, 'default_provider') or not self.default_provider:
            self.default_provider = "deepseek"
            self.logger.warning("default_provider 属性未设置，使用默认值: deepseek")

        # 确保 api_keys 属性存在，不存在则初始化
        if not hasattr(self, 'api_keys') or not self.api_keys:
            self.api_keys = {
                "openai": os.environ.get("OPENAI_API_KEY", ""),
                "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
                "google": os.environ.get("GOOGLE_API_KEY", ""),
                "deepseek": os.environ.get("DEEPSEEK_API_KEY", "")
            }
            self.logger.warning("api_keys 属性未设置，已初始化默认值")
            
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
                try:
                    from ..services.embedding_service import embedding_service
                    self.embedding_service = embedding_service
                    if not self.embedding_service.initialized:
                        logger.debug(f"[{request_id}] 嵌入服务需要显式初始化")
                        await self.embedding_service.initialize()
                except Exception as e:
                    logger.error(f"[{request_id}] 嵌入服务初始化失败: {str(e)}")
                    # 继续执行，不中断整个初始化过程
            embed_init_time = time.time() - embed_init_start
            logger.debug(f"[{request_id}] 嵌入服务初始化完成，耗时: {embed_init_time:.4f}秒")
            
            llm_init_time = time.time() - llm_init_start
            logger.debug(f"[{request_id}] LLM服务初始化完成，耗时: {llm_init_time:.4f}秒")
            
            # 获取Redis客户端
            redis_init_start = time.time()
            logger.debug(f"[{request_id}] 开始初始化Redis客户端")
            if not self.redis:
                try:
                    from ..common.redis_client import get_redis_client
                    self.redis = await asyncio.wait_for(get_redis_client(), timeout=5.0)
                except Exception as e:
                    logger.error(f"[{request_id}] Redis客户端初始化失败: {str(e)}")
                    # 继续执行，不中断整个初始化过程
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
            
            # 添加默认配置
            from app.config import settings
            if not hasattr(self, 'default_provider') or not self.default_provider:
                self.default_provider = getattr(settings, "DEFAULT_LLM_PROVIDER", "deepseek")
                logger.info(f"[{request_id}] 设置默认提供商: {self.default_provider}")
                
            if not hasattr(self, 'default_model') or not self.default_model:
                self.default_model = getattr(settings, "DEFAULT_LLM_MODEL", "deepseek-chat")
                logger.info(f"[{request_id}] 设置默认模型: {self.default_model}")
                
            if not hasattr(self, 'api_keys') or not self.api_keys:
                self.api_keys = {
                    "openai": getattr(settings, "OPENAI_API_KEY", ""),
                    "anthropic": getattr(settings, "ANTHROPIC_API_KEY", ""),
                    "google": getattr(settings, "GOOGLE_API_KEY", ""),
                    "deepseek": os.environ.get("DEEPSEEK_API_KEY", "")
                }
                logger.info(f"[{request_id}] 初始化API密钥")
            
            # 验证API密钥
            key_validate_start = time.time()
            logger.debug(f"[{request_id}] 开始验证API密钥")
            try:
                self._validate_api_keys()
            except Exception as e:
                logger.error(f"[{request_id}] API密钥验证失败: {str(e)}")
                # 继续执行，不中断整个初始化过程
            key_validate_time = time.time() - key_validate_start
            logger.debug(f"[{request_id}] API密钥验证完成，耗时: {key_validate_time:.4f}秒")
            
            self._initialized = True
            total_time = time.time() - start_time
            logger.info(f"[{request_id}] RAG服务初始化完成，总耗时: {total_time:.4f}秒")
        except Exception as e:
            total_time = time.time() - start_time
            error_message = f"[{request_id}] RAG服务初始化失败，耗时: {total_time:.4f}秒, 错误: {str(e)}"
            logger.error(error_message, exc_info=True)
            # 尝试初始化基本属性，使服务能够部分工作
            if not hasattr(self, 'default_provider') or not self.default_provider:
                self.default_provider = "deepseek"
            if not hasattr(self, 'api_keys') or not self.api_keys:
                self.api_keys = {"deepseek": os.environ.get("DEEPSEEK_API_KEY", "")}
            # 抛出异常以通知调用者
            raise ValueError(f"RAG服务初始化失败: {str(e)}")
        
        return True
    
    def should_skip_rag_analysis(self, question: str) -> tuple[bool, str]:
        """决定是否跳过RAG分析
        
        增强判断条件:
        1. 基于问题长度和复杂度评估
        2. 检测问题是否包含需要事实信息的关键词
        3. 识别闲聊vs信息查询类型问题
        
        返回:
            (是否跳过RAG, 原因说明)
        """
        # 简短问候语或简单指令不需要RAG
        if len(question) < 10:
            return True, "问题过短，无需RAG"
        
        # 检测是否为闲聊型问题
        chitchat_patterns = [
            r"你好[啊吗？！。,，]?$", 
            r"嗨[！。]?$",
            r"你[是]?[谁]?[？]?$", 
            r"谢谢[你！。]?$"
        ]
        for pattern in chitchat_patterns:
            if re.search(pattern, question):
                return True, "闲聊型问题，无需RAG"
        
        # 检测是否是请求事实信息的问题
        info_seeking_keywords = ["什么是", "如何", "为什么", "解释", "定义", 
                                 "方法", "步骤", "历史", "原因", "区别"]
        
        for keyword in info_seeking_keywords:
            if keyword in question:
                return False, f"问题包含信息查询关键词: {keyword}"
        
        # 复杂长问题可能需要RAG
        if len(question) > 50:
            return False, "复杂长问题，可能需要RAG"
            
        # 默认不使用RAG
        return True, "默认不使用RAG"

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
        检索与查询相关的文档
        
        参数:
            query: 查询文本
            
        返回:
            包含相关文档的列表
        """
        # 直接调用新的静默检索方法
        return await self._retrieve_knowledge_silent(query)
    
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
                redis_key = f"session:{session_id}"
                
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
        api_key: str = None,     # 保留此参数但不传递给generate_response
        stream: bool = True,
        role_id: str = None,
        auto_role_match: bool = False,
        show_thinking: bool = False,  # 是否显示思考过程
        rag_interface: str = None    # 新增参数，指定使用的RAG接口
    ) -> AsyncGenerator[Union[Dict, str], None]:
        """
        处理聊天请求，包括RAG流程

        参数:
            messages: 消息列表
            model: 模型名称
            session_id: 会话ID
            user_id: 用户ID
            enable_rag: 是否启用RAG
            lang: 语言
            provider: 提供商
            api_key: API密钥
            stream: 是否使用流式返回
            role_id: 角色ID
            auto_role_match: 是否启用自动角色匹配
            show_thinking: 是否显示思考过程
            rag_interface: 使用的RAG接口名称

        返回:
            消息生成的异步生成器
        """
        # 确保服务已初始化
        await self._ensure_initialized()
        
        # 获取用户最后一条消息
        user_question = ""
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                user_question = msg.get("content")
                break
        
        if not user_question:
            yield {"type": "error", "message": "未找到有效的用户消息"}
            return
        
        # 执行角色匹配（如果需要）
        role_info = None
        if auto_role_match:
            match_result = await self.match_role_for_chat(
                messages=messages,
                session_id=session_id,
                user_id=user_id,
                lang=lang,
                provider=provider,
                api_key=api_key
            )
            
            if match_result and match_result.get("success"):
                role_info = match_result.get("role", {})
                role_id = role_info.get("id") or role_info.get("role_id") or role_info.get("_id")
        
        # 如果指定了角色ID但没有角色信息，获取角色信息
        if role_id and not role_info:
            role_info = await self.get_role_info(session_id, role_id)
        
        # 确定是否使用RAG
        use_rag = False
        rag_reason = ""
        
        if enable_rag:
            should_skip, reason = self.should_skip_rag_analysis(user_question)
            use_rag = not should_skip
            rag_reason = reason
        
        # 准备处理流程
        if use_rag and show_thinking:
            # 使用RAG并显示思考过程
            # 1. 发送思考模式开启信号
            yield {"type": "thinking_mode", "enabled": True, "reason": rag_reason}
            
            # 2. 流式检索知识
            documents = []
            async for result in self._retrieve_knowledge(user_question, rag_interface):
                if result["type"] == "document":
                    # 收集文档，但不传递给客户端(这是内部类型)
                    documents.append(result["content"])
                elif result["type"] == "reference_chunk":
                    # 收集引用内容，同时将其添加到文档中
                    documents.append(result["content"])
                    # 传递给客户端
                    yield result
                else:
                    # 传递所有其他思考过程事件
                    yield result
            
            # 如果检索到文档，将其添加到上下文
            if documents:
                rag_messages = await self._add_rag_to_context(
                    messages.copy(), documents, role_info
                )
            else:
                # 未找到文档，使用原始消息
                rag_messages = messages
            
            # 准备带有角色信息的消息
            prepared_messages = await self._prepare_messages_with_role_info(
                rag_messages, role_id, role_info, session_id, user_id
            )
            
            # 生成回答 - 移除model_name和api_key参数
            async for chunk in self.generate_response(
                messages=prepared_messages,
                model=model,
                session_id=session_id,
                user_id=user_id,
                role_id=role_id,
                role_info=role_info,
                stream=True,
                provider=provider
            ):
                if isinstance(chunk, str):
                    yield {"type": "content", "content": chunk}
                else:
                    # 如果是引用或其他特殊类型数据，直接传递
                    yield chunk
        else:
            # 不显示思考过程的处理
            
            # 如果使用RAG但不显示思考过程，仍然进行检索
            rag_results = []
            if use_rag:
                try:
                    # 静默检索
                    rag_results = await self._retrieve_knowledge_silent(user_question, rag_interface)
                except Exception as e:
                    logger.error(f"静默RAG检索失败: {str(e)}")
            
            # 准备消息
            if use_rag and rag_results:
                # 添加RAG结果
                messages = await self._add_rag_to_context(
                    messages.copy(), rag_results, role_info
                )
            
            # 准备带有角色信息的消息
            prepared_messages = await self._prepare_messages_with_role_info(
                messages, role_id, role_info, session_id, user_id
            )
            
            # 生成回答
            async for chunk in self.generate_response(
                messages=prepared_messages,
                model=model,
                session_id=session_id,
                user_id=user_id,
                role_id=role_id,
                role_info=role_info,
                stream=stream,
                provider=provider
            ):
                yield chunk
    
    
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
                            provider=None, api_key=None):
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
            
            # 优先从Redis获取会话角色
            if session_id and self.session_role_manager:
                session_roles_start = time.time()
                try:
                    # 从会话中获取角色列表
                    session_roles = await self.session_role_manager.get_session_roles(session_id)
                    session_roles_time = time.time() - session_roles_start
                    logger.debug(f"[{request_id}] 获取会话角色列表完成，耗时: {session_roles_time:.4f}秒，找到 {len(session_roles)} 个角色")
                    
                    # 如果会话中没有角色，记录警告但继续执行
                    if not session_roles:
                        logger.warning(f"[{request_id}] 会话 {session_id} 中没有角色")
                    # 如果会话中只有一个角色，直接返回该角色
                    elif len(session_roles) == 1:
                        role = session_roles[0]
                        role_id = role.get("id") or role.get("role_id") or str(role.get("_id"))
                        role_name = role.get("name") or role.get("role_name", "未知角色")
                        logger.info(f"[{request_id}] 会话只有一个角色，直接使用: {role_name}")
                        
                        match_result = {
                            "success": True,
                            "role": role,
                            "match_score": 1.0,
                            "match_reason": "会话中只有一个可用角色，直接使用",
                            "matched_keywords": []
                        }
                        
                        # 如果会话ID存在，保存匹配角色到Redis
                        if session_id and role_id:
                            redis_start = time.time()
                            try:
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
            
            # 调用角色服务进行匹配（重要：确保传入session_id）
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
                    role = match_result.get("role", {})
                    role_id = role.get("id") or role.get("role_id") or str(role.get("_id"))
                    
                    if role_id:
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
    
    async def get_role_info(self, session_id: str ,role_id: str) -> Dict:
        """
        获取角色信息 - 从会话中获取简化的角色信息
        
        Args:
            role_id: 角色ID
            
        Returns:
            Dict: 简化的角色信息
        """
        # 使用新方法获取角色信息，不查询数据库
        return await self.get_session_role(session_id,role_id)
    
    async def _get_model_config(self, model, provider):
        """
        Get model configuration based on model and provider parameters
        
        Args:
            model: Model name or configuration
            provider: Provider name or configuration
            
        Returns:
            Dict: Model configuration
        """
        try:
            # Use _determine_model to get the model configuration
            model_config = self._determine_model(
                model=model,
                provider=provider,
                model_name=None,
                api_key=None
            )
            
            # If _determine_model returns a string (error message), return None
            if isinstance(model_config, str):
                logger.error(f"Failed to determine model: {model_config}")
                return None
                
            return model_config
        except Exception as e:
            logger.error(f"Error getting model configuration: {str(e)}")
            logger.exception(e)
            return None
        
    async def generate_response(
        self,
        messages: List[Dict],
        model: Union[str, Dict[str, Any]] = None,
        provider: Union[str, Dict[str, Any]] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = True,
        user_id: str = None,
        session_id: str = None,
        role_id: str = None,
        role_info: Dict = None,
        system_prompt: str = None,
    ) -> AsyncGenerator[Dict, None]:
        """生成回复
        
        参数:
            messages: 消息列表
            model: 模型名称或配置
            provider: 提供者名称或配置
            temperature: 温度参数
            top_p: top_p参数
            stream: 是否流式响应
            user_id: 用户ID
            session_id: 会话ID
            role_id: 角色ID
            role_info: 角色信息
            system_prompt: 系统提示，将覆盖role_info中的system_prompt
            
        返回:
            生成的回复
        """
        try:
            # 确保role_info是字典类型
            if role_info is None:
                role_info = {}
            elif not isinstance(role_info, dict):
                logger.warning(f"role_info不是字典类型，而是{type(role_info)}，将使用空字典替代")
                role_info = {}
                
            # 如果提供了系统提示，则覆盖角色中的系统提示
            if system_prompt:
                if not role_info:
                    role_info = {}
                role_info["system_prompt"] = system_prompt
                
            # 检查模型和提供者
            model_config = await self._get_model_config(model, provider)
            if not model_config:
                error_message = f"未找到有效的模型配置：model={model}, provider={provider}"
                logger.error(error_message)
                yield {"type": "error", "content": error_message}
                return
                
            # 准备消息
            processed_messages = await self._prepare_messages_with_role_info(
                messages=messages,
                role_id=role_id,
                role_info=role_info,
                session_id=session_id,
                user_id=user_id,
            )
            
            # 获取模型提供者和名称
            model_name = model_config["model_name"]
            provider_name = model_config["provider"]
            
            # 生成回复
            if stream:
                async for chunk in self.llm_service.generate_stream(
                    messages=processed_messages,
                ):
                    # 检查chunk是否为复杂对象，如果是则提取其文本内容
                    if hasattr(chunk, 'content'):
                        content = chunk.content
                    elif hasattr(chunk, 'text'):
                        content = chunk.text
                    elif isinstance(chunk, str):
                        content = chunk
                    else:
                        # 尝试转换为字符串
                        content = str(chunk)
                        
                    yield {"type": "content", "content": content}
            else:
                response = await self.llm_service.generate(
                    messages=processed_messages,
                )
                
                # 同样处理非流式响应
                if hasattr(response, 'content'):
                    content = response.content
                elif hasattr(response, 'text'):
                    content = response.text
                elif isinstance(response, str):
                    content = response
                else:
                    content = str(response)
                    
                yield {"type": "content", "content": content}
                
        except Exception as e:
            error_message = f"生成回复时出错: {str(e)}"
            logger.error(error_message)
            logger.exception(e)
            yield {"type": "error", "content": error_message}
    
    async def _retrieve_knowledge(self, query: str, rag_interface: str = None) -> AsyncGenerator[Dict, None]:
        """获取相关知识

        参数:
            query: 查询文本
            rag_interface: 指定的RAG接口

        返回:
            生成器，产生知识条目
        """
        try:
            # 构建请求
            url, headers, payload = self._prepare_retrieval_request(query, rag_interface)
            self.logger.info(f"准备向检索服务发送请求，URL: {url}")
            self.logger.debug(f"请求头: {headers}")
            self.logger.debug(f"请求载荷: {payload}")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                # 发送请求
                start_time = time.time()
                self.logger.info(f"开始发送检索请求...")
                
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=60.0
                )
                
                # 记录响应时间
                elapsed_time = time.time() - start_time
                self.logger.info(f"检索响应时间: {elapsed_time:.2f}秒")
                
                # 检查响应状态码
                if response.status_code == 401:
                    self.logger.error(f"检索API认证失败 (401 Unauthorized): {response.text}")
                    yield {"type": "error", "message": "知识库访问权限验证失败"}
                    return
                
                if response.status_code != 200:
                    self.logger.error(f"检索API请求失败 (HTTP {response.status_code}): {response.text}")
                    yield {"type": "error", "message": f"知识库请求失败: HTTP {response.status_code}"}
                    return
                
                # 记录原始响应
                self.logger.info(f"接收到响应，状态码: {response.status_code}")
                self.logger.debug(f"检索服务原始响应: {response.text[:500]}...")
                
                try:
                    # 记录关键调试信息
                    self.logger.info(f"开始解析JSON响应...")
                    
                    # 解析JSON响应
                    response_text = response.text
                    self.logger.debug(f"响应文本类型: {type(response_text)}")
                    
                    # 使用json.loads而不是response.json()以便更好地控制错误处理
                    response_data = json.loads(response_text)
                    
                    # 详细记录响应数据类型和内容
                    self.logger.info(f"响应数据类型: {type(response_data)}")
                    self.logger.debug(f"响应数据内容: {str(response_data)[:200]}...")
                    
                    # 类型检查
                    if not isinstance(response_data, dict):
                        self.logger.error(f"错误: 响应数据不是字典类型! 实际类型: {type(response_data)}")
                        yield {"type": "error", "message": "知识库返回了非预期的数据格式"}
                        return
                    
                    # 安全地获取code并记录
                    if "code" not in response_data:
                        self.logger.error("响应数据中缺少'code'字段")
                        response_code = None
                    else:
                        response_code = response_data["code"]
                        self.logger.info(f"响应代码: {response_code}")
                    
                    # 检查响应状态码
                    if response_code != 0:
                        error_msg = response_data.get("msg", "未知错误")
                        self.logger.error(f"检索服务返回错误代码 {response_code}: {error_msg}")
                        yield {"type": "error", "message": f"知识库返回错误: {error_msg}"}
                        return
                    
                    # 检查data字段
                    if "data" not in response_data:
                        self.logger.error("响应数据中缺少'data'字段")
                        yield {"type": "error", "message": "知识库返回的数据结构缺少必要字段"}
                        return
                    
                    # 类型检查data字段
                    data_field = response_data["data"]
                    self.logger.info(f"data字段类型: {type(data_field)}")
                    
                    if not isinstance(data_field, dict):
                        self.logger.error(f"data字段不是字典类型! 实际类型: {type(data_field)}")
                        yield {"type": "error", "message": "知识库返回了非预期的数据格式"}
                        return
                    
                    # 检查chunks字段
                    if "chunks" not in data_field:
                        self.logger.error("data字段中缺少'chunks'字段")
                        yield {"type": "error", "message": "知识库返回的数据结构缺少必要字段"}
                        return
                    
                    # 类型检查chunks字段
                    chunks_field = data_field["chunks"]
                    self.logger.info(f"chunks字段类型: {type(chunks_field)}")
                    
                    if not isinstance(chunks_field, list):
                        self.logger.error(f"chunks字段不是列表类型! 实际类型: {type(chunks_field)}")
                        yield {"type": "error", "message": "知识库返回了非预期的数据格式"}
                        return
                    
                    # 处理检索结果
                    self.logger.info(f"成功检索到 {len(chunks_field)} 个结果块")
                    
                    for chunk in chunks_field:
                        # 类型检查每个chunk
                        if not isinstance(chunk, dict):
                            self.logger.warning(f"跳过非字典类型的检索结果: {type(chunk)}")
                            continue
                        
                        # 记录每个chunk的内容
                        self.logger.debug(f"处理检索结果块: {str(chunk)[:100]}...")
                        
                        yield {
                            "type": "chunk",
                            "content": chunk.get("content", ""),
                            "source": chunk.get("source", "未知来源"),
                            "score": chunk.get("score", 0),
                            "metadata": chunk.get("metadata", {}),
                        }
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON解析错误: {str(e)}")
                    self.logger.error(f"无法解析的响应文本: {response.text[:200]}...")
                    yield {"type": "error", "message": "知识库返回了无效的数据格式"}
                    return
                
                except Exception as e:
                    self.logger.error(f"处理检索响应时发生异常: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    yield {"type": "error", "message": f"处理知识库响应时出错: {str(e)}"}
                    return
                
        except httpx.TimeoutException:
            self.logger.error("检索请求超时")
            yield {"type": "error", "message": "知识库请求超时"}
            
        except httpx.RequestError as e:
            self.logger.error(f"检索请求错误: {str(e)}")
            yield {"type": "error", "message": f"知识库请求错误: {str(e)}"}
            
        except Exception as e:
            self.logger.error(f"检索过程中发生异常: {str(e)}")
            self.logger.error(traceback.format_exc())
            yield {"type": "error", "message": f"知识库检索错误: {str(e)}"}

    def _prepare_retrieval_request(self, query: str, rag_interface: str = None) -> Tuple[str, Dict, Dict]:
        """准备检索请求的URL、头信息和载荷
        
        参数:
            query: 查询文本
            rag_interface: 指定的RAG接口名称
            
        返回:
            (url, headers, payload)元组
        """
        # 添加日志记录传入的参数
        self.logger.info(f"准备检索请求: query={query}, rag_interface={rag_interface}")
        
        # 获取RAG接口配置
        interface_config = None
        if hasattr(self.app_config, 'get_rag_interface'):
            self.logger.debug(f"使用app_config.get_rag_interface获取接口配置")
            interface_config = self.app_config.get_rag_interface(rag_interface)
            if interface_config is None:
                self.logger.error(f"获取到的RAG接口配置为None！rag_interface={rag_interface}")
                raise ValueError(f"获取RAG接口配置失败，接口名称: {rag_interface}")
            self.logger.info(f"获取到RAG接口配置: interface_name={interface_config.name}")
        else:
            # 兼容处理：如果app_config没有get_rag_interface方法，则创建一个默认配置
            from app.services.rag_interface_config import RagInterfaceConfig
            self.logger.warning("Config对象没有get_rag_interface方法，使用默认配置")
            # 导入os模块获取环境变量
            import os
            # 获取默认数据集ID
            default_dataset_id = os.getenv("DEFAULT_DATASET_ID", "default_dataset")
            self.logger.info(f"使用环境变量中的默认数据集ID: {default_dataset_id}")
            interface_config = RagInterfaceConfig(
                name="default",
                base_url=self.app_config.RETRIEVAL_SERVICE_URL.replace("/api/chat", ""),
                api_key=self.app_config.RETRIEVAL_API_KEY,
                dataset_ids=[default_dataset_id],  # 使用默认数据集ID
                document_ids=[]
            )
            
        # 再次验证接口配置不为空
        if interface_config is None:
            self.logger.error("无法获取有效的RAG接口配置！")
            raise ValueError("无法获取有效的RAG接口配置")
        
        # 准备请求URL
        retrieval_url = f"{interface_config.base_url}/api/v1/retrieval"
        
        # 准备API密钥
        api_key = interface_config.api_key
        if api_key and not api_key.startswith("ragflow-"):
            api_key = f"ragflow-{api_key.strip()}"
        
        # 记录调试信息
        if self.debug_mode:
            self.logger.debug(f"使用RAG接口: {interface_config.name}")
            self.logger.debug(f"检索URL: {retrieval_url}")
            self.logger.debug(f"API密钥前缀: {api_key[:10] if api_key else 'None'}...")
    
        # 准备请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 准备请求体
        payload = {
            "question": query,
            "page": 1,                     # 添加页码参数
            "page_size": 30,               # 添加每页大小参数
            "similarity_threshold": 0.5,
            "vector_similarity_weight": 0.8,
            "top_k": 1024,
            "keyword": False,
            "highlight": True
        }
        
        # 添加详细日志检查dataset_ids状态
        self.logger.info(f"检查dataset_ids: {interface_config.dataset_ids}")
        
        # 如果配置了数据集ID，添加到请求中
        if interface_config.dataset_ids:
            payload["dataset_ids"] = interface_config.dataset_ids
            self.logger.info(f"已添加dataset_ids到payload: {interface_config.dataset_ids}")
        else:
            # 缺少必要的dataset_ids参数
            self.logger.error("检索请求缺少必要的dataset_ids参数")
            # 记录环境变量状态以便调试
            import os
            self.logger.error(f"当前环境变量: DEFAULT_DATASET_ID={os.getenv('DEFAULT_DATASET_ID')}, SECONDARY_RAG_DATASET_IDS={os.getenv('SECONDARY_RAG_DATASET_IDS')}")
            raise ValueError("检索请求缺少必要的dataset_ids参数，请在.env文件中设置SECONDARY_RAG_DATASET_IDS环境变量")
            
        # 如果配置了文档ID，添加到请求中
        if interface_config.document_ids:
            payload["document_ids"] = interface_config.document_ids
            
        return retrieval_url, headers, payload

    async def _retrieve_knowledge_silent(self, query: str, rag_interface: str = None) -> List[Dict]:
        """
        从知识库检索相关信息 (无流式返回)
        
        参数:
            query: 用户查询
            rag_interface: 使用的RAG接口名称，如果为None则使用默认接口
            
        返回:
            检索结果列表
        """
        try:
            # 准备请求
            url, headers, payload = self._prepare_retrieval_request(query, rag_interface)
            self.logger.info(f"准备静默检索请求，URL: {url}")
            
            # 发送请求并获取结果
            return await self._send_retrieval_request(url, headers, payload)
            
        except Exception as e:
            self.logger.error(f"静默检索过程中发生异常: {str(e)}")
            self.logger.error(traceback.format_exc())
            return []

    async def _send_retrieval_request(self, url: str, headers: Dict, payload: Dict) -> List[Dict]:
        """发送检索请求并处理响应"""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # 发送请求
                start_time = time.time()
                self.logger.info(f"开始发送检索请求到: {url}")
                
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=60.0
                )
                
                # 记录响应时间
                elapsed_time = time.time() - start_time
                self.logger.info(f"检索响应时间: {elapsed_time:.2f}秒")
                
                # 记录状态码
                self.logger.info(f"检索响应状态码: {response.status_code}")
                
                # 检查响应状态码
                if response.status_code == 401:
                    self.logger.error(f"检索API认证失败 (401 Unauthorized): {response.text}")
                    return []
                
                if response.status_code != 200:
                    self.logger.error(f"检索API请求失败 (HTTP {response.status_code}): {response.text}")
                    return []
                
                # 记录原始响应
                self.logger.debug(f"检索服务原始响应: {response.text[:200]}...")
                
                try:
                    # 记录响应文本类型
                    response_text = response.text
                    self.logger.debug(f"响应文本类型: {type(response_text)}, 长度: {len(response_text)}")
                    
                    # 解析JSON响应前记录
                    self.logger.info(f"开始解析JSON响应...")
                    
                    # 解析JSON响应
                    try:
                        response_data = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON解析错误: {str(e)}")
                        self.logger.error(f"无法解析的响应文本: {response_text[:500]}...")
                        return []
                    
                    # 详细记录响应数据
                    self.logger.info(f"检索服务响应类型: {type(response_data)}")
                    self.logger.debug(f"响应数据: {str(response_data)[:300]}...")
                    
                    # 添加类型检查
                    if not isinstance(response_data, dict):
                        self.logger.error(f"错误: 响应数据不是字典类型! 实际类型: {type(response_data)}, 值: {str(response_data)[:200]}...")
                        return []
                    
                    # 安全地检查code字段
                    if "code" not in response_data:
                        self.logger.error("响应数据中缺少'code'字段")
                        return []
                    
                    # 检查响应状态码
                    response_code = response_data["code"]
                    self.logger.debug(f"响应状态码: {response_code}")
                    
                    if response_code != 0:
                        error_msg = response_data.get("msg", "未知错误")
                        self.logger.error(f"检索服务返回错误: {error_msg}")
                        return []
                    
                    # 检查data字段是否存在
                    if "data" not in response_data:
                        self.logger.error("响应数据中缺少'data'字段")
                        return []
                    
                    # 检查data字段类型
                    data_field = response_data["data"]
                    self.logger.debug(f"data字段类型: {type(data_field)}")
                    
                    if not isinstance(data_field, dict):
                        self.logger.error(f"data字段不是字典类型! 实际类型: {type(data_field)}")
                        return []
                    
                    # 检查chunks字段是否存在
                    if "chunks" not in data_field:
                        self.logger.error("data字段中缺少'chunks'字段")
                        return []
                    
                    # 获取检索结果并记录
                    data_chunks = data_field["chunks"]
                    self.logger.debug(f"chunks字段类型: {type(data_chunks)}")
                    
                    if not isinstance(data_chunks, list):
                        self.logger.error(f"chunks字段不是列表类型! 实际类型: {type(data_chunks)}")
                        return []
                    
                    self.logger.info(f"检索到 {len(data_chunks)} 个结果块")
                    return data_chunks
                    
                except json.JSONDecodeError:
                    self.logger.error(f"无法解析检索API响应: {response.text[:200]}...")
                    return []
                
                except Exception as e:
                    self.logger.error(f"处理检索响应异常: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    return []
                
        except httpx.TimeoutException:
            self.logger.error("检索请求超时")
            return []
            
        except httpx.RequestError as e:
            self.logger.error(f"检索请求错误: {str(e)}")
            return []
            
        except Exception as e:
            self.logger.error(f"检索过程中发生异常: {str(e)}")
            self.logger.error(traceback.format_exc())
            return []
    
    async def _add_rag_to_context(self, messages: List[dict], rag_results: List[Dict[str, Any]], role_info: Dict = None) -> List[dict]:
        """将RAG结果添加到消息上下文中，增强角色风格
        
        参数:
            messages: 原始消息列表
            rag_results: 检索到的相关文档
            role_info: 角色信息
            
        返回:
            增强后的消息列表
        """
        if not rag_results:
            return messages
        
        # 格式化检索结果
        context = self.format_retrieved_documents(rag_results)
        
        # 如果提供了角色信息，使用角色感知的RAG提示
        if role_info:
            system_prompt = self._build_role_aware_rag_prompt(context, role_info)
        else:
            # 使用默认格式
            system_prompt = f"""请基于以下参考信息回答用户的问题：

{context}

如果参考信息不足以回答问题，可以使用自己的知识，但请明确指出。
请直接回答问题，不要提及你在使用"参考信息"。
"""
        
        # 查找是否已有系统消息
        has_system_message = False
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                # 增强现有系统消息
                messages[i]["content"] = system_prompt
                has_system_message = True
                break
        
        # 如果没有系统消息，添加一个
        if not has_system_message:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        return messages
    
    async def _prepare_messages_with_role_info(
        self, 
        messages: List[dict], 
        role_id: str, 
        role_info: Dict[str, Any] = None, 
        session_id: str = None,
        user_id: str = None
    ) -> List[dict]:
        """
        Prepare a list of messages by using role information
        
        Args:
            messages: List of messages
            role_id: Role ID
            role_info: Role information, default is None
            session_id: Session ID, default is None
            user_id: User ID, default is None
            
        Returns:
            Updated list of messages
        """
        # 确保role_info是字典类型
        if role_info is None:
            role_info = {}
        elif not isinstance(role_info, dict):
            logger.warning(f"role_info不是字典类型，而是{type(role_info)}，将使用空字典替代")
            role_info = {}
            
        # 确保服务已初始化
        if not self.initialized:
            await self.initialize()

        # 获取历史消息
        processed_messages = []  # 创建一个新列表存储处理后的消息
        if session_id and user_id:
            try:
                # 验证会话是否存在
                session_exists = await self.verify_session_exists(session_id, user_id)
                if not session_exists:
                    logging.warning(f"Session {session_id} does not exist for user {user_id}")
                else:
                    # 获取 Redis 中的历史消息
                    redis_key = f"messages:{user_id}:{session_id}"
                    redis_messages = await self.redis.lrange(redis_key, 0, -1)
                    logger.info(f"从Redis获取到 {len(redis_messages)} 条消息")
                    
                    history = []
                    
                    # 解析消息记录
                    for msg_data in redis_messages:
                        try:
                            msg = json.loads(msg_data)
                            if msg.get("role") in ["user", "assistant", "system"]:
                                history.append(msg)
                        except json.JSONDecodeError:
                            logger.warning(f"解析消息失败: {msg_data[:100]}")
                            continue
                        
                    # 将历史消息添加到处理后的消息列表中
                    logger.info(f"成功解析 {len(history)} 条有效历史消息")
                    for msg in history:
                        processed_messages.append({
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", "")
                        })
            except Exception as e:
                logger.error(f"获取历史消息出错: {str(e)}", exc_info=True)
                # 如果获取历史消息失败，继续处理但记录错误
        
        # 添加当前消息
        for msg in messages:
            processed_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        # 构建完整的消息列表
        complete_messages = []
        
        # 添加系统消息
        system_prompt = "你是一个助手，请根据用户的问题提供有用的回答。"
        if isinstance(role_info, dict):
            system_prompt = role_info.get("system_prompt", system_prompt)
            
        complete_messages.append({"role": "system", "content": system_prompt})
        
        # 添加历史消息和当前消息
        complete_messages.extend(processed_messages)
        
        return complete_messages

    async def _prepare_messages(self, messages: List[dict], role_id: str, user_id: str = None) -> List[dict]:
        """
        Prepare messages for the LLM
        
        Args:
            messages: List of messages
            role_id: Used role ID
            user_id: User ID (optional)
            
        Returns:
            Updated list of messages
        """
        # 为保持兼容性，调用新的实现
        return await self._prepare_messages_with_role_info(messages, role_id, None, role_id, user_id)

    def _build_role_aware_rag_prompt(self, context: str, role_info: Dict) -> str:
        """构建保持角色风格的RAG提示模板
        
        参数:
            context: 检索到的上下文信息
            role_info: 角色信息
            
        返回:
            增强后的提示模板
        """
        # 确保role_info是字典类型
        if role_info is None:
            role_info = {}
        elif not isinstance(role_info, dict):
            logger.warning(f"role_info不是字典类型，而是{type(role_info)}，将使用空字典替代")
            role_info = {}
            
        role_name = role_info.get("name", "助手")
        personality = role_info.get("personality", "")
        speech_style = role_info.get("speech_style", "")
        
        # 获取原始系统提示
        original_prompt = role_info.get("system_prompt", "你是一个助手，请回答用户的问题。")
        
        # 构建增强提示
        rag_prompt = f"""{original_prompt}

在回答时，请参考以下信息:
{context}

请确保你的回答满足以下要求:
1. 保持角色设定: 你是{role_name}{", " + personality if personality else ""}
2. 使用适当的说话风格: {speech_style if speech_style else "自然、专业的语气"}
3. 回答要基于提供的参考信息，确保信息准确
4. 如果参考信息不足以回答问题，可以使用你的知识，但要清晰说明

请直接回答问题，不要提及你在使用"参考信息"。
"""
        return rag_prompt

    async def get_session_role(self, session_id: str, role_id: str = None, role_name: str = None) -> Dict:
        """
        从会话中获取角色信息，并返回简化的角色对象
        
        Args:
            session_id: 会话ID
            role_id: 角色ID（可选）
            role_name: 角色名称（可选）
            
        Returns:
            Dict: 简化的角色信息
        """
        if not session_id:
            logger.warning("未提供会话ID")
            return None
        
        try:
            # 使用session_role_manager获取会话角色列表
            if not self.session_role_manager:
                logger.warning("会话角色管理器未初始化")
                return None
                
            session_roles = await self.session_role_manager.get_session_roles(session_id)
            
            if not session_roles:
                logger.warning(f"会话 {session_id} 中没有角色")
                return None
            
            # 查找指定角色
            target_role = None
            
            # 1. 如果提供了role_id，按ID查找
            if role_id:
                for role in session_roles:
                    current_role_id = role.get("role_id") or role.get("id") or str(role.get("_id"))
                    if current_role_id and str(current_role_id) == str(role_id):
                        target_role = role
                        break
            
            # 2. 如果提供了role_name且没找到角色，按名称查找
            elif role_name and not target_role:
                for role in session_roles:
                    current_role_name = role.get("role_name") or role.get("name")
                    if current_role_name and current_role_name == role_name:
                        target_role = role
                        break
            
            # 3. 如果没有提供ID或名称且会话只有一个角色，直接使用该角色
            elif len(session_roles) == 1:
                target_role = session_roles[0]
            
            # 构建简化的角色信息对象
            if target_role:
                role_info = {
                    "role_id": target_role.get("role_id") or target_role.get("id") or str(target_role.get("_id")),
                    "name": target_role.get("role_name") or target_role.get("name", "未知角色"),
                    "system_prompt": target_role.get("system_prompt", "你是一个助手，请回答用户的问题。")
                }
                return role_info
            
            logger.warning(f"在会话 {session_id} 中未找到匹配的角色")
            return None
            
        except Exception as e:
            logger.error(f"从会话获取角色信息失败: {str(e)}")
            return None