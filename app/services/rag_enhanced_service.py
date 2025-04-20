from typing import Dict, Any, List, Optional, AsyncGenerator, Union, Tuple
import logging
import httpx
import re
import json
import uuid
from datetime import datetime

# 导入配置
from app.config import config

from ..services.llm_service import LLMService
from ..services.message_service import MessageService
from ..services.session_service import SessionService
from ..services.role_service import RoleService

class RAGEnhancedService:
    """RAG增强服务 - 提供自动判断是否需要RAG，以及优化的RAG检索和生成功能"""
    
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
        self.retrieval_url = config.RETRIEVAL_SERVICE_URL
        self.api_key = config.RETRIEVAL_API_KEY
        self.ragflow_chat_id = config.RAGFLOW_CHAT_ID
        self.stop_generation = {}  # 存储需要停止生成的消息ID
        self._initialized = False
    
    async def initialize(self):
        """异步初始化服务"""
        if self._initialized:
            return
            
        # 初始化消息服务
        if hasattr(self.message_service, 'initialize'):
            await self.message_service.initialize()
            
        # 初始化会话服务
        if hasattr(self.session_service, 'initialize'):
            await self.session_service.initialize()
            
        # 初始化角色服务
        if hasattr(self.role_service, 'initialize'):
            await self.role_service.initialize()
            
        self._initialized = True
        
    async def _ensure_initialized(self):
        """确保服务已初始化"""
        if not self._initialized:
            await self.initialize()
    
    async def analyze_question(self, question: str, model: str) -> Tuple[bool, str]:
        """
        分析问题，判断是否需要从知识库检索信息
        
        Args:
            question: 用户问题
            model: 使用的LLM模型 (已忽略，强制使用deepseek-chat)
            
        Returns:
            (need_rag, thinking): 是否需要RAG，分析思考过程
        """
        # 确保服务已初始化
        await self._ensure_initialized()
        
        self.logger.info(f"分析问题是否需要RAG: {question[:50]}...")
        
        # 强制使用deepseek-chat模型
        model = "deepseek-chat"
        
        # 构建分析提示词
        prompt = f"""请分析下面这个问题，并判断你是否需要检索外部知识库才能给出准确回答。

问题: {question}

请通过以下步骤分析:
1. 问题类型分析：这是什么类型的问题？是一般知识、特定领域专业知识，还是实时信息？
2. 必要知识评估：回答这个问题需要哪些具体知识？
3. 知识覆盖分析：你是否拥有回答所需的全部知识？有没有信息缺口？
4. RAG判断：是否需要检索外部知识库来补充缺失信息？

最后，请明确给出判断结果，格式为：
【需要检索】：是/否
【分析理由】：简要解释你的判断依据

请保持分析简洁、有逻辑，注重与问题的相关性。"""

        # 调用LLM进行分析
        try:
            response = await self.llm_service.chat_completion(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一位负责判断问题是否需要外部知识的分析助手。请分析给定问题并判断是否需要检索外部知识库。"},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                temperature=0.1,
                max_tokens=500
            )
            
            analysis = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            self.logger.info(f"问题分析结果: {analysis[:100]}...")
            
            # 提取是否需要RAG的判断
            need_rag = False
            if "【需要检索】：是" in analysis or "【需要检索】:是" in analysis:
                need_rag = True
            
            return need_rag, analysis
        except Exception as e:
            self.logger.error(f"分析问题失败: {str(e)}")
            # 出错时默认使用RAG
            return True, f"分析过程中出错，为安全起见将使用RAG。错误信息: {str(e)}"
    
    async def retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        从知识库中检索相关文档
        
        Args:
            query: 查询内容
            
        Returns:
            检索到的文档列表
        """
        # 确保服务已初始化
        await self._ensure_initialized()
        
        self.logger.info(f"从知识库检索内容: {query[:50]}...")
        
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
                    self.logger.error(f"从知识库检索失败: {response.status_code}, {response.text}")
                    return []
                
                result = response.json()
                documents = result.get("documents", [])
                
                if documents:
                    self.logger.info(f"检索到 {len(documents)} 个相关文档")
                else:
                    self.logger.warning("没有检索到相关文档")
                
                return documents
        except Exception as e:
            self.logger.error(f"检索文档时发生错误: {str(e)}")
            return []
    
    def format_retrieved_documents(self, documents: List[Dict[str, Any]]) -> str:
        """
        格式化检索到的文档，以便于在提示中使用
        
        Args:
            documents: 文档列表
            
        Returns:
            格式化后的文档字符串
        """
        # 注意：此方法不需要确保初始化，因为它不访问数据库
        
        if not documents:
            return "未找到相关参考资料。"
        
        formatted_text = "以下是相关参考资料：\n\n"
        
        for i, doc in enumerate(documents, 1):
            title = doc.get("title", "未知文档")
            content = doc.get("content", "")
            source = doc.get("source", "未知来源")
            score = doc.get("score", 0)
            
            formatted_text += f"[{i}] {title}\n"
            formatted_text += f"来源: {source}\n"
            formatted_text += f"相关度: {score:.2f}\n"
            formatted_text += f"内容: {content}\n\n"
        
        return formatted_text
    
    async def process_chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        role_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = True,
        message_id: Optional[str] = None,
        context_limit: Optional[int] = None,
        auto_title: bool = False
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """
        处理RAG增强的聊天
        
        Args:
            messages: 消息列表
            model: 使用的模型 (已忽略，强制使用deepseek-chat)
            session_id: 会话ID
            user_id: 用户ID
            role_id: 角色ID
            temperature: 温度参数
            max_tokens: 最大token数
            stream: 是否使用流式响应
            message_id: 消息ID，用于继续生成
            context_limit: 上下文窗口限制
            auto_title: 是否自动生成标题
            
        Yields:
            流式响应内容
        """
        # 确保服务已初始化
        await self._ensure_initialized()
        
        # 参数验证
        if not messages or not isinstance(messages, list) or len(messages) == 0:
            yield {"error": "消息列表为空或格式错误"}
            return
            
        # 验证必须的消息格式
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                yield {"error": "消息格式错误，必须包含role和content字段"}
                return
        
        # 强制使用deepseek-chat模型
        model = "deepseek-chat"
        
        # 参数规范化
        temperature = float(temperature) if temperature is not None else 0.7
        if temperature < 0 or temperature > 2:
            temperature = 0.7
            
        if max_tokens is not None:
            max_tokens = int(max_tokens)
            
        if context_limit is not None:
            context_limit = int(context_limit)
            
        # 生成消息ID
        current_message_id = message_id or str(uuid.uuid4())
        
        # 提取最后一个用户问题
        last_question = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_question = msg.get("content", "")
                break
                
        if not last_question:
            yield {"error": "未找到用户问题"}
            return
        
        # STEP 1: 分析问题
        yield "【分析问题中...】\n"
        need_rag, thinking = await self.analyze_question(last_question, model)
        
        # STEP 2: 如果需要RAG，检索文档
        documents = []
        references = []
        if need_rag:
            yield f"\n【问题分析】\n{thinking}\n\n【需要查询知识库】\n"
            documents = await self.retrieve_documents(last_question)
            
            if documents:
                yield "【找到相关资料】\n"
                
                # 准备引用信息
                for doc in documents:
                    references.append({
                        "title": doc.get("title", "未知文档"),
                        "content": doc.get("content", "")[:300],
                        "score": doc.get("score", 0),
                        "source": doc.get("source", "未知来源")
                    })
            else:
                yield "【未找到相关资料，将尝试直接回答】\n"
        else:
            yield f"\n【问题分析】\n{thinking}\n\n【无需查询知识库，可直接回答】\n"
            
        # STEP 3: 构建增强提示词
        formatted_docs = ""
        if documents:
            formatted_docs = self.format_retrieved_documents(documents)
            
        system_message = "你是一位知识渊博的助手。请基于用户问题回答，如果提供了参考资料，请参考这些资料。"
        
        # 如果有角色ID，获取角色系统提示词
        if role_id:
            try:
                # 确保服务已初始化（特别是角色服务）
                await self._ensure_initialized()
                
                # 通过角色服务获取角色系统提示词
                role_info = await self.role_service.get_role_by_id(role_id)
                if role_info and isinstance(role_info, dict):
                    # 使用字典语法获取system_prompt，避免KeyError
                    system_prompt_value = role_info.get("system_prompt")
                    if system_prompt_value:
                        system_message = system_prompt_value
            except Exception as e:
                self.logger.error(f"获取角色系统提示词失败: {str(e)}")
        
        # 构建增强系统消息
        system_message += "\n\n请在回答中清晰地分为两部分：【思考过程】和【最终回答】。在思考过程中分析问题和参考资料，然后给出准确的最终回答。"
        
        # 构建增强消息列表
        enhanced_messages = []
        
        # 替换系统消息
        system_found = False
        for msg in messages:
            if msg.get("role") == "system":
                enhanced_messages.append({"role": "system", "content": system_message})
                system_found = True
            else:
                enhanced_messages.append(msg)
                
        if not system_found:
            enhanced_messages.insert(0, {"role": "system", "content": system_message})
            
        # 如果有检索到的文档，添加到最后一个用户消息中
        if documents:
            # 找到最后一个用户消息并增强
            for i in range(len(enhanced_messages) - 1, -1, -1):
                if enhanced_messages[i].get("role") == "user":
                    enhanced_content = f"{enhanced_messages[i].get('content', '')}\n\n参考资料:\n{formatted_docs}"
                    enhanced_messages[i]["content"] = enhanced_content
                    break
            
        # STEP 4: 调用LLM API
        yield "\n【生成回答中...】\n\n"
        
        full_response = ""
        
        # 流式响应处理
        if stream:
            async for chunk in self.llm_service.chat_completion_stream(
                model=model,
                messages=enhanced_messages,
                temperature=temperature,
                max_tokens=max_tokens
            ):
                # 检查是否需要停止生成
                if current_message_id in self.stop_generation:
                    del self.stop_generation[current_message_id]
                    yield {"stopped": True, "message_id": current_message_id}
                    break
                
                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if content:
                    full_response += content
                    yield content
        else:
            response = await self.llm_service.chat_completion(
                model=model,
                messages=enhanced_messages,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            full_response = content
            yield content
        
        # 如果需要返回引用信息
        if references:
            yield {"references": references, "message_id": current_message_id}
    
    async def stop_message_generation(self, message_id: str) -> bool:
        """
        停止指定消息的生成
        
        Args:
            message_id: 消息ID
            
        Returns:
            是否成功停止生成
        """
        # 确保服务已初始化
        await self._ensure_initialized()
        
        if message_id not in self.stop_generation:
            return False
        
        self.stop_generation[message_id] = True
        return True 