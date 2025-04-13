"""
AI服务模块 - 封装AI接口调用

提供对各种AI模型的统一接口
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Callable, Optional, AsyncGenerator

logger = logging.getLogger(__name__)

# 尝试导入OpenAI SDK
try:
    import openai
    from openai import AsyncOpenAI
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False
    logging.warning("OpenAI SDK not installed. Using mock responses.")

# 导入降级服务
from app.services.fallback_service import FallbackService

class AIService:
    """AI服务，封装对ChatGPT等模型的调用"""
    
    def __init__(self, 
                api_key: Optional[str] = None,
                model: str = "gpt-3.5-turbo",
                timeout: int = 60):
        """
        初始化AI服务
        
        Args:
            api_key: OpenAI API密钥，如不提供则从环境变量中获取
            model: 要使用的模型名称
            timeout: API调用超时时间（秒）
        """
        self.model = model
        self.timeout = timeout
        
        # 检查API是否可用
        self.available = HAS_OPENAI_SDK
        if self.available:
            try:
                # 使用提供的密钥或环境变量
                api_key = api_key or os.environ.get("OPENAI_API_KEY")
                self.client = AsyncOpenAI(api_key=api_key, timeout=timeout)
                # 没有API密钥时也认为不可用
                if not api_key:
                    self.available = False
                    logger.warning("未设置OpenAI API密钥，将使用模拟响应")
            except Exception as e:
                self.available = False
                logger.error(f"初始化OpenAI客户端失败: {str(e)}")
        
    def format_conversation_history(self, 
                                  user_message: str,
                                  conversation_history: List[Dict[str, Any]], 
                                  role_prompt: str) -> List[Dict[str, str]]:
        """
        格式化对话历史，准备API调用
        
        Args:
            user_message: 用户最新消息
            conversation_history: 对话历史，每项包含role和content字段
            role_prompt: 角色提示
            
        Returns:
            格式化后的消息列表
        """
        messages = []
        
        # 添加系统提示
        if role_prompt:
            messages.append({"role": "system", "content": role_prompt})
        
        # 添加对话历史
        if conversation_history:
            for msg in conversation_history:
                if "role" in msg and "content" in msg:
                    role = msg["role"].lower()
                    # 确保角色字段为有效值
                    if role not in ["system", "user", "assistant"]:
                        role = "user" if role in ["human", "客户", "用户"] else "assistant"
                    messages.append({"role": role, "content": msg["content"]})
        
        # 添加最新的用户消息
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    async def generate_response(self,
                              user_message: str,
                              role_prompt: str,
                              conversation_history: List[Dict[str, Any]] = None,
                              temperature: float = 0.7,
                              max_tokens: int = 2000,
                              model: Optional[str] = None) -> str:
        """
        生成AI回复
        
        Args:
            user_message: 用户消息
            role_prompt: 角色提示
            conversation_history: 对话历史
            temperature: 温度参数，控制创造性
            max_tokens: 最大生成标记数
            model: 模型名称，如不提供则使用默认模型
            
        Returns:
            AI回复内容
        """
        if not self.available:
            logger.info("使用模拟响应代替真实API调用")
            return FallbackService.generate_mock_response(user_message, role_prompt)
        
        try:
            # 准备消息
            messages = self.format_conversation_history(
                user_message=user_message,
                conversation_history=conversation_history or [],
                role_prompt=role_prompt
            )
            
            # 使用提供的模型或默认模型
            model_name = model or self.model
            
            # 调用API
            logger.info(f"调用OpenAI API, 模型: {model_name}")
            
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 提取回复内容
            reply = response.choices[0].message.content
            logger.info(f"从API获取回复，长度: {len(reply)}")
            
            return reply
        except Exception as e:
            logger.error(f"生成AI回复时出错: {str(e)}")
            # 发生错误时使用模拟响应
            return FallbackService.generate_mock_response(user_message, role_prompt)
    
    async def generate_streamed_response(self,
                                       user_message: str,
                                       role_prompt: str,
                                       conversation_history: List[Dict[str, Any]] = None,
                                       temperature: float = 0.7,
                                       max_tokens: int = 2000,
                                       model: Optional[str] = None,
                                       callback: Optional[Callable[[str], None]] = None) -> AsyncGenerator[str, None]:
        """
        流式生成AI回复
        
        Args:
            user_message: 用户消息
            role_prompt: 角色提示
            conversation_history: 对话历史
            temperature: 温度参数，控制创造性
            max_tokens: 最大生成标记数
            model: 模型名称，如不提供则使用默认模型
            callback: 每生成一个块就调用的回调函数
            
        Yields:
            回复内容的块
        """
        if not self.available:
            # 使用模拟流式响应
            mock_response = FallbackService.generate_mock_response(user_message, role_prompt)
            logger.info("使用模拟流式响应")
            # 将回复分成小块
            chunks = [mock_response[i:i+20] for i in range(0, len(mock_response), 20)]
            
            # 模拟流式输出
            for chunk in chunks:
                if callback:
                    callback(chunk)
                yield chunk
                await asyncio.sleep(0.1)  # 添加延迟，使输出看起来更真实
            return
        
        try:
            # 准备消息
            messages = self.format_conversation_history(
                user_message=user_message,
                conversation_history=conversation_history or [],
                role_prompt=role_prompt
            )
            
            # 使用提供的模型或默认模型
            model_name = model or self.model
            
            # 调用API
            logger.info(f"调用流式OpenAI API, 模型: {model_name}")
            
            stream = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            # 流式处理回复
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    if callback:
                        callback(content)
                    yield content
        except Exception as e:
            logger.error(f"生成流式AI回复时出错: {str(e)}")
            error_message = f"生成回复时发生错误: {str(e)}"
            if callback:
                callback(error_message)
            yield error_message
    
    async def get_available_models(self) -> List[str]:
        """
        获取可用模型列表
        
        Returns:
            可用模型列表
        """
        if not self.available:
            return FallbackService.get_mock_models()
            
        try:
            models = await self.client.models.list()
            model_names = [model.id for model in models.data]
            return model_names
        except Exception as e:
            logger.error(f"获取可用模型列表时出错: {str(e)}")
            return FallbackService.get_mock_models()
    
    async def switch_model(self, model_name: str) -> bool:
        """
        切换使用的模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            是否切换成功
        """
        try:
            # 检查模型是否可用
            models = await self.get_available_models()
            if model_name not in models and self.available:
                logger.warning(f"模型 {model_name} 不可用")
                return False
                
            # 切换模型
            self.model = model_name
            logger.info(f"已切换到模型: {model_name}")
            return True
        except Exception as e:
            logger.error(f"切换模型时出错: {str(e)}")
            return False 