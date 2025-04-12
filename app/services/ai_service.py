"""
AI服务模块

负责与AI模型交互，生成回复内容，支持多种模型和流式输出
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from datetime import datetime

# TODO: 根据项目需求选择合适的AI API客户端库
# 如使用OpenAI API，则导入：from openai import AsyncOpenAI
# 如使用DeepSeek API，则导入适当的SDK
# 此处以OpenAI为例
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI SDK not installed. Using mock responses.")

logger = logging.getLogger(__name__)

class AIService:
    """
    AI服务类
    
    负责与AI模型交互，生成回复内容，支持多种模型和流式输出
    """
    
    def __init__(self, 
                api_key: Optional[str] = None,
                model: str = "gpt-3.5-turbo",
                timeout: int = 60):
        """
        初始化AI服务
        
        Args:
            api_key: API密钥
            model: 默认使用的模型名称
            timeout: API超时时间（秒）
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        
        # 初始化客户端（如果可用）
        if OPENAI_AVAILABLE and api_key:
            self.client = AsyncOpenAI(api_key=api_key, timeout=timeout)
            self.available = True
        else:
            self.client = None
            self.available = False
            
        logger.info(f"AI服务初始化完成，使用模型: {model}, API可用: {self.available}")
    
    def format_conversation_history(self, 
                                  user_message: str,
                                  conversation_history: List[Dict[str, Any]], 
                                  role_prompt: str) -> List[Dict[str, str]]:
        """
        格式化对话历史记录为模型可接受的格式
        
        Args:
            user_message: 当前用户消息
            conversation_history: 对话历史记录
            role_prompt: 角色提示
            
        Returns:
            格式化后的消息列表
        """
        # 创建系统提示
        messages = [{"role": "system", "content": role_prompt}]
        
        # 添加历史消息
        for msg in conversation_history:
            if msg.get("message_type") == "user":
                messages.append({
                    "role": "user", 
                    "content": msg.get("content", "")
                })
            elif msg.get("message_type") == "assistant":
                messages.append({
                    "role": "assistant", 
                    "content": msg.get("content", "")
                })
        
        # 添加最新的用户消息（如果不在历史记录中）
        if not conversation_history or conversation_history[-1].get("message_type") != "user" or conversation_history[-1].get("content") != user_message:
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
        生成AI回复（非流式）
        
        Args:
            user_message: 用户消息
            role_prompt: 角色提示信息
            conversation_history: 对话历史
            temperature: 生成温度
            max_tokens: 最大生成令牌数
            model: 使用的模型，默认使用初始化时指定的模型
            
        Returns:
            生成的回复内容
        """
        if not self.available:
            return self.generate_mock_response(user_message, role_prompt)
            
        try:
            # 格式化消息
            messages = self.format_conversation_history(
                user_message=user_message,
                conversation_history=conversation_history or [],
                role_prompt=role_prompt
            )
            
            # 调用API
            response = await self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 提取回复内容
            content = response.choices[0].message.content
            
            logger.info(f"AI响应生成成功: {len(content)} 字符")
            return content
            
        except Exception as e:
            logger.error(f"生成AI回复时出错: {str(e)}")
            raise
    
    async def generate_streamed_response(self,
                                       user_message: str,
                                       role_prompt: str,
                                       conversation_history: List[Dict[str, Any]] = None,
                                       temperature: float = 0.7,
                                       max_tokens: int = 2000,
                                       model: Optional[str] = None,
                                       callback: Optional[Callable[[str], None]] = None) -> AsyncGenerator[str, None]:
        """
        生成流式AI回复
        
        Args:
            user_message: 用户消息
            role_prompt: 角色提示信息
            conversation_history: 对话历史
            temperature: 生成温度
            max_tokens: 最大生成令牌数
            model: 使用的模型，默认使用初始化时指定的模型
            callback: 回调函数，用于处理流式输出的每个片段
            
        Yields:
            生成的文本片段
        """
        if not self.available:
            mock_response = self.generate_mock_response(user_message, role_prompt)
            # 模拟流式输出，将回复分成多个部分
            chunks = [mock_response[i:i+20] for i in range(0, len(mock_response), 20)]
            
            for chunk in chunks:
                await asyncio.sleep(0.1)  # 模拟延迟
                if callback:
                    callback(chunk)
                yield chunk
            return
            
        try:
            # 格式化消息
            messages = self.format_conversation_history(
                user_message=user_message,
                conversation_history=conversation_history or [],
                role_prompt=role_prompt
            )
            
            # 调用流式API
            stream = await self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            # 处理流式回复
            async for chunk in stream:
                if not chunk.choices:
                    continue
                    
                content = chunk.choices[0].delta.content
                if content:
                    if callback:
                        callback(content)
                    yield content
                    
        except Exception as e:
            logger.error(f"生成流式AI回复时出错: {str(e)}")
            error_message = f"生成回复时发生错误: {str(e)}"
            if callback:
                callback(error_message)
            yield error_message
    
    def generate_mock_response(self, user_message: str, role_prompt: str) -> str:
        """
        生成模拟回复（当API不可用时使用）
        
        Args:
            user_message: 用户消息
            role_prompt: 角色提示
            
        Returns:
            模拟的回复内容
        """
        # 从角色提示中提取角色名称（如果存在）
        role_name = "AI助手"
        if "我是" in role_prompt and "。" in role_prompt:
            role_intro = role_prompt.split("我是")[1].split("。")[0]
            if role_intro:
                role_name = role_intro.strip()
        
        # 生成模拟回复
        mock_responses = [
            f"我是{role_name}，很高兴为您服务！您说的"{user_message[:30]}..."是个很有趣的话题。我可以从多个角度为您解答。",
            f"作为{role_name}，我理解您询问的是关于"{user_message[:30]}..."。这是一个很好的问题，让我来回答。",
            f"感谢您的提问！"{user_message[:30]}..."确实值得探讨。作为{role_name}，我的看法是...",
            f"{role_name}收到您的问题了。关于"{user_message[:30]}..."，我想说的是...",
            f"您好！我是您的{role_name}。针对您提出的"{user_message[:30]}..."，我的回答是..."
        ]
        
        # 随机选择一个模拟回复
        import random
        base_response = random.choice(mock_responses)
        
        # 添加一些随机化的内容，使回复看起来更真实
        elaborations = [
            "这个问题可以从几个方面来看。首先...",
            "根据我的理解，这个问题的关键在于...",
            "如果深入分析，我们会发现...",
            "这是一个常见的问题，通常可以这样解决...",
            "从专业角度来看，应该注意以下几点..."
        ]
        
        conclusion = [
            "希望我的回答对您有所帮助！如果您有任何其他问题，请随时提问。",
            "以上就是我的回答，希望能解决您的疑问。",
            "总结一下，关键点是...希望这个回答对您有用！",
            "这就是我的看法，欢迎进一步讨论。",
            "如果您需要更多信息，请告诉我，我很乐意继续为您解答。"
        ]
        
        full_response = (
            base_response + "\n\n" + 
            random.choice(elaborations) + " " + 
            "实际上，这取决于具体情况和上下文。" + " " +
            "但通常来说，最佳做法是考虑所有可能的因素。" + "\n\n" + 
            random.choice(conclusion)
        )
        
        return full_response

    async def get_available_models(self) -> List[str]:
        """
        获取可用模型列表
        
        Returns:
            可用模型列表
        """
        if not self.available:
            return ["mock-model-1", "mock-model-2", "mock-model-3"]
            
        try:
            models = await self.client.models.list()
            model_names = [model.id for model in models.data]
            return model_names
        except Exception as e:
            logger.error(f"获取可用模型列表时出错: {str(e)}")
            return []
    
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