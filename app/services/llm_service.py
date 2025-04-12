"""
LLM集成服务模块

提供对多种LLM模型的统一集成接口，当前支持DeepSeek和智谱AI
包含提示模板处理、流式响应处理以及错误处理与重试功能
"""

import logging
import asyncio
import json
import time
import os
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator, Union, Tuple
from enum import Enum
import aiohttp
import backoff
from pydantic import BaseModel, Field

# 导入智谱AI认证模块
from app.services.zhipu_auth import get_zhipu_auth_headers

logger = logging.getLogger(__name__)

# 检查环境变量中是否有API密钥
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY", "")

# LLM提供商枚举
class LLMProvider(str, Enum):
    DEEPSEEK = "deepseek"
    ZHIPU = "zhipu"
    OPENAI = "openai"

# 消息角色枚举
class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

# 消息模型
class Message(BaseModel):
    role: MessageRole
    content: str

# 提示模板模型
class PromptTemplate(BaseModel):
    template: str
    variables: Dict[str, str] = Field(default_factory=dict)
    
    def format(self, **kwargs) -> str:
        """格式化提示模板，填充变量"""
        template_vars = self.variables.copy()
        template_vars.update(kwargs)
        return self.template.format(**template_vars)

# LLM配置模型
class LLMConfig(BaseModel):
    provider: LLMProvider
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 60
    retry_attempts: int = 3
    retry_backoff_factor: float = 2.0

# LLM响应模型
class LLMResponse(BaseModel):
    content: str
    model: str
    provider: LLMProvider
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

# 添加流式响应类
class StreamResponse(BaseModel):
    content: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    is_start: bool = False
    is_end: bool = False

class LLMService:
    """LLM服务类，提供对DeepSeek和智谱AI的统一接口"""
    
    def __init__(self, default_config: Optional[LLMConfig] = None):
        """
        初始化LLM服务
        
        Args:
            default_config: 默认LLM配置，如果未提供则使用环境变量中的配置
        """
        self.default_config = default_config or self._get_default_config()
        self.session = None  # aiohttp会话将在需要时初始化
        logger.info(f"LLM服务初始化完成，默认提供商: {self.default_config.provider}, 模型: {self.default_config.model_name}")
    
    def _get_default_config(self) -> LLMConfig:
        """获取默认配置，优先使用DeepSeek"""
        if DEEPSEEK_API_KEY:
            return LLMConfig(
                provider=LLMProvider.DEEPSEEK,
                model_name="deepseek-chat",
                api_key=DEEPSEEK_API_KEY
            )
        elif ZHIPU_API_KEY:
            return LLMConfig(
                provider=LLMProvider.ZHIPU,
                model_name="glm-4",
                api_key=ZHIPU_API_KEY
            )
        else:
            # 如果没有API密钥，使用空密钥，但会在调用时记录警告
            logger.warning("未找到有效的API密钥，服务将不可用")
            return LLMConfig(
                provider=LLMProvider.DEEPSEEK,
                model_name="deepseek-chat",
                api_key=""
            )
    
    async def _ensure_session(self):
        """确保aiohttp会话已初始化"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """关闭服务和相关资源"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def _format_messages_for_provider(self, 
                                     messages: List[Message], 
                                     provider: LLMProvider) -> List[Dict[str, str]]:
        """
        根据不同提供商格式化消息
        
        Args:
            messages: 消息列表
            provider: LLM提供商
            
        Returns:
            格式化后的消息列表
        """
        if provider == LLMProvider.DEEPSEEK:
            # DeepSeek格式与标准格式兼容
            return [{"role": msg.role, "content": msg.content} for msg in messages]
        elif provider == LLMProvider.ZHIPU:
            # 智谱AI格式也与标准格式兼容
            return [{"role": msg.role, "content": msg.content} for msg in messages]
        else:
            raise ValueError(f"不支持的LLM提供商: {provider}")
    
    def _get_api_endpoint(self, config: LLMConfig, stream: bool = False) -> str:
        """
        获取API端点URL
        
        Args:
            config: LLM配置
            stream: 是否使用流式API
            
        Returns:
            API端点URL
        """
        if config.provider == LLMProvider.DEEPSEEK:
            return "https://api.deepseek.com/v1/chat/completions"
        elif config.provider == LLMProvider.ZHIPU:
            # 智谱AI的API端点
            return "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        else:
            raise ValueError(f"不支持的LLM提供商: {config.provider}")
    
    def _get_request_headers(self, config: LLMConfig) -> Dict[str, str]:
        """
        获取请求头
        
        Args:
            config: LLM配置
            
        Returns:
            请求头字典
        """
        if config.provider == LLMProvider.DEEPSEEK:
            return {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json"
            }
        elif config.provider == LLMProvider.ZHIPU:
            # 使用专门的智谱AI认证模块生成JWT令牌
            return get_zhipu_auth_headers(config.api_key)
        else:
            raise ValueError(f"不支持的LLM提供商: {config.provider}")
    
    def _prepare_request_data(self, 
                             messages: List[Message], 
                             config: LLMConfig,
                             stream: bool = False) -> Dict[str, Any]:
        """
        准备请求数据
        
        Args:
            messages: 消息列表
            config: LLM配置
            stream: 是否使用流式API
            
        Returns:
            请求数据字典
        """
        formatted_messages = self._format_messages_for_provider(messages, config.provider)
        
        if config.provider == LLMProvider.DEEPSEEK:
            return {
                "model": config.model_name,
                "messages": formatted_messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "stream": stream
            }
        elif config.provider == LLMProvider.ZHIPU:
            return {
                "model": config.model_name,
                "messages": formatted_messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "stream": stream
            }
        else:
            raise ValueError(f"不支持的LLM提供商: {config.provider}")
    
    def _parse_response(self, 
                       response_data: Dict[str, Any], 
                       config: LLMConfig) -> LLMResponse:
        """
        解析API响应
        
        Args:
            response_data: API响应数据
            config: LLM配置
            
        Returns:
            解析后的LLM响应
        """
        try:
            if config.provider == LLMProvider.DEEPSEEK:
                content = response_data["choices"][0]["message"]["content"]
                finish_reason = response_data["choices"][0].get("finish_reason")
                tokens_used = response_data.get("usage", {}).get("total_tokens")
                
                return LLMResponse(
                    content=content,
                    model=config.model_name,
                    provider=config.provider,
                    tokens_used=tokens_used,
                    finish_reason=finish_reason,
                    raw_response=response_data
                )
            elif config.provider == LLMProvider.ZHIPU:
                content = response_data["choices"][0]["message"]["content"]
                finish_reason = response_data["choices"][0].get("finish_reason")
                tokens_used = response_data.get("usage", {}).get("total_tokens")
                
                return LLMResponse(
                    content=content,
                    model=config.model_name,
                    provider=config.provider,
                    tokens_used=tokens_used,
                    finish_reason=finish_reason,
                    raw_response=response_data
                )
            else:
                raise ValueError(f"不支持的LLM提供商: {config.provider}")
        except (KeyError, IndexError) as e:
            logger.error(f"解析响应时出错: {str(e)}, 响应数据: {response_data}")
            raise ValueError(f"无法解析LLM响应: {str(e)}")
    
    def _parse_stream_chunk(self, 
                           chunk_data: Dict[str, Any], 
                           config: LLMConfig) -> Tuple[Optional[str], Optional[str]]:
        """
        解析流式响应的数据块
        
        Args:
            chunk_data: 数据块
            config: LLM配置
            
        Returns:
            (内容片段, 结束原因)的元组，如果没有内容或非完成块则返回(None, None)
        """
        try:
            if config.provider == LLMProvider.DEEPSEEK:
                delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                finish_reason = chunk_data.get("choices", [{}])[0].get("finish_reason")
                return content, finish_reason
                
            elif config.provider == LLMProvider.ZHIPU:
                delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                finish_reason = chunk_data.get("choices", [{}])[0].get("finish_reason")
                return content, finish_reason
                
            else:
                raise ValueError(f"不支持的LLM提供商: {config.provider}")
        except (KeyError, IndexError) as e:
            logger.error(f"解析流式数据块时出错: {str(e)}, 数据块: {chunk_data}")
            return None, None
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError, ValueError),
        max_tries=3,
        factor=2
    )
    async def _make_api_request(self, 
                              endpoint: str, 
                              headers: Dict[str, str], 
                              data: Dict[str, Any],
                              timeout: int) -> Dict[str, Any]:
        """
        发送API请求
        
        Args:
            endpoint: API端点URL
            headers: 请求头
            data: 请求数据
            timeout: 超时时间（秒）
            
        Returns:
            API响应
        """
        await self._ensure_session()
        
        async with self.session.post(
            endpoint, 
            headers=headers, 
            json=data, 
            timeout=timeout
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"API请求失败: 状态码 {response.status}, 响应: {error_text}")
                response.raise_for_status()
                
            return await response.json()
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError, ValueError),
        max_tries=3,
        factor=2
    )
    async def generate_response(self, 
                             messages: List[Union[Message, Dict[str, str]]],
                             config: Optional[LLMConfig] = None) -> LLMResponse:
        """
        生成LLM回复（非流式）
        
        Args:
            messages: 消息列表，可以是Message对象或字典
            config: LLM配置，如果未提供则使用默认配置
            
        Returns:
            LLM响应
        """
        # 处理消息格式，确保是Message对象列表
        processed_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                processed_messages.append(Message(
                    role=msg.get("role", "user"),
                    content=msg.get("content", "")
                ))
            else:
                processed_messages.append(msg)
        
        # 使用默认配置或提供的配置
        used_config = config or self.default_config
        
        # 检查API密钥是否有效
        if not used_config.api_key:
            logger.error(f"缺少{used_config.provider}的API密钥")
            raise ValueError(f"缺少{used_config.provider}的API密钥")
        
        # 准备API请求
        endpoint = self._get_api_endpoint(used_config)
        headers = self._get_request_headers(used_config)
        data = self._prepare_request_data(processed_messages, used_config)
        
        try:
            # 发送API请求
            response_data = await self._make_api_request(
                endpoint=endpoint,
                headers=headers,
                data=data,
                timeout=used_config.timeout
            )
            
            # 解析响应
            return self._parse_response(response_data, used_config)
            
        except Exception as e:
            logger.error(f"生成回复时出错: {str(e)}")
            raise
    
    async def generate_stream_response(self, 
                                    messages: List[Union[Message, Dict[str, str]]],
                                    config: Optional[LLMConfig] = None,
                                    callback: Optional[Callable[[str], None]] = None) -> AsyncGenerator[str, None]:
        """
        生成流式LLM回复
        
        Args:
            messages: 消息列表，可以是Message对象或字典
            config: LLM配置，如果未提供则使用默认配置
            callback: 回调函数，用于处理每个生成的片段
            
        Yields:
            生成的文本片段
        """
        # 处理消息格式，确保是Message对象列表
        processed_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                processed_messages.append(Message(
                    role=msg.get("role", "user"),
                    content=msg.get("content", "")
                ))
            else:
                processed_messages.append(msg)
        
        # 使用默认配置或提供的配置
        used_config = config or self.default_config
        
        # 检查API密钥是否有效
        if not used_config.api_key:
            error_msg = f"缺少{used_config.provider}的API密钥"
            logger.error(error_msg)
            if callback:
                callback(error_msg)
            yield error_msg
            return
        
        # 准备API请求
        try:
            endpoint = self._get_api_endpoint(used_config, stream=True)
            headers = self._get_request_headers(used_config)
            data = self._prepare_request_data(processed_messages, used_config, stream=True)
        except Exception as prep_error:
            error_msg = f"准备API请求时出错: {str(prep_error)}"
            logger.error(error_msg)
            if callback:
                callback(error_msg)
            yield error_msg
            return
        
        await self._ensure_session()
        
        retry_count = 0
        max_retries = used_config.retry_attempts
        
        while retry_count <= max_retries:
            try:
                # 发送流式API请求
                async with self.session.post(
                    endpoint,
                    headers=headers,
                    json=data,
                    timeout=used_config.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"流式API请求失败: 状态码 {response.status}, 响应: {error_text}")
                        
                        if retry_count < max_retries:
                            retry_count += 1
                            wait_time = used_config.retry_backoff_factor ** retry_count
                            logger.info(f"正在重试 ({retry_count}/{max_retries})，等待 {wait_time} 秒...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            error_msg = f"API请求失败: {error_text}"
                            if callback:
                                callback(error_msg)
                            yield error_msg
                            return
                    
                    # 处理流式响应
                    async for line in response.content:
                        line = line.strip()
                        if not line or line == b'':
                            continue
                            
                        try:
                            if line.startswith(b'data: '):
                                line = line[6:]  # 移除'data: '前缀
                            
                            if line == b'[DONE]':
                                break
                                
                            chunk_data = json.loads(line)
                            content, finish_reason = self._parse_stream_chunk(chunk_data, used_config)
                            
                            if content:
                                if callback:
                                    callback(content)
                                yield content
                                
                            if finish_reason:
                                break
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"解析JSON时出错: {str(e)}, 行: {line}")
                            continue
                
                # 成功完成，跳出重试循环
                break
                
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"流式请求出错: {str(e)}")
                
                if retry_count < max_retries:
                    retry_count += 1
                    wait_time = used_config.retry_backoff_factor ** retry_count
                    logger.info(f"正在重试 ({retry_count}/{max_retries})，等待 {wait_time} 秒...")
                    await asyncio.sleep(wait_time)
                else:
                    error_msg = f"生成回复失败，重试{max_retries}次后仍然出错: {str(e)}"
                    logger.error(error_msg)
                    if callback:
                        callback(error_msg)
                    yield error_msg
                    return
            except Exception as e:
                # 捕获其他未预期的错误
                error_msg = f"流式生成时发生意外错误: {str(e)}"
                logger.error(error_msg, exc_info=True)  # 记录完整的错误堆栈
                if callback:
                    callback(error_msg)
                yield error_msg
                return
    
    def apply_prompt_template(self, 
                             template: Union[str, PromptTemplate], 
                             **variables) -> str:
        """
        应用提示模板，填充变量
        
        Args:
            template: 提示模板或模板字符串
            **variables: 要填充的变量
            
        Returns:
            格式化后的提示文本
        """
        if isinstance(template, str):
            template = PromptTemplate(template=template)
        
        return template.format(**variables)
    
    async def get_available_models(self, provider: Optional[LLMProvider] = None) -> List[str]:
        """
        获取指定提供商的可用模型列表
        
        Args:
            provider: LLM提供商，如果未提供则使用默认提供商
            
        Returns:
            可用模型列表
        """
        used_provider = provider or self.default_config.provider
        
        if used_provider == LLMProvider.DEEPSEEK:
            # DeepSeek可用模型列表
            return [
                "deepseek-chat",
                "deepseek-coder",
                "deepseek-lite"
            ]
        elif used_provider == LLMProvider.ZHIPU:
            # 智谱AI可用模型列表
            return [
                "glm-3-turbo",
                "glm-4",
                "glm-4v"
            ]
        else:
            return []
    
    async def generate_stream(self,
                              messages: List[Union[Message, Dict[str, str]]],
                              config: Optional[LLMConfig] = None) -> AsyncGenerator[StreamResponse, None]:
        """
        生成流式LLM回复，并封装为StreamResponse对象
        
        重要方法：这是流式生成必需的核心方法，请勿删除或重命名
        
        Args:
            messages: 消息列表，可以是Message对象或字典
            config: LLM配置，如果未提供则使用默认配置
            
        Yields:
            StreamResponse对象，包含生成的文本片段和元数据
        """
        # 初始化模型和提供商信息
        used_config = config or self.default_config
        provider_str = used_config.provider.value
        model_str = used_config.model_name
        
        # 发送开始事件
        yield StreamResponse(
            is_start=True,
            model=model_str,
            provider=provider_str
        )
        
        try:
            # 调用现有的流式生成方法
            async for content in self.generate_stream_response(
                messages=messages,
                config=config
            ):
                # 如果内容看起来像错误消息，包装成错误响应
                if content.startswith("API请求失败") or content.startswith("生成回复失败") or content.startswith("缺少"):
                    yield StreamResponse(
                        content=content,
                        model=model_str,
                        provider=provider_str
                    )
                    continue
                
                # 正常内容
                yield StreamResponse(
                    content=content,
                    model=model_str,
                    provider=provider_str
                )
                
            # 发送结束事件
            yield StreamResponse(
                is_end=True,
                model=model_str,
                provider=provider_str,
                tokens_used=None  # 在实际实现中可能需要从某处获取这个信息
            )
            
        except Exception as e:
            # 出错时发送错误消息
            error_msg = f"生成流式响应时出错: {str(e)}"
            logger.error(error_msg)
            yield StreamResponse(
                content=error_msg,
                model=model_str,
                provider=provider_str
            )
            
            # 确保发送结束事件
            yield StreamResponse(
                is_end=True,
                model=model_str,
                provider=provider_str
            )

def get_api_key(provider):
    """
    根据提供商获取API密钥
    
    Args:
        provider: LLMProvider枚举值
        
    Returns:
        对应的API密钥
    """
    if provider == LLMProvider.DEEPSEEK:
        return os.environ.get("DEEPSEEK_API_KEY", "")
    elif provider == LLMProvider.ZHIPU:
        return os.environ.get("ZHIPU_API_KEY", "")
    elif provider == LLMProvider.OPENAI:
        return os.environ.get("OPENAI_API_KEY", "")
    else:
        return None

# 确保关键方法在模块级别可用，防止导入问题
__all__ = [
    'LLMService', 'LLMConfig', 'LLMProvider', 
    'Message', 'MessageRole', 'PromptTemplate',
    'LLMResponse', 'StreamResponse', 'get_api_key'
] 