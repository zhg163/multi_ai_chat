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
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator, Union, Tuple, Set, Iterable
from enum import Enum
import aiohttp
import backoff
from pydantic import BaseModel, Field
import httpx
from dataclasses import dataclass
from aiohttp import ClientSession, ClientResponse, ClientError, ClientTimeout

# 导入智谱AI认证模块
from app.services.zhipu_auth import get_zhipu_auth_headers

logger = logging.getLogger(__name__)

# 检查环境变量中是否有API密钥
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY", "")

class LLMProvider(str, Enum):
    """LLM提供商枚举"""
    DEEPSEEK = "deepseek"
    ZHIPU = "zhipu"
    OPENAI = "openai"
    DEFAULT = "default"  # 系统默认提供商

class MessageRole(str, Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

@dataclass
class Message:
    """消息数据类"""
    role: MessageRole
    content: str

class PromptTemplate(BaseModel):
    template: str
    variables: Dict[str, str] = Field(default_factory=dict)
    
    def format(self, **kwargs) -> str:
        """格式化提示模板，填充变量"""
        template_vars = self.variables.copy()
        template_vars.update(kwargs)
        return self.template.format(**template_vars)

@dataclass
class LLMConfig:
    """LLM配置数据类"""
    provider: LLMProvider
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    
    def to_request_params(self) -> Dict[str, Any]:
        """转换为请求参数"""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop
        }
        
    def is_default_model(self) -> bool:
        """检查是否使用默认模型"""
        return self.model_name == "default"

@dataclass
class LLMResponse:
    """LLM响应数据类"""
    content: str
    finish_reason: Optional[str] = None
    
@dataclass
class StreamResponse:
    """流式响应数据类"""
    content: str
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    status: str = "generating"

class LLMService:
    """LLM服务类，提供对DeepSeek和智谱AI的统一接口"""
    
    def __init__(self, default_config: Optional[LLMConfig] = None):
        """
        初始化LLM服务
        
        Args:
            default_config: 默认LLM配置，如果未提供则使用环境变量中的配置
        """
        self.logger = logging.getLogger(__name__)
        self.default_config = default_config or self._get_default_config()
        self.session = None  # aiohttp会话将在需要时初始化
        logger.info(f"LLM服务初始化完成，默认提供商: {self.default_config.provider}, 模型: {self.default_config.model_name}")
    
    def _get_provider_str(self, config: Union[LLMConfig, Dict]) -> str:
        """
        从配置中获取提供商字符串
        
        Args:
            config: LLM配置，可以是LLMConfig对象或配置字典
            
        Returns:
            提供商字符串
        """
        if isinstance(config, dict):
            provider = config.get("provider", "deepseek")
            # 处理 "default" 的情况，使用系统默认提供商
            if provider == "default" or provider == LLMProvider.DEFAULT:
                return self.default_config.provider.value
            # 如果provider是枚举，获取其值
            if hasattr(provider, 'value'):
                return provider.value
            return str(provider).lower()
        else:
            # 处理LLMConfig对象
            provider = config.provider if hasattr(config, 'provider') else None
            # 处理 "default" 的情况，使用系统默认提供商
            if provider == "default" or provider == LLMProvider.DEFAULT:
                return self.default_config.provider.value
            # 如果provider是枚举，获取其值
            if hasattr(provider, 'value'):
                return provider.value
            return str(provider).lower()
    
    def _get_default_config(self) -> LLMConfig:
        """获取默认配置，优先使用DeepSeek"""
        if DEEPSEEK_API_KEY:
            api_key = DEEPSEEK_API_KEY
            if not api_key.startswith("sk-"):
                api_key = f"sk-{api_key}"
                logger.info(f"初始化时修正DeepSeek API密钥格式，添加sk-前缀")
                
            return LLMConfig(
                provider=LLMProvider.DEEPSEEK,
                model_name="deepseek-chat",
                api_key=api_key
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
                                     provider: Union[LLMProvider, str]) -> List[Dict[str, str]]:
        """
        根据不同提供商格式化消息
        
        Args:
            messages: 消息列表
            provider: LLM提供商，可以是LLMProvider枚举或字符串
            
        Returns:
            格式化后的消息列表
        """
        # 将provider标准化为小写字符串
        if hasattr(provider, 'value'):
            provider_str = provider.value.lower()
        else:
            provider_str = str(provider).lower()
        
        # 不同提供商的格式化处理
        if "deepseek" in provider_str:
            # DeepSeek格式与标准格式兼容
            return [{"role": msg.role, "content": msg.content} for msg in messages]
        elif "zhipu" in provider_str:
            # 智谱AI格式也与标准格式兼容
            return [{"role": msg.role, "content": msg.content} for msg in messages]
        else:
            # 默认使用标准格式
            logger.warning(f"未知提供商类型: {provider_str}，使用标准消息格式")
            return [{"role": msg.role, "content": msg.content} for msg in messages]
    
    def _get_api_endpoint(self, config, stream: bool = False) -> str:
        """
        获取API端点URL
        
        Args:
            config: LLM配置（可以是LLMConfig对象或字典）
            stream: 是否使用流式API
            
        Returns:
            API端点URL
        """
        # 判断config是LLMConfig对象还是字典
        if isinstance(config, dict):
            provider = config.get("provider")
            # 标准化提供商名称
            if provider is None:
                provider = "deepseek"  # 默认提供商
                logger.warning(f"未指定提供商，使用默认值: {provider}")
            
            # 处理不同格式的提供商名称
            provider_lower = provider.lower() if isinstance(provider, str) else str(provider).lower()
            if "deepseek" in provider_lower:
                return "https://api.deepseek.com/v1/chat/completions"
            elif "zhipu" in provider_lower:
                return "https://open.bigmodel.cn/api/paas/v4/chat/completions"
            elif "openai" in provider_lower:
                return "https://api.openai.com/v1/chat/completions"
        else:
                # 处理LLMConfig对象
                if hasattr(config, 'provider'):
                    provider = config.provider
                    # 如果provider是枚举，转换为字符串进行比较
                provider_str = provider.value if hasattr(provider, 'value') else str(provider)
                
                if "deepseek" in provider_str.lower():
                    return "https://api.deepseek.com/v1/chat/completions"
                elif "zhipu" in provider_str.lower():
                    return "https://open.bigmodel.cn/api/paas/v4/chat/completions"
                elif "openai" in provider_str.lower():
                    return "https://api.openai.com/v1/chat/completions"
                else:
                    logger.warning(f"不支持的LLM提供商: {provider}，默认使用DeepSeek端点")
                    return "https://api.deepseek.com/v1/chat/completions"

    def _get_request_headers(self, provider_type: str, api_key: str = None) -> Dict[str, str]:
        """
        获取请求头
        
        Args:
            provider_type: 提供商类型
            api_key: API密钥，如果提供则使用它，否则尝试从环境变量获取
            
        Returns:
            请求头字典
        """
        headers = {'Content-Type': 'application/json'}
        
        # 获取API密钥
        if not api_key:
            # 尝试从环境变量获取
            if provider_type == LLMProvider.DEEPSEEK.value:
                api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            elif provider_type == LLMProvider.ZHIPU.value:
                api_key = os.environ.get("ZHIPU_API_KEY", "")
            elif provider_type == "openai":
                api_key = os.environ.get("OPENAI_API_KEY", "")
            
        if not api_key:
            logger.error(f"无法获取API密钥，provider_type={provider_type}")
            raise ValueError(f"未配置API密钥: {provider_type}")
            
        # 根据不同的提供商处理API密钥格式
        if provider_type == "openai" or provider_type == "azure":
            # OpenAI/Azure格式：Bearer <api_key>
            headers['Authorization'] = f'Bearer {api_key}'
            
        elif provider_type == LLMProvider.DEEPSEEK.value or provider_type == "deepseek":
            # Deepseek格式：Bearer <api_key>
            # 确保不会有双前缀问题
            if api_key.startswith('sk-'):
                headers['Authorization'] = f'Bearer {api_key}'
            else:
                # Deepseek格式：Bearer <api_key>
                # 如果API密钥不是以sk-开头，则添加前缀
                if not api_key.startswith('sk-'):
                    api_key = f'sk-{api_key}'
                headers['Authorization'] = f'Bearer {api_key}'
                
                # 打印部分掩码后的API密钥进行日志记录
                masked_key = self._mask_api_key(api_key)
                logger.debug(f"Deepseek API密钥: {masked_key}")
                logger.debug(f"Deepseek Authorization头: {self._mask_header(headers.get('Authorization', ''))}")
            
        elif provider_type == LLMProvider.ZHIPU.value or provider_type == "zhipu":
                # 智谱格式: 直接使用API密钥
                headers['Authorization'] = api_key
            
                # 打印掩码后的头部信息
                masked_auth = self._mask_header(headers.get('Authorization', ''))
                logger.debug(f"Zhipu Authorization头: {masked_auth}")
            
        else:
                 # 默认使用Bearer格式
                headers['Authorization'] = f'Bearer {api_key}'
                logger.warning(f"未知的提供商类型: {provider_type}，使用默认的Bearer格式")
            
        return headers
    
    def _mask_api_key(self, api_key: str) -> str:
        """
        掩码API密钥，只显示前4位和后4位
        
        Args:
            api_key: 原始API密钥
            
        Returns:
            掩码后的API密钥
        """
        if not api_key or len(api_key) < 8:
            return "******"
            
        return f"{api_key[:4]}...{api_key[-4:]}"
        
    def _mask_header(self, header: str) -> str:
        """
        掩码请求头中的敏感信息
        
        Args:
            header: 原始头部信息
            
        Returns:
            掩码后的头部信息
        """
        if not header:
            return ""
            
        parts = header.split(" ")
        if len(parts) == 2 and parts[0] == "Bearer":
            return f"Bearer {self._mask_api_key(parts[1])}"
        return self._mask_api_key(header)
    
    def _prepare_request_data(self, 
                             messages: List[Message], 
                             config: Union[LLMConfig, Dict],
                             stream: bool = False) -> Dict[str, Any]:
        """
        准备请求数据
        
        Args:
            messages: 消息列表
            config: LLM配置，可以是LLMConfig对象或配置字典
            stream: 是否使用流式API
            
        Returns:
            请求数据字典
        """
        # 处理配置类型，获取标准化的提供商信息
        if isinstance(config, dict):
            provider = config.get("provider", "deepseek")
            provider_str = str(provider).lower()
            model_name = config.get("model_name", "deepseek-chat")
            temperature = config.get("temperature", 0.7)
            max_tokens = config.get("max_tokens", 2048)
            
            # 根据字典中的提供商参数格式化消息
            if not isinstance(messages[0], dict):
                formatted_messages = self._format_messages_for_provider(messages, provider)
            else:
                # 如果已经是字典格式，直接使用
                formatted_messages = messages
        else:
            # 处理LLMConfig对象
            provider = config.provider if hasattr(config, 'provider') else None
            provider_str = provider.value.lower() if hasattr(provider, 'value') else str(provider).lower()
            model_name = config.model_name
            temperature = config.temperature
            max_tokens = config.max_tokens
            
            # 格式化消息
            if not isinstance(messages[0], dict):
                formatted_messages = self._format_messages_for_provider(messages, provider)
            else:
                # 如果已经是字典格式，直接使用
                formatted_messages = messages
        
        # 根据提供商准备请求数据
        if "deepseek" in provider_str:
            return {
                "model": model_name,
                "messages": formatted_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }
        elif "zhipu" in provider_str:
            return {
                "model": model_name,
                "messages": formatted_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }
        else:
            # 使用默认格式
            logger.warning(f"未知提供商类型: {provider_str}，使用默认请求格式")
            return {
                "model": model_name,
                "messages": formatted_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }
    
    def _parse_response(self, 
                       response_data: Dict[str, Any], 
                       config: Union[LLMConfig, Dict]) -> LLMResponse:
        """
        解析API响应
        
        Args:
            response_data: API响应数据
            config: LLM配置，可以是LLMConfig对象或配置字典
            
        Returns:
            解析后的LLM响应
        """
        try:
            # 处理配置类型，获取标准化的提供商信息
            if isinstance(config, dict):
                provider = config.get("provider", "deepseek")
                provider_str = str(provider).lower()
            else:
                # 处理LLMConfig对象
                provider = config.provider if hasattr(config, 'provider') else None
                provider_str = provider.value.lower() if hasattr(provider, 'value') else str(provider).lower()
            
            # 根据提供商解析响应
            if "deepseek" in provider_str:
                content = response_data["choices"][0]["message"]["content"]
                finish_reason = response_data["choices"][0].get("finish_reason")
                tokens_used = response_data.get("usage", {}).get("total_tokens")
                
                return LLMResponse(
                    content=content,
                    finish_reason=finish_reason
                )
            elif "zhipu" in provider_str:
                content = response_data["choices"][0]["message"]["content"]
                finish_reason = response_data["choices"][0].get("finish_reason")
                tokens_used = response_data.get("usage", {}).get("total_tokens")
                
                return LLMResponse(
                    content=content,
                    finish_reason=finish_reason
                )
            else:
                # 通用格式解析
                logger.warning(f"未知提供商类型: {provider_str}，尝试通用格式解析")
                content = response_data["choices"][0]["message"]["content"]
                finish_reason = response_data["choices"][0].get("finish_reason")
                
                return LLMResponse(
                    content=content,
                    finish_reason=finish_reason
                )
        except (KeyError, IndexError) as e:
            logger.error(f"解析响应时出错: {str(e)}, 响应数据: {response_data}")
            raise ValueError(f"无法解析LLM响应: {str(e)}")
    
    def _parse_stream_chunk(self, 
                           chunk_data: Dict[str, Any], 
                           config: Union[LLMConfig, Dict]) -> Tuple[Optional[str], Optional[str]]:
        """
        解析流式响应的数据块
        
        Args:
            chunk_data: 数据块
            config: LLM配置，可以是LLMConfig对象或配置字典
            
        Returns:
            (内容片段, 结束原因)的元组，如果没有内容或非完成块则返回(None, None)
        """
        try:
            # 处理配置类型，获取标准化的提供商名称
            if isinstance(config, dict):
                provider = config.get("provider", "deepseek")
                provider_str = str(provider).lower()
            else:
                # 处理LLMConfig对象
                provider = config.provider if hasattr(config, 'provider') else None
                # 如果provider是枚举，获取其值
                provider_str = provider.value.lower() if hasattr(provider, 'value') else str(provider).lower()
            
            # 根据提供商解析响应
            if "deepseek" in provider_str:
                delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                finish_reason = chunk_data.get("choices", [{}])[0].get("finish_reason")
                return content, finish_reason
                
            elif "zhipu" in provider_str:
                delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                finish_reason = chunk_data.get("choices", [{}])[0].get("finish_reason")
                return content, finish_reason
                
            else:
                # 默认采用通用的格式处理
                logger.warning(f"未知的提供商类型: {provider_str}，尝试通用格式解析")
                delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                finish_reason = chunk_data.get("choices", [{}])[0].get("finish_reason")
                return content, finish_reason
                
        except (KeyError, IndexError) as e:
            logger.error(f"解析流式数据块时出错: {str(e)}, 数据块: {chunk_data}")
            return None, None
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError, ValueError),
        max_tries=1,
        factor=2
    )
    async def _make_api_request(self, endpoint: str, headers: Dict[str, str], data: Dict) -> Dict:
        """
        进行API请求
        
        Args:
            endpoint: API地址
            headers: 请求头
            data: 请求数据
            
        Returns:
            响应数据
        """
        # 记录请求信息（脱敏）
        logger.info(f"Sending request to endpoint: {endpoint}")
        
        # 记录脱敏的授权头
        if "Authorization" in headers:
            auth_header = headers["Authorization"]
            # 只显示前12位和后4位
            masked_auth = f"{auth_header[:12]}...{auth_header[-4:] if len(auth_header) > 16 else ''}"
            logger.info(f"请求头部Authorization: {masked_auth}")
        else:
            logger.warning("请求中缺少Authorization头部")
        
        logger.debug(f"Request data: {json.dumps(data, ensure_ascii=False)}")
        
        try:
            logger.info("创建临时会话发送请求")
            timeout = aiohttp.ClientTimeout(total=60)  # 设置60秒超时
            async with aiohttp.ClientSession(timeout=timeout) as session:
                logger.debug(f"发送POST请求到 {endpoint}")
                async with session.post(endpoint, headers=headers, json=data) as response:
                    logger.info(f"收到状态码: {response.status}")
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"请求失败详情 - 状态码: {response.status}, 内容: {error_text[:500]}")
                        logger.error(f"请求失败详情 - 请求头: {headers}")
                        logger.error(f"请求失败详情 - 请求数据: {json.dumps(data, ensure_ascii=False)[:500]}")
                        response.raise_for_status()  # 抛出错误以触发重试机制
                    
                    # 读取响应内容
                    response_body = await response.read()
                    response_status = response.status
                    response_headers = {k: v for k, v in response.headers.items()}  # 转换为普通字典
                    response_reason = response.reason
                    response_content_type = response.headers.get('Content-Type', '')

                    # 创建自定义响应对象（不使用ClientResponse克隆）
                    class SimpleResponse:
                        def __init__(self):
                            self.status = response_status
                            self.reason = response_reason
                            self.headers = response_headers
                            self._body = response_body
                            self._content_type = response_content_type
                        
                        async def read(self):
                            return self._body
                        
                        async def text(self):
                            return self._body.decode('utf-8')
                        
                        async def json(self):
                            import json
                            return json.loads(await self.text())

                    # 返回简化的响应对象
                    return SimpleResponse()
                
        except aiohttp.ClientError as e:
            logger.error(f"API请求客户端错误: {str(e)}")
            raise
        except asyncio.TimeoutError:
            logger.error(f"API请求超时")
            raise
        except Exception as e:
            logger.error(f"API请求未预期错误: {str(e)}", exc_info=True)
            raise ValueError(f"API请求失败: {str(e)}")
            
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError, ValueError),
        max_tries=3,
        factor=2
    )
    async def generate_response(self, 
                             messages: List[Union[Message, Dict[str, str]]],
                             config: Optional[Union[LLMConfig, Dict]] = None) -> LLMResponse:
        """
        生成LLM回复（非流式）
        
        Args:
            messages: 消息列表，可以是Message对象或字典
            config: LLM配置，如果未提供则使用默认配置，可以是LLMConfig对象或配置字典
            
        Returns:
            LLM响应
        """
        # 处理消息格式，确保是消息对象或字典列表
        processed_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                if not isinstance(msg, Message):
                    # 确保字典有必要的字段
                    if "role" not in msg:
                        msg["role"] = "user"
                    if "content" not in msg:
                        msg["content"] = ""
                processed_messages.append(msg)
            elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                # 如果是Message对象，创建一个字典
                processed_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            else:
                logger.warning(f"无法处理的消息格式: {type(msg)}")
                # 默认处理为用户消息
                processed_messages.append({"role": "user", "content": str(msg)})
        
        # 使用默认配置或提供的配置
        used_config = config or self.default_config
        
        # 获取提供商信息
        if isinstance(used_config, dict):
            provider = used_config.get("provider", "deepseek")
            api_key = used_config.get("api_key")
            if not api_key:
                # 添加防御性检查，确保provider不是None
                if provider is not None:
                    api_key = os.environ.get(f"{provider.upper()}_API_KEY", "")
                    if provider.lower() == "deepseek" and api_key and not api_key.startswith("sk-"):
                        api_key = f"sk-{api_key}"
                else:
                    logger.error("提供商为None，无法获取API密钥")
                    return LLMResponse(content="配置错误：提供商为None，无法获取API密钥", finish_reason="error")
        else:
            # 使用LLMConfig对象
            provider = used_config.provider.value if hasattr(used_config.provider, 'value') else str(used_config.provider)
            api_key = used_config.api_key
        
        # 检查API密钥是否有效
        if not api_key:
            logger.error(f"缺少{provider}的API密钥")
            return LLMResponse(content=f"配置错误：缺少{provider}的API密钥", finish_reason="error")
        
        # 准备API请求
        try:
            endpoint = self._get_api_endpoint(used_config)
            headers = self._get_request_headers(provider, api_key)
            data = self._prepare_request_data(processed_messages, used_config)
        except Exception as e:
            logger.error(f"准备API请求时出错: {str(e)}", exc_info=True)
            return LLMResponse(content=f"API请求准备失败: {str(e)}", finish_reason="error")
        
        try:
            # 发送API请求
            response = await self._make_api_request(
                endpoint=endpoint,
                headers=headers,
                data=data
            )
            
            # 检查响应是否为None
            if response is None:
                logger.error("API响应为None，无法处理")
                error_msg = "获取API响应失败，服务可能暂时不可用"
                return LLMResponse(content=error_msg, finish_reason="error")
            
            # 安全地解析JSON响应
            try:
                response_json = await response.json()
                return self._parse_response(response_json, used_config)
            except (json.JSONDecodeError, aiohttp.ContentTypeError) as json_err:
                logger.error(f"解析响应JSON时出错: {str(json_err)}", exc_info=True)
                # 尝试读取原始响应内容作为备用
                try:
                    response_text = await response.text()
                    logger.error(f"原始响应内容: {response_text[:500]}")
                    error_msg = f"无法解析API响应: {str(json_err)}"
                except Exception as text_err:
                    logger.error(f"读取响应内容时出错: {str(text_err)}", exc_info=True)
                    error_msg = "无法读取或解析API响应"
                
                return LLMResponse(content=error_msg, finish_reason="error")
            
        except Exception as e:
            logger.error(f"生成回复时出错: {str(e)}", exc_info=True)
            return LLMResponse(content=f"生成响应时出错: {str(e)}", finish_reason="error")
    
    async def generate_stream_response(self, messages: List[Union[Dict, Message]], config: Optional[Dict] = None) -> AsyncGenerator[StreamResponse, None]:
        """
        通过API生成流式响应
        
        Args:
            messages: 消息列表，可以是字典或Message对象
            config: 配置参数，可选
            
        Yields:
            生成的StreamResponse对象
        """
        logger.info(f"调用API生成流式响应, 模型: {self.default_config.model_name}, 提供商: {self.default_config.provider}")
        
        # 处理传入的消息格式
        processed_messages = []
        for message in messages:
            if isinstance(message, dict):
                processed_messages.append(message)
            elif isinstance(message, Message):
                processed_messages.append(message.to_dict())
            else:
                raise ValueError(f"不支持的消息类型: {type(message)}")
                
        # 准备API请求
        api_url = self._get_api_endpoint(self.default_config, stream=True)
        headers = self._get_request_headers(self.default_config.provider.value, self.default_config.api_key)
        
        data = {
            "model": self.default_config.model_name,
            "messages": processed_messages,
            "stream": True
        }
        
        # 调整API请求的结构，使其符合提供商的要求
        if self.default_config.provider == "anthropic":
            # 适配Anthropic API格式
            data = {
                "model": self.default_config.model_name,
                "messages": processed_messages,
                "stream": True
            }
        elif self.default_config.provider == "openai":
            # OpenAI API格式，默认已兼容
            pass
        elif self.default_config.provider == "azure":
            # 适配Azure OpenAI格式
            pass
        
        # 创建临时会话
        session = None
        try:
            session = aiohttp.ClientSession()
            logger.info(f"创建临时会话以请求API: {api_url}")
            
            # 发送API请求
            async with session.post(api_url, headers=headers, json=data, timeout=60) as response:
                # 检查响应状态
                if response.status != 200:
                    error_text = await response.text()
                    error_message = f"错误: API请求失败，状态码: {response.status}, 错误: {error_text}"
                    logger.error(error_message)
                    yield StreamResponse(content="", finish_reason="error", status="error", 
                                 error=f"API request failed with status {response.status}: {error_text}")
                    return
                
                # 处理流式响应
                buffer = ""
                async for line in response.content:
                    line = line.decode('utf-8')
                    
                    # 处理可能包含多行数据的响应
                    for subline in line.split('\n'):
                        if not subline.strip():
                            continue
                            
                        if subline.startswith('data: '):
                            subline = subline[6:]  # 删除'data: '前缀
                        
                        if subline.strip() == '[DONE]':
                            break
                            
                        try:
                            chunk = json.loads(subline)
                            if self.default_config.provider == "anthropic":
                                if chunk.get('type') == 'content_block_delta':
                                    text = chunk.get('delta', {}).get('text', '')
                                    if text:
                                        yield StreamResponse(content=text, status="generating")
                            elif self.default_config.provider == "openai":
                                choices = chunk.get('choices', [])
                                if choices and 'delta' in choices[0]:
                                    delta = choices[0]['delta']
                                    if 'content' in delta and delta['content']:
                                        yield StreamResponse(content=delta['content'], status="generating")
                                elif self.default_config.provider == "deepseek" or self.default_config.provider == LLMProvider.DEEPSEEK:
                                    choices = chunk.get('choices', [])
                                    if choices and 'delta' in choices[0]:
                                        delta = choices[0]['delta']
                                        if 'content' in delta and delta['content']:
                                            yield StreamResponse(content=delta['content'], status="generating")
                                elif self.default_config.provider == "zhipu" or self.default_config.provider == LLMProvider.ZHIPU:
                                    choices = chunk.get('choices', [])
                                    if choices and 'delta' in choices[0]:
                                        delta = choices[0]['delta']
                                        if 'content' in delta and delta['content']:
                                            yield StreamResponse(content=delta['content'], status="generating")
                        except json.JSONDecodeError:
                            logger.warning(f"无法解析JSON响应: {subline}")
                        except Exception as e:
                            logger.error(f"处理响应片段时出错: {str(e)}", exc_info=True)
                            yield StreamResponse(content=f"处理错误: {str(e)}", status="error", error=str(e))
        except aiohttp.ClientError as e:
            error_message = f"网络错误: {str(e)}"
            logger.error(error_message, exc_info=True)
            yield StreamResponse(content=error_message, finish_reason="error", status="error", error=str(e))
        except Exception as e:
            error_message = f"错误: {str(e)}"
            logger.error(error_message, exc_info=True)
            yield StreamResponse(content=error_message, finish_reason="error", status="error", error=str(e))
        finally:
            # 确保会话在完成后关闭
            if session:
                try:
                    await session.close()
                    logger.info("关闭临时创建的会话")
                except Exception as e:
                    logger.error(f"关闭会话时出错: {str(e)}", exc_info=True)
    
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
    
    async def generate_stream(self, messages: List[Message], config: Optional[Union[LLMConfig, Dict]] = None, 
                           message_id: str = "", stop_generation: bool = False) -> AsyncGenerator[StreamResponse, None]:
        """
        生成流式响应
        
        Args:
            messages: 消息列表
            config: LLM配置，可选，可以是LLMConfig对象或配置字典
            message_id: 消息ID，用于日志和停止生成
            stop_generation: 是否应停止生成
            
        Yields:
            流式响应对象
        """
        # 如果请求停止生成，则立即返回
        if stop_generation:
            yield StreamResponse(content="", finish_reason="stop", status="stopped", 
                                error="Generation stopped by user request")
            return
        
        if not messages:
            logger.error("生成流式响应时消息列表为空")
            yield StreamResponse(content="", finish_reason="error", status="error", 
                                error="Empty message list")
            return
        
        # 使用默认配置或输入配置
        used_config = config or self.default_config
        
        # 处理消息格式，确保能够正确访问属性
        processed_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                # 如果是字典，创建一个包含role和content的新字典
                processed_msg = {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                }
                processed_messages.append(processed_msg)
            elif isinstance(msg, Message):
                # 如果是Message对象，转换为字典
                processed_msg = {
                    "role": msg.role.value if isinstance(msg.role, MessageRole) else msg.role,
                    "content": msg.content
                }
                processed_messages.append(processed_msg)
            else:
                # 不支持的消息类型
                logger.error(f"生成流式响应：不支持的消息类型: {type(msg)}")
                yield StreamResponse(content="", finish_reason="error", status="error", 
                                    error=f"Unsupported message type: {type(msg)}")
                return
                
        if len(processed_messages) == 0:
            logger.error("生成流式响应：所有消息都无法处理")
            yield StreamResponse(content="", finish_reason="error", status="error", 
                                error="All messages could not be processed")
            return
        
        # 获取提供商的API端点和请求头
        provider_str = self._get_provider_str(used_config)
        api_url = self._get_api_endpoint(used_config, stream=True)
        headers = self._get_request_headers(provider_str, used_config.api_key if hasattr(used_config, 'api_key') else "")
        
        # 准备请求数据
        data = {}
        try:
            # 准备用于特定提供商的请求数据
            data = self._prepare_request_data(processed_messages, used_config, stream=True)
        except Exception as e:
            logger.error(f"准备API请求时出错: {str(e)}")
            yield StreamResponse(content="", finish_reason="error", status="error", 
                                error=f"Failed to prepare API request: {str(e)}")
            return
            
        logger.info(f"开始流式请求 {api_url}，消息ID: {message_id}")
        session = None
        
        try:
            session = aiohttp.ClientSession()
            async with session.post(api_url, json=data, headers=headers, timeout=120) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API请求失败: {response.status} - {error_text}")
                    logger.error(f"请求失败详情 - 请求头: {headers}")
                    yield StreamResponse(content="", finish_reason="error", status="error", 
                                        error=f"API request failed with status {response.status}: {error_text}")
                    return
                
                # 处理流式响应
                buffer = ""
                try:
                    async for line in response.content:
                        # 转换bytes为字符串
                        line = line.decode('utf-8')
                        
                        # 处理可能包含多行数据的响应
                        for subline in line.split('\n'):
                            if not subline.strip():
                                continue
                                
                            if subline.startswith('data: '):
                                subline = subline[6:]  # 删除'data: '前缀
                                
                                if subline.strip() == '[DONE]':
                                    break
                
                                try:
                                    chunk = json.loads(subline)
                                    if self.default_config.provider == "anthropic":
                                        if chunk.get('type') == 'content_block_delta':
                                            text = chunk.get('delta', {}).get('text', '')
                                            if text:
                                                yield StreamResponse(content=text, status="generating")
                                    elif self.default_config.provider == "openai":
                                        choices = chunk.get('choices', [])
                                        if choices and 'delta' in choices[0]:
                                            delta = choices[0]['delta']
                                            if 'content' in delta and delta['content']:
                                                yield StreamResponse(content=delta['content'], status="generating")
                                    elif self.default_config.provider == "deepseek" or self.default_config.provider == LLMProvider.DEEPSEEK:
                                        choices = chunk.get('choices', [])
                                        if choices and 'delta' in choices[0]:
                                            delta = choices[0]['delta']
                                            if 'content' in delta and delta['content']:
                                                yield StreamResponse(content=delta['content'], status="generating")
                                    elif self.default_config.provider == "zhipu" or self.default_config.provider == LLMProvider.ZHIPU:
                                        choices = chunk.get('choices', [])
                                        if choices and 'delta' in choices[0]:
                                            delta = choices[0]['delta']
                                            if 'content' in delta and delta['content']:
                                                yield StreamResponse(content=delta['content'], status="generating")
                                except json.JSONDecodeError:
                                    logger.warning(f"无法解析JSON响应: {subline}")
                except Exception as e:
                                    logger.error(f"处理响应片段时出错: {str(e)}", exc_info=True)
                                    yield StreamResponse(content=f"处理错误: {str(e)}", status="error", error=str(e))
        except Exception as e:
                    logger.error(f"读取响应内容时出错: {str(e)}", exc_info=True)
                    yield StreamResponse(content="", finish_reason="error", status="error", 
                                        error=f"Error reading response content: {str(e)}")
        except aiohttp.ClientError as e:
            logger.exception(f"HTTP客户端错误: {str(e)}")
            yield StreamResponse(content="", finish_reason="error", status="error", 
                                error=f"HTTP client error: {str(e)}")
        
        except asyncio.TimeoutError:
            logger.error(f"API请求超时")
            yield StreamResponse(content="", finish_reason="error", status="error", 
                                error="Request timeout")
        
        except Exception as e:
            logger.exception(f"生成流式响应时发生错误: {str(e)}")
            yield StreamResponse(content="", finish_reason="error", status="error", 
                                error=f"Unexpected error: {str(e)}")
        
        finally:
            # 确保会话始终被关闭
            if session and not session.closed:
                await session.close()
                logger.debug("客户端会话已关闭")

    async def generate(self, **kwargs):
        """
        generate方法，作为generate_response方法的别名
        
        为了提供对llm_routes.py的向后兼容性
        支持model, temperature, max_tokens等参数
        """
        logger.info("使用generate别名方法，调用generate_response")
        
        # 提取消息列表
        messages = kwargs.get("messages", [])
        
        # 创建配置对象
        config = None
        if "model" in kwargs or "temperature" in kwargs or "max_tokens" in kwargs:
            # 获取提供商枚举
            provider = None
            if "provider" in kwargs and kwargs["provider"]:
                try:
                    provider = LLMProvider(kwargs["provider"])
                except ValueError:
                    provider = self.default_config.provider
            else:
                provider = self.default_config.provider
                
            # 确保max_tokens为整数或使用默认值
            max_tokens_value = kwargs.get("max_tokens")
            if max_tokens_value is None:
                max_tokens_value = self.default_config.max_tokens
            else:
                max_tokens_value = int(max_tokens_value)
                
            # 创建配置对象
            config = LLMConfig(
                provider=provider,
                model_name=kwargs.get("model", self.default_config.model_name),
                api_key=get_api_key(provider) or "",
                temperature=kwargs.get("temperature", self.default_config.temperature),
                max_tokens=max_tokens_value
            )
        
        # 调用generate_response
        response = await self.generate_response(messages, config)
        
        # 转换LLMResponse对象为字典，保持与chat_completion一致
        return {
            "choices": [
                {
                    "message": {"content": response.content},
                    "finish_reason": response.finish_reason
                }
            ],
            "usage": {"total_tokens": response.tokens_used},
            "model": response.model,
            "provider": response.provider.value if response.provider else None
        }
    
    async def chat_completion(self, **kwargs):
        """
        chat_completion方法，作为generate_response方法的别名
        
        为了提供对RAGEnhancedService的向后兼容性
        支持model, temperature, max_tokens等参数
        """
        logger.info("使用chat_completion别名方法，调用generate_response")
        
        try:
            # 提取消息列表
            messages = kwargs.get("messages", [])
        
            # 处理消息格式，确保能够正确访问属性
            processed_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    # 如果是字典，创建一个包含role和content的新字典
                    processed_msg = {
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    }
                    processed_messages.append(processed_msg)
                elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                    # 如果是Message对象或类似对象
                    processed_msg = {
                        "role": msg.role,
                        "content": msg.content
                    }
                    processed_messages.append(processed_msg)
                else:
                    logger.warning(f"无法处理的消息格式: {type(msg)}")
                    # 默认处理为用户消息
                    processed_messages.append({"role": "user", "content": str(msg)})
            
            # 获取模型信息
            model = kwargs.get("model", "deepseek-chat")
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 2048)
            provider = kwargs.get("provider", "deepseek")
            
            # 检查API密钥
            api_key = kwargs.get("api_key") or os.environ.get(f"{provider.upper()}_API_KEY", "")
            if not api_key:
                logger.warning(f"未配置{provider}的API密钥，返回模拟响应")
                return {
                    "choices": [
                        {
                            "message": {"content": "由于配置问题，暂时无法连接到LLM服务。请检查服务配置。"},
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {"total_tokens": 0},
                    "model": model,
                    "provider": provider
                }
            
            # 准备API请求
            try:
                api_url = self._get_api_endpoint({"provider": provider}, stream=False)
                headers = self._get_request_headers(provider, api_key)
                
                # 准备请求数据
                data = {
                    "model": model,
                    "messages": processed_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                }
                
                # 创建临时会话发送请求
                timeout = aiohttp.ClientTimeout(total=60)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(api_url, headers=headers, json=data) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"API请求失败: {response.status} - {error_text}")
                            return {
                                "choices": [
                                    {
                                        "message": {"content": f"API请求失败: {error_text}"},
                                        "finish_reason": "error"
                                    }
                                ],
                                "error": error_text
                            }
                        
                        # 处理响应
                        json_response = await response.json()
                        content = "未能从API响应中提取内容"
                        
                        # 提取内容，处理不同响应格式
                        if "choices" in json_response and json_response["choices"]:
                            if "message" in json_response["choices"][0]:
                                content = json_response["choices"][0]["message"].get("content", "")
                        
                        return {
                            "choices": [
                                {
                                    "message": {"content": content},
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": json_response.get("usage", {"total_tokens": 0}),
                            "model": model,
                            "provider": provider
                        }
            except Exception as e:
                logger.error(f"API请求失败: {str(e)}")
                return {
                    "choices": [
                        {
                            "message": {"content": f"API请求失败: {str(e)}"},
                            "finish_reason": "error"
                        }
                    ],
                    "error": str(e)
                }
                
        except Exception as e:
            logger.error(f"chat_completion执行失败: {str(e)}")
            return {
                "choices": [
                    {
                        "message": {"content": f"发生错误: {str(e)}"},
                        "finish_reason": "error"
                    }
                ],
                "error": str(e)
            }
    
    async def chat_completion_stream(self, **kwargs):
        """
        chat_completion_stream方法，作为generate_stream方法的别名
        
        为了提供对RAGEnhancedService的向后兼容性
        支持model, temperature, max_tokens等参数
        """
        logger.info("使用chat_completion_stream别名方法，调用generate_stream")
        
        # 提取消息列表和相关参数
        messages = kwargs.get("messages", [])
        message_id = kwargs.get("message_id")
        stop_generation = kwargs.get("stop_generation")
        model = kwargs.get("model")
        provider = kwargs.get("provider")
        api_key = kwargs.get("api_key")
        
        # 创建配置对象
        config = None
        if "model" in kwargs or "temperature" in kwargs or "max_tokens" in kwargs:
            # 获取提供商枚举
            provider_enum = None
            if provider:
                try:
                    provider_enum = LLMProvider(provider)
                except ValueError:
                    provider_enum = self.default_config.provider
            else:
                provider_enum = self.default_config.provider
                
            # 确保max_tokens为整数或使用默认值
            max_tokens_value = kwargs.get("max_tokens")
            if max_tokens_value is None:
                max_tokens_value = self.default_config.max_tokens
            else:
                max_tokens_value = int(max_tokens_value)
                
            # 创建配置对象
            config = {
                "provider": provider_enum.value,
                "model_name": model or self.default_config.model_name,
                "api_key": api_key or get_api_key(provider_enum) or "",
                "temperature": kwargs.get("temperature", self.default_config.temperature),
                "max_tokens": max_tokens_value
            }
        
        # 调用底层方法
        async for chunk in self.generate_stream(
            messages, 
            model=model,
            provider=provider,
            api_key=api_key,
            config=config,
            message_id=message_id,
            stop_generation=stop_generation
        ):
            # 转换StreamResponse对象为RAGEnhancedService期望的格式
            if chunk.status == "starting":
                # 开始消息，转换成字典格式
                yield {
                    "choices": [{"delta": {"role": "assistant"}}],
                    "model": getattr(chunk, "model", "unknown-model")
                }
            elif chunk.status == "completed" or (chunk.finish_reason == "stop" and chunk.status != "generating"):
                # 结束消息，转换成特殊字典格式
                # 注意：不要返回字符串，而是返回一个标记结束的字典
                yield {
                    "choices": [{"finish_reason": chunk.finish_reason or "stop"}],
                    "model": getattr(chunk, "model", "unknown-model"),
                    "is_done": True  # 添加标记以帮助识别结束
                }
            elif chunk.status == "stopped":
                # 处理停止生成的情况
                yield {
                    "choices": [{"delta": {"content": chunk.content}, "finish_reason": "stopped"}],
                    "model": getattr(chunk, "model", "unknown-model"),
                    "stopped": True
                }
            elif chunk.content is not None:
                # 内容块，转换成标准格式的字典
                yield {
                    "choices": [{"delta": {"content": chunk.content}}],
                    "model": getattr(chunk, "model", "unknown-model")
                }

    def _get_request_config(self, config: Optional[LLMConfig], provider: Optional[LLMProvider], model: Optional[str]) -> LLMConfig:
        """获取请求配置，处理默认值和覆盖值"""
        # 使用默认配置作为基础
        result_config = self.default_config
        
        # 如果提供了配置，应用它
        if config:
            # 处理provider（如果为default，使用默认提供商）
            if config.provider == LLMProvider.DEFAULT:
                provider_value = result_config.provider
            else:
                provider_value = config.provider
            
            # 处理model_name（如果为default，使用默认模型）
            if config.model_name == "default":
                model_name_value = result_config.model_name
            else:
                model_name_value = config.model_name
            
            # 创建新配置对象
            result_config = LLMConfig(
                provider=provider_value,
                model_name=model_name_value,
                api_key=config.api_key or result_config.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                stop=config.stop
            )
        
        # 如果提供了provider参数，覆盖配置中的provider
        if provider:
            if provider == LLMProvider.DEFAULT:
                result_config.provider = self.default_config.provider
            else:
                # 检查provider是否为字符串或枚举
                if isinstance(provider, str):
                    try:
                        result_config.provider = LLMProvider(provider)
                    except ValueError:
                        logger.warning(f"无效的提供商: {provider}，使用默认值: {result_config.provider}")
                else:
                    result_config.provider = provider
        
        # 如果提供了model参数，覆盖配置中的model_name
        if model:
            # 检查model是否包含provider信息（如"deepseek/deepseek-chat"）
            if '/' in model:
                parts = model.split('/', 1)
                try:
                    result_config.provider = LLMProvider(parts[0])
                    result_config.model_name = parts[1]
                except ValueError:
                    logger.warning(f"无效的提供商格式: {parts[0]}，使用默认提供商")
                    if model == "default":
                        result_config.model_name = self.default_config.model_name
                    else:
                        result_config.model_name = model
            else:
                if model == "default":
                    result_config.model_name = self.default_config.model_name
                else:
                    result_config.model_name = model
        
        # 尝试获取API密钥
        if not result_config.api_key:
            try:
                result_config.api_key = self._get_api_key_for_provider(result_config.provider)
            except Exception as e:
                logger.warning(f"获取API密钥失败: {str(e)}，使用空API密钥")
        
        logger.debug(f"生成请求配置: 提供商={result_config.provider}, 模型={result_config.model_name}")
        return result_config
    
    def _get_api_key_for_provider(self, provider: LLMProvider) -> str:
        """根据提供商获取API密钥"""
        if provider == LLMProvider.DEEPSEEK:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            if not api_key.startswith("sk-"):
                api_key = f"sk-{api_key}"
            return api_key
        elif provider == LLMProvider.ZHIPU:
            return os.environ.get("ZHIPU_API_KEY", "")
        elif provider == LLMProvider.OPENAI:
            return os.environ.get("OPENAI_API_KEY", "")
        else:
            return ""

    def _get_request_url(self, config: LLMConfig) -> str:
        """
        获取请求URL
        
        Args:
            config: LLM配置
            
        Returns:
            请求URL
        """
        return self._get_api_endpoint(config)

    def _get_response_parser(self, config: LLMConfig):
        """
        获取响应解析器
        
        Args:
            config: LLM配置
            
        Returns:
            响应解析器对象
        """
        # 简单的解析器实现
        class ResponseParser:
            def __init__(self, config):
                self.config = config
                
            def parse(self, chunk):
                """解析流式响应块"""
                if self.config.provider == LLMProvider.DEEPSEEK:
                    if 'choices' in chunk:
                        for choice in chunk['choices']:
                            if 'delta' in choice and 'content' in choice['delta']:
                                content = choice['delta']['content']
                                finish_reason = choice.get('finish_reason')
                                return StreamResponse(
                                    content=content,
                                    finish_reason=finish_reason
                                )
                elif self.config.provider == LLMProvider.ZHIPU:
                    if 'choices' in chunk:
                        for choice in chunk['choices']:
                            if 'delta' in choice and 'content' in choice['delta']:
                                content = choice['delta']['content']
                                finish_reason = choice.get('finish_reason')
                                return StreamResponse(
                                    content=content,
                                    finish_reason=finish_reason
                                )
                return None
            
        return ResponseParser(config)

    def _get_provider_string(self, provider_name=None):
        """Get the correct provider string to use for API calls."""
        # 添加防御性检查
        if provider_name is None:
            # 获取默认提供商，确保不为None
            if hasattr(self.default_config, 'provider'):
                if hasattr(self.default_config.provider, 'value'):
                    provider_name = self.default_config.provider.value
                else:
                    provider_name = str(self.default_config.provider)
            else:
                provider_name = "deepseek"  # 强制设置一个默认值
            
            logger.debug(f"Provider is None, using default: {provider_name}")
        
        # 确保provider_name是字符串类型
        if not isinstance(provider_name, str):
            try:
                provider_name = str(provider_name)
            except Exception as e:
                logger.error(f"无法将provider_name转换为字符串: {str(e)}", exc_info=True)
                provider_name = "deepseek"  # 强制设置一个默认值
        
        # 原始逻辑保持不变
        provider = provider_name.upper()
        if provider == "AZURE_OPENAI":
            return "AZURE_OPENAI"
        elif provider == "OPENAI":
            return "OPENAI"
        elif provider == "HUGGINGFACE":
            return "HUGGINGFACE"
        elif provider == "DEEPSEEK":
            return "DEEPSEEK"
        elif provider == "CLAUDE":
            return "CLAUDE"
        elif provider == "ZHIPU":
            return "ZHIPU"
        else:
            logger.warning(f"Unknown provider: {provider_name}, using default.")
            return "DEEPSEEK"  # 直接返回固定值而不是依赖self.default_config

def get_api_key(provider):
    """
    根据提供商获取API密钥
    
    Args:
        provider: LLMProvider枚举值
        
    Returns:
        对应的API密钥
    """
    if provider == LLMProvider.DEEPSEEK:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        # 确保DeepSeek API密钥格式正确，必须以"sk-"开头
        if api_key and not api_key.startswith("sk-"):
            api_key = f"sk-{api_key}"
            logger.info(f"get_api_key时修正DeepSeek API密钥格式，添加sk-前缀")
        return api_key
    elif provider == LLMProvider.ZHIPU:
        return os.environ.get("ZHIPU_API_KEY", "")
    elif provider == LLMProvider.OPENAI:
        return os.environ.get("OPENAI_API_KEY", "")
    else:
        return None

# 创建默认配置
default_config = LLMConfig(
    provider=LLMProvider.DEEPSEEK,
    model_name="deepseek-chat",
    api_key=get_api_key(LLMProvider.DEEPSEEK),
    temperature=0.7,
    max_tokens=2048
)

# 创建LLM服务实例
llm_service = LLMService(default_config=default_config)

# 确保关键方法在模块级别可用，防止导入问题
__all__ = [
    'LLMService', 'LLMConfig', 'LLMProvider', 
    'Message', 'MessageRole', 'PromptTemplate',
    'LLMResponse', 'StreamResponse', 'get_api_key',
    'llm_service'  # 添加新创建的llm_service实例
] 