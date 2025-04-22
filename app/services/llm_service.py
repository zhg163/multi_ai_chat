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
    
    # 禁用Protected命名空间检查
    model_config = {
        'protected_namespaces': ()
    }

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
    text: Optional[str] = None  # 添加text字段以兼容旧代码
    event_type: Optional[str] = None  # 添加event_type字段以兼容旧代码
    model: Optional[str] = None
    provider: Optional[str] = None
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    is_start: bool = False
    is_end: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 添加metadata字段存储额外信息

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
    
    def _get_request_headers(self, provider: LLMProvider, api_key: Optional[str] = None) -> Dict[str, str]:
        """
        获取请求头部
        
        Args:
            provider: LLM提供商
            api_key: 可选的API密钥，如果不提供则使用默认配置
            
        Returns:
            请求头部字典
        """
        # 获取API密钥
        if api_key is None:
            api_key = self.get_api_key(provider)
            
        headers = {
            "Content-Type": "application/json"
        }
        
        # DeepSeek API特殊处理
        if provider == LLMProvider.DEEPSEEK:
            # 记录原始API密钥格式（部分脱敏）
            masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
            logger.info(f"Deepseek API密钥原始格式: {masked_key}")
            
            # 确保API密钥以sk-开头
            if not api_key.startswith("sk-"):
                logger.warning("Deepseek API密钥不是以sk-开头，自动添加前缀")
                api_key = f"sk-{api_key}"
                masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "***"
                logger.info(f"修正后的API密钥格式: {masked_key}")
            
            # 构造Authorization头
            headers["Authorization"] = f"Bearer {api_key}"
            masked_auth = f"{headers['Authorization'][:12]}...{headers['Authorization'][-4:]}" if len(headers["Authorization"]) > 16 else "***"
            logger.info(f"Deepseek Authorization头部: {masked_auth}")
            
        # ZHIPU API特殊处理
        elif provider == LLMProvider.ZHIPU:
            headers["Authorization"] = api_key
            
        return headers
    
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
    async def _make_api_request(self, endpoint: str, headers: Dict[str, str], data: Dict) -> aiohttp.ClientResponse:
        """
        进行API请求
        
        Args:
            endpoint: API地址
            headers: 请求头
            data: 请求数据
            
        Returns:
            API响应
        """
        # 记录请求信息（脱敏）
        logger.info(f"Sending request to endpoint: {endpoint}")
        
        # 对请求头部进行检查，特别是Deepseek API
        if "deepseek.com" in endpoint and "Authorization" in headers:
            auth_header = headers["Authorization"]
            masked_auth = f"{auth_header[:12]}...{auth_header[-4:] if len(auth_header) > 16 else ''}"
            logger.info(f"请求头部Authorization: {masked_auth}")
            
            # 确保Authorization头部格式正确（以Bearer开头）
            if not auth_header.startswith("Bearer ") and auth_header.startswith("sk-"):
                logger.warning("检测到Authorization头部未以Bearer开头，进行修正")
                headers["Authorization"] = f"Bearer {auth_header}"
                logger.info(f"修正后的Authorization: {headers['Authorization'][:12]}...{headers['Authorization'][-4:] if len(headers['Authorization']) > 16 else ''}")
        
        logger.debug(f"Request data: {json.dumps(data, ensure_ascii=False)}")
        
        try:
            logger.info("创建临时会话发送请求")
            # 创建临时会话发送请求
            session = aiohttp.ClientSession()
            try:
                response = await session.post(endpoint, headers=headers, json=data)
                logger.info(f"API请求成功，响应状态: {response.status}")
                return response
            except Exception as e:
                logger.error(f"API请求失败: {str(e)}")
                raise
            finally:
                await session.close()
                logger.info("临时会话已关闭")
        except Exception as e:
            logger.error(f"创建会话或发送请求时出错: {str(e)}")
            raise
    
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
        headers = self._get_request_headers(used_config.provider, used_config.api_key)
        data = self._prepare_request_data(processed_messages, used_config)
        
        try:
            # 发送API请求
            response = await self._make_api_request(
                endpoint=endpoint,
                headers=headers,
                data=data
            )
            
            # 解析响应
            return self._parse_response(await response.json(), used_config)
            
        except Exception as e:
            logger.error(f"生成回复时出错: {str(e)}")
            raise
    
    async def generate_stream_response(self, messages: List[Union[Dict, Message]], config: Optional[Dict] = None) -> AsyncGenerator[str, None]:
        """
        通过API生成流式响应
        
        Args:
            messages: 消息列表，可以是字典或Message对象
            config: 配置参数，可选
            
        Yields:
            生成的文本片段
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
        headers = self._get_request_headers(self.default_config.provider)
        
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
                    yield error_message
                    return
                
                # 处理流式响应
                buffer = ""
                async for line in response.content:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        line = line[6:]  # 删除'data: '前缀
                        
                        if line.strip() == '[DONE]':
                            break
                            
                        try:
                            chunk = json.loads(line)
                            if self.default_config.provider == "anthropic":
                                if chunk.get('type') == 'content_block_delta':
                                    text = chunk.get('delta', {}).get('text', '')
                                    if text:
                                        yield text
                            elif self.default_config.provider == "openai":
                                choices = chunk.get('choices', [])
                                if choices and 'delta' in choices[0]:
                                    delta = choices[0]['delta']
                                    if 'content' in delta and delta['content']:
                                        yield delta['content']
                        except json.JSONDecodeError:
                            logger.warning(f"无法解析JSON响应: {line}")
                        except Exception as e:
                            logger.error(f"处理响应片段时出错: {str(e)}", exc_info=True)
                            yield f"处理错误: {str(e)}"
        except aiohttp.ClientError as e:
            error_message = f"网络错误: {str(e)}"
            logger.error(error_message, exc_info=True)
            yield error_message
        except Exception as e:
            error_message = f"错误: {str(e)}"
            logger.error(error_message, exc_info=True)
            yield error_message
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
    
    async def generate_stream(
        self, 
        messages: List[Union[Dict, Message]], 
        model: str = None,
        provider: str = None,
        api_key: str = None,
        config: Optional[Dict] = None,
        message_id: str = None,
        stop_generation: Optional[Dict[str, bool]] = None
    ) -> AsyncGenerator[StreamResponse, None]:
        """
        生成流式响应
        
        Args:
            messages: 消息列表
            model: 模型名称（可选）
            provider: 提供商（可选）
            api_key: API密钥（可选）
            config: 配置字典（可选）
            message_id: 消息ID（可选）
            stop_generation: 停止生成的标志字典（可选）
            
        Yields:
            流式响应数据
        """
        # 确保messages是Message对象列表
        msgs = []
        for item in messages:
            if isinstance(item, dict):
                msgs.append(Message(
                    role=item.get("role", "user"),
                    content=item.get("content", "")
                ))
            else:
                msgs.append(item)
                
        # 确定配置
        llm_config = self.default_config
        
        # 如果提供了配置字典，使用它覆盖默认配置
        if config:
            logger.info(f"使用提供的配置覆盖默认配置")
            for key, value in config.items():
                if hasattr(llm_config, key):
                    setattr(llm_config, key, value)
        
        # 如果单独提供了提供商、模型或API密钥，优先使用它们
        if provider:
            logger.info(f"使用提供的提供商: {provider}")
            try:
                llm_config.provider = provider if isinstance(provider, LLMProvider) else LLMProvider(provider)
            except ValueError:
                logger.warning(f"无效的提供商: {provider}, 使用默认提供商: {llm_config.provider}")
        
        if model:
            logger.info(f"使用提供的模型: {model}")
            llm_config.model_name = model
        
        if api_key:
            # 记录API密钥的长度和格式（部分脱敏）
            key_format = f"{api_key[:5]}...{api_key[-4:]}" if len(api_key) > 10 else "格式不正确"
            logger.info(f"使用提供的API密钥: 长度={len(api_key)}, 格式={key_format}, 是否以sk-开头={api_key.startswith('sk-')}")
            llm_config.api_key = api_key
            
        # 记录最终使用的配置（脱敏）
        logger.info(f"开始生成流式响应，模型：{llm_config.model_name}，提供商：{llm_config.provider}")
        
        # API密钥脱敏记录
        safe_key = llm_config.api_key
        if safe_key:
            safe_key = f"{safe_key[:5]}...{safe_key[-4:]}" if len(safe_key) > 10 else "[密钥格式不正确]"
            key_format = f"长度={len(llm_config.api_key)}, 格式={safe_key}, 是否以sk-开头={llm_config.api_key.startswith('sk-')}"
            logger.info(f"最终使用的API密钥信息: {key_format}")
            
        logger.info(f"调用API生成流式响应, 模型: {llm_config.model_name}, 提供商: {llm_config.provider}")
        
        # 发送开始事件
        yield StreamResponse(
            text="",
            event_type="start",
            is_start=True,
            metadata={"model": llm_config.model_name, "provider": str(llm_config.provider)}
        )
        
        try:
            endpoint = self._get_api_endpoint(llm_config, stream=True)
            headers = self._get_request_headers(llm_config.provider, llm_config.api_key)
            data = self._prepare_request_data(msgs, llm_config, stream=True)
            
            # 创建完整文本字符串，用于累积生成的内容
            full_text = ""
            
            # 确保已创建会话
            await self._ensure_session()
            
            # 记录开始API调用
            logger.info(f"开始调用流式API: {endpoint}")
            
            # 创建超时对象
            timeout_obj = aiohttp.ClientTimeout(total=llm_config.timeout)
            
            # 发送API请求并处理流式响应
            async with self.session.post(endpoint, 
                                        headers=headers, 
                                        json=data, 
                                        timeout=timeout_obj) as response:
                if response.status != 200:
                    error_text = await response.text()
                    error_msg = f"API请求失败: {response.status}, 错误: {error_text}"
                    logger.error(error_msg)
                    
                    # 记录更详细的错误信息
                    safe_headers = headers.copy()
                    if "Authorization" in safe_headers:
                        auth_val = safe_headers["Authorization"]
                        safe_headers["Authorization"] = f"{auth_val[:12]}...{auth_val[-4:]}"
                    logger.error(f"请求失败详情 - 状态码: {response.status}, URL: {endpoint}")
                    logger.error(f"请求失败详情 - 请求头: {safe_headers}")
                    
                    # 发送错误事件
                    yield StreamResponse(
                        text=error_msg,
                        event_type="error",
                        metadata={"error": error_text, "status": response.status}
                    )
                    return
                
                # 处理流式响应
                async for line in response.content:
                    line = line.strip()
                    if not line or line == b"":
                        continue
                    
                    if line.startswith(b"data:"):
                        line = line[5:].strip()
                    
                    # 如果设置了停止标志，中断生成
                    if stop_generation and message_id in stop_generation and stop_generation[message_id]:
                        logger.info(f"收到停止生成请求, 消息ID: {message_id}")
                        break
                    
                    try:
                        chunk = json.loads(line)
                        text_chunk, finish_reason = self._parse_stream_chunk(chunk, llm_config)
                        
                        if text_chunk:
                            full_text += text_chunk
                            yield StreamResponse(
                                content=text_chunk,
                                text=text_chunk,
                                event_type="content",
                                metadata={"model": llm_config.model_name}
                            )
                        
                        if finish_reason:
                            logger.info(f"生成完成, 结束原因: {finish_reason}")
                            break
                    except json.JSONDecodeError:
                        logger.warning(f"无法解析JSON: {line}")
                    except Exception as e:
                        logger.error(f"处理流式响应块时出错: {str(e)}")
            
            # 发送结束事件
            yield StreamResponse(
                content="",
                text="",
                event_type="end",
                is_end=True,
                metadata={"model": llm_config.model_name, "provider": str(llm_config.provider), "full_text": full_text}
            )
        except Exception as e:
            logger.error(f"生成流式响应时发生错误: {str(e)}")
            # 发送错误事件
            yield StreamResponse(
                content=str(e),
                text=str(e),
                event_type="error",
                metadata={"error": str(e)}
            )

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
    
    # 添加别名方法，以兼容RAGEnhancedService的调用
    async def chat_completion(self, **kwargs):
        """
        chat_completion方法，作为generate_response方法的别名
        
        为了提供对RAGEnhancedService的向后兼容性
        支持model, temperature, max_tokens等参数
        """
        logger.info("使用chat_completion别名方法，调用generate_response")
        
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
        
        # 调用底层方法
        response = await self.generate_response(messages, config)
        
        # 将LLMResponse对象转换为字典，以便RAGEnhancedService可以使用get()方法
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
            if chunk.is_start:
                # 开始消息，转换成字典格式
                yield {
                    "choices": [{"delta": {"role": "assistant"}}],
                    "model": chunk.model or "unknown-model"
                }
            elif chunk.is_end:
                # 结束消息，转换成特殊字典格式
                # 注意：不要返回字符串，而是返回一个标记结束的字典
                yield {
                    "choices": [{"finish_reason": "stop"}],
                    "model": chunk.model or "unknown-model",
                    "is_done": True  # 添加标记以帮助识别结束
                }
            elif chunk.event_type == "stopped":
                # 处理停止生成的情况
                yield {
                    "choices": [{"delta": {"content": chunk.text}, "finish_reason": "stopped"}],
                    "model": chunk.model or "unknown-model",
                    "stopped": True
                }
            elif chunk.content is not None:
                # 内容块，转换成标准格式的字典
                yield {
                    "choices": [{"delta": {"content": chunk.content}}],
                    "model": chunk.model or "unknown-model"
                }

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

# 确保关键方法在模块级别可用，防止导入问题
__all__ = [
    'LLMService', 'LLMConfig', 'LLMProvider', 
    'Message', 'MessageRole', 'PromptTemplate',
    'LLMResponse', 'StreamResponse', 'get_api_key'
] 