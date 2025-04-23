"""
Two-Phase LLM Streaming Service - 用于角色选择和流式回复生成的两阶段服务

第一阶段：根据用户输入选择最合适的角色
第二阶段：根据选定的角色生成流式回复
"""

import logging
import random
import asyncio
import json
from typing import Dict, List, Tuple, Optional, Any, AsyncGenerator

from app.services.llm_service import LLMService, Message, MessageRole, LLMConfig, LLMProvider, StreamResponse
from app.services.llm_service_two_phase import TwoPhaseService

logger = logging.getLogger(__name__)

class TwoPhaseStreamService:
    """Two-Phase LLM Service with streaming capabilities"""

    @staticmethod
    async def select_role(message: str, roles: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], float, str]:
        """
        根据用户消息和可用角色，使用LLM选择最合适的角色
        
        Args:
            message: 用户消息
            roles: 可用角色列表
            
        Returns:
            tuple: (选中的角色, 匹配分数, 匹配原因)
        """
        # 复用普通TwoPhaseService中的角色选择功能
        return await TwoPhaseService.select_role(message, roles)

    @staticmethod
    async def generate_stream_response(
        message: str, 
        role: Dict[str, Any]
    ) -> AsyncGenerator[StreamResponse, None]:
        """
        根据用户消息和选定角色生成流式回复
        
        Args:
            message: 用户消息
            role: 选定的角色
            
        Yields:
            StreamResponse: 流式响应片段
        """
        # 创建LLM服务
        llm_service = LLMService()
        
        # 获取角色信息
        role_name = role.get("role_name", "Unknown")
        system_prompt = role.get("system_prompt", "")
        
        if not system_prompt:
            logger.warning(f"角色 {role_name} 没有system_prompt")
            system_prompt = f"你是{role_name}，请以自然的方式回复用户。"
        
        # 准备消息
        system_message = Message(role=MessageRole.SYSTEM, content=system_prompt)
        user_message = Message(role=MessageRole.USER, content=message)
        
        try:
            # 获取角色的温度设置
            temperature = float(role.get("temperature", 0.7))
            
            # 记录开始生成流式响应
            logger.info(f"开始为角色 {role_name} 生成流式回复")
            
            # 调用流式API
            async for chunk in llm_service.generate_stream(
                messages=[system_message, user_message],
                config=LLMConfig(
                    provider=llm_service.default_config.provider,
                    model_name=llm_service.default_config.model_name,
                    api_key=llm_service.default_config.api_key,
                    temperature=temperature
                )
            ):
                # 传递每个响应片段
                yield chunk
                
            # 生成完成
            logger.info(f"角色 {role_name} 的流式回复生成完成")
            
        except Exception as e:
            logger.exception(f"生成流式回复时发生错误: {e}")
            yield StreamResponse(
                content=f"生成响应时出错: {str(e)}", 
                finish_reason="error",
                status="error",
                error=str(e)
            )

    @staticmethod
    async def improve_stream_response(
        original_message: str, 
        original_response: str, 
        role: Dict[str, Any]
    ) -> AsyncGenerator[StreamResponse, None]:
        """
        根据用户反馈生成改进的流式回复
        
        Args:
            original_message: 原始用户消息
            original_response: 原始回复
            role: 选定的角色
            
        Yields:
            StreamResponse: 流式响应片段
        """
        # 创建LLM服务
        llm_service = LLMService()
        
        # 获取角色信息
        role_name = role.get("role_name", "Unknown")
        system_prompt = role.get("system_prompt", "")
        
        if not system_prompt:
            logger.warning(f"角色 {role_name} 没有system_prompt")
            system_prompt = f"你是{role_name}，请以自然的方式回复用户。"
        
        # 构建改进提示
        improvement_prompt = f"""你之前的回复被用户拒绝，需要改进。

原始用户消息: "{original_message}"

你之前的回复: "{original_response}"

请根据角色设定提供一个更好的回复。特别注意:
1. 确保回复与角色人设一致
2. 提供更有帮助、更详细的信息
3. 调整语气和表达方式以更好地符合角色特点
4. 避免重复之前回复中的问题

请直接给出改进后的回复，不要解释你做了什么改变。"""

        # 准备消息
        system_message = Message(role=MessageRole.SYSTEM, content=system_prompt)
        user_message1 = Message(role=MessageRole.USER, content=original_message)
        assistant_message = Message(role=MessageRole.ASSISTANT, content=original_response)
        user_message2 = Message(role=MessageRole.USER, content=improvement_prompt)
        
        try:
            # 获取角色的温度设置，改进时使用稍高的温度
            temperature = float(role.get("temperature", 0.7)) + 0.1
            temperature = min(temperature, 1.0)  # 确保不超过1.0
            
            # 记录开始生成改进的流式响应
            logger.info(f"开始为角色 {role_name} 生成改进的流式回复")
            
            # 调用流式API
            async for chunk in llm_service.generate_stream(
                messages=[system_message, user_message1, assistant_message, user_message2],
                config=LLMConfig(
                    provider=llm_service.default_config.provider,
                    model_name=llm_service.default_config.model_name,
                    api_key=llm_service.default_config.api_key,
                    temperature=temperature
                )
            ):
                # 传递每个响应片段
                yield chunk
                
            # 生成完成
            logger.info(f"角色 {role_name} 的改进流式回复生成完成")
            
        except Exception as e:
            logger.exception(f"生成改进流式回复时发生错误: {e}")
            yield StreamResponse(
                content=f"生成改进响应时出错: {str(e)}", 
                finish_reason="error",
                status="error",
                error=str(e)
            ) 