"""
Two-Phase LLM Service - 用于角色选择和回复生成的两阶段服务

第一阶段：根据用户输入选择最合适的角色
第二阶段：根据选定的角色生成回复
"""

import logging
import random
import asyncio
from typing import Dict, List, Tuple, Optional, Any

from app.services.llm_service import LLMService, Message, MessageRole, LLMConfig, LLMProvider

logger = logging.getLogger(__name__)

class TwoPhaseService:
    """Two-Phase LLM Service for role selection and response generation"""

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
        if not roles:
            logger.warning("没有提供角色列表，无法进行角色选择")
            return {}, 0.0, "没有可用角色"
        
        # 如果只有一个角色，直接返回
        if len(roles) == 1:
            logger.info(f"只有一个角色可用，直接选择: {roles[0].get('role_name', 'Unknown')}")
            return roles[0], 0.9, "只有一个可用角色"
        
        # 创建LLM服务
        llm_service = LLMService()
        
        # 构建提示
        role_descriptions = "\n".join([
            f"{i+1}. {role.get('role_name', 'Unknown')}: {role.get('system_prompt', 'No description')[:200]}..."
            for i, role in enumerate(roles)
        ])
        
        prompt = f"""作为一个角色匹配系统，你需要从以下角色中选择一个最适合回答用户消息的角色：

{role_descriptions}

用户消息: "{message}"

请分析用户消息与角色描述进行匹配并选择最合适的角色。返回JSON格式如下:
{{
  "selected_role_index": 角色索引(1到{len(roles)}),
  "confidence_score": 置信度(0.0到1.0),
  "reasoning": "选择该角色的原因"
}}
"""
        
        # 发送请求给LLM
        try:
            system_message = Message(role=MessageRole.SYSTEM, content="你是一个精确的角色选择系统，能够根据用户消息匹配最合适的角色。")
            user_message = Message(role=MessageRole.USER, content=prompt)
            
            response = await llm_service.generate_response(
                messages=[system_message, user_message],
                config=LLMConfig(
                    provider=llm_service.default_config.provider,
                    model_name=llm_service.default_config.model_name,
                    api_key=llm_service.default_config.api_key,
                    temperature=0.3  # 低温度，确保结果的确定性
                )
            )
            
            # 解析LLM返回的JSON
            import json
            import re
            
            # 使用正则表达式提取JSON部分
            content = response.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                try:
                    result = json.loads(json_str)
                    
                    selected_index = int(result.get("selected_role_index", 1)) - 1  # 转换为0索引
                    if selected_index < 0 or selected_index >= len(roles):
                        selected_index = 0  # 默认第一个
                        
                    confidence = float(result.get("confidence_score", 0.7))
                    reasoning = result.get("reasoning", "未提供理由")
                    
                    selected_role = roles[selected_index]
                    logger.info(f"LLM选择角色: {selected_role.get('role_name', 'Unknown')}, 置信度: {confidence}")
                    
                    return selected_role, confidence, reasoning
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"解析LLM响应失败: {e}, 响应内容: {content}")
            
            # 如果解析失败，随机选择
            selected_index = random.randint(0, len(roles) - 1)
            selected_role = roles[selected_index]
            logger.warning(f"LLM响应解析失败，随机选择角色: {selected_role.get('role_name', 'Unknown')}")
            return selected_role, 0.5, "LLM响应解析失败，随机选择"
            
        except Exception as e:
            logger.exception(f"角色选择过程中发生错误: {e}")
            
            # 发生错误时随机选择一个角色
            selected_index = random.randint(0, len(roles) - 1)
            selected_role = roles[selected_index]
            logger.warning(f"由于错误，随机选择角色: {selected_role.get('role_name', 'Unknown')}")
            return selected_role, 0.5, f"选择过程出错: {str(e)}"

    @staticmethod
    async def generate_response(message: str, role: Dict[str, Any]) -> str:
        """
        根据用户消息和选定角色生成回复

        Args:
            message: 用户消息
            role: 选定的角色

        Returns:
            str: 生成的回复
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
            
            # 发送请求给LLM
            response = await llm_service.generate_response(
                messages=[system_message, user_message],
                config=LLMConfig(
                    provider=llm_service.default_config.provider,
                    model_name=llm_service.default_config.model_name,
                    api_key=llm_service.default_config.api_key,
                    temperature=temperature
                )
            )
            
            logger.info(f"为角色 {role_name} 生成回复，长度: {len(response.content)}")
            return response.content
            
        except Exception as e:
            logger.exception(f"生成回复时发生错误: {e}")
            return f"生成响应时出错: {str(e)}"

    @staticmethod
    async def improve_response(original_message: str, original_response: str, role: Dict[str, Any]) -> str:
        """
        根据用户反馈生成改进的回复

        Args:
            original_message: 原始用户消息
            original_response: 原始回复
            role: 选定的角色

        Returns:
            str: 改进后的回复
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
            
            # 发送请求给LLM
            response = await llm_service.generate_response(
                messages=[system_message, user_message1, assistant_message, user_message2],
                config=LLMConfig(
                    provider=llm_service.default_config.provider,
                    model_name=llm_service.default_config.model_name,
                    api_key=llm_service.default_config.api_key,
                    temperature=temperature
                )
            )
            
            logger.info(f"为角色 {role_name} 生成改进回复，长度: {len(response.content)}")
            return response.content
            
        except Exception as e:
            logger.exception(f"生成改进回复时发生错误: {e}")
            return f"生成改进响应时出错: {str(e)}" 