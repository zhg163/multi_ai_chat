"""
摘要服务 - 处理对话摘要生成和向量嵌入

使用DeepSeek模型进行摘要生成和向量嵌入
"""

import os
import logging
import aiohttp
import json
from typing import List, Dict, Any, Optional
import numpy as np
from app.config import memory_settings
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)

class SummaryService:
    """
    摘要服务类，负责生成对话摘要和生成向量表示
    """
    
    def __init__(self):
        """初始化摘要服务"""
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
        self.model = memory_settings.SUMMARY_MODEL
        
        if not self.api_key:
            logger.warning("未设置DeepSeek API密钥，摘要服务可能无法正常工作")
            
    async def generate_summary(self, messages: List[Dict]) -> str:
        """
        生成对话摘要
        
        Args:
            messages: 对话消息列表，每条消息包含role和content
            
        Returns:
            摘要文本
        """
        try:
            # 格式化消息作为提示词
            formatted_messages = []
            for msg in messages:
                formatted_messages.append(f"{msg['role']}: {msg['content']}")
                
            chat_history = "\n".join(formatted_messages)
            
            # 构建提示词
            prompt = f"""请对以下对话内容进行摘要总结，保留关键信息点，摘要应该简洁明了但包含对话的要点：

对话内容：
{chat_history}

摘要："""

            # 构建API请求
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,  # 较低温度以获得更加确定性的回答
                "max_tokens": 512     # 摘要长度限制
            }
            
            # 发送请求
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base}/chat/completions", 
                    headers=headers, 
                    json=data
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"生成摘要API请求失败: {error_text}")
                        # 返回简单摘要作为回退
                        return self._generate_fallback_summary(messages)
                        
                    response_data = await response.json()
                    summary = response_data["choices"][0]["message"]["content"]
                    
                    logger.info(f"已生成摘要，长度: {len(summary)}")
                    return summary
                    
        except Exception as e:
            logger.error(f"生成摘要时发生错误: {str(e)}")
            # 返回简单摘要作为回退
            return self._generate_fallback_summary(messages)
            
    def _generate_fallback_summary(self, messages: List[Dict]) -> str:
        """
        生成简单的回退摘要，当API请求失败时使用
        
        Args:
            messages: 对话消息列表
            
        Returns:
            简单摘要
        """
        try:
            # 获取第一条和最后一条用户消息
            first_user_msg = None
            last_user_msg = None
            
            for msg in messages:
                if msg["role"] == "user":
                    if first_user_msg is None:
                        first_user_msg = msg["content"]
                    last_user_msg = msg["content"]
                    
            # 获取消息数量
            user_msg_count = sum(1 for msg in messages if msg["role"] == "user")
            ai_msg_count = sum(1 for msg in messages if msg["role"] == "assistant")
            
            # 构建简单摘要
            summary = f"对话包含{user_msg_count}个用户消息和{ai_msg_count}个AI回复。"
            
            if first_user_msg:
                # 截取前50个字符
                first_preview = first_user_msg[:50] + "..." if len(first_user_msg) > 50 else first_user_msg
                summary += f" 开始于: \"{first_preview}\""
                
            if last_user_msg and last_user_msg != first_user_msg:
                # 截取前50个字符
                last_preview = last_user_msg[:50] + "..." if len(last_user_msg) > 50 else last_user_msg
                summary += f" 结束于: \"{last_preview}\""
                
            return summary
            
        except Exception as e:
            logger.error(f"生成回退摘要失败: {str(e)}")
            return f"对话包含{len(messages)}条消息"
            
    def generate_embedding(self, text: str) -> List[float]:
        """
        生成文本的向量表示
        
        Args:
            text: 文本内容
            
        Returns:
            向量表示
        """
        try:
            # 使用项目现有的嵌入服务
            vector = embedding_service.encode_text(text)
            
            # 转换为列表
            if isinstance(vector, np.ndarray):
                return vector.tolist()
            
            return vector
            
        except Exception as e:
            logger.error(f"生成文本向量失败: {str(e)}")
            return []
            
    def should_generate_summary(self, messages: List[Dict], token_count: int) -> bool:
        """
        判断是否应该生成摘要
        
        Args:
            messages: 对话消息列表
            token_count: token数量
            
        Returns:
            是否应该生成摘要
        """
        # 对话轮次超过指定数量
        if len(messages) >= 50:
            return True
            
        # Token消耗超过模型限制的70%
        if token_count > memory_settings.MAX_TOKENS:
            return True
            
        # TODO: 实现话题切换检测
        
        return False

# 创建全局实例
summary_service = SummaryService() 