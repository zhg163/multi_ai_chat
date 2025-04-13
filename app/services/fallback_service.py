"""
降级服务模块 - 提供系统降级和备选实现

当主要服务不可用或出错时，提供备选功能和数据
"""

import random
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class FallbackService:
    """提供各种降级和备选功能的服务"""
    
    @staticmethod
    def generate_mock_response(user_message: str, role_prompt: str) -> str:
        """
        生成模拟AI回复（当API不可用时使用）
        
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
            f"我是{role_name}，很高兴为您服务！您说的\"{user_message[:30]}...\"是个很有趣的话题。我可以从多个角度为您解答。",
            f"作为{role_name}，我理解您询问的是关于\"{user_message[:30]}...\"。这是一个很好的问题，让我来回答。",
            f"感谢您的提问！\"{user_message[:30]}...\"确实值得探讨。作为{role_name}，我的看法是...",
            f"{role_name}收到您的问题了。关于\"{user_message[:30]}...\"，我想说的是...",
            f"您好！我是您的{role_name}。针对您提出的\"{user_message[:30]}...\"，我的回答是..."
        ]
        
        # 随机选择一个模拟回复
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
    
    @staticmethod
    def get_sample_users(as_dict=False):
        """返回示例用户数据，可以选择返回字典或对象"""
        sample_users = [
            {
                "id": "507f1f77bcf86cd799439001" if not as_dict else None,
                "name": "张三",
                "username": "zhangsan",
                "email": "zhang@example.com",
                "avatar": "https://example.com/avatars/user1.png",
                "description": "普通用户",
                "tags": ["电影", "篮球"],
                "is_active": True
            },
            {
                "id": "507f1f77bcf86cd799439002" if not as_dict else None,
                "name": "李四",
                "username": "lisi",
                "email": "li@example.com",
                "avatar": "https://example.com/avatars/user2.png",
                "description": "技术专家",
                "tags": ["编程", "AI"],
                "is_active": True
            },
            {
                "id": "507f1f77bcf86cd799439003" if not as_dict else None,
                "name": "王五",
                "username": "wangwu",
                "email": "wang@example.com",
                "avatar": "https://example.com/avatars/user3.png",
                "description": "文学爱好者",
                "tags": ["阅读", "写作"],
                "is_active": True
            }
        ]
        
        if as_dict:
            # 为数据库插入准备的格式，没有id字段
            return [
                {k: v for k, v in user.items() if k != 'id'}
                for user in sample_users
            ]
        
        return sample_users
    
    @staticmethod
    def get_default_roles():
        """获取默认角色数据"""
        from app.config.defaults import DEFAULT_ROLES
        return DEFAULT_ROLES

    @staticmethod
    def get_mock_models() -> List[str]:
        """返回模拟的模型列表"""
        return ["mock-model-1", "mock-model-2", "mock-model-3"] 