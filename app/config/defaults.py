"""
默认配置值模块 - 定义了应用程序的各种默认值和配置
"""

# 默认角色配置
DEFAULT_ROLES = {
    "role_id_1": {
        "name": "AI助手",
        "description": "智能助手角色，可以回答各种问题并提供帮助",
        "system_prompt": "你是一个名为'AI助手'的智能助理，致力于提供友好、准确的信息和帮助。",
        "personality": "友好、专业、乐于助人",
        "speech_style": "正式但亲切",
        "temperature": 0.7
    },
    "role_id_2": {
        "name": "技术专家",
        "description": "专业的技术顾问，精通编程和IT领域的知识",
        "system_prompt": "你是一个名为'技术专家'的技术顾问，熟悉各种编程语言和技术领域，能够提供专业的技术建议和解决方案。",
        "personality": "专业、理性、精确",
        "speech_style": "技术性、专业、含有适量术语",
        "temperature": 0.5
    },
    "role_id_3": {
        "name": "创意顾问",
        "description": "富有创意的思想家，擅长提供独特的想法和创新的视角",
        "system_prompt": "你是一个名为'创意顾问'的创意型人工智能，善于从独特的角度思考问题，提供有创意和灵感的回答。",
        "personality": "富有想象力、创造性、开放",
        "speech_style": "灵活、生动、比喻丰富",
        "temperature": 0.8
    },
    "default": {
        "name": "通用助手",
        "description": "默认角色，当找不到指定角色时使用",
        "system_prompt": "你是一个通用人工智能助手，旨在友好地回答问题并提供帮助。",
        "personality": "平和、有礼、全面",
        "speech_style": "中性、适应性强",
        "temperature": 0.7
    }
}

# 默认LLM配置
DEFAULT_LLM_CONFIG = {
    "models": {
        "default": "default-model",
        "alternatives": ["model-1", "model-2"]
    },
    "providers": {
        "default": "default-provider",
        "alternatives": ["provider-1", "provider-2"]
    },
    "params": {
        "temperature": 0.7,
        "max_tokens": 2000,
        "top_p": 1.0
    }
}

# 默认消息处理配置
DEFAULT_MESSAGE_CONFIG = {
    "max_history_length": 50,
    "max_message_length": 10000,
    "stream_chunk_size": 10
}

# 默认路由相关配置
DEFAULT_ROUTE_CONFIG = {
    "prefix": "/api",
    "rate_limits": {
        "standard": "60/minute",
        "premium": "120/minute"
    }
}

# 默认会话配置
DEFAULT_SESSION_CONFIG = {
    "timeout_minutes": 60,
    "max_active_sessions": 10,
    "archive_after_days": 30
}

# 默认系统提示词
DEFAULT_SYSTEM_PROMPTS = {
    "assistant": "你是一个有用的AI助手。你应该礼貌、简洁且有帮助地回答问题。",
    "expert": "你是一位知识渊博的专家。你应该提供详细、准确和专业的回答。",
    "poet": "你是一位富有创造力的诗人。你应该用优美、富有诗意的语言回答问题。"
}

# 降级配置
FALLBACK_CONFIG = {
    # 是否启用模拟响应
    "enable_mock_responses": True,
    
    # 是否启用模拟数据库
    "enable_mock_database": True,
    
    # 最大重试次数
    "max_retries": 3,
    
    # 重试延迟（秒）
    "retry_delay": 1.0
}

# 默认用户配置
DEFAULT_USER = {
    "name": "访客用户",
    "username": "guest",
    "email": "guest@example.com",
    "avatar": "https://example.com/avatars/default.png",
    "description": "临时访客用户",
    "tags": ["访客"],
    "is_active": True
}

# 添加RAG服务配置
import os

# RAG服务配置
RETRIEVAL_SERVICE_URL = os.getenv("RETRIEVAL_SERVICE_URL", "http://localhost:9222/api/chat")
RETRIEVAL_API_KEY = os.getenv("RETRIEVAL_API_KEY", "")
RAGFLOW_CHAT_ID = os.getenv("RAGFLOW_CHAT_ID", "ragflow-default")

# 启用RAG功能的模型列表
RAG_ENABLED_MODELS = ["deepseek-chat"] 