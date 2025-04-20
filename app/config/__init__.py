"""
配置模块 - 包含应用程序的各种配置和默认值
"""

# 从memory_settings.py导入所有配置
from app.config.memory_settings import (
    MEMORY_CONFIG,
    VECTOR_STORE_CONFIG,
    MEMORY_WORKFLOW,
    MEMORY_TYPES,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_PASSWORD,
    REDIS_DB,
    MAX_CHAT_ROUNDS,
    SUMMARY_MODEL,
    SUMMARY_MAX_TOKENS,
    SUMMARY_TEMPERATURE,
    SUMMARY_PROMPT_TEMPLATE,
    DEFAULT_SUMMARY,
    MAX_TOKENS
)

# 导入RAG相关配置
from app.config.defaults import (
    RETRIEVAL_SERVICE_URL,
    RETRIEVAL_API_KEY,
    RAGFLOW_CHAT_ID,
    RAG_ENABLED_MODELS
)

# 为了向后兼容，将重要的配置直接暴露在模块级别
memory_settings = {
    # 记忆配置项
    "MEMORY_CONFIG": MEMORY_CONFIG,
    "VECTOR_STORE_CONFIG": VECTOR_STORE_CONFIG,
    "MEMORY_WORKFLOW": MEMORY_WORKFLOW,
    "MEMORY_TYPES": MEMORY_TYPES,
    
    # Redis配置
    "REDIS_HOST": REDIS_HOST,
    "REDIS_PORT": REDIS_PORT,
    "REDIS_PASSWORD": REDIS_PASSWORD,
    "REDIS_DB": REDIS_DB,
    
    # 聊天相关
    "MAX_CHAT_ROUNDS": MAX_CHAT_ROUNDS,
    
    # 摘要相关
    "SUMMARY_MODEL": SUMMARY_MODEL,
    "SUMMARY_MAX_TOKENS": SUMMARY_MAX_TOKENS,
    "SUMMARY_TEMPERATURE": SUMMARY_TEMPERATURE,
    "SUMMARY_PROMPT_TEMPLATE": SUMMARY_PROMPT_TEMPLATE,
    "DEFAULT_SUMMARY": DEFAULT_SUMMARY,
    "MAX_TOKENS": MAX_TOKENS
}

# 创建Config类作为访问配置的统一接口
class Config:
    """应用配置类"""
    
    # 基础配置
    DEBUG = True
    
    # RAG服务配置
    RETRIEVAL_SERVICE_URL = RETRIEVAL_SERVICE_URL
    RETRIEVAL_API_KEY = RETRIEVAL_API_KEY
    RAGFLOW_CHAT_ID = RAGFLOW_CHAT_ID
    
    # 启用RAG功能的模型列表
    RAG_ENABLED_MODELS = RAG_ENABLED_MODELS
    
    # 需要维护与memory_settings一致的设置项，确保兼容
    @property
    def MONGO_URI(self):
        from app.config.memory_settings import MONGO_URI
        return MONGO_URI
    
    @property
    def MONGO_DB_NAME(self):
        from app.config.memory_settings import MONGO_DB_NAME
        return MONGO_DB_NAME
    
    @property
    def REDIS_HOST(self):
        return REDIS_HOST
    
    @property
    def REDIS_PORT(self):
        return REDIS_PORT
    
    @property
    def REDIS_PASSWORD(self):
        return REDIS_PASSWORD

# 创建配置实例
config = Config() 