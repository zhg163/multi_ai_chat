"""
应用配置文件 - 包含记忆模块的配置项
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any

class MemorySettings(BaseSettings):
    # Redis配置
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6378"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "!qaz2wsX")
    
    # MongoDB配置
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "multi_ai_chat")
    
    # 对话配置
    MAX_CHAT_ROUNDS: int = int(os.getenv("MAX_CHAT_ROUNDS", "2"))  # 保留的最大对话轮次
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "11000"))  # 触发摘要的Token阈值(70%的16k)
    SUMMARY_MODEL: str = os.getenv("SUMMARY_MODEL", "deepseek-chat")  # 用于生成摘要的模型

    # 向量索引配置
    VECTOR_DIMENSIONS: int = 384  # 向量维度
    VECTOR_SIMILARITY: str = "cosine"  # 向量相似度计算方法

memory_settings = MemorySettings() 

class Config:
    """应用配置类"""
    
    # 基础配置
    DEBUG: bool = os.getenv("DEBUG", "True") == "True"
    
    # RAG服务配置
    RETRIEVAL_SERVICE_URL: str = os.getenv("RETRIEVAL_SERVICE_URL", "http://localhost:9222/api/chat")
    RETRIEVAL_API_KEY: str = os.getenv("RETRIEVAL_API_KEY", "")
    RAGFLOW_CHAT_ID: str = os.getenv("RAGFLOW_CHAT_ID", "ragflow-default")
    
    # 启用RAG功能的模型列表
    RAG_ENABLED_MODELS: list = ["gpt-3.5-turbo", "gpt-4", "deepseek-chat"]
    
    # 角色系统配置
    ENABLE_ROLE_BASED_CHAT: bool = True
    DEFAULT_SYSTEM_PROMPT: str = "你是一位知识渊博的助手。请基于用户问题回答，如果提供了参考资料，请参考这些资料。"
    
    # 需要维护与memory_settings一致的设置项，确保兼容
    @property
    def MONGO_URI(self):
        return memory_settings.MONGO_URI
    
    @property
    def MONGO_DB_NAME(self):
        return memory_settings.MONGO_DB_NAME
    
    @property
    def REDIS_HOST(self):
        return memory_settings.REDIS_HOST
    
    @property
    def REDIS_PORT(self):
        return memory_settings.REDIS_PORT
    
    @property
    def REDIS_PASSWORD(self):
        return memory_settings.REDIS_PASSWORD

# 创建配置实例
config = Config() 