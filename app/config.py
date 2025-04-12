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