"""
内存管理相关配置

定义了记忆模块的各种设置，包括短期和长期记忆的配置参数
"""

import os

# Redis配置
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6378"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "!qaz2wsX")
REDIS_DB = 0

# 最大聊天轮数
MAX_CHAT_ROUNDS = 2

# 摘要生成配置
SUMMARY_MIN_MESSAGES = 5
SUMMARY_MAX_TOKENS = 1000
SUMMARY_MODEL = "gpt-3.5-turbo-16k"  # 用于生成摘要的模型
SUMMARY_TEMPERATURE = 0.3        # 摘要生成温度，较低以保持一致性
SUMMARY_PROMPT_TEMPLATE = "请总结以下对话内容，提取关键信息和主题：\n\n{context}"
DEFAULT_SUMMARY = "这是一个关于多种主题的对话。"
MAX_TOKENS = 4000                # 触发摘要生成的token阈值

# 内存管理器配置
MEMORY_CONFIG = {
    # 短期记忆配置
    "short_term": {
        "max_messages": 100,          # 每个会话最大消息数
        "max_message_length": 10000,  # 单条消息最大长度
        "max_sessions": 50,           # 最大会话数
        "default_ttl": 86400,         # 默认生存时间（秒，24小时）
    },
    
    # 长期记忆配置
    "long_term": {
        "enabled": True,              # 是否启用长期记忆
        "collection_name": "memories", # 向量数据库集合名称
        "embedding_dimensions": 1536,  # 嵌入向量维度
        "chunk_size": 500,            # 文本分块大小
        "chunk_overlap": 50,          # 分块重叠字符数
        "similarity_threshold": 0.75,  # 相似度阈值
        "max_results": 5,             # 查询返回最大结果数
    },
    
    # 记忆检索配置
    "retrieval": {
        "max_context_items": 10,      # 检索的最大上下文项数
        "recency_bias": 0.1,          # 时间衰减因子
        "relevance_threshold": 0.6,   # 相关性阈值
    },
    
    # 缓存配置
    "cache": {
        "enabled": True,              # 是否启用缓存
        "ttl": 3600,                  # 缓存生存时间（秒）
        "max_size": 1000,             # 最大缓存项数
    }
}

# 向量存储配置
VECTOR_COLLECTION_NAME = "chat_memory"
VECTOR_DIMENSION = 1536

# 记忆工作流配置
MEMORY_WORKFLOW = {
    "summarize_threshold": 20,        # 触发会话总结的消息数
    "importance_threshold": 0.7,      # 记忆重要性阈值
    "combine_similar": True,          # 是否合并相似记忆
    "forget_threshold": 0.3           # 遗忘阈值 - 低于此值的记忆可能被遗忘
}

# 记忆类型配置
MEMORY_TYPES = {
    "conversation": {"weight": 1.0, "ttl": 30 * 86400},  # 对话记忆（30天）
    "fact": {"weight": 1.2, "ttl": 90 * 86400},         # 事实记忆（90天）
    "preference": {"weight": 1.5, "ttl": 180 * 86400},   # 偏好记忆（180天）
    "important": {"weight": 2.0, "ttl": 365 * 86400}     # 重要记忆（365天）
}

# 向量存储配置（添加缺失的配置）
VECTOR_STORE_CONFIG = {
    "type": "simple",             # 向量存储类型：simple, chroma, faiss 等
    "path": "app/data/vector_store",  # 向量存储文件路径
    "collection_name": "message_embeddings",  # 集合名称
    "distance_metric": "cosine",      # 相似度计算方法
    "embedding_function": "openai",   # 嵌入函数类型
    "persist": True                   # 是否持久化存储
} 