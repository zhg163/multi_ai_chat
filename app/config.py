"""
应用配置文件 - 包含记忆模块的配置项
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any, List
from app.services.rag_interface_config import RagInterfaceConfig

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
    
    # RAG服务配置 (用于向后兼容)
    RETRIEVAL_SERVICE_URL: str = os.getenv("RETRIEVAL_SERVICE_URL", os.getenv("RAG_BASE_URL", "http://localhost:9222/api/chat"))
    RETRIEVAL_API_KEY: str = os.getenv("RETRIEVAL_API_KEY", os.getenv("RAG_API_KEY", ""))
    RAGFLOW_CHAT_ID: str = os.getenv("RAGFLOW_CHAT_ID", "ragflow-default")
    
    # 多接口RAG配置
    DEFAULT_RAG_INTERFACE: str = os.getenv("DEFAULT_RAG_INTERFACE", "primary")
    DEFAULT_DATASET_ID: str = os.getenv("DEFAULT_DATASET_ID", "default_dataset")
    
    # 初始化RAG接口配置
    def __init__(self):
        # 设置日志记录器
        import logging
        self.logger = logging.getLogger("config")
        
        # 记录当前环境变量状态
        self.logger.info(f"Config.__init__: DEFAULT_RAG_INTERFACE={self.DEFAULT_RAG_INTERFACE}, DEFAULT_DATASET_ID={self.DEFAULT_DATASET_ID}")
        
        # 设置默认数据集ID环境变量，确保RagInterfaceConfig能使用它
        if not os.getenv("DEFAULT_DATASET_ID"):
            os.environ["DEFAULT_DATASET_ID"] = self.DEFAULT_DATASET_ID
            self.logger.info(f"设置环境变量DEFAULT_DATASET_ID={self.DEFAULT_DATASET_ID}")
        else:
            self.logger.info(f"环境变量DEFAULT_DATASET_ID已存在: {os.getenv('DEFAULT_DATASET_ID')}")
        
        # 初始化RAG接口
        self.rag_interfaces = {}
        
        # 添加主要RAG接口
        self.logger.info("初始化主要RAG接口(primary)")
        primary_interface = RagInterfaceConfig.from_env(name="primary")
        self.rag_interfaces["primary"] = primary_interface
        self.logger.info(f"primary接口配置: dataset_ids={primary_interface.dataset_ids}")
        
        # 添加次要RAG接口
        self.logger.info("初始化次要RAG接口(secondary)")
        secondary_interface = RagInterfaceConfig.from_env(prefix="SECONDARY", name="secondary")
        self.rag_interfaces["secondary"] = secondary_interface
        self.logger.info(f"secondary接口配置: dataset_ids={secondary_interface.dataset_ids}")
    
    # 启用RAG功能的模型列表
    RAG_ENABLED_MODELS: list = ["gpt-3.5-turbo", "gpt-4", "deepseek-chat"]
    
    # 角色系统配置
    ENABLE_ROLE_BASED_CHAT: bool = True
    DEFAULT_SYSTEM_PROMPT: str = "你是一位知识渊博的助手。请基于用户问题回答，如果提供了参考资料，请参考这些资料。"
    
    # 获取RAG接口配置
    def get_rag_interface(self, interface_name: str = None) -> RagInterfaceConfig:
        """获取指定的RAG接口配置"""
        # 添加日志记录
        import logging
        logger = logging.getLogger("config")
        
        # 防止接口名为None时取到None值
        if interface_name is None:
            interface_name = self.DEFAULT_RAG_INTERFACE
            logger.info(f"接口名为None，使用默认接口: {interface_name}")
        
        name = interface_name
        logger.info(f"get_rag_interface调用: interface_name={interface_name}, 使用的name={name}")
        
        # 确保rag_interfaces已初始化
        if not hasattr(self, 'rag_interfaces') or not self.rag_interfaces:
            logger.error("rag_interfaces未初始化！尝试重新初始化")
            # 重新初始化接口
            self.rag_interfaces = {}
            primary_interface = RagInterfaceConfig.from_env(name="primary")
            self.rag_interfaces["primary"] = primary_interface
            secondary_interface = RagInterfaceConfig.from_env(prefix="SECONDARY", name="secondary")
            self.rag_interfaces["secondary"] = secondary_interface
            logger.info(f"已重新初始化接口: {list(self.rag_interfaces.keys())}")
        
        # 记录所有可用的接口
        logger.info(f"可用的RAG接口: {list(self.rag_interfaces.keys())}")
        
        # 获取接口，确保至少有一个fallback
        interface = self.rag_interfaces.get(name)
        if interface is None:
            logger.warning(f"找不到指定的接口 '{name}'，尝试使用'primary'")
            interface = self.rag_interfaces.get("primary")
            
        # 最后的保障：如果仍然为None，创建一个默认配置
        if interface is None:
            logger.error("无法获取任何有效的接口配置，创建默认配置")
            from app.services.rag_interface_config import RagInterfaceConfig
            import os
            # 获取默认数据集ID
            default_dataset_id = os.getenv("DEFAULT_DATASET_ID", "default_dataset")
            interface = RagInterfaceConfig(
                name="emergency_default",
                base_url=self.RETRIEVAL_SERVICE_URL.replace("/api/chat", ""),
                api_key=self.RETRIEVAL_API_KEY,
                dataset_ids=[default_dataset_id],
                document_ids=[]
            )
        
        logger.info(f"返回的接口配置: name={interface.name}, dataset_ids={interface.dataset_ids}")
        return interface
    
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