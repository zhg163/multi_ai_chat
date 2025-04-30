"""
RAG接口配置类 - 为RAG服务提供配置结构
"""
import os
from typing import List
import logging

class RagInterfaceConfig:
    """RAG接口配置类"""
    def __init__(self, 
                 name: str, 
                 base_url: str, 
                 api_key: str, 
                 dataset_ids: List[str] = None, 
                 document_ids: List[str] = None):
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.dataset_ids = dataset_ids or []
        self.document_ids = document_ids or []
    
    @classmethod
    def from_env(cls, prefix: str = "", name: str = "primary"):
        """从环境变量创建配置"""
        # 添加日志记录
        logger = logging.getLogger("rag_interface")
        logger.info(f"RagInterfaceConfig.from_env 调用: prefix='{prefix}', name='{name}'")
        
        # 获取基础URL和API Key
        if prefix:
            base_url = os.getenv(f"{prefix}_RAG_BASE_URL", os.getenv("RAG_BASE_URL", "http://localhost:9222"))
            api_key = os.getenv(f"{prefix}_RAG_API_KEY", os.getenv("RAG_API_KEY", ""))
            dataset_ids_str = os.getenv(f"{prefix}_RAG_DATASET_IDS", "")
            document_ids_str = os.getenv(f"{prefix}_RAG_DOCUMENT_IDS", "")
            logger.info(f"使用前缀'{prefix}'查找环境变量: {prefix}_RAG_DATASET_IDS='{dataset_ids_str}'")
        else:
            base_url = os.getenv("RAG_BASE_URL", "http://localhost:9222")
            api_key = os.getenv("RAG_API_KEY", "")
            dataset_ids_str = os.getenv("RAG_DATASET_IDS", "")
            document_ids_str = os.getenv("RAG_DOCUMENT_IDS", "")
            logger.info(f"使用无前缀环境变量: RAG_DATASET_IDS='{dataset_ids_str}'")
        
        # 记录默认数据集ID环境变量
        default_dataset_id = os.getenv("DEFAULT_DATASET_ID", "default_dataset")
        logger.info(f"当前DEFAULT_DATASET_ID环境变量值: '{default_dataset_id}'")
        
        # 解析数据集ID和文档ID（如果有）
        dataset_ids = [id.strip() for id in dataset_ids_str.split(",")] if dataset_ids_str else []
        document_ids = [id.strip() for id in document_ids_str.split(",")] if document_ids_str else []
        
        logger.info(f"解析后的dataset_ids: {dataset_ids}")
        
        # 确保即使没有配置环境变量，也至少使用一个默认的数据集ID
        if not dataset_ids:
            # 使用默认数据集ID
            dataset_ids = [default_dataset_id]
            logger.warning(f"警告: 未配置数据集ID，使用默认ID: {default_dataset_id}")
            print(f"警告: 未配置数据集ID，使用默认ID: {default_dataset_id}")
        
        config = cls(name=name, base_url=base_url, api_key=api_key, 
                 dataset_ids=dataset_ids, document_ids=document_ids)
        logger.info(f"创建的RagInterfaceConfig: name={name}, dataset_ids={dataset_ids}")
        return config 