"""
嵌入服务 - 用于将文本转换为语义向量表示
简化版实现 - 不依赖于sentence_transformers库
"""

import numpy as np
import logging
import os
import json
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name="simplified-embedding-model"):
        self.model = None
        self.model_name = model_name
        self.embedding_cache = {}  # 角色ID -> 向量的缓存
        self.cache_file = Path("app/data/embedding_cache.json")
        self.cache_dir = Path("app/data")
        self.initialized = False
        
    async def initialize(self):
        """初始化服务和缓存"""
        try:
            logger.info(f"初始化简化版嵌入服务")
            self.initialized = True
            
            # 尝试从文件加载缓存
            await self._load_cache()
            logger.info("简化版嵌入服务初始化完成")
        except Exception as e:
            logger.error(f"初始化向量服务失败: {str(e)}")
            
    async def _load_cache(self):
        """从文件加载缓存"""
        try:
            # 确保目录存在
            os.makedirs(self.cache_dir, exist_ok=True)
            
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                # 转换回NumPy数组
                for role_id, vector_list in cache_data.items():
                    self.embedding_cache[role_id] = np.array(vector_list, dtype=np.float32)
                    
                logger.info(f"已加载向量缓存，包含 {len(self.embedding_cache)} 个角色")
            else:
                logger.info("向量缓存文件不存在，将创建新缓存")
        except Exception as e:
            logger.error(f"加载向量缓存失败: {str(e)}")
            
    async def _save_cache(self):
        """将缓存保存到文件"""
        try:
            # 确保目录存在
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # 将NumPy数组转换为列表
            cache_data = {}
            for role_id, vector in self.embedding_cache.items():
                cache_data[role_id] = vector.tolist()
                
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f)
                
            logger.info(f"已保存向量缓存，包含 {len(self.embedding_cache)} 个角色")
        except Exception as e:
            logger.error(f"保存向量缓存失败: {str(e)}")
            
    async def precompute_all_role_embeddings(self):
        """预计算所有角色的向量表示"""
        try:
            # 获取数据库连接
            from app.database.mongodb import get_db
            db = await get_db()
            
            # 获取所有活跃角色
            roles = await db.roles.find({"is_active": True}).to_list(None)
            
            updated = 0
            for role in roles:
                role_id = str(role["_id"])
                # 检查是否需要更新
                if role_id not in self.embedding_cache:
                    await self.get_role_embedding(role, refresh=True)
                    updated += 1
                    
            if updated > 0:
                logger.info(f"已更新 {updated} 个角色的向量表示")
                # 保存更新后的缓存
                await self._save_cache()
                
        except Exception as e:
            logger.error(f"预计算角色向量失败: {str(e)}")
    
    async def get_role_embedding(self, role, refresh=False):
        """获取角色的向量表示"""
        role_id = str(role["_id"]) if "_id" in role else role["id"]
        
        # 如果缓存中已存在且不需要刷新，直接返回
        if role_id in self.embedding_cache and not refresh:
            return self.embedding_cache[role_id]
        
        # 构建角色的特征文本
        features = []
        if role.get("name"):
            features.append(f"名称: {role['name']}")
        if role.get("description"):
            features.append(f"描述: {role['description']}")
        if role.get("personality"):
            features.append(f"性格: {role['personality']}")
        if role.get("speech_style"):
            features.append(f"说话风格: {role['speech_style']}")
        if role.get("keywords") and isinstance(role["keywords"], list):
            features.append(f"关键词: {' '.join(role['keywords'])}")
            
        # 将特征拼接成文本
        feature_text = " ".join(features)
        
        # 生成简化的向量表示（使用哈希函数替代语义模型）
        embedding = self._simplified_encode(feature_text)
        
        # 存入缓存
        self.embedding_cache[role_id] = embedding
        
        return embedding
        
    def encode_text(self, text):
        """编码文本为向量表示（简化版）"""
        if not self.initialized:
            logger.error("嵌入服务尚未初始化")
            return np.zeros(384, dtype=np.float32)  # 返回零向量作为备用
            
        return self._simplified_encode(text)
    
    def _simplified_encode(self, text):
        """简化版文本编码函数，基于文本哈希生成伪向量
        
        这不是一个真正的语义嵌入，只是一个临时替代方案
        """
        # 使用文本的哈希值生成伪随机向量
        # 维度设为384，与多语言MiniLM模型一致
        dim = 384
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        # 使用哈希值的字节来生成伪随机向量
        # 重复哈希字节以达到所需维度
        seed = int.from_bytes(hash_bytes[:4], byteorder='little')
        np.random.seed(seed)
        
        # 生成随机向量并归一化
        vector = np.random.randn(dim)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm  # 单位化向量
        
        return vector.astype(np.float32)

# 全局单例
embedding_service = EmbeddingService() 