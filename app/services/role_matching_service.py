"""
角色匹配服务 - 根据用户消息选择最合适的角色
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import jieba
import re
import traceback

from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)

class RoleMatchingService:
    def __init__(self):
        # 配置权重
        self.weights = {
            "semantic": 0.6,     # 语义相似度权重
            "keyword": 0.3,      # 关键词匹配权重
            "context": 0.1,      # 上下文关联权重
        }
        self.min_score = 0.2     # 最低匹配分数
        
        # 加载结巴分词词典
        try:
            jieba.initialize()
            logger.info("结巴分词初始化完成")
        except Exception as e:
            logger.error(f"结巴分词初始化失败: {str(e)}")

    async def find_matching_roles(self, 
                                 message: str, 
                                 session_id: Optional[str] = None,
                                 limit: int = 3) -> List[Dict[str, Any]]:
        """
        根据消息内容找到匹配的角色
        
        Args:
            message: 用户消息内容
            session_id: 会话ID，用于获取上下文
            limit: 返回的最大角色数量
            
        Returns:
            匹配角色列表，每项包含角色信息和匹配分数
        """
        try:
            # 获取数据库连接
            from app.db.mongodb import get_db
            db = await get_db()
            
            # 1. 获取所有活跃角色
            roles = await db.roles.find({"is_active": True}).to_list(None)
            if not roles:
                logger.warning("没有找到活跃角色")
                return []
                
            # 2. 对消息进行预处理
            processed_message = self._preprocess_text(message)
            
            # 3. 计算消息的向量表示
            message_embedding = embedding_service.encode_text(processed_message)
            if message_embedding is None:
                logger.error("无法计算消息的向量表示")
                return []
            
            # 4. 计算每个角色的匹配分数
            results = []
            
            for role in roles:
                # 获取角色向量
                role_embedding = await embedding_service.get_role_embedding(role)
                
                # 计算语义相似度分数
                semantic_score = self._compute_semantic_similarity(message_embedding, role_embedding)
                
                # 计算关键词匹配分数
                keyword_score = self._compute_keyword_match(processed_message, role)
                
                # 计算上下文关联分数
                context_score = 0
                if session_id:
                    context_score = await self._compute_context_relevance(message, role, session_id)
                
                # 综合计算最终分数
                final_score = (
                    self.weights["semantic"] * semantic_score +
                    self.weights["keyword"] * keyword_score +
                    self.weights["context"] * context_score
                )
                
                # 记录结果
                if final_score >= self.min_score:
                    results.append({
                        "role_id": str(role["_id"]),
                        "role_name": role["name"],
                        "score": round(float(final_score), 3),
                        "details": {
                            "semantic": round(float(semantic_score), 3),
                            "keyword": round(float(keyword_score), 3),
                            "context": round(float(context_score), 3)
                        }
                    })
            
            # 5. 按分数排序并限制返回数量
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"角色匹配失败: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 转小写
        text = text.lower()
        # 使用结巴分词
        words = jieba.cut(text)
        return " ".join(words)
    
    def _compute_semantic_similarity(self, message_embedding, role_embedding) -> float:
        """计算语义相似度 (余弦相似度)"""
        # 归一化向量
        message_norm = np.linalg.norm(message_embedding)
        role_norm = np.linalg.norm(role_embedding)
        
        # 避免除零错误
        if message_norm == 0 or role_norm == 0:
            return 0
            
        # 计算余弦相似度
        similarity = np.dot(message_embedding, role_embedding) / (message_norm * role_norm)
        
        # 转换到 [0,1] 范围
        return float((similarity + 1) / 2)
    
    def _compute_keyword_match(self, processed_message: str, role: Dict[str, Any]) -> float:
        """计算关键词匹配分数"""
        keywords = role.get("keywords", [])
        if not keywords:
            return 0
            
        # 将消息分词为集合
        message_words = set(processed_message.split())
        
        # 计算匹配的关键词数
        matched_keywords = 0
        for keyword in keywords:
            keyword = keyword.lower()
            # 检查完整短语匹配
            if keyword in processed_message:
                matched_keywords += 1
            # 检查单词匹配
            elif any(kw in message_words for kw in keyword.split()):
                matched_keywords += 0.5
                
        # 计算匹配比例
        if len(keywords) > 0:
            return min(1.0, matched_keywords / len(keywords))
        return 0
    
    async def _compute_context_relevance(self, message: str, role: Dict[str, Any], session_id: str) -> float:
        """计算上下文关联分数"""
        try:
            # 获取数据库连接
            from app.db.mongodb import get_db
            db = await get_db()
            
            # 获取最近的10条消息
            recent_messages = await db.messages.find(
                {"session_id": session_id}
            ).sort("created_at", -1).limit(10).to_list(None)
            
            # 如果没有历史消息，返回基础分数
            if not recent_messages:
                return 0.5  # 默认中间值
                
            # 检查角色在历史消息中的出现频率
            role_id = str(role["_id"])
            role_message_count = sum(1 for msg in recent_messages if msg.get("role_id") == role_id)
            
            # 计算上下文关联分数
            if len(recent_messages) > 0:
                return role_message_count / len(recent_messages)
            return 0
            
        except Exception as e:
            logger.error(f"计算上下文关联失败: {str(e)}")
            return 0

# 全局单例
role_matching_service = RoleMatchingService() 