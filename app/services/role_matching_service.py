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
        self.min_score = 0.15    # 最低匹配分数 - 降低阈值增加匹配几率
        
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
            from app.database.mongodb import get_db
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
                    # 确定匹配的主要原因
                    scores = {
                        "semantic": semantic_score * self.weights["semantic"],
                        "keyword": keyword_score * self.weights["keyword"],
                        "context": context_score * self.weights["context"]
                    }
                    max_score_type = max(scores, key=scores.get)
                    match_reason = "未知原因"
                    if max_score_type == "semantic":
                        match_reason = "基于语义相似度匹配"
                    elif max_score_type == "keyword":
                        match_reason = "基于关键词匹配"
                    elif max_score_type == "context":
                        match_reason = "基于历史对话上下文匹配"
                    
                    results.append({
                        "role_id": str(role["_id"]),
                        "role_name": role["name"],
                        "score": round(float(final_score), 3),
                        "match_reason": match_reason,
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
    
    def _compute_keyword_match(self, processed_text: str, role: Dict[str, Any]) -> float:
        """计算关键词匹配分数"""
        try:
            # 获取角色关键词
            keywords = role.get("keywords", [])
            if not keywords:
                # 如果角色没有定义关键词，尝试从描述中提取
                description = role.get("description", "")
                name = role.get("name", "")
                if description:
                    # 简单分词
                    words = self._preprocess_text(description).split()
                    # 取最多5个词作为关键词
                    keywords = [w for w in words if len(w) > 1][:5]
                
                # 如果仍然没有关键词，使用角色名称
                if not keywords and name:
                    keywords = [name]
                    
                # 如果仍然没有关键词，返回基础分数
                if not keywords:
                    logger.warning(f"角色没有关键词: {role.get('name', '未知角色')}")
                    return 0.2  # 基础分数
            
            logger.info(f"角色 '{role.get('name', '未知角色')}' 的关键词: {keywords}")
            
            # 统计关键词匹配数量
            words = processed_text.split()
            word_set = set(words)
            
            # 匹配结果
            matched_keywords = []
            exact_matches = []
            partial_matches = []
            
            for keyword in keywords:
                # 预处理关键词
                processed_keyword = self._preprocess_text(keyword)
                
                # 检查完全匹配 (关键词作为整体出现在文本中)
                if processed_keyword in processed_text:
                    exact_matches.append(keyword)
                    matched_keywords.append(keyword)
                    continue
                
                # 检查部分匹配 (关键词的单词在文本中出现)
                keyword_words = processed_keyword.split()
                if len(keyword_words) > 1:
                    matched_words = [w for w in keyword_words if w in word_set]
                    if matched_words and len(matched_words) / len(keyword_words) >= 0.5:
                        partial_matches.append(keyword)
                        matched_keywords.append(keyword)
                
                # 如果关键词很短 (小于3个字符)，仅当完全匹配时才计算
                if len(processed_keyword) < 3 and processed_keyword not in word_set:
                    continue
                    
                # 检查单词级别的匹配 (关键词作为单词出现在文本中)
                if processed_keyword in word_set:
                    matched_keywords.append(keyword)
            
            # 记录匹配情况
            if matched_keywords:
                logger.info(f"角色 '{role.get('name', '未知角色')}' 匹配到关键词:")
                if exact_matches:
                    logger.info(f"  - 完全匹配: {exact_matches}")
                if partial_matches:
                    logger.info(f"  - 部分匹配: {partial_matches}")
                
                other_matches = [k for k in matched_keywords if k not in exact_matches and k not in partial_matches]
                if other_matches:
                    logger.info(f"  - 单词匹配: {other_matches}")
                
            # 计算匹配分数
            # 完全匹配的关键词获得1.0的分数
            # 部分匹配的关键词获得0.6的分数
            # 单词匹配的关键词获得0.3的分数
            total_score = 0
            for keyword in matched_keywords:
                if keyword in exact_matches:
                    total_score += 1.0
                elif keyword in partial_matches:
                    total_score += 0.6
                else:
                    total_score += 0.3
                    
            # 根据匹配的关键词数量和总关键词数量计算最终分数
            if len(keywords) > 0:
                # 归一化分数，但给多个匹配项加权
                match_ratio = min(1.0, total_score / max(5, len(keywords)))
                
                # 如果匹配了多个关键词，给予额外奖励
                if len(matched_keywords) > 1:
                    match_ratio = min(1.0, match_ratio * (1 + 0.1 * (len(matched_keywords) - 1)))
                
                # 更好地平衡匹配数量和匹配比例
                # 这使得匹配了更多关键词的角色会获得更高的分数
                # 例如，匹配2/16比匹配1/7获得更高的分数
                absolute_bonus = min(0.5, len(matched_keywords) * 0.05)
                match_ratio += absolute_bonus
                
                # 确保分数不超过1.0
                match_ratio = min(1.0, match_ratio)
                    
                logger.info(f"角色 '{role.get('name', '未知角色')}' 关键词匹配分数: {match_ratio:.3f}")
                return match_ratio
            else:
                return 0
                
        except Exception as e:
            logger.error(f"计算关键词匹配分数失败: {str(e)}")
            logger.error(traceback.format_exc())
            return 0
    
    async def _compute_context_relevance(self, message: str, role: Dict[str, Any], session_id: str) -> float:
        """计算上下文关联分数"""
        try:
            # 获取数据库连接
            from app.database.mongodb import get_db
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

    async def find_matching_role(self, message: str, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        根据消息内容找到最匹配的单个角色(异步版本)
        
        Args:
            message: 用户消息内容
            session_id: 会话ID，用于限制角色范围
            
        Returns:
            最匹配的角色信息，如果没有匹配返回None
        """
        try:
            # 获取Redis客户端
            from app.services.redis_service import redis_service
            import json
            
            redis_client = await redis_service.get_redis()
            if not redis_client:
                logger.error("无法获取Redis客户端")
                return None
            
            roles = []
            
            # 如果提供了会话ID，则只从会话中的角色中匹配
            if session_id:
                logger.info(f"从会话 {session_id} 中的角色范围内进行匹配")
                
                # 获取会话数据
                session_key = f"session:{session_id}"
                session_data = await redis_client.hgetall(session_key)
                
                if not session_data:
                    logger.warning(f"找不到会话数据: {session_id}")
                    return None
                
                # 解析角色数据
                roles_data = session_data.get("roles", "[]")
                
                try:
                    roles = json.loads(roles_data)
                    if not isinstance(roles, list):
                        logger.warning(f"会话角色数据格式不正确: {type(roles)}")
                        return None
                    
                    # # 如果会话没有角色，返回None
                    # if not session_roles:
                    #     logger.warning(f"会话 {session_id} 没有角色")
                    #     return None
                    
                    # # 获取会话中每个角色的详细信息
                    # for role_info in session_roles:
                    #     role_id = role_info.get("role_id")
                    #     role_name = role_info.get("role_name")
                    #     system_prompt = role_info.get("system_prompt")
                        
                    #     if role_id:
                    #         role_key = f"role:{role_id}"
                    #         role_data = await redis_client.get(role_key)
                    #         if role_data:
                    #             role = json.loads(role_data)
                    #             if role.get("is_active", True):  # 默认为活跃
                    #                 roles.append(role)
                except json.JSONDecodeError:
                    logger.error(f"会话角色数据解析失败: {roles_data[:100]}...")
                    return None
                
                logger.info(f"会话中找到 {len(roles)} 个角色: {[r.get('role_name', '未知') for r in roles]}")

            
            if not roles:
                logger.warning("没有找到活跃角色可供匹配")
                return None
            
            # 对消息进行预处理
            processed_message = self._preprocess_text(message)
            
            # 由于嵌入服务可能未初始化，暂时只使用关键词匹配
            results = []
            
            for role in roles:
                # 计算关键词匹配分数 (1.0倍权重)
                keyword_score = self._compute_keyword_match(processed_message, role)
                
                # 使用关键词分数作为最终分数
                final_score = keyword_score
                
                # 记录匹配分数
                logger.info(f"角色 '{role.get('name', '未知角色')}' 关键词匹配分数: {final_score:.3f}")
                
                # 仅当分数超过最小阈值时才考虑
                if final_score >= self.min_score:
                    match_reason = "基于关键词匹配"
                    
                    results.append({
                        "role_id": role.get("id", ""),
                        "name": role.get("name", "未知角色"),
                        "score": round(float(final_score), 3),
                        "match_reason": match_reason,
                        "details": {
                            "keyword": round(float(keyword_score), 3)
                        }
                    })
            
            # 按分数排序
            results.sort(key=lambda x: x["score"], reverse=True)
            
            # 返回分数最高的角色
            if results:
                logger.info(f"找到最佳匹配角色: {results[0]['name']}, 分数: {results[0]['score']}")
                return results[0]
            else:
                logger.warning("未找到合适的匹配角色")
                return None
            
        except Exception as e:
            logger.error(f"角色匹配失败: {str(e)}")
            logger.error(traceback.format_exc())
            return None

# 全局单例
role_matching_service = RoleMatchingService() 