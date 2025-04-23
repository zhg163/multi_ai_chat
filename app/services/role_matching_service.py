"""
角色匹配服务 - 根据用户消息选择最合适的角色
"""

import numpy as np
import random
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
            roles = []
            session_roles = []  # 存储从Redis获取的原始角色列表
            
            # 如果提供了会话ID，从Redis获取该会话的角色集合
            if session_id:
                logger.info(f"正在根据会话ID从Redis获取角色集合: {session_id}")
                # 获取Redis客户端
                from app.services.redis_service import redis_service
                import json
                
                redis_client = await redis_service.get_redis()
                if not redis_client:
                    logger.error("无法获取Redis客户端")
                    return []
                
                # 获取会话数据
                session_key = f"session:{session_id}"
                session_data = await redis_client.hgetall(session_key)
                
                if not session_data:
                    logger.warning(f"找不到会话数据: {session_id}")
                    return []
                
                # 解析角色数据
                roles_data = session_data.get("roles", "[]")
                
                try:
                    session_roles = json.loads(roles_data)  # 保存原始角色列表，用于可能的随机选择
                    if not isinstance(session_roles, list):
                        logger.warning(f"会话角色数据格式不正确: {type(session_roles)}")
                        return []
                    
                    roles = session_roles  # 设置待处理角色列表
                    logger.info(f"从Redis获取到 {len(roles)} 个角色")
                    # 添加日志，记录每个角色的信息
                    for i, role in enumerate(roles):
                        logger.info(f"角色 {i+1}:")
                        logger.info(f"  - role_id: {role.get('role_id', 'N/A')}")
                        logger.info(f"  - role_name: {role.get('role_name', 'N/A')}")
                        logger.info(f"  - keywords: {role.get('keywords', 'N/A')}")
                    
                except json.JSONDecodeError:
                    logger.error(f"角色数据解析失败: {roles_data[:100]}...")
                    return []
            else:
                # 如果没有提供会话ID，则从MongoDB获取所有活跃角色
                logger.info("未提供会话ID，从MongoDB获取所有活跃角色")
            from app.database.mongodb import get_db
            db = await get_db()
            
                # 获取所有活跃角色
            roles = await db.roles.find({"is_active": True}).to_list(None)
                
            if not roles:
                logger.warning("没有找到可用角色")
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
                # 根据角色数据来源调整字段名称
                role_id = str(role.get("_id", role.get("role_id", "")))
                role_name = role.get("name", role.get("role_name", "未知角色"))
                
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
                        "role_id": role_id,
                        "role_name": role_name,
                        "score": round(float(final_score), 3),
                        "match_reason": match_reason,
                        "details": {
                            "semantic": round(float(semantic_score), 3),
                            "keyword": round(float(keyword_score), 3),
                            "context": round(float(context_score), 3)
                        }
                    })
            
            # 5. 如果没有匹配到任何角色，且提供了会话ID，从会话角色中随机选择一个
            if not results and session_id and session_roles:
                logger.info("未找到匹配角色，将从会话角色中随机选择一个")
                random_role = random.choice(session_roles)
                role_id = str(random_role.get("_id", random_role.get("role_id", "")))
                role_name = random_role.get("name", random_role.get("role_name", "未知角色"))
                
                results.append({
                    "role_id": role_id,
                    "role_name": role_name,
                    "score": 0.1,  # 设置一个较低的分数表示是随机选择的
                    "match_reason": "随机选择",
                    "details": {
                        "semantic": 0.0,
                        "keyword": 0.0,
                        "context": 0.0,
                        "random": True
                    }
                })
                logger.info(f"随机选择了角色: {role_name} (ID: {role_id})")
            
            # 6. 按分数排序并限制返回数量
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
    
    def _compute_keyword_match(self, processed_message: str, role: Dict) -> float:
        """
        计算关键词匹配分数
        
        Args:
            processed_message: 预处理后的消息
            role: 角色信息字典
            
        Returns:
            关键词匹配分数 (0-1之间)
        """
        try:
            import json
            
            # 获取角色关键词 (支持多种可能的格式)
            keywords = role.get("keywords", [])
            
            # 关键词可能是JSON字符串，尝试解析
            if isinstance(keywords, str):
                try:
                    # 如果是JSON格式的字符串
                    if keywords.startswith('[') and keywords.endswith(']'):
                        keywords = json.loads(keywords)
                    # 如果是逗号分隔的字符串
                    else:
                        keywords = [k.strip() for k in keywords.split(',') if k.strip()]
                except json.JSONDecodeError:
                    # 如果解析失败，尝试作为逗号分隔的字符串处理
                    keywords = [k.strip() for k in keywords.split(',') if k.strip()]
                
            # 确保关键词是列表类型
            if not isinstance(keywords, list):
                logger.warning(f"角色关键词不是列表格式: {type(keywords)}")
                return 0.0
                    
            # 如果关键词为空，返回0分
            if not keywords:
                logger.debug("角色没有设置关键词，无法进行关键词匹配")
                return 0.0
            
            # 记录关键词
            role_name = role.get("name", role.get("role_name", "未知角色"))
            logger.info(f"角色 '{role_name}' 的关键词: {keywords}")
            
            # 计算匹配分数
            exact_matches = 0
            partial_matches = 0
            
            for keyword in keywords:
                if not keyword or not isinstance(keyword, str):
                    continue
                
                keyword = keyword.lower().strip()
                if not keyword:
                    continue
                    
                # 精确匹配 (1.0权重)
                if keyword in processed_message:
                    exact_matches += 1
                    logger.info(f"精确匹配关键词: '{keyword}'")
                    
                # 部分匹配 (0.6权重)
                elif any(keyword in token or token in keyword for token in self._tokenize(processed_message)):
                    partial_matches += 1
                    logger.info(f"部分匹配关键词: '{keyword}'")
                
            # 如果关键词列表为空，返回0分
            valid_keywords = len([k for k in keywords if isinstance(k, str) and k.strip()])
            if valid_keywords == 0:
                return 0.0
            
            # 计算总分
            exact_score = exact_matches / valid_keywords
            partial_score = (partial_matches / valid_keywords) * 0.6
            
            # 最终分数为精确匹配和部分匹配的和，最大为1.0
            final_score = min(1.0, exact_score + partial_score)
                    
            logger.info(f"关键词匹配分数: {final_score:.3f} (精确: {exact_matches}, 部分: {partial_matches}, 总关键词: {valid_keywords})")
            return final_score
                
        except Exception as e:
            logger.error(f"计算关键词匹配分数时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.0
    
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
            role_id = str(role.get("_id", role.get("role_id", "")))
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
                    
                    # 添加角色细节日志帮助调试
                    for i, role in enumerate(roles):
                        logger.info(f"会话角色 {i+1}:")
                        logger.info(f"  - role_id: {role.get('role_id', 'N/A')}")
                        logger.info(f"  - role_name: {role.get('role_name', 'N/A')}")
                        logger.info(f"  - keywords: {role.get('keywords', 'N/A')}")
                    
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
                # 修正字段名，确保使用正确的字段获取角色ID和名称
                role_id = role.get("role_id", "")
                role_name = role.get("role_name", "未知角色")
                
                # 计算关键词匹配分数 (1.0倍权重)
                keyword_score = self._compute_keyword_match(processed_message, role)
                
                # 使用关键词分数作为最终分数
                final_score = keyword_score
                
                # 记录匹配分数
                logger.info(f"角色 '{role_name}' 关键词匹配分数: {final_score:.3f}")
                
                # 仅当分数超过最小阈值时才考虑
                if final_score >= self.min_score:
                    match_reason = "基于关键词匹配"
                    
                    results.append({
                        "role_id": role_id,
                        "name": role_name,
                        "score": round(float(final_score), 3),
                        "match_reason": match_reason,
                        "details": {
                            "keyword": round(float(keyword_score), 3)
                        }
                    })
            
            # 按分数排序
            results.sort(key=lambda x: x["score"], reverse=True)
            
            # 如果找到了匹配的角色，返回分数最高的
            if results:
                logger.info(f"找到最佳匹配角色: {results[0]['name']}, 分数: {results[0]['score']}")
                return results[0]
            else:
                # 当没有匹配到角色时，随机选择一个
                if roles:
                    random_role = random.choice(roles)
                    role_id = random_role.get("role_id", "")
                    role_name = random_role.get("role_name", "未知角色")
                    
                    logger.info(f"未找到匹配角色，随机选择了: {role_name}")
                    return {
                        "role_id": role_id,
                        "name": role_name,
                        "score": 0.1,  # 设置一个较低的分数表示是随机选择的
                        "match_reason": "随机选择",
                        "details": {
                            "keyword": 0.0,
                            "random": True
                        }
                    }
                else:
                    logger.warning("未找到合适的匹配角色")
                    return None
            
        except Exception as e:
            logger.error(f"角色匹配失败: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def _tokenize(self, text: str) -> List[str]:
        """
        简单的文本分词函数，用于关键词匹配
        
        Args:
            text: 要分词的文本
            
        Returns:
            分词后的单词列表
        """
        if not text or not isinstance(text, str):
            return []
            
        # 预处理文本
        text = text.lower()
        
        # 简单的空格分词
        words = [word.strip() for word in text.split() if word.strip()]
        
        # 添加字符级别的n-gram (对中文更友好)
        char_ngrams = []
        if any(c for c in text if '\u4e00' <= c <= '\u9fff'):  # 检测是否包含中文字符
            # 为中文文本添加2-gram和3-gram
            text_no_space = ''.join(text.split())
            for i in range(len(text_no_space)):
                if i + 2 <= len(text_no_space):
                    char_ngrams.append(text_no_space[i:i+2])
                if i + 3 <= len(text_no_space):
                    char_ngrams.append(text_no_space[i:i+3])
        
        # 合并结果
        return words + char_ngrams

# 全局单例
role_matching_service = RoleMatchingService() 