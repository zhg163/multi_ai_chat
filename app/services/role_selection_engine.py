"""
角色选择引擎服务

负责根据用户消息内容智能选择最适合的AI角色进行回复
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import json
import random
from datetime import datetime

from app.services.role_service import RoleService
from app.services.session_service import SessionService
from app.models.role import Role

logger = logging.getLogger(__name__)

class RoleSelectionEngine:
    """
    角色选择引擎
    
    根据用户消息内容和上下文，智能选择最适合的AI角色进行回复
    """
    
    def __init__(self, session_service=None, role_service=None):
        """
        初始化角色选择引擎
        
        Args:
            session_service: 会话服务，用于获取会话相关信息
            role_service: 角色服务，用于获取角色信息
        """
        self.session_service = session_service or SessionService()
        self.role_service = role_service or RoleService()
        
    async def select_role_for_message(
        self, 
        session_id: str, 
        message: str,
        user_id: str,
        context_messages: Optional[List[Dict[str, Any]]] = None,
        preferred_role_id: Optional[str] = None,
        selection_mode: str = "auto"
    ) -> Tuple[Optional[Dict[str, Any]], float, str]:
        """
        为用户消息选择最合适的角色
        
        Args:
            session_id: 会话ID
            message: 用户消息内容
            user_id: 用户ID
            context_messages: 上下文消息列表，如果为None则自动获取
            preferred_role_id: 用户指定的优先角色ID
            selection_mode: 角色选择模式，可选值:
                - "auto": 自动选择最合适的角色
                - "round_robin": 轮流选择会话中的所有角色
                - "continue": 继续使用上一条消息的角色
                - "highest_score": 仅使用关键词匹配得分最高的角色
                - "random": 从会话中随机选择角色
                
        Returns:
            Tuple(选择的角色, 匹配分数, 选择原因)
        """
        # 记录开始时间，用于性能监控
        start_time = datetime.utcnow()
        
        # 获取会话中可用的角色
        session_roles = await self.session_service.get_session_roles(session_id, user_id)
        if not session_roles:
            logger.warning(f"Session {session_id} has no active roles")
            return None, 0.0, "No active roles in session"
        
        # 如果指定了优先角色且角色在会话中，直接使用该角色
        if preferred_role_id:
            for role in session_roles:
                if str(role["_id"]) == preferred_role_id:
                    logger.info(f"Using preferred role {preferred_role_id}")
                    return role, 1.0, "User preferred role"
        
        # 根据选择模式进行角色选择
        if selection_mode == "random":
            # 随机选择角色
            selected_role = random.choice(session_roles)
            return selected_role, 0.5, "Random selection"
            
        elif selection_mode == "round_robin":
            # 获取上下文消息，如果未提供则从数据库获取
            if context_messages is None:
                context = await self.session_service.get_recent_messages(session_id, limit=10)
                context_messages = context.get("items", [])
            
            # 找出上一条助手消息使用的角色
            last_role_id = None
            for msg in reversed(context_messages):
                if msg.get("message_type") == "assistant" and msg.get("role_id"):
                    last_role_id = msg.get("role_id")
                    break
            
            # 确定下一个角色
            if last_role_id:
                # 找出当前角色的索引
                current_index = -1
                for i, role in enumerate(session_roles):
                    if str(role["_id"]) == last_role_id:
                        current_index = i
                        break
                
                # 选择下一个角色（循环）
                next_index = (current_index + 1) % len(session_roles)
                return session_roles[next_index], 0.5, "Round-robin selection"
            else:
                # 如果没有找到上一个角色，使用第一个角色
                return session_roles[0], 0.5, "First-role in round-robin"
                
        elif selection_mode == "continue":
            # 获取上下文消息，如果未提供则从数据库获取
            if context_messages is None:
                context = await self.session_service.get_recent_messages(session_id, limit=10)
                context_messages = context.get("items", [])
            
            # 找出上一条助手消息使用的角色
            last_role_id = None
            for msg in reversed(context_messages):
                if msg.get("message_type") == "assistant" and msg.get("role_id"):
                    last_role_id = msg.get("role_id")
                    break
            
            # 如果找到了上一个角色并且在会话角色列表中，继续使用
            if last_role_id:
                for role in session_roles:
                    if str(role["_id"]) == last_role_id:
                        return role, 0.7, "Continuing previous role"
            
            # 如果没有找到上一个角色，使用关键词匹配
            logger.info(f"No previous role found, falling back to keyword matching")
        
        # 使用关键词匹配（用于auto和highest_score模式，或其他模式的回退）
        # 首先创建session_role_id到role的映射
        role_map = {str(role["_id"]): role for role in session_roles}
        
        # 获取消息关键词匹配的角色，限制为会话中的角色
        matched_roles = await self.role_service.match_roles_by_keywords(
            message=message,
            limit=len(session_roles),
            min_score=0.1
        )
        
        # 过滤出会话中的角色
        session_matched_roles = []
        for role in matched_roles:
            role_id = str(role["_id"])
            if role_id in role_map:
                session_matched_roles.append((role, role.get("match_score", 0)))
        
        # 如果找到匹配的角色
        if session_matched_roles:
            # 对于highest_score模式，直接返回得分最高的角色
            if selection_mode == "highest_score":
                best_role, score = session_matched_roles[0]
                return best_role, score, f"Highest keyword match score: {score:.2f}"
            
            # 对于auto模式，考虑更复杂的选择逻辑
            # 这里可以添加更多启发式规则，如考虑历史交互、用户偏好等
            
            # 基本实现：在得分较高的角色中引入一些随机性
            # 只有当得分差距不大时才考虑随机选择，避免选择明显不匹配的角色
            top_roles = []
            highest_score = session_matched_roles[0][1]
            
            # 将得分接近最高分的角色加入候选池
            score_threshold = max(0.2, highest_score * 0.8)  # 至少0.2的得分或不低于最高分的80%
            
            for role, score in session_matched_roles:
                if score >= score_threshold:
                    top_roles.append((role, score))
                else:
                    break  # 已按分数排序，不需要继续检查
            
            # 从候选池中随机选择，但得分越高的角色被选中概率越大
            total_weight = sum(score for _, score in top_roles)
            r = random.random() * total_weight
            
            cumulative_weight = 0
            for role, score in top_roles:
                cumulative_weight += score
                if r <= cumulative_weight:
                    return role, score, f"Weighted random selection, score: {score:.2f}"
            
            # 保险起见，如果上面的逻辑没有选择角色，返回得分最高的
            best_role, score = top_roles[0]
            return best_role, score, f"Fallback to highest score: {score:.2f}"
        
        # 如果没有找到匹配的角色，随机选择一个
        selected_role = random.choice(session_roles)
        return selected_role, 0.1, "No keyword match, random fallback"
    
    async def get_role_prompt(self, role_id: str, message: str, conversation_history: List[Dict[str, Any]]) -> str:
        """
        根据选定的角色生成合适的提示
        
        Args:
            role_id: 角色ID
            message: 当前用户消息
            conversation_history: 对话历史
            
        Returns:
            生成的提示文本
        """
        # 获取角色信息
        role = await self.role_service.get_role_by_id(role_id)
        if not role:
            logger.error(f"Role {role_id} not found")
            return "You are a helpful assistant. Please respond to the user's message."
        
        # 获取角色的默认模板
        template = await self.role_service.get_default_template(role_id)
        template_content = ""
        
        if template:
            template_content = template.get("content", "")
        
        # 如果没有模板，使用角色的基本信息创建默认提示
        if not template_content:
            system_prompt = role.get("system_prompt", "")
            personality = role.get("personality", "")
            speech_style = role.get("speech_style", "")
            
            template_content = f"""你是{role.get('name', '助手')}。

{system_prompt}

{'' if not personality else f'你的性格特点：{personality}'}

{'' if not speech_style else f'你的说话风格：{speech_style}'}

请根据上面的角色设定，回复用户的消息。回复时保持角色的一致性。"""
        
        # 处理提示模板中的变量替换
        processed_prompt = template_content.replace(
            "{role_name}", role.get("name", "助手")
        ).replace(
            "{role_description}", role.get("description", "")
        ).replace(
            "{personality}", role.get("personality", "")
        ).replace(
            "{speech_style}", role.get("speech_style", "")
        )
        
        return processed_prompt
    
    async def log_selection(
        self, 
        session_id: str, 
        message_id: str, 
        selected_role_id: str, 
        score: float, 
        selection_reason: str
    ) -> None:
        """
        记录角色选择结果，用于分析和改进
        
        Args:
            session_id: 会话ID
            message_id: 消息ID
            selected_role_id: 选中的角色ID
            score: 匹配分数
            selection_reason: 选择理由
        """
        # 记录选择结果到日志
        logger.info(
            f"Role selection: session={session_id}, message={message_id}, "
            f"role={selected_role_id}, score={score:.2f}, reason={selection_reason}"
        )
        
        # 可以在此处添加将选择结果写入数据库的逻辑
        # 例如记录到专门的角色选择日志集合中
        # await self.db.role_selection_logs.insert_one({
        #     "session_id": session_id,
        #     "message_id": message_id,
        #     "selected_role_id": selected_role_id,
        #     "score": score,
        #     "selection_reason": selection_reason,
        #     "timestamp": datetime.utcnow()
        # }) 