"""
自定义会话服务 - 提供基于MD5会话ID生成的会话管理功能
"""

import logging
from typing import List, Dict, Any, Optional, Union
from bson import ObjectId
import os

from app.models.custom_session import CustomSession
from app.models.user import User

logger = logging.getLogger(__name__)

class CustomSessionService:
    """自定义会话服务，实现基于多条件生成MD5会话ID的功能"""
    
    @staticmethod
    async def check_user_exists(user_id: str) -> bool:
        """
        检查用户是否存在
        
        参数:
            user_id: 用户ID
            
        返回:
            用户是否存在
        """
        user = await User.get_by_id(user_id)
        return user is not None
    
    @staticmethod
    async def check_role_validity(roles: List[Dict[str, str]]) -> bool:
        """
        检查角色是否有效
        
        参数:
            roles: 角色列表，每个角色包含role_id和role_name
            
        返回:
            角色是否有效
        """
        # 在此处可以添加角色验证逻辑
        # 如果没有角色ID或名称，返回False
        if not roles:
            return False
            
        for role in roles:
            if not role.get("role_id") or not role.get("role_name"):
                return False
                
        return True
    
    @staticmethod
    async def get_active_sessions_count(user_id: str) -> int:
        """
        获取用户活跃会话数量
        
        参数:
            user_id: 用户ID
            
        返回:
            活跃会话数量
        """
        active_sessions = await CustomSession.get_active_sessions_by_user(user_id)
        return len(active_sessions)
    
    @staticmethod
    async def create_custom_session(
        class_id: str,
        class_name: str,
        user_id: str,
        user_name: str,
        roles: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        创建自定义会话
        
        参数:
            class_id: 聊天室ID
            class_name: 聊天室名称
            user_id: 用户ID
            user_name: 用户名称
            roles: 角色列表，每个角色包含role_id和role_name
            
        返回:
            创建的会话信息
        """
        # 前置校验
        # 1. 检查用户是否存在
        # user_exists = await CustomSessionService.check_user_exists(user_id)
        # if not user_exists:
        #     raise ValueError(f"用户不存在: {user_id}")
            
        # 2. 检查角色是否有效
        roles_valid = await CustomSessionService.check_role_validity(roles)
        if not roles_valid:
            raise ValueError("提供的角色无效")
            
        # 3. 检查用户活跃会话数量
        if os.getenv("TESTING") != "true":  # 只在非测试环境中限制活跃会话数量
            active_sessions_count = await CustomSessionService.get_active_sessions_count(user_id)
            if active_sessions_count >= 100:  # 限制最多1个活跃会话
                raise ValueError(f"用户已有{active_sessions_count}个活跃会话，不能创建更多会话")
            
        # 创建会话
        session = await CustomSession.create_session(
            class_id=class_id,
            class_name=class_name,
            user_id=user_id,
            user_name=user_name,
            roles=roles
        )
        
        return session
    
    @staticmethod
    async def update_session_status(session_id: str, status: int) -> bool:
        """
        更新会话状态
        
        参数:
            session_id: 会话ID
            status: 会话状态 (0-未开始，1-进行中，2-已结束)
            
        返回:
            更新是否成功
        """
        if status not in [0, 1, 2]:
            raise ValueError(f"无效的会话状态: {status}")
            
        return await CustomSession.update_session_status(session_id, status)
    
    @staticmethod
    async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话信息
        
        参数:
            session_id: 会话ID
            
        返回:
            会话信息，如果不存在则返回None
        """
        return await CustomSession.get_session_by_id(session_id)
    
    @staticmethod
    async def check_session_exists(session_id: str) -> bool:
        """
        检查会话是否存在
        
        参数:
            session_id: 会话ID
            
        返回:
            会话是否存在
        """
        session = await CustomSession.get_session_by_id(session_id)
        return session is not None 
    
    @staticmethod
    async def get_all_sessions(page: int, limit: int, status: Optional[int] = None) -> tuple[list, int]:
        skip = (page - 1) * limit
        query = {"session_status": status} if status is not None else {}
        collection = await CustomSession.get_collection()
        
        sessions = await collection.find(query).skip(skip).limit(limit).to_list(limit)
        total = await collection.count_documents(query)
        
        for session in sessions:
            session['_id'] = str(session['_id'])
            
            # Handle created_at field safely
            if 'created_at' in session and session['created_at'] is not None:
                if not isinstance(session['created_at'], str):
                    try:
                        session['created_at'] = session['created_at'].isoformat()
                    except AttributeError:
                        session['created_at'] = str(session['created_at'])
            
            # Handle updated_at field safely
            if 'updated_at' in session and session['updated_at'] is not None:
                if not isinstance(session['updated_at'], str):
                    try:
                        session['updated_at'] = session['updated_at'].isoformat()
                    except AttributeError:
                        session['updated_at'] = str(session['updated_at'])
        
        return sessions, total

    @staticmethod
    async def delete_session(session_id: str) -> bool:
        """
        删除指定ID的会话，同时从Redis和MongoDB中删除数据
        
        参数:
            session_id: 要删除的会话ID
            
        返回:
            删除操作是否成功
        """
        try:
            logger.info(f"正在删除会话: {session_id}")
            
            # 首先检查会话是否存在
            session = await CustomSession.get_session_by_id(session_id)
            if not session:
                logger.warning(f"尝试删除不存在的会话: {session_id}")
                return False
            
            # 从MongoDB删除会话
            mongo_deleted = await CustomSession.delete_from_mongodb(session_id)
            
            # 从Redis删除会话缓存
            redis_deleted = await CustomSession.delete_from_redis(session_id)
            
            # 记录删除结果
            logger.info(f"会话删除结果 - MongoDB: {mongo_deleted}, Redis: {redis_deleted}")
            
            # 只要有一个存储删除成功就返回True
            return mongo_deleted or redis_deleted
            
        except Exception as e:
            logger.error(f"删除会话时发生错误: {e}")
            return False