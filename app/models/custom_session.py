"""
自定义会话模型 - 基于MD5生成会话ID的实现
"""

import hashlib
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from bson import ObjectId
import logging

from app.database.mongodb import get_db
from app.database.fallback import MockCollection

logger = logging.getLogger(__name__)

class CustomSession:
    collection = None

    @classmethod
    async def get_collection(cls):
        """获取会话集合"""
        if cls.collection is None:
            db = await get_db()
            if db is None:
                cls.collection = MockCollection("sessions")
            else:
                cls.collection = db.sessions
        return cls.collection

    @classmethod
    async def generate_session_id(cls, class_name: str, user_name: str, roles: List[Dict[str, str]]) -> str:
        """
        根据规则生成会话ID: MD5(class_name + user_name + role_names + timestamp)
        
        参数:
            class_name: 聊天室名称
            user_name: 用户名称
            roles: 角色列表，每个角色是包含role_id和role_name的字典
            
        返回:
            生成的会话ID
        """
        # 从角色列表中提取角色名称并排序
        role_names = sorted([role.get("role_name", "") for role in roles])
        role_names_str = "".join(role_names)
        
        # 生成MD5
        timestamp = str(int(time.time()))
        input_string = f"{class_name}{user_name}{role_names_str}{timestamp}"
        
        # 生成MD5哈希
        session_id = hashlib.md5(input_string.encode()).hexdigest()
        
        # 验证唯一性
        collection = await cls.get_collection()
        existing = await collection.find_one({"session_id": session_id})
        
        # 如果已存在，添加随机数重试
        if existing:
            logger.warning(f"会话ID冲突: {session_id}，正在重新生成")
            import random
            random_suffix = str(random.randint(1000, 9999))
            session_id = hashlib.md5((input_string + random_suffix).encode()).hexdigest()
        
        return session_id

    @classmethod
    async def create_session(cls, 
                             class_id: str, 
                             class_name: str, 
                             user_id: str, 
                             user_name: str, 
                             roles: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        创建新会话
        
        参数:
            class_id: 聊天室ID
            class_name: 聊天室名称
            user_id: 用户ID
            user_name: 用户名称
            roles: 角色列表，每个角色是包含role_id和role_name的字典
            
        返回:
            创建的会话信息
        """
        # 生成会话ID
        session_id = await cls.generate_session_id(class_name, user_name, roles)
        
        # 获取当前时间
        now = datetime.utcnow()
        
        # 准备会话数据
        session_data = {
            "session_id": session_id,
            "class_id": class_id,
            "class_name": class_name,
            "user_id": user_id,
            "user_name": user_name,
            "roles": roles,
            "session_obj": {},  # 空会话内容
            "session_status": 0,  # 0-未开始
            "created_at": now,
            "updated_at": now
        }
        
        # 插入数据库
        collection = await cls.get_collection()
        result = await collection.insert_one(session_data)
        
        # 验证插入
        if not result.inserted_id:
            logger.error(f"创建会话失败: {session_data}")
            raise Exception("创建会话失败")
        
        # 同步到Redis
        try:
            from app.memory.buffer_memory import ShortTermMemory
            from app.memory.memory_manager import get_memory_manager
            
            # 获取内存管理器
            memory_manager = await get_memory_manager()
            
            # 同步到Redis
            if memory_manager and memory_manager.short_term:
                # 存储更完整的会话信息到Redis，与MongoDB保持一致
                redis_key = f"session:{user_id}:{session_id}"
                redis_data = {
                    "session_id": session_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "user_id": user_id,
                    "user_name": user_name,
                    "session_status": str(session_data["session_status"]),  # 转为字符串但保持数值一致性
                    "created_at": now.timestamp(),
                    "updated_at": now.timestamp()
                }
                
                # 直接使用Redis命令保存，而不是使用memory模块的方法
                redis_client = memory_manager.short_term.redis
                redis_client.hmset(redis_key, redis_data)
                
                # 设置过期时间，例如7天
                redis_client.expire(redis_key, 7 * 24 * 60 * 60)
                
                logger.info(f"会话已同步到Redis: {session_id}, 数据与MongoDB保持一致")
            else:
                logger.warning(f"无法获取内存管理器，会话未同步到Redis: {session_id}")
        except Exception as e:
            logger.error(f"同步会话到Redis失败: {str(e)}")
            # 不中断流程，继续返回会话信息
        
        # 返回会话信息
        return {
            "session_id": session_id,
            "class_name": class_name,
            "user_name": user_name
        }

    @classmethod
    async def update_session_status(cls, session_id: str, status: int) -> bool:
        """
        更新会话状态
        
        参数:
            session_id: 会话ID
            status: 会话状态 (0-未开始，1-进行中，2-已结束)
            
        返回:
            更新是否成功
        """
        collection = await cls.get_collection()
        result = await collection.update_one(
            {"session_id": session_id},
            {"$set": {
                "session_status": status,
                "updated_at": datetime.utcnow()
            }}
        )
        
        # 同步到Redis
        if result.modified_count > 0:
            try:
                from app.memory.memory_manager import get_memory_manager
                
                # 获取内存管理器
                memory_manager = await get_memory_manager()
                
                # 获取会话信息
                session = await collection.find_one({"session_id": session_id})
                if not session:
                    logger.warning(f"要更新的会话不存在: {session_id}")
                    return result.modified_count > 0
                
                # 同步到Redis
                if memory_manager and memory_manager.short_term:
                    # 直接使用Redis命令更新，保持与MongoDB一致的数据结构
                    redis_key = f"session:{session['user_id']}:{session_id}"
                    redis_client = memory_manager.short_term.redis
                    
                    # 更新Redis中的状态
                    redis_data = {
                        "session_status": str(status),  # 转为字符串但保持数值
                        "updated_at": datetime.utcnow().timestamp()
                    }
                    
                    redis_client.hmset(redis_key, redis_data)
                    
                    # 如果是结束状态，设置较短的过期时间
                    if status == 2:  # 已结束
                        redis_client.expire(redis_key, 1 * 24 * 60 * 60)  # 1天后过期
                        logger.info(f"会话状态已同步到Redis(已结束): {session_id}")
                    else:
                        logger.info(f"会话状态已同步到Redis: {session_id}, 状态: {status}")
            except Exception as e:
                logger.error(f"同步会话状态到Redis失败: {str(e)}")
                # 不中断流程，继续返回更新状态
        
        return result.modified_count > 0

    @classmethod
    async def get_session_by_id(cls, session_id: str) -> Optional[Dict[str, Any]]:
        """
        根据会话ID获取会话信息
        
        参数:
            session_id: 会话ID
            
        返回:
            会话信息，如果不存在则返回None
        """
        collection = await cls.get_collection()
        return await collection.find_one({"session_id": session_id})

    @classmethod
    async def get_active_sessions_by_user(cls, user_id: str) -> List[Dict[str, Any]]:
        """
        获取用户的活跃会话列表
        
        参数:
            user_id: 用户ID
            
        返回:
            活跃会话列表
        """
        collection = await cls.get_collection()
        cursor = collection.find({
            "user_id": user_id,
            "session_status": {"$in": [0, 1]}  # 未开始或进行中
        })
        return await cursor.to_list(length=100)  # 限制最多返回100个会话 