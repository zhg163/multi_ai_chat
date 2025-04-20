"""
自定义会话模型 - 基于MD5生成会话ID的实现
"""

import hashlib
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from bson import ObjectId
import logging
import json

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
        
        # 准备会话数据 - 统一的会话数据模型
        session_data = {
            "session_id": session_id,
            "class_id": class_id,
            "class_name": class_name,
            "user_id": user_id,
            "user_name": user_name,
            "roles": roles,
            "session_status": 0,  # 0-未开始，1-进行中，2-已结束
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
            from app.memory.memory_manager import get_memory_manager
            
            # 获取内存管理器
            memory_manager = await get_memory_manager()
            
            # 同步到Redis
            if memory_manager and memory_manager.short_term and memory_manager.short_term.redis:
                redis_client = memory_manager.short_term.redis
                
                # 使用统一格式的会话数据 - 两种存储用同样的结构
                # 1. 会话信息键
                redis_session_key = f"session:{user_id}:{session_id}"
                
                # 2. 准备会话数据 - 与MongoDB存储完全相同的格式
                redis_data = {
                    "id": session_id,
                    "session_id": session_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "user_id": user_id,
                    "user_name": user_name,
                    "status": "0",  # Redis中以字符串存储状态
                    "session_status": "0",  # 保持与MongoDB一致的字段名
                    "start_time": now.timestamp(),
                    "created_at": now.timestamp(),
                    "updated_at": now.timestamp()
                }
                
                # 将角色数组序列化为JSON字符串 - 与MongoDB使用相同的结构
                redis_data["roles"] = json.dumps(roles)
                
                # 向Redis写入数据
                redis_client.hmset(redis_session_key, redis_data)
                
                # 设置过期时间，例如7天
                redis_client.expire(redis_session_key, 7 * 24 * 60 * 60)
                
                logger.info(f"会话已同步到Redis: {session_id}, 键: {redis_session_key}, 使用与MongoDB一致的角色存储结构")
                
                # 创建会话消息列表键
                message_key = f"messages:{user_id}:{session_id}"
                
                # 预创建空消息列表
                if not redis_client.exists(message_key):
                    # 添加一个系统消息作为初始消息
                    system_message = {
                        "role": "system",
                        "content": f"会话开始：{class_name}",
                        "timestamp": now.isoformat(),
                        "metadata": {
                            "session_id": session_id,
                            "user_id": user_id,
                            "roles": [role.get("role_name", "") for role in roles]
                        }
                    }
                    redis_client.rpush(message_key, json.dumps(system_message))
                    logger.info(f"为会话 {session_id} 创建初始消息列表")
            else:
                logger.warning(f"无法获取内存管理器或Redis客户端，会话未同步到Redis: {session_id}")
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
        
        # 获取当前会话信息，用于获取用户ID
        session = await collection.find_one({"session_id": session_id})
        if not session:
            logger.warning(f"要更新的会话不存在: {session_id}")
            return False
            
        # 提取用户ID，用于Redis键构建
        user_id = session.get("user_id")
        if not user_id:
            logger.warning(f"会话缺少用户ID: {session_id}")
            return False
        
        # 更新MongoDB中的状态
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
                
                # 同步到Redis
                if memory_manager and memory_manager.short_term and memory_manager.short_term.redis:
                    # 1. 构建Redis键
                    redis_session_key = f"session:{user_id}:{session_id}"
                    redis_client = memory_manager.short_term.redis
                    
                    # 检查会话是否存在于Redis中
                    if redis_client.exists(redis_session_key):
                        # 2. 更新Redis中的状态字段，保持与MongoDB一致
                        update_time = datetime.utcnow().timestamp()
                        redis_data = {
                            "session_status": str(status),  # 保持与MongoDB相同的字段名
                            "status": str(status),          # 兼容旧字段
                            "updated_at": update_time
                        }
                        
                        # 3. 执行更新
                        redis_client.hmset(redis_session_key, redis_data)
                        
                        # 4. 如果是结束状态，设置较短的过期时间
                        if status == 2:  # 已结束
                            redis_client.expire(redis_session_key, 1 * 24 * 60 * 60)  # 1天后过期
                            logger.info(f"会话状态已同步到Redis并设置为结束状态，过期时间1天: {session_id}")
                        else:
                            # 重置为标准过期时间
                            redis_client.expire(redis_session_key, 7 * 24 * 60 * 60)  # 7天后过期
                            logger.info(f"会话状态已同步到Redis: {session_id}, 状态: {status}, 过期时间7天")
                    else:
                        # 如果Redis中没有此会话记录，则创建一个
                        logger.warning(f"Redis中不存在会话，尝试从MongoDB同步: {session_id}")
                        
                        # 创建基本会话信息
                        basic_session = {
                            "id": session_id,
                            "session_id": session_id,
                            "user_id": user_id,
                            "user_name": session.get("user_name", "未知用户"),
                            "class_id": session.get("class_id", ""),
                            "class_name": session.get("class_name", ""),
                            "status": str(status),
                            "session_status": str(status),
                            "updated_at": datetime.utcnow().timestamp(),
                            "created_at": session.get("created_at", datetime.utcnow()).timestamp() if isinstance(session.get("created_at"), datetime) else datetime.utcnow().timestamp()
                        }
                        
                        # 添加角色信息 - 仅使用JSON格式存储角色列表，与MongoDB保持一致
                        roles = session.get("roles", [])
                        basic_session["roles"] = json.dumps(roles)
                        
                        # 保存到Redis
                        redis_client.hmset(redis_session_key, basic_session)
                        
                        # 设置过期时间
                        if status == 2:  # 已结束
                            redis_client.expire(redis_session_key, 1 * 24 * 60 * 60)  # 1天后过期
                        else:
                            redis_client.expire(redis_session_key, 7 * 24 * 60 * 60)  # 7天后过期
                            
                        logger.info(f"从MongoDB同步会话到Redis成功: {session_id}")
                else:
                    logger.warning(f"无法获取内存管理器或Redis客户端，无法同步状态到Redis: {session_id}")
            except Exception as e:
                logger.error(f"同步会话状态到Redis失败: {str(e)}")
                # 不中断流程，继续返回MongoDB更新状态
        else:
            logger.warning(f"MongoDB中会话状态未更新: {session_id}, 状态: {status}")
        
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
    async def delete_from_redis(cls, session_id: str) -> bool:
        """
        从Redis中删除会话数据
        
        参数:
            session_id: 要删除的会话ID
            
        返回:
            删除操作是否成功
        """
        try:
            # 获取Redis客户端
            from app.memory.memory_manager import get_memory_manager
            memory_manager = await get_memory_manager()
            
            if not memory_manager or not memory_manager.short_term or not memory_manager.short_term.redis:
                logger.error("无法获取内存管理器或Redis客户端")
                return False
            
            redis_client = memory_manager.short_term.redis
            
            # 查找所有可能的会话键
            session_keys = redis_client.keys(f"session:*:{session_id}")
            
            if not session_keys:
                logger.warning(f"Redis中未找到会话: {session_id}")
                return False
            
            # 删除所有相关键
            deleted_count = 0
            for key in session_keys:
                # 删除会话信息
                redis_client.delete(key)
                deleted_count += 1
                
                # 尝试提取用户ID
                parts = key.split(":")
                if len(parts) >= 3:
                    user_id = parts[1]
                    
                    # 删除会话消息列表
                    message_key = f"messages:{user_id}:{session_id}"
                    if redis_client.exists(message_key):
                        redis_client.delete(message_key)
                        logger.info(f"已删除会话消息列表: {message_key}")
            
            logger.info(f"从Redis中删除了{deleted_count}个会话键: {session_id}")
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"从Redis删除会话失败: {str(e)}")
            return False

    @classmethod
    async def delete_from_mongodb(cls, session_id: str) -> bool:
        """
        从MongoDB中删除会话数据
        
        参数:
            session_id: 要删除的会话ID
            
        返回:
            删除操作是否成功
        """
        try:
            collection = await cls.get_collection()
            result = await collection.delete_one({"session_id": session_id})
            
            if result.deleted_count > 0:
                logger.info(f"已从MongoDB删除会话: {session_id}")
                return True
            else:
                logger.warning(f"MongoDB中未找到会话: {session_id}")
                return False
            
        except Exception as e:
            logger.error(f"从MongoDB删除会话失败: {str(e)}")
            return False

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

    @classmethod
    async def sync_session_to_redis(cls, session_id: str) -> bool:
        """
        将MongoDB中的会话数据同步到Redis
        
        参数:
            session_id: 会话ID
            
        返回:
            同步是否成功
        """
        try:
            # 从MongoDB获取会话信息
            collection = await cls.get_collection()
            session = await collection.find_one({"session_id": session_id})
            
            if not session:
                logger.warning(f"要同步的会话不存在于MongoDB: {session_id}")
                return False
                
            # 获取用户ID
            user_id = session.get("user_id")
            if not user_id:
                logger.warning(f"会话缺少用户ID: {session_id}")
                return False
                
            # 获取Redis客户端
            from app.memory.memory_manager import get_memory_manager
            memory_manager = await get_memory_manager()
            
            if not memory_manager or not memory_manager.short_term or not memory_manager.short_term.redis:
                logger.error("无法获取内存管理器或Redis客户端")
                return False
                
            redis_client = memory_manager.short_term.redis
            
            # 构建Redis键
            redis_session_key = f"session:{user_id}:{session_id}"
            
            # 处理MongoDB时间字段，转换为timestamp
            created_at = session.get("created_at")
            updated_at = session.get("updated_at")
            
            # 准备Redis数据
            redis_data = {
                "id": session_id,
                "session_id": session_id,
                "user_id": user_id,
                "user_name": session.get("user_name", "未知用户"),
                "class_id": session.get("class_id", ""),
                "class_name": session.get("class_name", ""),
                "status": str(session.get("session_status", 0)),
                "session_status": str(session.get("session_status", 0)),
                "updated_at": updated_at.timestamp() if isinstance(updated_at, datetime) else time.time(),
                "created_at": created_at.timestamp() if isinstance(created_at, datetime) else time.time(),
                "start_time": created_at.timestamp() if isinstance(created_at, datetime) else time.time()
            }
            
            # 添加角色信息 - 仅使用JSON格式存储角色列表，与MongoDB保持一致
            roles = session.get("roles", [])
            redis_data["roles"] = json.dumps(roles)
            
            # 保存到Redis
            redis_client.hmset(redis_session_key, redis_data)
            
            # 设置过期时间
            session_status = session.get("session_status", 0)
            if session_status == 2:  # 已结束
                redis_client.expire(redis_session_key, 1 * 24 * 60 * 60)  # 1天后过期
            else:
                redis_client.expire(redis_session_key, 7 * 24 * 60 * 60)  # 7天后过期
                
            logger.info(f"成功将会话从MongoDB同步到Redis: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"同步会话到Redis失败: {str(e)}")
            return False
    
    @classmethod
    async def sync_session_to_mongodb(cls, session_id: str, user_id: str) -> bool:
        """
        将Redis中的会话数据同步到MongoDB
        
        参数:
            session_id: 会话ID
            user_id: 用户ID
            
        返回:
            同步是否成功
        """
        try:
            # 获取Redis客户端
            from app.memory.memory_manager import get_memory_manager
            memory_manager = await get_memory_manager()
            
            if not memory_manager or not memory_manager.short_term or not memory_manager.short_term.redis:
                logger.error("无法获取内存管理器或Redis客户端")
                return False
                
            redis_client = memory_manager.short_term.redis
            
            # 构建Redis键
            redis_session_key = f"session:{user_id}:{session_id}"
            
            # 检查Redis中是否存在会话
            if not redis_client.exists(redis_session_key):
                logger.warning(f"Redis中不存在会话: {session_id}")
                return False
                
            # 获取Redis会话数据
            redis_data = redis_client.hgetall(redis_session_key)
            
            if not redis_data:
                logger.warning(f"Redis会话数据为空: {session_id}")
                return False
                
            # 检查MongoDB中是否已存在该会话
            collection = await cls.get_collection()
            existing = await collection.find_one({"session_id": session_id})
            
            # 解析角色信息
            roles = []
            try:
                # 使用JSON字符串中的角色数据
                if "roles" in redis_data and redis_data["roles"]:
                    roles = json.loads(redis_data["roles"])
                # 如果没有roles字段或解析失败，则创建空角色列表
            except Exception as e:
                logger.error(f"解析Redis角色数据失败: {str(e)}")
                # 尝试使用空角色列表继续执行
            
            # 处理时间字段，转换为datetime
            created_at = float(redis_data.get("created_at", time.time()))
            updated_at = float(redis_data.get("updated_at", time.time()))
            
            # 准备MongoDB数据
            mongo_data = {
                "session_id": session_id,
                "user_id": user_id,
                "user_name": redis_data.get("user_name", "未知用户"),
                "class_id": redis_data.get("class_id", ""),
                "class_name": redis_data.get("class_name", ""),
                "session_status": int(redis_data.get("session_status", redis_data.get("status", "0"))),
                "roles": roles,
                "created_at": datetime.fromtimestamp(created_at),
                "updated_at": datetime.fromtimestamp(updated_at)
            }
            
            # 更新或插入到MongoDB
            if existing:
                # 更新现有会话
                result = await collection.update_one(
                    {"session_id": session_id},
                    {"$set": mongo_data}
                )
                
                if result.modified_count > 0:
                    logger.info(f"成功更新MongoDB中的会话: {session_id}")
                else:
                    logger.warning(f"MongoDB会话未更新: {session_id}")
            else:
                # 创建新会话
                result = await collection.insert_one(mongo_data)
                
                if result.inserted_id:
                    logger.info(f"成功在MongoDB中创建会话: {session_id}")
                else:
                    logger.error(f"创建MongoDB会话失败: {session_id}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"同步会话到MongoDB失败: {str(e)}")
            return False
    
    @classmethod
    async def get_session_inconsistencies(cls, limit: int = 100) -> list:
        """
        检测Redis和MongoDB中的会话不一致情况
        
        参数:
            limit: 最多检查的会话数量
            
        返回:
            不一致的会话ID列表和原因
        """
        try:
            # 获取Redis客户端
            from app.memory.memory_manager import get_memory_manager
            memory_manager = await get_memory_manager()
            
            if not memory_manager or not memory_manager.short_term or not memory_manager.short_term.redis:
                logger.error("无法获取内存管理器或Redis客户端")
                return [{"error": "无法获取Redis客户端"}]
                
            redis_client = memory_manager.short_term.redis
            
            # 获取MongoDB会话
            collection = await cls.get_collection()
            mongo_sessions = await collection.find({}).limit(limit).to_list(limit)
            
            inconsistencies = []
            
            # 检查每个MongoDB会话在Redis中是否存在
            for session in mongo_sessions:
                session_id = session.get("session_id")
                user_id = session.get("user_id")
                
                if not session_id or not user_id:
                    continue
                
                redis_key = f"session:{user_id}:{session_id}"
                
                # 检查Redis中是否存在
                if not redis_client.exists(redis_key):
                    inconsistencies.append({
                        "session_id": session_id,
                        "user_id": user_id,
                        "reason": "Redis中不存在此会话",
                        "source": "mongodb"
                    })
                    continue
                
                # 获取Redis数据
                redis_data = redis_client.hgetall(redis_key)
                
                # 比较状态值
                mongo_status = str(session.get("session_status", "0"))
                redis_status = redis_data.get("session_status", redis_data.get("status", ""))
                
                if mongo_status != redis_status:
                    inconsistencies.append({
                        "session_id": session_id,
                        "user_id": user_id,
                        "reason": f"状态不一致: MongoDB={mongo_status}, Redis={redis_status}",
                        "source": "both"
                    })
                
                # 比较角色列表
                mongo_roles = session.get("roles", [])
                redis_roles = []
                try:
                    if "roles" in redis_data and redis_data["roles"]:
                        redis_roles = json.loads(redis_data["roles"])
                except Exception as e:
                    logger.warning(f"无法解析Redis角色数据: {str(e)}")
                
                # 检查角色列表长度是否一致
                if len(mongo_roles) != len(redis_roles):
                    inconsistencies.append({
                        "session_id": session_id,
                        "user_id": user_id,
                        "reason": f"角色列表长度不一致: MongoDB={len(mongo_roles)}, Redis={len(redis_roles)}",
                        "source": "both"
                    })
                    continue
                
                # 深入比较角色数据内容
                mongo_role_ids = sorted([role.get("role_id", "") for role in mongo_roles])
                redis_role_ids = sorted([role.get("role_id", "") for role in redis_roles])
                
                if mongo_role_ids != redis_role_ids:
                    inconsistencies.append({
                        "session_id": session_id,
                        "user_id": user_id,
                        "reason": "角色数据不一致",
                        "source": "both",
                        "mongo_roles": mongo_role_ids,
                        "redis_roles": redis_role_ids
                    })
            
            # 检查是否有Redis会话不在MongoDB中
            # 获取所有会话键
            session_keys = redis_client.keys("session:*")
            
            for key in session_keys[:limit]:  # 限制检查数量
                parts = key.split(":")
                if len(parts) != 3:
                    continue
                    
                user_id = parts[1]
                session_id = parts[2]
                
                # 检查MongoDB中是否存在
                mongo_session = await collection.find_one({"session_id": session_id})
                if not mongo_session:
                    inconsistencies.append({
                        "session_id": session_id,
                        "user_id": user_id,
                        "reason": "MongoDB中不存在此会话",
                        "source": "redis",
                        "key": key
                    })
            
            return inconsistencies
            
        except Exception as e:
            logger.error(f"检测会话不一致失败: {str(e)}")
            return [{"error": str(e)}]

    @classmethod
    async def migrate_redis_roles_format(cls, limit: int = 100) -> Dict[str, Any]:
        """
        迁移Redis中的角色数据格式为统一的JSON格式
        
        将老格式(角色单独存储在单独的字段)迁移到新格式(仅使用JSON字符串存储角色列表)
        
        参数:
            limit: 最多处理的会话数量
            
        返回:
            迁移结果统计
        """
        try:
            # 获取Redis客户端
            from app.memory.memory_manager import get_memory_manager
            memory_manager = await get_memory_manager()
            
            if not memory_manager or not memory_manager.short_term or not memory_manager.short_term.redis:
                logger.error("无法获取内存管理器或Redis客户端")
                return {"success": False, "error": "无法获取Redis客户端"}
                
            redis_client = memory_manager.short_term.redis
            
            # 获取所有会话键
            session_keys = redis_client.keys("session:*")
            
            # 迁移统计
            stats = {
                "total": len(session_keys),
                "processed": 0,
                "migrated": 0,
                "already_migrated": 0,
                "errors": 0,
                "error_details": []
            }
            
            # 处理每个会话
            for key in session_keys[:limit]:
                stats["processed"] += 1
                
                try:
                    # 获取会话数据
                    session_data = redis_client.hgetall(key)
                    
                    # 检查是否已经是新格式(仅有roles字段)
                    if "roles" in session_data:
                        # 检查是否仍有旧格式的角色字段
                        has_old_format = any(k.startswith("role_") and (k.endswith("_id") or k.endswith("_name")) for k in session_data.keys())
                        
                        if not has_old_format:
                            stats["already_migrated"] += 1
                            continue
                    
                    # 构建角色列表
                    roles = []
                    try:
                        # 如果已经有roles字段，尝试解析
                        if "roles" in session_data and session_data["roles"]:
                            roles = json.loads(session_data["roles"])
                        else:
                            # 从旧格式构建角色列表
                            roles_count = int(session_data.get("roles_count", "0"))
                            for i in range(roles_count):
                                role_id = session_data.get(f"role_{i}_id")
                                role_name = session_data.get(f"role_{i}_name")
                                if role_id and role_name:
                                    roles.append({"role_id": role_id, "role_name": role_name})
                    except Exception as e:
                        logger.error(f"构建角色列表失败: {str(e)}, key={key}")
                        stats["error_details"].append({"key": key, "error": f"构建角色列表失败: {str(e)}"})
                        stats["errors"] += 1
                        continue
                    
                    # 保存角色列表为JSON字符串
                    redis_client.hset(key, "roles", json.dumps(roles))
                    
                    # 删除旧格式的角色字段
                    fields_to_delete = [k for k in session_data.keys() if (k.startswith("role_") and (k.endswith("_id") or k.endswith("_name"))) or k == "roles_count"]
                    if fields_to_delete:
                        redis_client.hdel(key, *fields_to_delete)
                    
                    stats["migrated"] += 1
                    
                except Exception as e:
                    logger.error(f"迁移会话失败: {str(e)}, key={key}")
                    stats["error_details"].append({"key": key, "error": str(e)})
                    stats["errors"] += 1
            
            return {
                "success": True,
                "stats": stats
            }
        
        except Exception as e:
            logger.error(f"角色数据格式迁移失败: {str(e)}")
            return {"success": False, "error": str(e)} 