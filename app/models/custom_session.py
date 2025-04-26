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
import asyncio

from app.database.mongodb import get_db

logger = logging.getLogger(__name__)

class SessionStatus:
    """会话状态常量"""
    ACTIVE = "active"     # 当前活跃会话
    ARCHIVED = "archived" # 已归档会话
    DELETED = "deleted"   # 已删除但未永久移除

class CustomSession:
    collection = None

    @classmethod
    async def get_collection(cls):
        """获取会话集合"""
        if cls.collection is None:
            db = await get_db()
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
            
            # 添加重试机制
            max_retries = 3
            retry_delay = 1  # 秒
            
            for attempt in range(max_retries):
                try:
                    # 同步到Redis - 修复属性名称
                    if memory_manager and memory_manager.short_term_memory and memory_manager.short_term_memory.redis:
                        redis_client = memory_manager.short_term_memory.redis
                
                        # 使用新标准格式的会话键 - 不包含user_id
                        redis_session_key = f"session:{session_id}"
                
                        # 准备会话数据 - 与MongoDB存储完全相同的格式
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
                
                        # 处理角色信息 - 使用MongoDB存储一致的格式
                        roles = session_data["roles"]
                        logger.info(f"同步到Redis前的角色数据: {roles}")

                        # 从MongoDB获取完整角色信息
                        try:
                            from app.database.connection import get_database
                            from bson.objectid import ObjectId
                            
                            # 获取数据库连接
                            db = await get_database()
                            if db is not None:
                                # 遍历每个角色
                                for i, role in enumerate(roles):
                                    role_id = role.get("role_id")
                                    if role_id:
                                        try:
                                            # 查询MongoDB获取完整角色信息
                                            object_role_id = ObjectId(role_id)
                                            role_info = await db.roles.find_one({"_id": object_role_id})
                                            
                                            if role_info:
                                                logger.info(f"从MongoDB获取到角色 {role_id} 的完整信息")
                                                # 添加keywords字段
                                                if "keywords" in role_info:
                                                    role["keywords"] = role_info["keywords"]
                                                    logger.info(f"更新角色keywords: {role['keywords']}")
                                                else:
                                                    role["keywords"] = []
                                                    logger.warning(f"MongoDB中角色 {role_id} 也缺少keywords字段")
                                            else:
                                                logger.warning(f"未能从MongoDB找到角色: {role_id}")
                                                role["keywords"] = []
                                        except Exception as e:
                                            logger.error(f"获取角色 {role_id} 信息时出错: {str(e)}")
                                            role["keywords"] = []
                                    else:
                                        logger.warning(f"角色缺少role_id，无法获取完整信息")
                                        role["keywords"] = []
                        except Exception as e:
                            logger.error(f"获取角色完整信息时出错: {str(e)}")

                        # 检查每个角色是否包含keywords字段
                        for i, role in enumerate(roles):
                            if "keywords" not in role:
                                logger.warning(f"角色{i}（{role.get('role_name', 'unknown')}）缺少keywords字段")
                                role["keywords"] = []
                            else:
                                logger.info(f"角色{i}（{role.get('role_name', 'unknown')}）的keywords: {role['keywords']}")
                        
                        redis_data["roles"] = json.dumps(roles)
                        logger.info(f"序列化后的角色数据: {redis_data['roles'][:100]}...")
                
                        # 向Redis写入数据
                        await redis_client.hmset(redis_session_key, redis_data)
                
                        # 设置过期时间，例如7天
                        await redis_client.expire(redis_session_key, 7 * 24 * 60 * 60)
                
                        logger.info(f"会话已同步到Redis: {session_id}, 键: {redis_session_key}")
                
                        # 创建会话消息列表键
                        message_key = f"messages:{user_id}:{session_id}"
                
                        # 预创建空消息列表
                        if not await redis_client.exists(message_key):
                            # 添加一个系统消息作为初始消息
                            system_message = {
                                "role": "system",
                                "content": f"会话开始：{class_name}",
                                "timestamp": now.isoformat(),
                                "metadata": {
                                    "session_id": session_id,
                                    "user_id": user_id,
                                    "roles": [role.get("role_name", "") for role in roles],
                                    "role_prompts": {role.get("role_id", ""): role.get("system_prompt", "") for role in roles if "system_prompt" in role}
                                }
                            }
                            await redis_client.rpush(message_key, json.dumps(system_message))
                            logger.info(f"为会话 {session_id} 创建初始消息列表")
                        
                        # 同步成功，跳出重试循环
                        break
                    else:
                        # 记录更详细的信息，帮助诊断问题
                        logger_detail = {
                            "memory_manager_exists": memory_manager is not None,
                            "short_term_memory_exists": hasattr(memory_manager, 'short_term_memory') if memory_manager else False,
                            "redis_exists": hasattr(memory_manager.short_term_memory, 'redis') if memory_manager and hasattr(memory_manager, 'short_term_memory') else False
                        }
                        logger.warning(f"无法获取内存管理器或Redis客户端，会话未同步到Redis: {session_id}, 详情: {logger_detail}")
                        
                        # 如果不是最后一次尝试，等待后重试
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay * (2 ** attempt))  # 指数退避策略
                        else:
                            logger.error(f"会话同步到Redis重试{max_retries}次后失败: {session_id}")
                except Exception as retry_error:
                    logger.warning(f"同步会话到Redis尝试 {attempt+1}/{max_retries} 失败: {str(retry_error)}")
                    
                    # 如果不是最后一次尝试，等待后重试
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (2 ** attempt))  # 指数退避策略
                    else:
                        logger.error(f"会话同步到Redis重试{max_retries}次后失败: {str(retry_error)}")
                        raise  # 重新抛出异常，让外层catch处理
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
                if memory_manager and memory_manager.short_term_memory and memory_manager.short_term_memory.redis:
                    # 使用新标准格式的会话键 - 不包含user_id
                    redis_session_key = f"session:{session_id}"
                    redis_client = memory_manager.short_term_memory.redis
                    
                    # 添加重试机制
                    max_retries = 3
                    retry_delay = 1  # 秒
                    
                    for attempt in range(max_retries):
                        try:
                            # 检查会话是否存在于Redis中
                            if await redis_client.exists(redis_session_key):
                                # 更新Redis中的状态字段，保持与MongoDB一致
                                update_time = datetime.utcnow().timestamp()
                                redis_data = {
                                    "session_status": str(status),  # 兼容旧字段
                                    "status": str(status),          # 兼容旧字段
                                    "updated_at": update_time
                                }
                                
                                # 执行更新
                                await redis_client.hmset(redis_session_key, redis_data)
                                
                                # 如果是结束状态，设置较短的过期时间
                                if status == 2:  # 已结束
                                    await redis_client.expire(redis_session_key, 1 * 24 * 60 * 60)  # 1天后过期
                                    logger.info(f"会话状态已同步到Redis并设置为结束状态，过期时间1天: {session_id}")
                                else:
                                    # 重置为标准过期时间
                                    await redis_client.expire(redis_session_key, 7 * 24 * 60 * 60)  # 7天后过期
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
                                await redis_client.hmset(redis_session_key, basic_session)
                                
                                # 设置过期时间
                                ttl = 1 * 24 * 60 * 60 if status == 2 else 7 * 24 * 60 * 60
                                await redis_client.expire(redis_session_key, ttl)
                            
                            # 同步成功，跳出重试循环
                            break
                        except Exception as retry_error:
                            logger.warning(f"同步会话状态到Redis尝试 {attempt+1}/{max_retries} 失败: {str(retry_error)}")
                            
                            # 如果不是最后一次尝试，等待后重试
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_delay * (2 ** attempt))  # 指数退避策略
                else:
                    logger.error(f"同步会话状态到Redis重试{max_retries}次后失败: {str(retry_error)}")
                    # 记录详细错误，但允许继续
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
    async def delete_from_redis(cls, session_id: str, user_id: str = None) -> bool:
        """
        从Redis删除会话
        
        参数:
            session_id: 会话ID
            user_id: 可选的用户ID，如果提供则用于构建消息键
            
        返回:
            是否删除成功
        """
        logger = logging.getLogger("app.models.custom_session")
        
        try:
            # 获取Redis客户端
            from app.memory.memory_manager import get_memory_manager
            
            memory_manager = await get_memory_manager()
            
            if memory_manager and memory_manager.short_term_memory and memory_manager.short_term_memory.redis:
                redis_client = memory_manager.short_term_memory.redis
            
                # 如果未提供user_id，尝试从MongoDB获取
                if not user_id:
                    logger.info(f"用户ID未提供，尝试从MongoDB获取: {session_id}")
                    collection = await cls.get_collection()
                    session_data = await collection.find_one({"session_id": session_id})
            
                    if session_data and "user_id" in session_data:
                        user_id = session_data.get("user_id")
                        logger.info(f"成功从MongoDB获取用户ID: {user_id}")
                    else:
                        logger.warning(f"在MongoDB中找不到会话或用户ID: {session_id}")
                
                # 删除所有相关会话键
                keys_to_delete = []
                
                # 1. 标准会话键: session:{session_id}
                keys_to_delete.append(f"session:{session_id}")
                
                # 3. 消息列表键: messages:{user_id}:{session_id}
                if user_id:
                    keys_to_delete.append(f"messages:{user_id}:{session_id}")
                
                # 4. 旧会话键格式: session:{user_id}:{session_id} (用于兼容)
                if user_id:
                    keys_to_delete.append(f"session:{user_id}:{session_id}")
                
                # 执行删除
                for key in keys_to_delete:
                    logger.info(f"尝试删除Redis键: {key}")
                    if await redis_client.exists(key):
                        await redis_client.delete(key)
                        logger.info(f"已删除Redis键: {key}")
                    else:
                        logger.warning(f"键不存在，无需删除: {key}")
                
                return True
            else:
                logger.error("无法获取Redis客户端")
                return False
            
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
    async def sync_session_to_redis(cls, session_id: str, user_id: str) -> bool:
        """
        将会话数据从MongoDB同步到Redis
        
        参数:
            session_id: 会话ID
            user_id: 用户ID
            
        返回:
            是否同步成功
        """
        logger = logging.getLogger("app.models.custom_session")
        logger.info(f"开始将会话 {session_id} 同步到Redis")
        
        # 添加重试机制
        max_retries = 3
        retry_delay = 1  # 秒
        
        for attempt in range(max_retries):
            try:
                # 1. 从MongoDB获取会话数据
                collection = await cls.get_collection()
                session_data = await collection.find_one({"session_id": session_id})
            
                if not session_data:
                    logger.warning(f"找不到会话: {session_id}")
                    return False
                
                # 验证会话包含用户ID
                if "user_id" not in session_data:
                    logger.error(f"会话数据不包含user_id: {session_id}")
                    return False
                
                stored_user_id = session_data.get("user_id", "")
                
                # 如果传入的用户ID不为空，验证与存储的是否一致
                if user_id and stored_user_id != user_id:
                    logger.warning(f"用户ID不匹配: 存储={stored_user_id}, 请求={user_id}")
                    # 使用数据库中的user_id作为正确值
                    user_id = stored_user_id
                
                # 2. 获取内存管理器
                from app.memory.memory_manager import get_memory_manager
                
                memory_manager = await get_memory_manager()
                
                # 3. 同步到Redis
                if memory_manager and memory_manager.short_term_memory and memory_manager.short_term_memory.redis:
                    redis_client = memory_manager.short_term_memory.redis
            
                    # 使用新的标准格式的会话键 (不包含user_id)
                    redis_session_key = f"session:{session_id}"
                    logger.info(f"使用的Redis会话键: {redis_session_key}")
            
                    # 准备会话数据 - 与MongoDB存储完全相同的格式
                    redis_data = {
                        "id": session_id,
                        "session_id": session_id,
                        "class_id": str(session_data.get("class_id", "")),
                        "class_name": session_data.get("class_name", ""),
                        "user_id": user_id,
                        "user_name": session_data.get("user_name", ""),
                        "status": str(session_data.get("session_status", "0")),
                        "session_status": str(session_data.get("session_status", "0")),
                        "start_time": session_data.get("created_at", datetime.utcnow()).timestamp(),
                        "created_at": session_data.get("created_at", datetime.utcnow()).timestamp(),
                        "updated_at": session_data.get("updated_at", datetime.utcnow()).timestamp()
                    }
            
                    # 处理角色信息
                    roles = session_data.get("roles", [])
                    logger.info(f"角色数据: {roles}")
                    
                    # 将角色数据转换为字符串
                    redis_data["roles"] = json.dumps(roles)
            
                    # 向Redis写入数据
                    await redis_client.hmset(redis_session_key, redis_data)
            
                    # 设置过期时间，例如7天
                    await redis_client.expire(redis_session_key, 7 * 24 * 60 * 60)
                
                    logger.info(f"会话 {session_id} 已成功同步到Redis")
                    return True
                else:
                    logger.error(f"内存管理器或Redis客户端不可用")
                    return False
            
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"同步到Redis失败，尝试重试 ({attempt+1}/{max_retries}): {str(e)}")
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"同步到Redis失败，已达到最大重试次数: {str(e)}")
                    return False
        
        return False
    
    @classmethod
    async def sync_session_to_mongodb(cls, session_id: str, user_id: str) -> bool:
        """
        将会话从Redis同步到MongoDB
        
        参数:
            session_id: 会话ID
            user_id: 用户ID
            
        返回:
            同步是否成功
        """
        try:
            # 获取内存管理器
            from app.memory.memory_manager import get_memory_manager
            memory_manager = await get_memory_manager()
            
            # 检查内存管理器和Redis可用性
            if not memory_manager or not memory_manager.short_term_memory or not memory_manager.short_term_memory.redis:
                logger.error(f"内存管理器或Redis不可用，无法同步会话: {session_id}")
                return False
                
            # 使用短期记忆的Redis客户端
            redis_client = memory_manager.short_term_memory.redis
            
            # 构建Redis键
            redis_session_key = f"session:{session_id}"
            
            # 添加重试机制
            max_retries = 3
            retry_delay = 1  # 秒
            
            # 用于存储Redis数据
            redis_data = None
            
            for attempt in range(max_retries):
                try:
                    # 从Redis获取会话数据
                    redis_data = await redis_client.hgetall(redis_session_key)
            
                    if not redis_data:
                        logger.warning(f"Redis中不存在会话: {session_id}")
                    return False
                
                    # 获取成功，跳出循环
                    break
                except Exception as retry_error:
                    logger.warning(f"从Redis获取会话数据尝试 {attempt+1}/{max_retries} 失败: {str(retry_error)}")
            
                    # 如果不是最后一次尝试，等待后重试
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (2 ** attempt))  # 指数退避策略
                    else:
                        logger.error(f"从Redis获取会话数据重试{max_retries}次后失败: {str(retry_error)}")
                        return False
            
            # 准备MongoDB数据
            mongodb_data = {
                "session_id": session_id,
                "user_id": user_id,
                "updated_at": datetime.utcnow()
            }
            
            # 从Redis数据转换为MongoDB数据
            if "user_name" in redis_data:
                mongodb_data["user_name"] = redis_data["user_name"]
                
            if "class_id" in redis_data:
                mongodb_data["class_id"] = redis_data["class_id"]
                
            if "class_name" in redis_data:
                mongodb_data["class_name"] = redis_data["class_name"]
                
            if "session_status" in redis_data:
                try:
                    mongodb_data["session_status"] = int(redis_data["session_status"])
                except ValueError:
                    mongodb_data["session_status"] = 0
                    
            # 处理角色数据
            if "roles" in redis_data:
                try:
                    roles_json = redis_data["roles"]
                    logger.info(f"从Redis读取的角色数据JSON: {roles_json[:100]}...")
                    roles = json.loads(roles_json)
                    logger.info(f"解析后的角色数据: {roles}")
                    
                    # 补充角色信息 - 确保所有角色都有完整数据
                    from app.database.connection import get_database
                    from bson.objectid import ObjectId
                    
                    db = await get_database()
                    if db is not None:
                        for i, role in enumerate(roles):
                            # 检查角色是否缺少keywords或keywords为空
                            role_id = role.get("role_id")
                            if role_id and ("keywords" not in role or not role["keywords"]):
                                try:
                                    # 从数据库获取完整角色信息
                                    role_data = await db.roles.find_one({"_id": ObjectId(role_id)})
                                    if role_data and "keywords" in role_data:
                                        logger.info(f"从数据库补充角色 {role_id} 的keywords: {role_data['keywords']}")
                                        role["keywords"] = role_data["keywords"]
                                except Exception as role_error:
                                    logger.error(f"获取角色 {role_id} 完整信息失败: {str(role_error)}")
                    
                    # 检查每个角色是否包含keywords字段
                    for i, role in enumerate(roles):
                        if "keywords" not in role:
                            logger.warning(f"MongoDB同步中角色{i}（{role.get('role_name', 'unknown')}）缺少keywords字段")
                            role["keywords"] = []
                        else:
                            logger.info(f"MongoDB同步中角色{i}（{role.get('role_name', 'unknown')}）的keywords: {role['keywords']}")
                    
                    mongodb_data["roles"] = roles
                except json.JSONDecodeError as json_error:
                    logger.error(f"解析角色JSON失败: {str(json_error)}, 原始数据: {roles_json[:100]}...")
                    mongodb_data["roles"] = []
            
            # 检查MongoDB中是否存在会话
            collection = await cls.get_collection()
            existing = await collection.find_one({"session_id": session_id})
            
            if existing:
                # 更新现有会话
                result = await collection.update_one(
                    {"session_id": session_id},
                    {"$set": mongodb_data}
                )
                
                if result.modified_count > 0:
                    logger.info(f"已更新MongoDB中的会话: {session_id}")
                    return True
                else:
                    logger.warning(f"MongoDB会话更新无效（可能无变化）: {session_id}")
                    return True  # 仍然返回成功，因为会话存在
            else:
                # 创建新会话
                mongodb_data["created_at"] = datetime.utcnow()
                
                result = await collection.insert_one(mongodb_data)
                
                if result.inserted_id:
                    logger.info(f"已在MongoDB中创建新会话: {session_id}")
                    return True
                else:
                    logger.error(f"创建MongoDB会话失败: {session_id}")
                    return False
            
        except Exception as e:
            logger.error(f"将会话从Redis同步到MongoDB失败: {str(e)}")
            return False
    
    @classmethod
    async def get_session_inconsistencies(cls, limit: int = 100) -> list:
        """
        检查MongoDB和Redis之间的会话数据一致性
        
        参数:
            limit: 最大检查会话数量
            
        返回:
            不一致的会话列表，每项包含session_id和不一致原因
        """
        logger = logging.getLogger("app.models.custom_session")
        logger.info(f"开始检查MongoDB和Redis之间的会话数据一致性，限制 {limit} 条")
        
        try:
            # 获取内存管理器
            from app.memory.memory_manager import get_memory_manager
            memory_manager = await get_memory_manager()
            
            # 检查内存管理器和Redis可用性
            if not memory_manager or not memory_manager.short_term_memory or not memory_manager.short_term_memory.redis:
                logger.error("内存管理器或Redis不可用，无法检查一致性")
                return [{"error": "内存管理器或Redis不可用"}]
                
            # 使用短期记忆的Redis客户端
            redis_client = memory_manager.short_term_memory.redis
            
            # 获取MongoDB会话
            collection = await cls.get_collection()
            mongo_sessions = await collection.find(
                {"session_status": {"$ne": 2}},  # 排除已完成的会话
                {"session_id": 1, "user_id": 1, "session_status": 1}
            ).limit(limit).to_list(length=limit)
            
            inconsistencies = []
            
            # 检查每个会话
            for session in mongo_sessions:
                session_id = session.get("session_id")
                user_id = session.get("user_id")
                
                if not session_id or not user_id:
                    logger.warning(f"MongoDB会话缺少session_id或user_id: {session}")
                    continue
                
                # 检查标准会话键
                session_key = f"session:{session_id}"
                
                # 检查旧格式键 (兼容)
                old_session_key = f"session:{user_id}:{session_id}"
                
                # 检查消息键
                message_key = f"messages:{user_id}:{session_id}"
                
                # 检查各种键是否存在
                session_exists = await redis_client.exists(session_key)
                old_session_exists = await redis_client.exists(old_session_key)
                message_exists = await redis_client.exists(message_key)
                
                if not session_exists and not old_session_exists:
                    # Redis中完全没有这个会话
                    inconsistencies.append({
                        "session_id": session_id,
                        "user_id": user_id,
                        "reason": "Redis中完全缺少会话",
                        "source": "mongodb"
                    })
                    logger.warning(f"会话 {session_id} (用户 {user_id}) 在Redis中不存在")
                    continue
                
                # 获取实际的Redis数据 - 优先使用标准会话键
                redis_data = None
                actual_key = None
                
                if session_exists:
                    redis_data = await redis_client.hgetall(session_key)
                    actual_key = session_key
                    logger.info(f"使用标准会话键: {session_key}")
                elif old_session_exists:
                    redis_data = await redis_client.hgetall(old_session_key)
                    actual_key = old_session_key
                    logger.warning(f"使用旧格式会话键: {old_session_key} - 需要迁移")
                
                if not redis_data:
                    inconsistencies.append({
                        "session_id": session_id,
                        "user_id": user_id,
                        "reason": f"Redis中会话数据为空 (键: {actual_key})",
                        "source": "redis_empty"
                    })
                    logger.warning(f"会话 {session_id} 在Redis中存在但数据为空")
                    continue
                
                # 检查关键字段一致性
                if redis_data.get("session_id") != session_id:
                    inconsistencies.append({
                        "session_id": session_id,
                        "user_id": user_id,
                        "reason": "会话ID不匹配",
                        "mongodb_id": session_id,
                        "redis_id": redis_data.get("session_id"),
                        "redis_key": actual_key
                    })
                    logger.warning(f"会话ID不匹配: MongoDB={session_id}, Redis={redis_data.get('session_id')}")
                
                # 检查用户ID一致性 - 考虑字符串和整数类型的差异
                if str(redis_data.get("user_id", "")) != str(user_id):
                    inconsistencies.append({
                        "session_id": session_id,
                        "user_id": user_id,
                        "reason": "用户ID不匹配",
                        "mongodb_user_id": str(user_id),
                        "redis_user_id": str(redis_data.get("user_id", "")),
                        "redis_key": actual_key
                    })
                    logger.warning(f"用户ID不匹配: MongoDB={user_id}, Redis={redis_data.get('user_id')}")
                
                # 检查会话状态 - 尝试处理不同的格式
                mongo_status = session.get("session_status", 0)
                
                # 处理Redis中可能存在的两种状态字段
                redis_status = None
                if "session_status" in redis_data:
                    redis_status_str = redis_data.get("session_status", "0")
                    redis_status = int(redis_status_str) if redis_status_str.isdigit() else -1
                elif "status" in redis_data:
                    redis_status_str = redis_data.get("status", "0")
                    redis_status = int(redis_status_str) if redis_status_str.isdigit() else -1
                else:
                    redis_status = -1
                
                if mongo_status != redis_status:
                    inconsistencies.append({
                        "session_id": session_id,
                        "user_id": user_id,
                        "reason": "会话状态不匹配",
                        "mongodb_status": mongo_status,
                        "redis_status": redis_status,
                        "redis_key": actual_key
                    })
                    logger.warning(f"会话状态不匹配: MongoDB={mongo_status}, Redis={redis_status}")
                
                # 检查消息键是否存在
                if not message_exists:
                    inconsistencies.append({
                        "session_id": session_id,
                        "user_id": user_id,
                        "reason": "缺少消息列表",
                        "message_key": message_key
                    })
                    logger.warning(f"缺少消息列表: {message_key}")
            
            logger.info(f"一致性检查完成，找到 {len(inconsistencies)} 个不一致项")
            return inconsistencies
            
        except Exception as e:
            logger.error(f"检查会话一致性失败: {str(e)}")
            return [{"error": str(e)}]

    @classmethod
    async def migrate_redis_roles_format(cls, limit: int = 100) -> Dict[str, Any]:
        """
        迁移Redis中的角色数据格式并标准化键格式
        
        参数:
            limit: 最大迁移会话数量
            
        返回:
            迁移结果统计
        """
        logger = logging.getLogger("app.models.custom_session")
        logger.info(f"开始迁移Redis中的角色数据格式和键格式，限制 {limit} 条")
        
        try:
            # 获取内存管理器
            from app.memory.memory_manager import get_memory_manager
            memory_manager = await get_memory_manager()
            
            # 检查内存管理器和Redis可用性
            if not memory_manager or not memory_manager.short_term_memory or not memory_manager.short_term_memory.redis:
                logger.error("内存管理器或Redis不可用，无法迁移角色格式")
                return {"error": "内存管理器或Redis不可用"}
                
            # 使用短期记忆的Redis客户端
            redis_client = memory_manager.short_term_memory.redis
            
            # 统计变量
            results = {
                "scanned": 0,
                "updated": 0,
                "skipped": 0,
                "errors": 0,
                "session_keys_migrated": 0
            }
            
            # 获取所有会话键
            session_keys = await redis_client.keys("session:*")
            session_keys = session_keys[:limit]  # 限制处理数量
            
            results["scanned"] = len(session_keys)
            logger.info(f"找到 {len(session_keys)} 个会话键")
            
            # 分类会话键
            standard_keys = []  # session:{session_id}
            old_format_keys = []  # session:{user_id}:{session_id}
            
            for key in session_keys:
                parts = key.split(":")
                
                if len(parts) == 2:
                    # 标准格式: session:{session_id}
                    standard_keys.append(key)
                elif len(parts) == 3:
                    # 旧格式: session:{user_id}:{session_id}
                    old_format_keys.append(key)
                else:
                    logger.warning(f"无法识别的键格式: {key}")
            
            logger.info(f"键分类: 标准格式={len(standard_keys)}, 旧格式={len(old_format_keys)}")
            
            # 处理旧格式键，迁移到新格式
            for key in old_format_keys:
                try:
                    # 解析旧格式键 session:{user_id}:{session_id}
                    parts = key.split(":")
                    if len(parts) != 3:
                        results["errors"] += 1
                        continue
                    
                    user_id = parts[1]
                    session_id = parts[2]
                    
                    # 构建新格式键
                    new_session_key = f"session:{session_id}"
                    
                    # 获取旧键数据
                    old_data = await redis_client.hgetall(key)
                        
                    if not old_data:
                        results["skipped"] += 1
                        continue
                    
                    # 检查是否已存在新格式键
                    if await redis_client.exists(new_session_key):
                        # 删除旧键
                        await redis_client.delete(key)
                        results["session_keys_migrated"] += 1
                        logger.info(f"删除已迁移的旧格式键: {key}")
                        continue
                    
                    # 迁移数据到新格式键
                    await redis_client.hmset(new_session_key, old_data)
                    
                    # 设置TTL
                    ttl = await redis_client.ttl(key)
                    if ttl > 0:
                        await redis_client.expire(new_session_key, ttl)
                    else:
                        # 设置默认过期时间
                        await redis_client.expire(new_session_key, 7 * 24 * 60 * 60)
                    
                    # 删除旧格式键
                    await redis_client.delete(key)
                    
                    results["session_keys_migrated"] += 1
                    logger.info(f"迁移会话键: {key} -> {new_session_key}")
                    
                    # 处理角色数据格式
                    await cls._migrate_roles_for_key(new_session_key, redis_client, results)
                    
                except Exception as e:
                    logger.error(f"处理旧格式键 {key} 失败: {str(e)}")
                    results["errors"] += 1
            
            # 处理标准格式键的角色数据
            for key in standard_keys:
                try:
                    # 处理角色数据格式
                    await cls._migrate_roles_for_key(key, redis_client, results)
        
                except Exception as e:
                    logger.error(f"处理标准格式键 {key} 失败: {str(e)}")
                    results["errors"] += 1
            
            logger.info(f"迁移完成: {results}")
            return results
            
        except Exception as e:
            logger.error(f"迁移过程中出错: {str(e)}")
            return {"error": str(e)}
    
    @classmethod
    async def _migrate_roles_for_key(cls, key: str, redis_client, results: dict) -> None:
        """
        为单个键迁移角色数据格式
        
        参数:
            key: Redis键
            redis_client: Redis客户端
            results: 结果统计字典
        """
        logger = logging.getLogger("app.models.custom_session")
        
        # 获取会话数据
        session_data = await redis_client.hgetall(key)
        
        if not session_data:
            results["skipped"] += 1
            return
        
        # 检查roles字段
        if "roles" not in session_data:
            results["skipped"] += 1
            return
        
        # 检查是否需要更新
        roles_value = session_data["roles"]
        
        try:
            # 尝试解析JSON
            json.loads(roles_value)
            # 已经是JSON格式，不需要更新
            results["skipped"] += 1
            return
        except json.JSONDecodeError:
            # 不是JSON格式，需要迁移
            pass
        
        # 提取会话ID
        parts = key.split(":")
        if len(parts) != 3:
            results["errors"] += 1
            return
        
        user_id = parts[1]
        session_id = parts[2]
        
        # 从MongoDB获取角色数据
        mongo_session = await cls.get_session_by_id(session_id)
        
        if not mongo_session or "roles" not in mongo_session:
            # 如果MongoDB中没有数据，尝试将当前数据转换为JSON
            try:
                # 尝试解析当前数据
                roles_data = roles_value.strip()
                
                # 检查是否已经是列表格式的字符串
                if roles_data.startswith("[") and roles_data.endswith("]"):
                    try:
                        json.loads(roles_data)
                        # 已经是有效的JSON，不需要更改
                        return
                    except:
                        pass
                
                # 尝试将字符串转换为列表格式
                roles = []
                for role_item in roles_data.split(","):
                    role_item = role_item.strip()
                    if role_item:
                        role_parts = role_item.split(":")
                        if len(role_parts) >= 2:
                            role_id = role_parts[0].strip()
                            role_name = role_parts[1].strip()
                            
                            roles.append({
                                "role_id": role_id,
                                "role_name": role_name,
                                "keywords": []
                            })
                
                # 转换为JSON
                roles_json = json.dumps(roles)
                
                # 更新Redis
                await redis_client.hset(key, "roles", roles_json)
                results["updated"] += 1
                logger.info(f"已转换角色数据并更新Redis: {key}")
                return
                
            except Exception as conv_error:
                logger.error(f"尝试转换角色数据失败: {str(conv_error)}")
                results["errors"] += 1
                return
        
        # 使用MongoDB中的角色数据更新Redis
        roles = mongo_session["roles"]
        roles_json = json.dumps(roles)
        
        # 更新Redis
        await redis_client.hset(key, "roles", roles_json)
        
        results["updated"] += 1
        logger.info(f"已从MongoDB获取角色数据并更新Redis: {key}")
        return 