"""
短期记忆模块 - 使用Redis实现的缓冲记忆
"""

import json
import time
import logging
from app.memory.schemas import Message, ChatSession
from app.config import memory_settings
from typing import Dict, Tuple, Optional
import os
import uuid
from datetime import datetime
from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

class ShortTermMemory:
    """
    短期记忆实现类，基于Redis
    使用Redis列表存储对话历史，按消息添加顺序排列
    """
    
    def __init__(self, max_rounds: int = 4):
        """初始化BufferMemory"""
        self.redis = None
        self.max_rounds = max_rounds
        self.redis_url = None
        
    async def initialize(self):
        """初始化Redis连接"""
        try:
            # 从环境变量或内存设置中获取配置
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6378"))
            redis_password = os.getenv("REDIS_PASSWORD", "!qaz2wsX")
            
            self.redis_url = f"redis://{redis_host}:{redis_port}"
            if redis_password:
                self.redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}"
                
            self.redis = await Redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            # 测试连接
            await self.redis.ping()
            logger.info(f"成功连接到Redis服务器: {self.redis_url}")
            
        except RedisError as e:
            logger.error(f"Redis连接失败: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"初始化短期记忆时出错: {str(e)}")
            raise
            
    async def start_session(self, session_id: str, user_id: str, selected_username: str = None) -> str:
        """开始一个新的会话
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            selected_username: 前端选中的用户名称（可选）
        
        Returns:
            会话ID
        """
        try:
            # 使用新的标准格式
            session_key = f"session:{session_id}"
            session = ChatSession(id=session_id, user_id=user_id)
            
            # 存储会话信息
            await self.redis.hset(session_key, "id", session.id)
            await self.redis.hset(session_key, "user_id", session.user_id)
            await self.redis.hset(session_key, "status", session.status)
            await self.redis.hset(session_key, "start_time", str(session.start_time))
            
            # 如果有选中的用户名称，存储到会话信息中
            if selected_username:
                await self.redis.hset(session_key, "selected_username", selected_username)
            
            logger.info(f"已开始新会话: {session_id}, 会话键: {session_key}")
            return session_id
            
        except Exception as e:
            logger.error(f"开始会话时出错: {str(e)}")
            raise
        
    async def end_session(self, session_id: str, user_id: str) -> bool:
        """结束会话"""
        try:
            # 使用新的标准格式
            session_key = f"session:{session_id}"
            
            # 检查会话是否存在
            if not await self.redis.exists(session_key):
                logger.warning(f"会话不存在: {session_id}")
                return False
                
            # 更新会话状态
            await self.redis.hset(session_key, "status", "completed")
            await self.redis.hset(session_key, "end_time", str(datetime.now()))
            
            logger.info(f"结束会话: {session_id}, 用户: {user_id}")
            return True
        except Exception as e:
            logger.error(f"结束会话失败: {str(e)}")
            raise
    
    async def get_role_info(self, role_id: str) -> dict:
        """从MongoDB获取角色信息
        
        Args:
            role_id: 角色ID
            
        Returns:
            角色信息字典，失败时返回None
        """
        logger = logging.getLogger("app.memory.buffer_memory")
        logger.info(f"正在从MongoDB获取角色信息: {role_id}")
        
        try:
            from app.database.connection import get_database
            from bson.objectid import ObjectId
            
            # 尝试将role_id转换为ObjectId
            logger.info(f"角色ID类型: {type(role_id)}, 值: {role_id}")
            
            try:
                # 尝试将role_id转换为ObjectId
                object_role_id = ObjectId(role_id)
                logger.info(f"成功创建ObjectId: {object_role_id}")
                
                # 获取数据库连接
                db = await get_database()
                if db is not None:
                    logger.info(f"成功获取数据库连接, 开始查询角色: collection={db.roles.name}")
                    
                    # 打印更详细的查询信息
                    logger.info(f"执行查询: db.roles.find_one({{'_id': ObjectId('{role_id}')}})")
                    
                    # 执行查询
                    role_info = await db.roles.find_one({"_id": object_role_id})
                    logger.info(f"MongoDB查询结果: {role_info}")
                    
                    if not role_info:
                        logger.warning(f"未找到角色: _id={object_role_id}")
                        # 尝试列出几个角色ID作为参考
                        try:
                            cursor = db.roles.find({}).limit(5)
                            sample_roles = await cursor.to_list(length=5)
                            sample_info = [{"_id": str(r["_id"]), "name": r.get("name", "无名称")} for r in sample_roles] if sample_roles else []
                            logger.debug(f"数据库中的样例角色: {sample_info}")
                        except Exception as sample_error:
                            logger.debug(f"获取样例角色失败: {str(sample_error)}")
                    
                    return role_info
                else:
                    logger.error("无法获取数据库连接")
                    return None
            except Exception as oid_error:
                logger.error(f"ObjectId转换失败: {str(oid_error)}, 原始值: {role_id}")
                return None
        except Exception as error:
            logger.error(f"获取角色信息失败: {str(error)}", exc_info=True)
            return None
            
    async def get_user_info(self, user_id: str) -> dict:
        """从MongoDB获取用户信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户信息字典，失败时返回None
        """
        logger = logging.getLogger("app.memory.buffer_memory")
        logger.info(f"正在从MongoDB获取用户信息: {user_id}")
        
        try:
            from app.database.connection import get_database
            from bson.objectid import ObjectId
            
            # 尝试将user_id转换为ObjectId
            logger.info(f"用户ID类型: {type(user_id)}, 值: {user_id}")
            
            try:
                # 尝试将user_id转换为ObjectId
                object_user_id = ObjectId(user_id)
                logger.info(f"成功创建用户ObjectId: {object_user_id}")
                
                # 获取数据库连接
                db = await get_database()
                if db is not None:
                    logger.info(f"成功获取数据库连接, 开始查询用户: collection={db.users.name}")
                    
                    # 打印更详细的查询信息
                    logger.info(f"执行查询: db.users.find_one({{'_id': ObjectId('{user_id}')}})")
                    
                    # 执行查询
                    user_info = await db.users.find_one({"_id": object_user_id})
                    logger.info(f"MongoDB用户查询结果: {user_info}")
                    
                    if not user_info:
                        logger.warning(f"未找到用户: _id={object_user_id}")
                    
                    return user_info
                else:
                    logger.error("无法获取数据库连接")
                    return None
            except Exception as oid_error:
                logger.error(f"用户ObjectId转换失败: {str(oid_error)}, 原始值: {user_id}")
                return None
        except Exception as error:
            logger.error(f"获取用户信息失败: {str(error)}", exc_info=True)
            return None

    def _get_message_key(self, session_id: str, user_id: str) -> str:
        """
        构建消息键
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            str: Redis键名
        """
        return f"messages:{user_id}:{session_id}"
        
    async def add_message(self, session_id: str, user_id: str, message: Message) -> None:
        """添加消息到会话历史
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            message: 消息对象
        """
        try:
            # 使用新的标准格式
            session_key = f"session:{session_id}"
            message_key = self._get_message_key(session_id, user_id)
            
            # 检查会话是否存在
            if not await self.redis.exists(session_key):
                raise ValueError(f"会话不存在: {session_id}")
                
            # 序列化消息
            message_data = {
                "id": message.id,
                "content": message.content,
                "role": message.role,
                "timestamp": str(message.timestamp)
            }
            
            # 添加到消息列表
            await self.redis.rpush(message_key, json.dumps(message_data))
            
            # 如果历史记录超过最大轮数，删除最旧的消息
            history_length = await self.redis.llen(message_key)
            if history_length > self.max_rounds * 2:  # 每轮包含用户和AI的消息
                await self.redis.lpop(message_key)
                
            logger.info(f"已添加消息到会话 {session_id}: {message.content[:50]}...")
            
        except Exception as e:
            logger.error(f"添加消息时出错: {str(e)}")
            raise
            
    async def get_session_history(self, session_id: str, user_id: str) -> Tuple[ChatSession, list]:
        """获取会话历史
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            (会话对象, 消息列表)
        """
        try:
            # 使用新的标准格式
            session_key = f"session:{session_id}"
            message_key = self._get_message_key(session_id, user_id)
            
            # 检查会话是否存在
            if not await self.redis.exists(session_key):
                raise ValueError(f"会话不存在: {session_id}")
                
            # 获取会话信息
            session_data = await self.redis.hgetall(session_key)
            session = ChatSession(
                id=session_data["id"],
                user_id=session_data["user_id"],
                status=session_data["status"],
                start_time=datetime.fromisoformat(session_data["start_time"])
            )
            
            # 获取历史消息
            history = []
            message_data_list = await self.redis.lrange(message_key, 0, -1)
            for msg_data in message_data_list:
                msg_dict = json.loads(msg_data)
                history.append(Message(
                    id=msg_dict["id"],
                    content=msg_dict["content"],
                    role=msg_dict["role"],
                    timestamp=datetime.fromisoformat(msg_dict["timestamp"])
                ))
                
            logger.info(f"已获取会话 {session_id} 的历史记录，共 {len(history)} 条消息")
            return session, history
            
        except Exception as e:
            logger.error(f"获取会话历史时出错: {str(e)}")
            raise
            
    async def clear_session(self, session_id: str, user_id: str) -> None:
        """清除会话数据
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
        """
        try:
            # 使用新的标准格式
            session_key = f"session:{session_id}"
            message_key = self._get_message_key(session_id, user_id)
            
            # 删除会话和消息记录（移除内存键）
            await self.redis.delete(session_key, message_key)
            logger.info(f"已清除会话数据: {session_id}")
            
        except Exception as e:
            logger.error(f"清除会话数据时出错: {str(e)}")
            raise
            
    async def get_session_messages(self, session_id: str, user_id: str) -> list:
        """获取会话消息列表
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            消息列表
        """
        try:
            # 获取消息列表
            message_key = self._get_message_key(session_id, user_id)
            message_data_list = await self.redis.lrange(message_key, 0, -1)
            
            # 解析消息
            messages = []
            for msg_data in message_data_list:
                try:
                    msg_dict = json.loads(msg_data)
                    messages.append(msg_dict)
                except json.JSONDecodeError:
                    logger.error(f"无法解析消息: {msg_data}")
            
            logger.info(f"获取会话 {session_id} 的消息列表，共 {len(messages)} 条")
            return messages
            
        except Exception as e:
            logger.error(f"获取会话消息列表时出错: {str(e)}")
            return []
            
    async def get_session_info(self, session_id: str, user_id: str) -> dict:
        """获取会话信息
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            会话信息
        """
        try:
            # 使用新的标准格式
            session_key = f"session:{session_id}"
            
            # 检查会话是否存在
            if not await self.redis.exists(session_key):
                logger.warning(f"会话不存在: {session_id}")
                return None
                
            # 获取会话信息
            session_data = await self.redis.hgetall(session_key)
            logger.info(f"获取会话信息: {session_id}")
            return session_data
            
        except Exception as e:
            logger.error(f"获取会话信息时出错: {str(e)}")
            return None
            
    async def count_tokens(self, session_id: str, user_id: str) -> int:
        """计算会话中的token数量
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            token数量
        """
        try:
            # 使用新的消息key格式
            message_key = self._get_message_key(user_id, session_id)
            
            # 获取所有消息
            messages = await self.redis.lrange(message_key, 0, -1)
            
            if not messages:
                return 0
                
            # 解析消息并计算token
            token_count = 0
            for message_json in messages:
                message = json.loads(message_json)
                content = message.get("content", "")
                token_count += len(content) // 4  # 简单估算
                
            logger.info(f"会话 {session_id} 的token数量: {token_count}")
            return token_count
            
        except Exception as e:
            logger.error(f"计算token数量时出错: {str(e)}")
            return 0
    
    async def list_active_sessions(self, user_id: str) -> list:
        """列出用户的所有活跃会话
        
        Args:
            user_id: 用户ID
            
        Returns:
            活跃会话列表
        """
        try:
            # 获取所有会话
            sessions = await self.redis.keys(f"session:{user_id}:*")
            
            active_sessions = []
            for session_key in sessions:
                # 获取会话信息
                session_data = await self.redis.hgetall(session_key)
                if not session_data:
                    continue
                    
                # 检查会话状态
                if session_data.get("status") == "active":
                    session_id = session_key.decode().split(":")[-1]
                    active_sessions.append({
                        "session_id": session_id,
                        "start_time": session_data.get("start_time", ""),
                        "message_count": await self.redis.llen(f"history:{user_id}:{session_id}")
                    })
                    
            logger.info(f"用户 {user_id} 的活跃会话数量: {len(active_sessions)}")
            return active_sessions
            
        except Exception as e:
            logger.error(f"获取活跃会话列表时出错: {str(e)}")
            raise

    async def update_role_names(self, user_id: str, session_id: str) -> dict:
        """
        更新会话中所有消息的角色名称
        根据role_id字段从MongoDB中获取最新的角色名称
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            
        Returns:
            dict: 更新结果统计
        """
        try:
            message_key = self._get_message_key(session_id, user_id)
            
            # 检查会话是否存在
            if not self.redis.exists(message_key):
                logger.warning(f"尝试更新不存在的会话: {session_id}")
                return {"status": "error", "message": "会话不存在", "updated": 0, "total": 0}
            
            # 获取会话中的所有消息
            message_jsons = self.redis.lrange(message_key, 0, -1)
            if not message_jsons:
                logger.warning(f"会话 {session_id} 没有消息")
                return {"status": "success", "message": "会话没有消息", "updated": 0, "total": 0}
            
            # 统计需要更新的消息
            total_messages = len(message_jsons)
            updated_count = 0
            role_cache = {}  # 缓存已查询过的角色信息
            
            logger.info(f"开始更新会话 {session_id} 的角色名称，共 {total_messages} 条消息")
            
            # 导入数据库连接
            from app.database.connection import get_database
            from bson.objectid import ObjectId
            db = await get_database()
            
            # 检查数据库连接
            if not db:
                logger.error("无法获取数据库连接")
                return {"status": "error", "message": "数据库连接失败", "updated": 0, "total": total_messages}
            
            # 处理每条消息
            for i, message_json in enumerate(message_jsons):
                try:
                    # 解析消息
                    message = json.loads(message_json)
                    
                    # 检查是否有角色ID
                    role_id = message.get("role_id")
                    
                    if not role_id or role_id == "null" or (isinstance(role_id, str) and not role_id.strip()):
                        logger.debug(f"消息 #{i+1} 没有有效的角色ID")
                        continue
                    
                    # 记录原始角色名
                    original_role = message.get("role", "")
                    logger.info(f"消息 #{i+1}: 角色ID={role_id}, 原始角色名={original_role}")
                    
                    # 尝试从缓存中获取角色信息
                    if role_id in role_cache:
                        role_info = role_cache[role_id]
                        logger.debug(f"使用缓存的角色信息: {role_id} -> {role_info}")
                    else:
                        # 从MongoDB获取角色信息
                        try:
                            object_role_id = ObjectId(role_id)
                            role_info = await db.roles.find_one({"_id": object_role_id})
                            
                            # 缓存角色信息
                            role_cache[role_id] = role_info
                            logger.info(f"从MongoDB获取到角色信息: {role_id} -> {role_info}")
                        except Exception as e:
                            logger.error(f"获取角色信息失败: {str(e)}")
                            continue
                    
                    # 更新角色名称
                    if role_info and "name" in role_info and role_info["name"]:
                        new_role_name = role_info["name"]
                        
                        # 检查角色名是否需要更新
                        if new_role_name != original_role:
                            # 更新角色名称
                            message["role"] = new_role_name
                            
                            # 将更新后的消息保存回Redis
                            updated_json = json.dumps(message)
                            self.redis.lset(message_key, i, updated_json)
                            
                            updated_count += 1
                            logger.info(f"已更新消息 #{i+1} 的角色名称: {original_role} -> {new_role_name}")
                        else:
                            logger.debug(f"消息 #{i+1} 的角色名已是最新: {original_role}")
                    elif role_info:
                        logger.warning(f"角色 {role_id} 没有name字段或name为空: {role_info}")
                        
                        # 尝试其他可能的名称字段
                        for field in ['title', 'display_name', 'nickname']:
                            if field in role_info and role_info[field]:
                                message["role"] = role_info[field]
                                
                                # 将更新后的消息保存回Redis
                                updated_json = json.dumps(message)
                                self.redis.lset(message_key, i, updated_json)
                                
                                updated_count += 1
                                logger.info(f"已使用替代字段 {field} 更新消息 #{i+1} 的角色名称: {original_role} -> {role_info[field]}")
                                break
                    else:
                        logger.warning(f"未找到角色信息: {role_id}")
                except json.JSONDecodeError:
                    logger.error(f"解析消息 #{i+1} 时出错，可能是JSON格式不正确")
                except Exception as e:
                    logger.error(f"处理消息 #{i+1} 时出错: {str(e)}")
            
            logger.info(f"会话 {session_id} 角色名称更新完成: 共 {total_messages} 条消息，已更新 {updated_count} 条")
            
            return {
                "status": "success",
                "message": f"已更新 {updated_count}/{total_messages} 条消息的角色名称",
                "updated": updated_count,
                "total": total_messages
            }
        except Exception as e:
            logger.error(f"更新角色名称失败: {str(e)}", exc_info=True)
            return {"status": "error", "message": str(e), "updated": 0, "total": 0} 