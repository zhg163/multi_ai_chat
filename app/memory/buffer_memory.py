"""
短期记忆模块 - 使用Redis实现的缓冲记忆
"""

import redis
import json
import time
import logging
from app.memory.schemas import Message, ChatSession
from app.config import memory_settings
from typing import Dict
import os
import uuid

logger = logging.getLogger(__name__)

class ShortTermMemory:
    """
    短期记忆实现类，基于Redis
    使用Redis列表存储对话历史，按消息添加顺序排列
    """
    
    def __init__(self, redis_client: redis.Redis, max_rounds: int = 4):
        """初始化BufferMemory"""
        self.redis = redis_client
        self.max_rounds = max_rounds
        
    def start_session(self, session_id: str, user_id: str, selected_username: str = None) -> str:
        """开始一个新的会话
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            selected_username: 前端选中的用户名称（可选）
        
        Returns:
            会话ID
        """
        try:
            session_key = f"session:{user_id}:{session_id}"
            session = ChatSession(id=session_id, user_id=user_id)
            
            # 存储会话信息
            self.redis.hset(session_key, "id", session.id)
            self.redis.hset(session_key, "user_id", session.user_id)
            self.redis.hset(session_key, "status", session.status)
            self.redis.hset(session_key, "start_time", str(session.start_time))
            
            # 如果有选中的用户名称，存储到会话信息中
            if selected_username:
                self.redis.hset(session_key, "selected_username", selected_username)
                logger.info(f"会话关联用户名称: {selected_username}")
            
            logger.info(f"开始新会话: {session_id}, 用户: {user_id}")
            return session_id
        except Exception as e:
            logger.error(f"创建会话失败: {str(e)}")
            raise
        
    def end_session(self, session_id: str, user_id: str) -> bool:
        """结束会话"""
        try:
            session_key = f"session:{user_id}:{session_id}"
            
            # 检查会话是否存在
            if not self.redis.exists(session_key):
                logger.warning(f"会话不存在: {session_id}")
                return False
                
            # 更新会话状态
            self.redis.hset(session_key, "status", "completed")
            self.redis.hset(session_key, "end_time", str(time.time()))
            
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

    async def add_message(self, session_id: str, user_id: str, role: str, content: str, role_id: str = None, message_id: str = None) -> tuple:
        """
        添加消息到会话
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            role: 消息角色（user/assistant/system）
            content: 消息内容
            role_id: 角色ID，对应MongoDB roles表中的_id
            message_id: 消息ID，如果不提供则自动生成
            
        Returns:
            (bool, dict): 添加是否成功，以及可能需要归档的消息
        """
        try:
            logger.info(f"【role跟踪】ShortTermMemory - 接收到role='{role}', roleid='{role_id}'")
            message_key = f"messages:{user_id}:{session_id}"
            
            # 处理roleid为字符串"null"的情况
            if role_id == "null" or not role_id or (isinstance(role_id, str) and not role_id.strip()):
                role_id = None
                logger.warning("接收到无效的role_id值，已将其转换为None")
                logger.info("【role跟踪】无效role_id，设置为None")
            
            # 记录传入的role_id
            logger.info(f"接收到的原始roleid: {role_id}, 类型: {type(role_id).__name__}")
            
            # 创建基本消息对象
            message = Message(role=role, content=content)
            original_role = role  # 保存原始角色名用于日志记录
            logger.info(f"【role跟踪】消息对象创建 - role='{message.role}', original_role='{original_role}'")
            
            # 设置roleid，确保值不是null字符串
            if role_id:
                message.roleid = role_id 
                logger.info(f"设置message.roleid为: {role_id}")
            else:
                message.roleid = None
                logger.info(f"roleid为空值，设置message.roleid为: None")
            
            # 先将消息转换为字典，确保message_dict已定义
            message_dict = message.dict()
            logger.info(f"【role跟踪】序列化前 - role='{message_dict.get('role')}', roleid='{message_dict.get('roleid')}'")
            
            # 设置消息ID
            if message_id:
                message_dict["message_id"] = message_id
                logger.info(f"使用提供的消息ID: {message_id}")
            else:
                # 生成一个唯一的消息ID
                message_dict["message_id"] = f"msg_{uuid.uuid4()}"
                logger.info(f"生成新的消息ID: {message_dict['message_id']}")
            
            # 如果角色是"user"，尝试处理用户名称
            if original_role == "user":
                # 先检查是否为anonymous_user，如果是则尝试使用会话中存储的selected_username
                if user_id == "anonymous_user":
                    try:
                        # 获取会话信息
                        session_info = self.get_session_info(session_id, user_id)
                        if session_info and "selected_username" in session_info:
                            selected_username = session_info["selected_username"]
                            logger.info(f"从会话信息中获取到选中的用户名称: {selected_username}")
                            message_dict["role"] = selected_username
                            logger.info(f"【role跟踪】用户名称替换 - 从'{original_role}'到'{selected_username}'")
                        else:
                            logger.error("会话中没有选中的用户名称，但用户是匿名用户")
                            # 抛出异常，强制客户端选择用户名
                            raise ValueError("未选择用户，请先选择一个用户名再发送消息")
                    except ValueError:
                        # 重新抛出明确的错误
                        raise
                    except Exception as e:
                        logger.error(f"获取会话选中用户名称失败: {str(e)}")
                        raise ValueError("处理用户信息失败，请重新选择用户名再尝试")
                
                # 如果上面的处理没有替换角色名称，尝试从MongoDB获取用户信息
                if message_dict["role"] == original_role:
                    try:
                        user_info = await self.get_user_info(user_id)
                        logger.info(f"从数据库获取到用户信息: {user_info}")
                        
                        if user_info and "name" in user_info and user_info["name"]:
                            logger.info(f"使用用户中文名称: {user_info['name']}，替换原始角色: {original_role}")
                            message_dict["role"] = user_info["name"]
                            logger.info(f"【role跟踪】用户名称替换 - 从'{original_role}'到'{message_dict['role']}'")
                        else:
                            # 尝试其他可能的用户名称字段
                            for field in ['username', 'nickname', 'display_name']:
                                if user_info and field in user_info and user_info[field]:
                                    logger.info(f"使用用户字段{field}: {user_info[field]}，替换原始角色: {original_role}")
                                    message_dict["role"] = user_info[field]
                                    logger.info(f"【role跟踪】用户名称替换 - 从'{original_role}'到'{message_dict['role']}'")
                                    break
                            else:
                                logger.info(f"【role跟踪】未找到合适的用户名称字段，保持为'{original_role}'")
                    except Exception as e:
                        logger.error(f"获取用户信息失败: {str(e)}")
                        logger.info(f"【role跟踪】获取用户信息失败，保持为'{original_role}'")
            # 如果有角色ID，尝试查询角色信息并更新角色名称
            elif role_id:
                try:
                    role_info = await self.get_role_info(role_id)
                    logger.info(f"从数据库获取到角色信息: {role_info}")

                    # 修改此处逻辑，允许角色被替换
                    if role_info and "name" in role_info:
                        logger.info(f"使用角色中文名称: {role_info['name']}，替换原始角色: {original_role}")
                        message_dict["role"] = role_info["name"]
                        logger.info(f"【role跟踪】角色名称替换 - 从'{original_role}'到'{message_dict['role']}'")
                    else:
                        logger.info(f"【role跟踪】角色信息中没有name字段，保持为'{original_role}'")
                except Exception as e:
                    logger.error(f"获取角色信息失败: {str(e)}")
                    logger.info(f"【role跟踪】获取角色信息失败，保持为'{original_role}'")
            else:
                logger.info(f"【role跟踪】无需替换角色名称，保持为'{original_role}'")
            
            # 获取当前消息数量
            message_count = self.redis.llen(message_key)
            
            # 获取要归档的消息
            oldest_message = None
            should_archive = message_count >= (self.max_rounds * 2)  # 一轮包含用户和助手两条消息
            
            if should_archive:
                # 获取最旧的消息(最后一条)用于归档
                oldest_json = self.redis.lindex(message_key, -1)
                if oldest_json:
                    try:
                        oldest_message = json.loads(oldest_json)
                        logger.info(f"准备归档消息: {oldest_message.get('role', '未知')}, 内容前20字符: {oldest_message.get('content', '')[:20]}")
                    except json.JSONDecodeError:
                        logger.error("解析最旧消息时出错，可能是JSON格式不正确")
            
            # 确保roleid字段存在且不是空字符串或"null"字符串
            if not message_dict.get("roleid") or message_dict.get("roleid") == "null" or message_dict.get("roleid") == "":
                message_dict["roleid"] = None
                logger.info("检测到roleid为空或无效，已将其设置为None")
            
            # 关键修改：为所有角色提供二次替换机会，不再限制只有assistant角色
            if role_id and message_dict.get("role") == original_role and original_role != "user":
                logger.info(f"【role跟踪】检测到角色需要二次替换，当前role='{message_dict.get('role')}'")
                try:
                    from app.database.connection import get_database
                    from bson.objectid import ObjectId
                    
                    db = await get_database()
                    if db is not None:
                        role_info_second = await db.roles.find_one({"_id": ObjectId(role_id)})
                        logger.info(f"【role跟踪】二次查询结果 - role_info={role_info_second}")
                        if role_info_second and "name" in role_info_second and role_info_second["name"]:
                            role_name = role_info_second["name"]
                            logger.info(f"【角色替换】将'{original_role}'替换为实际角色名称: {role_name}")
                            logger.info(f"【role跟踪】最终替换 - 从'{message_dict['role']}'到'{role_name}'")
                            message_dict["role"] = role_name
                except Exception as replace_error:
                    logger.error(f"尝试替换角色名称时出错: {str(replace_error)}")
                    logger.info(f"【role跟踪】二次替换异常，保持原值'{message_dict.get('role')}'")
            
            # 确保role字段是有效值    
            if not message_dict.get("role"):
                logger.warning(f"发现role字段为空值，恢复为原始值: {original_role}")
                logger.info(f"【role跟踪】role字段为空，恢复为原始值'{original_role}'")
                message_dict["role"] = original_role
                
            # 记录最终保存的数据    
            logger.info(f"最终保存到Redis的消息数据: {message_dict}")
            logger.info(f"【role跟踪】最终保存到Redis - role='{message_dict.get('role')}', roleid='{message_dict.get('roleid')}'")
            
            # 序列化消息并添加到Redis
            message_json = json.dumps(message_dict)
            result = self.redis.lpush(message_key, message_json)
            
            if not result:
                logger.error(f"Redis lpush返回失败: {result}")
                logger.info("【role跟踪】Redis保存失败")
                return False, None
            
            # 二次检查：如果角色名称仍为原始值，尝试更新它（非user角色）
            if message_dict["role"] == original_role and role_id and original_role != "user":
                try:
                    updated_dict = message_dict.copy()
                    updated_dict["role"] = role_info["name"] if role_info and "name" in role_info else original_role
                    if updated_dict["role"] != original_role:
                        updated_json = json.dumps(updated_dict)
                        self.redis.lset(message_key, 0, updated_json)
                        logger.info(f"【角色替换】后置更新 - 将'{original_role}'替换为'{updated_dict['role']}'")
                except Exception as e:
                    logger.error(f"后置更新角色名称失败: {str(e)}")
                
            # 维持队列大小
            self.redis.ltrim(message_key, 0, (self.max_rounds * 2) - 1)
            
            logger.info(f"已添加消息到会话 {session_id}: role={message.role}, roleid={message.roleid or '无'}, content前20字符={content[:20]}...")
            logger.info(f"【role跟踪】消息成功保存到Redis")
            
            # 返回添加状态和需要归档的消息
            return (True, oldest_message) if should_archive else (True, None)
        except Exception as e:
            logger.error(f"添加消息失败: {str(e)}", exc_info=True)  # 添加完整堆栈信息
            logger.info(f"【role跟踪】处理消息异常：{str(e)}")
            return False, None
            
    async def add_message_with_retry(self, session_id: str, user_id: str, role: str, content: str, role_id: str = None, message_id: str = None, max_retries=3) -> tuple:
        """
        添加消息到会话，带重试机制
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            role: 消息角色
            content: 消息内容
            role_id: 角色ID
            message_id: 消息ID
            max_retries: 最大重试次数
            
        Returns:
            (bool, dict): 添加是否成功，以及可能需要归档的消息
        """
        # 初始化
        retries = 0
        result = False
        message = None
        
        # 重试循环
        while retries < max_retries and not result:
            if retries > 0:
                logger.info(f"正在尝试第{retries+1}次添加消息...")
                
            try:
                result, message = await self.add_message(
                    session_id=session_id,
                    user_id=user_id,
                    role=role,
                    content=content,
                    role_id=role_id,
                    message_id=message_id
                )
                
                if result:
                    logger.info(f"成功添加消息，尝试次数: {retries+1}")
                    break
                    
            except redis.ConnectionError as e:
                retries += 1
                logger.warning(f"Redis连接错误，尝试重新连接 ({retries}/{max_retries}): {str(e)}")
                time.sleep(0.5)  # 短暂延迟后重试
                # 尝试重新初始化连接
                try:
                    self.redis = redis.Redis(
                        host=os.getenv("REDIS_HOST", "localhost"),
                        port=int(os.getenv("REDIS_PORT", "6378")),
                        password=os.getenv("REDIS_PASSWORD", "!qaz2wsX"),
                        decode_responses=True
                    )
                    # 测试连接
                    self.redis.ping()
                except Exception as conn_error:
                    logger.error(f"Redis重连失败: {str(conn_error)}")
                
        # 如果重试失败，记录错误并返回失败
        logger.error(f"添加消息到Redis失败，已重试{max_retries}次")
        return False, None
        
    def get_session_messages(self, session_id: str, user_id: str) -> list:
        """获取会话的所有消息，按时间顺序排列（旧->新）"""
        try:
            message_key = f"messages:{user_id}:{session_id}"
            
            # 获取所有消息
            messages_json = self.redis.lrange(message_key, 0, -1)
            
            # 解析消息并反转列表（使其按时间顺序排列）
            messages = [json.loads(msg) for msg in messages_json]
            messages.reverse()  # 由于lpush，我们需要反转列表
            
            return messages
        except Exception as e:
            logger.error(f"获取会话消息失败: {str(e)}")
            return []
            
    def get_session_info(self, session_id: str, user_id: str) -> dict:
        """获取会话信息"""
        try:
            session_key = f"session:{user_id}:{session_id}"
            
            # 检查会话是否存在
            if not self.redis.exists(session_key):
                logger.warning(f"会话不存在: {session_id}")
                return None
                
            # 获取会话信息
            session_data = self.redis.hgetall(session_key)
            
            return session_data
        except Exception as e:
            logger.error(f"获取会话信息失败: {str(e)}")
            return None
            
    def count_tokens(self, session_id: str, user_id: str) -> int:
        """估算会话中的token数量"""
        try:
            messages = self.get_session_messages(session_id, user_id)
            if not messages:
                return 0
                
            # 简单估算：假设平均每个字符是1.5个token
            text = ""
            for msg in messages:
                text += msg["content"]
                
            return int(len(text) * 1.5)  # 简单估算
        except Exception as e:
            logger.error(f"计算token数量失败: {str(e)}")
            return 0
    
    def list_active_sessions(self, user_id: str) -> list:
        """列出用户的所有活跃会话"""
        try:
            # 查找所有会话键
            pattern = f"session:{user_id}:*"
            session_keys = self.redis.keys(pattern)
            
            active_sessions = []
            for key in session_keys:
                # 获取会话状态
                status = self.redis.hget(key, "status")
                if status == "active":
                    session_id = key.split(":")[-1]
                    session_data = self.redis.hgetall(key)
                    active_sessions.append(session_data)
            
            return active_sessions
        except Exception as e:
            logger.error(f"列出活跃会话失败: {str(e)}")
            return []

    async def update_role_names(self, user_id: str, session_id: str) -> dict:
        """
        更新会话中所有消息的角色名称
        根据roleid字段从MongoDB中获取最新的角色名称
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            
        Returns:
            dict: 更新结果统计
        """
        try:
            message_key = f"messages:{user_id}:{session_id}"
            
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
                    role_id = message.get("roleid")
                    
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