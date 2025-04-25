#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import time
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import asyncio
import hashlib
from datetime import datetime
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import ENABLE_ROLE_BASED_CHAT, REDIS_ROLE_TTL, ROLE_USAGE_LIMIT, settings
from app.models.session_role import SessionRole
from app.utils.role_utils import parse_role_from_session, get_role_name, get_role_prompt, is_role_expired, normalize_role_data

from ..models.custom_session import CustomSession

# 配置日志记录器
logger = logging.getLogger(__name__)

class SessionRoleManager:
    """
    会话角色管理器 - 管理会话中的角色数据，提供角色查询、使用统计等功能
    """
    
    def __init__(self, redis_client=None):
        """初始化会话角色管理器"""
        # 初始化Redis客户端
        self.redis = redis_client
        
        # 如果没有提供Redis客户端，使用连接信息创建
        if self.redis is None:
            # 使用默认Redis DB 0，而不是依赖于配置中的REDIS_DB属性
            redis_db = getattr(settings, "REDIS_DB", 0)
            self.redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{redis_db}"
            
            # 处理密码配置，放在if self.redis is None内部
            if settings.REDIS_PASSWORD:
                self.redis_url = f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}/{redis_db}"
        
        # 初始化日志记录器
        self.logger = logging.getLogger("session_role_manager")
        
        self.custom_session = CustomSession()
    
    async def _ensure_redis_connected(self):
        """确保Redis连接已建立"""
        if self.redis is None:
            try:
                self.redis = await Redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                self.logger.info("已成功连接到Redis")
            except Exception as e:
                self.logger.error(f"连接Redis失败: {str(e)}")
                raise
    
    async def get_session_roles(self, session_id: str) -> List[Dict[str, Any]]:
        """
        获取会话的所有角色
        
        Args:
            session_id: 会话ID
            
        Returns:
            角色列表
        """
        try:
            await self._ensure_redis_connected()
            # 从Redis获取会话数据
            session_key = f"session:{session_id}"
            session_data = await self.redis.hgetall(session_key)
            
            if not session_data:
                self.logger.warning(f"找不到会话数据: {session_id}")
                return []
                
            # 解析角色数据
            roles_data = session_data.get("roles", "[]")
            
            # 尝试解析JSON
            try:
                roles = json.loads(roles_data)
                if not isinstance(roles, list):
                    self.logger.warning(f"角色数据格式不正确: {type(roles)}")
                    return []
                return roles
            except json.JSONDecodeError:
                self.logger.error(f"角色数据解析失败: {roles_data[:100]}...")
                return []
                
        except Exception as e:
            self.logger.error(f"获取会话角色时出错: {str(e)}")
            return []
    
    async def get_role_by_id(self, session_id: str, role_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取角色信息
        
        Args:
            session_id: 会话ID
            role_id: 角色ID
            
        Returns:
            角色信息字典，如果找不到则返回None
        """
        try:
            await self._ensure_redis_connected()
            # 从Redis获取会话数据
            session_key = f"session:{session_id}"
            session_data = await self.redis.hgetall(session_key)
            
            if not session_data:
                self.logger.warning(f"找不到会话数据: {session_id}")
                return None
                
            # 解析角色数据
            roles_data = session_data.get("roles", "[]")
            
            # 尝试解析JSON
            try:
                roles = json.loads(roles_data)
                if not isinstance(roles, list):
                    self.logger.warning(f"角色数据格式不正确: {type(roles)}")
                    return None
                    
                # 查找指定ID的角色
                for role in roles:
                    if role.get("role_id") == role_id:
                        return role
                        
                self.logger.warning(f"找不到指定的角色ID: {role_id}")
                return None
                
            except json.JSONDecodeError:
                self.logger.error(f"角色数据解析失败: {roles_data[:100]}...")
                return None
                
        except Exception as e:
            self.logger.error(f"根据ID获取角色时出错: {str(e)}")
            return None
            
    async def get_default_role(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取默认角色
        
        Args:
            session_id: 会话ID
            
        Returns:
            默认角色信息字典，如果找不到则返回None
        """
        try:
            await self._ensure_redis_connected()
            # 从Redis获取会话数据
            session_key = f"session:{session_id}"
            session_data = await self.redis.hgetall(session_key)
            
            if not session_data:
                self.logger.warning(f"找不到会话数据: {session_id}")
                return None
                
            # 解析角色数据
            roles_data = session_data.get("roles", "[]")
            
            # 尝试解析JSON
            try:
                roles = json.loads(roles_data)
                if not isinstance(roles, list):
                    self.logger.warning(f"角色数据格式不正确: {type(roles)}")
                    return None
                    
                # 返回第一个角色作为默认角色
                if roles:
                    return roles[0]
                    
                self.logger.warning(f"会话中没有定义角色: {session_id}")
                return None
                
            except json.JSONDecodeError:
                self.logger.error(f"角色数据解析失败: {roles_data[:100]}...")
                return None
                
        except Exception as e:
            self.logger.error(f"获取默认角色时出错: {str(e)}")
            return None
            
    async def update_role_usage_count(self, session_id: str, role_id: str) -> bool:
        """
        更新角色使用次数
        
        Args:
            session_id: 会话ID
            role_id: 角色ID
            
        Returns:
            是否更新成功
        """
        try:
            await self._ensure_redis_connected()
            # 从Redis获取会话数据
            session_key = f"session:{session_id}"
            session_data = await self.redis.hgetall(session_key)
            
            if not session_data:
                self.logger.warning(f"找不到会话数据: {session_id}")
                return False
                
            # 解析角色数据
            roles_data = session_data.get("roles", "[]")
            
            # 尝试解析JSON
            try:
                roles = json.loads(roles_data)
                if not isinstance(roles, list):
                    self.logger.warning(f"角色数据格式不正确: {type(roles)}")
                    return False
                    
                # 更新指定角色的使用次数
                updated = False
                for role in roles:
                    if role.get("role_id") == role_id or role.get("id") == role_id:
                        role["usage_count"] = role.get("usage_count", 0) + 1
                        self.logger.info(f"更新角色 {role_id} 使用次数，当前: {role['usage_count']}")
                        updated = True
                        break
                
                if not updated:
                    self.logger.warning(f"找不到要更新的角色: {role_id}")
                    return False
                        
                # 将更新后的角色数据保存回Redis
                await self.redis.hset(session_key, "roles", json.dumps(roles))
                self.logger.info(f"已更新会话 {session_id} 的角色使用次数")
                return True
                
            except json.JSONDecodeError:
                self.logger.error(f"角色数据解析失败: {roles_data[:100]}...")
                return False
                
        except Exception as e:
            self.logger.error(f"更新角色使用次数时出错: {str(e)}")
            return False

    async def get_role(self, role_id: str) -> Optional[Dict[str, Any]]:
        """
        获取角色信息
        
        Args:
            role_id: 角色ID
            
        Returns:
            角色信息字典，如果找不到则返回None
        """
        if not ENABLE_ROLE_BASED_CHAT:
            self.logger.info("角色系统未启用")
            return None
            
        try:
            await self._ensure_redis_connected()
            # 从Redis获取角色数据
            role_key = f"role:{role_id}"
            role_data = await self.redis.get(role_key)
            
            if not role_data:
                self.logger.warning(f"找不到角色数据: {role_id}")
                
                # 回退策略：尝试从会话数据中查找角色
                self.logger.info(f"尝试从会话数据中查找角色: {role_id}")
                
                # 先获取所有会话键
                session_keys = await self.redis.keys("session:*")
                
                for key in session_keys:
                    session_data = await self.redis.hgetall(key)
                    if not session_data or "roles" not in session_data:
                        continue
                        
                    try:
                        roles = json.loads(session_data.get("roles", "[]"))
                        if not isinstance(roles, list):
                            continue
                            
                        # 查找匹配的角色
                        for role in roles:
                            if role.get("role_id") == role_id or role.get("id") == role_id:
                                self.logger.info(f"在会话 {key} 中找到角色 {role_id}")
                                
                                # 保存找到的角色到独立的角色存储中，以便将来使用
                                try:
                                    await self.redis.set(role_key, json.dumps(role))
                                    self.logger.info(f"已将角色 {role_id} 保存到 Redis")
                                except Exception as e:
                                    self.logger.warning(f"保存角色到Redis失败: {str(e)}")
                                    
                                return role
                    except json.JSONDecodeError:
                        continue
                        
                self.logger.warning(f"在所有会话中都找不到角色: {role_id}")
                return None
                
            # 解析角色数据
            try:
                role = json.loads(role_data)
                return role
            except json.JSONDecodeError:
                self.logger.error(f"角色数据解析失败: {role_data[:100]}...")
                return None
                
        except Exception as e:
            self.logger.error(f"获取角色信息时出错: {str(e)}")
            return None
            
    async def get_available_roles(self) -> List[Dict[str, Any]]:
        """
        获取所有可用角色
        
        Returns:
            角色信息字典列表
        """
        if not ENABLE_ROLE_BASED_CHAT:
            self.logger.info("角色系统未启用")
            return []
            
        try:
            await self._ensure_redis_connected()
            # 获取所有角色键
            role_keys = await self.redis.keys("role:*")
            
            roles = []
            for key in role_keys:
                role_data = await self.redis.get(key)
                if role_data:
                    try:
                        role = json.loads(role_data)
                        roles.append(role)
                    except json.JSONDecodeError:
                        self.logger.error(f"角色数据解析失败: {role_data[:100]}...")
                        
            return roles
            
        except Exception as e:
            self.logger.error(f"获取可用角色列表时出错: {str(e)}")
            return []
            
    async def create_or_update_role(self, role: SessionRole) -> bool:
        """
        创建或更新角色
        
        Args:
            role: 角色对象
            
        Returns:
            是否操作成功
        """
        if not ENABLE_ROLE_BASED_CHAT:
            self.logger.info("角色系统未启用")
            return False
            
        try:
            await self._ensure_redis_connected()
            # 准备角色数据
            role_data = {
                "role_id": role.role_id,
                "name": role.name,
                "description": role.description,
                "prompt": role.prompt,
                "created_at": role.created_at,
                "updated_at": int(time.time()),
                "usage_count": role.usage_count,
                "usage_limit": role.usage_limit,
                "expired_at": role.expired_at
            }
            
            # 保存到Redis
            role_key = f"role:{role.role_id}"
            await self.redis.set(role_key, json.dumps(role_data))
            
            # 设置过期时间
            if role.expired_at:
                ttl = role.expired_at - int(time.time())
                if ttl > 0:
                    await self.redis.expire(role_key, ttl)
                    
            return True
            
        except Exception as e:
            self.logger.error(f"创建或更新角色时出错: {str(e)}")
            return False
            
    async def delete_role(self, role_id: str) -> bool:
        """
        删除角色
        
        Args:
            role_id: 角色ID
            
        Returns:
            是否删除成功
        """
        if not ENABLE_ROLE_BASED_CHAT:
            self.logger.info("角色系统未启用")
            return False
            
        try:
            await self._ensure_redis_connected()
            # 删除角色数据
            role_key = f"role:{role_id}"
            result = await self.redis.delete(role_key)
            
            return result > 0
            
        except Exception as e:
            self.logger.error(f"删除角色时出错: {str(e)}")
            return False
    
    async def check_role_usage_limit(self, session_id: str, user_id: str, role_id: str) -> Tuple[bool, int]:
        """
        检查角色使用次数是否已达到限制
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            role_id: 角色ID
            
        Returns:
            (是否可以使用, 剩余次数)
        """
        if not ENABLE_ROLE_BASED_CHAT:
            # 角色系统未启用，无限制
            return True, -1
            
        # 判断是否设置了使用次数限制
        if ROLE_USAGE_LIMIT <= 0:
            # 无限制
            return True, -1
            
        try:
            await self._ensure_redis_connected()
            # 获取会话中角色的使用记录
            usage_key = f"role_usage:{session_id}:{user_id}:{role_id}"
            usage_count = await self.redis.zcard(usage_key)
            
            # 计算剩余次数
            remaining = ROLE_USAGE_LIMIT - usage_count
            if remaining <= 0:
                self.logger.warning(f"角色使用达到限制: {role_id}, 会话: {session_id}, 用户: {user_id}")
                return False, 0
            
            self.logger.info(f"角色使用检查通过: {role_id}, 剩余使用次数: {remaining}")
            return True, remaining
        except Exception as e:
            self.logger.error(f"检查角色使用限制失败: {str(e)}")
            # 出错时允许使用
            return True, -1

    async def get_role_for_session(self, session_id: str, role_id: Optional[str] = None) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        获取会话的角色信息
        
        Args:
            session_id: 会话ID
            role_id: 角色ID（可选）
            
        Returns:
            Tuple[bool, Optional[Dict[str, Any]], str]：
                - 是否成功
                - 角色信息字典
                - 错误消息（如果失败）
        """
        try:
            await self._ensure_redis_connected()
            # 获取会话数据
            session_data = await self.custom_session.get_session(session_id)
            if not session_data:
                return False, None, f"会话 {session_id} 不存在"
                
            # 解析角色信息
            role_info = parse_role_from_session(session_data, role_id)
            if not role_info:
                return False, None, f"找不到角色信息 (session_id={session_id}, role_id={role_id})"
                
            # 标准化角色数据
            normalized_role = normalize_role_data(role_info)
            
            return True, normalized_role, ""
            
        except Exception as e:
            error_msg = f"获取会话角色信息时出错: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg
            
    async def update_role_usage(self, session_id: str, role_id: str) -> bool:
        """
        更新角色使用次数
        
        Args:
            session_id: 会话ID
            role_id: 角色ID
            
        Returns:
            bool: 是否成功更新
        """
        try:
            await self._ensure_redis_connected()
            # 获取会话数据
            session_data = await self.custom_session.get_session(session_id)
            if not session_data:
                logger.error(f"更新角色使用次数失败: 会话 {session_id} 不存在")
                return False
                
            # 获取角色列表
            roles_data = session_data.get("roles", [])
            roles = []
            
            if isinstance(roles_data, str):
                try:
                    roles = json.loads(roles_data)
                except json.JSONDecodeError:
                    logger.error(f"角色数据解析失败: {roles_data[:100]}...")
                    return False
            elif isinstance(roles_data, list):
                roles = roles_data
            else:
                logger.error(f"无效的角色数据类型: {type(roles_data)}")
                return False
                
            # 更新指定角色的使用次数
            updated = False
            for role in roles:
                if role.get("role_id") == role_id:
                    role["usage_count"] = role.get("usage_count", 0) + 1
                    updated = True
                    break
                    
            if not updated:
                logger.warning(f"找不到要更新的角色: {role_id}")
                return False
                
            # 更新会话中的角色数据
            session_data["roles"] = roles
            success = await self.custom_session.update_session(session_id, session_data)
            
            return success
            
        except Exception as e:
            logger.error(f"更新角色使用次数时出错: {str(e)}")
            return False
            
    async def check_role_availability(self, session_id: str, role_id: Optional[str] = None) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        检查角色是否可用（存在且未过期）
        
        Args:
            session_id: 会话ID
            role_id: 角色ID（可选）
            
        Returns:
            Tuple[bool, str, Optional[Dict[str, Any]]]:
                - 是否可用
                - 错误消息（如果不可用）
                - 角色信息字典（如果可用）
        """
        # 获取角色信息
        success, role_info, error_msg = await self.get_role_for_session(session_id, role_id)
        if not success:
            return False, error_msg, None
            
        # 检查角色是否过期
        if is_role_expired(role_info):
            return False, f"角色 {get_role_name(role_info)} 已达到使用次数上限", None
            
        return True, "", role_info
        
    async def get_role_system_prompt(self, session_id: str, role_id: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
        """
        获取角色的系统提示
        
        Args:
            session_id: 会话ID
            role_id: 角色ID（可选）
            
        Returns:
            Tuple[bool, str, Optional[str]]：
                - 是否成功
                - 错误消息（如果失败）
                - 系统提示（如果成功）
        """
        # 如果提供了role_id，直接获取角色信息
        if role_id:
            role_info = await self.get_role(role_id)
            if not role_info:
                # 尝试从会话中获取角色信息
                success, role_info, error_msg = await self.get_role_for_session(session_id, role_id)
                if not success:
                    return False, error_msg, None
        else:
            # 否则从会话获取角色信息
            success, role_info, error_msg = await self.get_role_for_session(session_id, role_id)
            if not success:
                return False, error_msg, None
                
            # 从角色信息中提取系统提示（尝试多种可能的字段名）
        system_prompt = None
        possible_fields = ["system_prompt", "systemPrompt", "prompt", "content", "description"]
        
        for field in possible_fields:
            if field in role_info and role_info[field]:
                system_prompt = role_info[field]
                self.logger.info(f"从字段 '{field}' 获取到系统提示")
                break
            
        if not system_prompt:
            # 如果没找到系统提示，使用角色名称和描述构建一个基本提示
            name = role_info.get("name", role_info.get("role_name", "未知角色"))
            description = role_info.get("description", "")
            
            if description:
                system_prompt = f"你是{name}。{description}"
            else:
                system_prompt = f"你是{name}。请以这个角色的方式回答问题。"
            
            self.logger.info(f"未找到系统提示，使用自动生成的提示: {system_prompt[:50]}...")
            
        return True, "", system_prompt 