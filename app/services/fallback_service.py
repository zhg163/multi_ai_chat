#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
后备服务 - 提供当主要服务不可用时的替代功能
"""

import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class FallbackService:
    """
    后备服务类 - 当主要服务不可用时提供基本功能
    """
    
    def __init__(self):
        """初始化后备服务"""
        self.logger = logging.getLogger("fallback_service")
        self.logger.info("后备服务初始化")
        
    async def get_default_user(self) -> Dict[str, Any]:
        """
        获取默认用户信息
        
        Returns:
            Dict[str, Any]: 默认用户信息
        """
        return {
            "id": "default_user",
            "name": "默认用户",
            "role": "user",
            "is_active": True
        }
        
    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取用户信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            Optional[Dict[str, Any]]: 用户信息，如果不存在则返回None
        """
        self.logger.warning(f"使用后备服务获取用户信息: {user_id}")
        if user_id == "default_user":
            return await self.get_default_user()
        return None
        
    async def list_users(self, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """
        获取用户列表
        
        Args:
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            List[Dict[str, Any]]: 用户列表
        """
        self.logger.warning("使用后备服务获取用户列表")
        return [await self.get_default_user()]
        
    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建用户
        
        Args:
            user_data: 用户数据
            
        Returns:
            Dict[str, Any]: 创建的用户信息
        """
        self.logger.warning("使用后备服务创建用户，未实际创建")
        return {**user_data, "id": "temporary_user_id"}
        
    async def update_user(self, user_id: str, update_data: Dict[str, Any]) -> bool:
        """
        更新用户信息
        
        Args:
            user_id: 用户ID
            update_data: 更新数据
            
        Returns:
            bool: 是否更新成功
        """
        self.logger.warning(f"使用后备服务更新用户 {user_id}，未实际更新")
        return True
        
    async def delete_user(self, user_id: str) -> bool:
        """
        删除用户
        
        Args:
            user_id: 用户ID
            
        Returns:
            bool: 是否删除成功
        """
        self.logger.warning(f"使用后备服务删除用户 {user_id}，未实际删除")
        return True

# 全局单例
fallback_service = FallbackService() 