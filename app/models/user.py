from datetime import datetime
from bson import ObjectId
from typing import Dict, Any, List, Optional

from app.database.connection import Database

class User:
    collection = None

    @classmethod
    def get_collection(cls):
        """获取用户集合"""
        return Database.db.users

    @classmethod
    async def create(cls, username, preferences=None, favorite_roles=None):
        """创建新用户"""
        now = datetime.utcnow()
        user_data = {
            "username": username,
            "created_at": now,
            "last_active": now,
            "preferences": preferences or {"theme": "default", "notification_settings": {}},
            "favorite_roles": favorite_roles or []
        }
        
        result = await cls.get_collection().insert_one(user_data)
        user_data["_id"] = result.inserted_id
        return user_data

    @classmethod
    async def get_by_id(cls, user_id):
        """根据ID获取用户"""
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        return await cls.get_collection().find_one({"_id": user_id})

    @classmethod
    async def get_by_username(cls, username):
        """根据用户名获取用户"""
        return await cls.get_collection().find_one({"username": username})

    @classmethod
    async def update(cls, user_id, update_data):
        """更新用户信息"""
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
            
        # 自动更新last_active字段
        if 'last_active' not in update_data:
            update_data['last_active'] = datetime.utcnow()
            
        result = await cls.get_collection().update_one(
            {"_id": user_id},
            {"$set": update_data}
        )
        return result.modified_count > 0

    @classmethod
    async def delete(cls, user_id):
        """删除用户"""
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        result = await cls.get_collection().delete_one({"_id": user_id})
        return result.deleted_count > 0

    @classmethod
    async def add_favorite_role(cls, user_id, role_id):
        """添加收藏角色"""
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        if isinstance(role_id, str):
            role_id = ObjectId(role_id)
            
        result = await cls.get_collection().update_one(
            {"_id": user_id},
            {"$addToSet": {"favorite_roles": role_id}}
        )
        return result.modified_count > 0

    @classmethod
    async def remove_favorite_role(cls, user_id, role_id):
        """移除收藏角色"""
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        if isinstance(role_id, str):
            role_id = ObjectId(role_id)
            
        result = await cls.get_collection().update_one(
            {"_id": user_id},
            {"$pull": {"favorite_roles": role_id}}
        )
        return result.modified_count > 0 