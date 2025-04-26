"""角色模型和相关操作"""

from datetime import datetime
from bson import ObjectId
from typing import Dict, Any, List, Optional

from app.database.connection import Database

class Role:
    collection = None

    @classmethod
    def get_collection(cls):
        """获取角色集合"""
        return Database.db.roles

    @classmethod
    async def create(cls, name, description, personality, speech_style, 
                     keywords=None, temperature=0.7, 
                     prompt_template=None, system_prompt=None):
        """创建新角色"""
        now = datetime.utcnow()
        role_data = {
            "name": name,
            "description": description,
            "personality": personality,
            "speech_style": speech_style,
            "keywords": keywords or [],
            "temperature": temperature,
            "prompt_template": prompt_template or "",
            "system_prompt": system_prompt or "",
            "created_at": now,
            "updated_at": now,
            "active": True
        }
        
        result = await cls.get_collection().insert_one(role_data)
        role_data["_id"] = result.inserted_id
        return role_data

    @classmethod
    async def get_by_id(cls, role_id):
        """根据ID获取角色"""
        if isinstance(role_id, str):
            role_id = ObjectId(role_id)
        return await cls.get_collection().find_one({"_id":role_id})

    @classmethod
    async def get_by_name(cls, name):
        """根据名称获取角色"""
        return await cls.get_collection().find_one({"name": name})

    @classmethod
    async def update(cls, role_id, update_data):
        """更新角色信息"""
        if isinstance(role_id, str):
            role_id = ObjectId(role_id)
            
        # 自动更新updated_at字段
        update_data['updated_at'] = datetime.utcnow()
            
        result = await cls.get_collection().update_one(
            {"_id": role_id},
            {"$set": update_data}
        )
        return result.modified_count > 0

    @classmethod
    async def update_keywords(cls, role_id, keywords):
        """更新角色关键词"""
        if isinstance(role_id, str):
            role_id = ObjectId(role_id)
            
        result = await cls.get_collection().update_one(
            {"_id": role_id},
            {
                "$set": {
                    "keywords": keywords,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0

    @classmethod
    async def list_active(cls, limit=50, offset=0):
        """列出所有活跃角色"""
        cursor = cls.get_collection().find(
            {"active": True}
        ).sort("name", 1).skip(offset).limit(limit)
        
        return await cursor.to_list(length=limit)

    @classmethod
    async def deactivate(cls, role_id):
        """停用角色"""
        if isinstance(role_id, str):
            role_id = ObjectId(role_id)
            
        result = await cls.get_collection().update_one(
            {"_id": role_id},
            {
                "$set": {
                    "active": False,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0

    @classmethod
    async def activate(cls, role_id):
        """启用角色"""
        if isinstance(role_id, str):
            role_id = ObjectId(role_id)
            
        result = await cls.get_collection().update_one(
            {"_id": role_id},
            {
                "$set": {
                    "active": True,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0

    @classmethod
    async def add_prompt_template(cls, role_id, template_name, template_content, 
                                 is_default=False, description=""):
        """为角色添加提示模板"""
        if isinstance(role_id, str):
            role_id = ObjectId(role_id)
            
        template = {
            "id": str(ObjectId()),  # 为模板生成唯一ID
            "name": template_name,
            "content": template_content,
            "description": description,
            "created_at": datetime.utcnow()
        }
        
        # 如果是默认模板，更新role的default_template_id
        update_data = {
            "$push": {"prompt_templates": template},
            "$set": {"updated_at": datetime.utcnow()}
        }
        
        if is_default:
            update_data["$set"]["default_template_id"] = template["id"]
        
        result = await cls.get_collection().update_one(
            {"_id": role_id},
            update_data
        )
        
        return result.modified_count > 0, template["id"]

    @classmethod
    async def get_prompt_templates(cls, role_id):
        """获取角色的所有提示模板"""
        if isinstance(role_id, str):
            role_id = ObjectId(role_id)
            
        role = await cls.get_by_id(role_id)
        if not role:
            return []
            
        return role.get("prompt_templates", [])

    @classmethod
    async def update_prompt_template(cls, role_id, template_id, update_data):
        """更新角色提示模板"""
        if isinstance(role_id, str):
            role_id = ObjectId(role_id)
            
        # 构建更新的数组元素
        updates = {}
        for key, value in update_data.items():
            updates[f"prompt_templates.$.{key}"] = value
        
        updates["updated_at"] = datetime.utcnow()
        
        result = await cls.get_collection().update_one(
            {
                "_id": role_id,
                "prompt_templates.id": template_id
            },
            {"$set": updates}
        )
        
        return result.modified_count > 0

    @classmethod
    async def delete_prompt_template(cls, role_id, template_id):
        """删除角色提示模板"""
        if isinstance(role_id, str):
            role_id = ObjectId(role_id)
            
        # 获取角色信息，检查是否为默认模板
        role = await cls.get_by_id(role_id)
        if not role:
            return False
            
        # 如果要删除的是默认模板，需要重置default_template_id字段
        update_data = {
            "$pull": {"prompt_templates": {"id": template_id}},
            "$set": {"updated_at": datetime.utcnow()}
        }
        
        if role.get("default_template_id") == template_id:
            # 找到剩余模板中的第一个作为默认值，如果没有则设置为空
            templates = role.get("prompt_templates", [])
            remaining_templates = [t for t in templates if t["id"] != template_id]
            if remaining_templates:
                update_data["$set"]["default_template_id"] = remaining_templates[0]["id"]
            else:
                update_data["$set"]["default_template_id"] = ""
        
        result = await cls.get_collection().update_one(
            {"_id": role_id},
            update_data
        )
        
        return result.modified_count > 0

    @classmethod
    async def set_default_template(cls, role_id, template_id):
        """设置角色默认提示模板"""
        if isinstance(role_id, str):
            role_id = ObjectId(role_id)
            
        # 验证模板存在
        role = await cls.get_by_id(role_id)
        if not role:
            return False
            
        templates = role.get("prompt_templates", [])
        template_exists = any(t["id"] == template_id for t in templates)
        
        if not template_exists:
            return False
            
        result = await cls.get_collection().update_one(
            {"_id": role_id},
            {
                "$set": {
                    "default_template_id": template_id,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        return result.modified_count > 0

    def __init__(self, **kwargs):
        """初始化角色实例"""
        for key, value in kwargs.items():
            setattr(self, key, value) 