from datetime import datetime
from typing import List, Optional, Dict, Any
from bson.objectid import ObjectId
from app.database.connection import Database

class SessionStatus:
    """会话状态常量"""
    ACTIVE = "active"     # 当前活跃会话
    ARCHIVED = "archived" # 已归档会话
    DELETED = "deleted"   # 已删除但未永久移除

class Session:
    collection = None

    @classmethod
    def get_collection(cls):
        """获取会话集合"""
        if cls.collection is None:
            if Database.db is None:
                from tests.conftest import MockCollection
                cls.collection = MockCollection("sessions")
            else:
                cls.collection = Database.db.sessions
        return cls.collection

    @classmethod
    async def create(cls, user_id, title=None, description=None, 
                    role_ids=None, settings=None):
        """创建新会话"""
        now = datetime.utcnow()
        
        # 如果没有提供标题，创建默认标题
        if not title:
            title = f"会话 {now.strftime('%Y-%m-%d %H:%M')}"
            
        session_data = {
            "user_id": user_id,
            "title": title,
            "description": description or "",
            "role_ids": role_ids or [],
            "settings": settings or {
                "history_enabled": True,
                "context_window": 10,
                "memory_enabled": False,
                "system_prompt": ""
            },
            "status": "active",
            "created_at": now,
            "updated_at": now,
            "last_message_at": now
        }
        
        result = await cls.get_collection().insert_one(session_data)
        session_data["_id"] = result.inserted_id
        return session_data

    @classmethod
    async def get_by_id(cls, session_id, user_id=None):
        """根据ID获取会话"""
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)
            
        query = {"_id": session_id}
        if user_id:
            query["user_id"] = user_id
            
        return await cls.get_collection().find_one(query)

    @classmethod
    async def list_by_user(cls, user_id, limit=20, offset=0, status="active"):
        """获取用户的会话列表"""
        query = {"user_id": user_id}
        if status:
            query["status"] = status
            
        cursor = cls.get_collection().find(query)\
            .sort("last_message_at", -1)\
            .skip(offset)\
            .limit(limit)
            
        return await cursor.to_list(length=limit)

    @classmethod
    async def update(cls, session_id, update_data, user_id=None):
        """更新会话信息"""
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)
            
        # 自动更新updated_at字段
        update_data["updated_at"] = datetime.utcnow()
        
        query = {"_id": session_id}
        if user_id:
            query["user_id"] = user_id
            
        result = await cls.get_collection().update_one(
            query,
            {"$set": update_data}
        )
        return result.modified_count > 0

    @classmethod
    async def update_roles(cls, session_id, role_ids, user_id=None):
        """更新会话角色"""
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)
            
        query = {"_id": session_id}
        if user_id:
            query["user_id"] = user_id
            
        result = await cls.get_collection().update_one(
            query,
            {
                "$set": {
                    "role_ids": role_ids,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0

    @classmethod
    async def update_settings(cls, session_id, settings, user_id=None):
        """更新会话设置"""
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)
            
        query = {"_id": session_id}
        if user_id:
            query["user_id"] = user_id
            
        result = await cls.get_collection().update_one(
            query,
            {
                "$set": {
                    "settings": settings,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0

    @classmethod
    async def update_status(cls, session_id, status, user_id=None):
        """更新会话状态"""
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)
            
        valid_statuses = ["active", "archived", "deleted"]
        if status not in valid_statuses:
            raise ValueError(f"无效的会话状态: {status}")
            
        query = {"_id": session_id}
        if user_id:
            query["user_id"] = user_id
            
        result = await cls.get_collection().update_one(
            query,
            {
                "$set": {
                    "status": status,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0

    @classmethod
    async def delete(cls, session_id, user_id=None):
        """删除会话（标记为已删除）"""
        return await cls.update_status(session_id, "deleted", user_id)

    @classmethod
    async def add_roles(cls, session_id, role_ids):
        """向会话添加角色"""
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)
            
        # 转换所有role_id为ObjectId
        role_object_ids = []
        for role_id in role_ids:
            if isinstance(role_id, str):
                role_object_ids.append(ObjectId(role_id))
            else:
                role_object_ids.append(role_id)
                
        result = await cls.get_collection().update_one(
            {"_id": session_id},
            {
                "$addToSet": {"role_ids": {"$each": role_object_ids}},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        return result.modified_count > 0

    @classmethod
    async def remove_roles(cls, session_id, role_ids):
        """从会话移除角色"""
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)
            
        # 转换所有role_id为ObjectId
        role_object_ids = []
        for role_id in role_ids:
            if isinstance(role_id, str):
                role_object_ids.append(ObjectId(role_id))
            else:
                role_object_ids.append(role_id)
                
        result = await cls.get_collection().update_one(
            {"_id": session_id},
            {
                "$pullAll": {"role_ids": role_object_ids},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        return result.modified_count > 0

    @classmethod
    async def update_last_active(cls, session_id):
        """更新会话最后活跃时间"""
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)
            
        result = await cls.get_collection().update_one(
            {"_id": session_id},
            {"$set": {"updated_at": datetime.utcnow()}}
        )
        return result.modified_count > 0

    @classmethod
    async def archive_inactive_sessions(cls, days=30):
        """归档长时间不活跃的会话"""
        from datetime import timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        result = await cls.get_collection().update_many(
            {
                "updated_at": {"$lt": cutoff_date},
                "status": "active"
            },
            {"$set": {"status": "archived"}}
        )
        return result.modified_count

    @classmethod
    async def permanently_delete(cls, session_id, user_id=None):
        """永久删除会话（从数据库中彻底删除）"""
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)
            
        query = {"_id": session_id}
        if user_id:
            query["user_id"] = user_id
        
        # 确保只删除标记为"deleted"状态的会话
        query["status"] = "deleted"
            
        result = await cls.get_collection().delete_one(query)
        return result.deleted_count > 0
    
    @classmethod
    async def permanently_delete_all_user_deleted(cls, user_id, older_than_days=None):
        """永久删除用户所有已标记删除的会话
        
        参数:
            user_id: 用户ID
            older_than_days: 如果提供，只删除超过指定天数的已删除会话
            
        返回:
            删除的会话数量
        """
        query = {
            "user_id": user_id,
            "status": "deleted"
        }
        
        # 如果指定了时间限制，添加到查询条件
        if older_than_days:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
            query["updated_at"] = {"$lt": cutoff_date}
            
        result = await cls.get_collection().delete_many(query)
        return result.deleted_count

    @classmethod
    async def search_sessions(cls, user_id, query=None, filters=None, sort_by="updated_at", 
                         sort_direction=-1, limit=20, offset=0):
        """
        高级会话搜索
        
        参数:
            user_id: 用户ID
            query: 搜索关键词(搜索标题和描述)
            filters: 筛选条件字典，如 {"status": "active", "role_ids": "role_123"}
            sort_by: 排序字段，默认为更新时间
            sort_direction: 排序方向，1为升序，-1为降序
            limit: 最大返回数量
            offset: 跳过的记录数
            
        返回:
            匹配的会话列表和总数
        """
        # 构建基础查询条件
        search_query = {"user_id": user_id}
        
        # 处理文本搜索
        if query and len(query.strip()) > 0:
            text_query = {"$or": [
                {"title": {"$regex": query, "$options": "i"}},
                {"description": {"$regex": query, "$options": "i"}}
            ]}
            search_query.update(text_query)
        
        # 处理筛选条件
        if filters:
            if "status" in filters and filters["status"]:
                search_query["status"] = filters["status"]
            
            if "role_ids" in filters and filters["role_ids"]:
                search_query["role_ids"] = {"$in": [ObjectId(filters["role_ids"])]}
                
            if "created_after" in filters and filters["created_after"]:
                search_query["created_at"] = {"$gte": filters["created_after"]}
                
            if "created_before" in filters and filters["created_before"]:
                if "created_at" not in search_query:
                    search_query["created_at"] = {}
                search_query["created_at"]["$lte"] = filters["created_before"]
                
            if "updated_after" in filters and filters["updated_after"]:
                search_query["updated_at"] = {"$gte": filters["updated_after"]}
                
            if "updated_before" in filters and filters["updated_before"]:
                if "updated_at" not in search_query:
                    search_query["updated_at"] = {}
                search_query["updated_at"]["$lte"] = filters["updated_before"]
        
        # 获取总数
        total = await cls.get_collection().count_documents(search_query)
        
        # 获取数据
        cursor = cls.get_collection().find(search_query)\
            .sort(sort_by, sort_direction)\
            .skip(offset)\
            .limit(limit)
            
        sessions = await cursor.to_list(length=limit)
        
        return sessions, total 