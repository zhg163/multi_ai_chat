from datetime import datetime
from bson.objectid import ObjectId
from app.database.connection import Database

class Memory:
    collection = None

    @classmethod
    def get_collection(cls):
        """获取记忆集合"""
        if cls.collection is None:
            cls.collection = Database.db.memories
        return cls.collection

    @classmethod
    async def create(cls, session_id, role_id, content, source_messages=None, 
                   vector_embedding=None, metadata=None):
        """创建新记忆"""
        # 确保ID格式正确
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)
        if isinstance(role_id, str):
            role_id = ObjectId(role_id)
            
        # 转换所有source_message_id为ObjectId
        source_message_ids = []
        if source_messages:
            for msg_id in source_messages:
                if isinstance(msg_id, str):
                    source_message_ids.append(ObjectId(msg_id))
                else:
                    source_message_ids.append(msg_id)
        
        memory_data = {
            "session_id": session_id,
            "role_id": role_id,
            "content": content,
            "source_messages": source_message_ids,
            "vector_embedding": vector_embedding or [],
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow(),
            "access_count": 0,
            "metadata": metadata or {}
        }
        
        result = await cls.get_collection().insert_one(memory_data)
        memory_data["_id"] = result.inserted_id
        return memory_data

    @classmethod
    async def get_by_id(cls, memory_id):
        """根据ID获取记忆"""
        if isinstance(memory_id, str):
            memory_id = ObjectId(memory_id)
        return await cls.get_collection().find_one({"_id": memory_id})

    @classmethod
    async def update_embedding(cls, memory_id, vector_embedding):
        """更新记忆的向量嵌入"""
        if isinstance(memory_id, str):
            memory_id = ObjectId(memory_id)
            
        result = await cls.get_collection().update_one(
            {"_id": memory_id},
            {"$set": {"vector_embedding": vector_embedding}}
        )
        return result.modified_count > 0

    @classmethod
    async def record_access(cls, memory_id):
        """记录记忆访问"""
        if isinstance(memory_id, str):
            memory_id = ObjectId(memory_id)
            
        result = await cls.get_collection().update_one(
            {"_id": memory_id},
            {
                "$set": {"last_accessed": datetime.utcnow()},
                "$inc": {"access_count": 1}
            }
        )
        return result.modified_count > 0

    @classmethod
    async def get_session_memories(cls, session_id, role_id=None, limit=100, offset=0):
        """获取会话记忆"""
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)
        
        query = {"session_id": session_id}
        if role_id:
            if isinstance(role_id, str):
                role_id = ObjectId(role_id)
            query["role_id"] = role_id
            
        cursor = cls.get_collection().find(query).sort(
            "created_at", -1
        ).skip(offset).limit(limit)
        
        return await cursor.to_list(length=limit)

    @classmethod
    async def find_by_source_message(cls, message_id):
        """查找引用特定消息的记忆"""
        if isinstance(message_id, str):
            message_id = ObjectId(message_id)
            
        cursor = cls.get_collection().find(
            {"source_messages": message_id}
        )
        
        return await cursor.to_list(length=None) 