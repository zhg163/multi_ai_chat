import asyncio
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from app.database.connection import Database

async def create_indices():
    """创建所有必要的索引"""
    # 用户集合索引
    await Database.db.users.create_indexes([
        IndexModel([("username", ASCENDING)], unique=True),
        IndexModel([("last_active", DESCENDING)])
    ])
    
    # 角色集合索引
    await Database.db.roles.create_indexes([
        IndexModel([("name", ASCENDING)], unique=True),
        IndexModel([("active", ASCENDING)]),
        IndexModel([("keywords", ASCENDING)])
    ])
    
    # 会话集合索引
    await Database.db.sessions.create_indexes([
        IndexModel([("user_id", ASCENDING)]),
        IndexModel([("roles", ASCENDING)]),
        IndexModel([("status", ASCENDING)]),
        IndexModel([("last_active", DESCENDING)])
    ])
    
    # 消息集合索引
    await Database.db.messages.create_indexes([
        IndexModel([("session_id", ASCENDING)]),
        IndexModel([("sender_id", ASCENDING)]),
        IndexModel([("created_at", ASCENDING)]),
        IndexModel([("responding_role_id", ASCENDING)])
    ])
    
    # 记忆集合索引
    await Database.db.memories.create_indexes([
        IndexModel([("session_id", ASCENDING)]),
        IndexModel([("role_id", ASCENDING)]),
        IndexModel([("source_messages", ASCENDING)]),
        IndexModel([("created_at", DESCENDING)]),
        IndexModel([("last_accessed", DESCENDING)])
    ])
    
    # 针对向量搜索的索引需要专门API，如Atlas Search
    print("所有索引创建完成")

if __name__ == "__main__":
    # 运行脚本创建索引
    asyncio.run(create_indices()) 