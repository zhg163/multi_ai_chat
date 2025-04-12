# migrations/create_indexes.py
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT

def run_migration():
    client = MongoClient('mongodb://root:example@localhost:27017/')
    db = client['multi_ai_chat']
    
    # 用户集合索引
    db.users.create_index('username', unique=True)
    
    # 角色集合索引
    db.roles.create_index('name', unique=True)
    db.roles.create_index([('keywords', TEXT)])
    
    # 会话集合索引
    db.sessions.create_index([('user_id', ASCENDING), ('created_at', DESCENDING)])
    db.sessions.create_index('status')
    
    # 消息集合索引
    db.messages.create_index([('session_id', ASCENDING), ('created_at', ASCENDING)])
    
    # 记忆集合索引
    db.memory.create_index([('session_id', ASCENDING), ('role_id', ASCENDING)])
    
    print("所有索引创建完成")

if __name__ == "__main__":
    run_migration()