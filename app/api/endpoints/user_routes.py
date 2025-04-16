from fastapi import APIRouter, HTTPException
from pymongo import MongoClient
from app.database.mongodb import get_db
from app.models.user import User

router = APIRouter()

@router.post("/api/users/init")
async def init_users(users: list[User]):
    db = get_db()
    try:
        # 检查已存在用户
        existing_names = set()
        for user in db.users.find({"name": {"$in": [user.name for user in users]}}):
            existing_names.add(user["name"])
        
        # 筛选新用户
        new_users = [user.dict() for user in users if user.name not in existing_names]
        
        if not new_users:
            return {"message": "所有用户已存在，无需添加新用户"}
        
        # 添加新用户
        result = db.users.insert_many(new_users)
        return {"message": f"成功添加 {len(result.inserted_ids)} 个用户", "inserted_ids": result.inserted_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))