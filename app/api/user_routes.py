from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Path, status, Body, Request
from pydantic import BaseModel, Field, ConfigDict
from bson import ObjectId
import traceback
import logging
from datetime import datetime, timedelta

from app.database.mongodb import get_db
from app.services.fallback_service import FallbackService
from app.auth.auth_handler import create_access_token
from app.memory.memory_manager import get_memory_manager

# 配置日志
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/users", tags=["users"])

class UserBase(BaseModel):
    """用户基础数据模型"""
    name: str = Field(..., description="用户名称")
    username: str = Field(..., description="用户登录名")
    email: Optional[str] = Field(None, description="用户邮箱")
    avatar: Optional[str] = Field(None, description="用户头像")
    description: Optional[str] = Field(None, description="用户描述")
    tags: Optional[List[str]] = Field(default_factory=list, description="用户标签")
    is_active: Optional[bool] = Field(True, description="是否激活")

class UserResponse(UserBase):
    """用户信息响应模型"""
    id: str

    # 使用最新的Pydantic V2配置方式
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "name": "张三",
                "username": "zhangsan",
                "email": "zhang@example.com",
                "avatar": "https://example.com/avatar.png",
                "description": "普通用户",
                "tags": ["喜欢电影", "户外运动"],
                "is_active": True
            }
        }
    )

class UserCreate(UserBase):
    """创建用户的请求模型"""
    pass

class UserUpdate(BaseModel):
    """更新用户的请求模型"""
    name: Optional[str] = Field(None, description="用户名称")
    username: Optional[str] = Field(None, description="用户登录名")
    email: Optional[str] = Field(None, description="用户邮箱")
    avatar: Optional[str] = Field(None, description="用户头像")
    description: Optional[str] = Field(None, description="用户描述")
    tags: Optional[List[str]] = Field(None, description="用户标签")
    is_active: Optional[bool] = Field(None, description="是否激活")

@router.get("/", response_model=List[UserResponse])
async def list_users(
    is_active: Optional[bool] = None,
    limit: int = Query(100, gt=0, le=100),
    offset: int = Query(0, ge=0),
    db=Depends(get_db)
):
    """获取用户列表，支持分页和活跃状态过滤"""
    try:
        # 检查数据库连接
        if db is None:
            logger.error("数据库连接失败: db对象为None")
            # 返回空列表而不是错误，这样前端仍然可以显示
            return FallbackService.get_sample_users()
            
        # 检查users集合是否存在
        try:
            if not hasattr(db, 'users'):
                # 创建一个用户集合
                logger.warning("数据库中未找到users集合，将返回内存中的示例用户")
                return FallbackService.get_sample_users()
        except Exception as e:
            logger.error(f"检查users集合时出错: {str(e)}")
            return FallbackService.get_sample_users()
            
        query = {}
        if is_active is not None:
            query["is_active"] = is_active
        
        logger.info(f"Fetching users with query: {query}, limit: {limit}, offset: {offset}")
        
        try:
            # 尝试从users集合获取数据
            cursor = db.users.find(query).skip(offset).limit(limit)
            users_list = await cursor.to_list(length=limit)
        except Exception as e:
            logger.error(f"查询用户数据时出错: {str(e)}")
            return FallbackService.get_sample_users()
        
        # 如果没有用户数据，添加一些示例用户
        if len(users_list) == 0:
            logger.info("No users found, creating sample users")
            try:
                sample_users = FallbackService.get_sample_users(as_dict=True)
                
                # 将示例用户添加到数据库
                result = await db.users.insert_many(sample_users)
                
                # 获取刚插入的用户数据
                cursor = db.users.find(query).skip(offset).limit(limit)
                users_list = await cursor.to_list(length=limit)
            except Exception as e:
                logger.error(f"插入示例用户时出错: {str(e)}")
                return FallbackService.get_sample_users()
        
        # 处理ObjectId - 重命名_id为id
        result_users = []
        for user in users_list:
            # 转换_id为id
            user["id"] = str(user.pop("_id"))
            result_users.append(user)
        
        logger.info(f"Found {len(result_users)} users")
        return result_users
        
    except Exception as e:
        logger.error(f"Error fetching users: {str(e)}")
        logger.error(traceback.format_exc())
        # 返回示例用户而不是抛出异常
        return FallbackService.get_sample_users()

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, db=Depends(get_db)):
    """根据ID获取用户详情"""
    try:
        object_id = ObjectId(user_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    
    user = await db.users.find_one({"_id": object_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # 处理ObjectId
    user["id"] = str(user.pop("_id"))
    
    return user 

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate, db=Depends(get_db)):
    """创建新用户"""
    try:
        # 检查用户是否已存在
        existing_user = await db.users.find_one({"name": user.name})
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"用户名 '{user.name}' 已存在"
            )
        
        user_dict = user.model_dump(exclude_unset=True)
        result = await db.users.insert_one(user_dict)
        created_user = await db.users.find_one({"_id": result.inserted_id})
        
        # 处理ObjectId
        created_user["id"] = str(created_user.pop("_id"))
        
        return created_user
    except Exception as e:
        logger.error(f"创建用户失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建用户失败: {str(e)}"
        )

@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, user_update: UserUpdate, db=Depends(get_db)):
    """更新用户信息"""
    try:
        # 验证ObjectId
        try:
            object_id = ObjectId(user_id)
        except:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="无效的用户ID格式"
            )
        
        # 检查用户是否存在
        existing_user = await db.users.find_one({"_id": object_id})
        if not existing_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"用户ID '{user_id}' 不存在"
            )
        
        # 准备更新数据
        update_data = user_update.model_dump(exclude_unset=True, exclude_none=True)
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="没有提供需要更新的字段"
            )
        
        # 执行更新
        result = await db.users.update_one(
            {"_id": object_id},
            {"$set": update_data}
        )
        
        if result.modified_count == 0:
            logger.warning(f"用户 {user_id} 未被修改，可能提供的数据与现有数据相同")
            
        # 返回更新后的用户数据
        updated_user = await db.users.find_one({"_id": object_id})
        updated_user["id"] = str(updated_user.pop("_id"))
        
        return updated_user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新用户失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新用户失败: {str(e)}"
        )

@router.post("/select-user")
async def select_user_login(
    request: Request,
    user_data: Dict = Body(...)
):
    """选择用户并登录"""
    try:
        user_id = user_data.get("user_id")
        session_id = user_data.get("session_id")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="缺少用户ID")
            
        # 验证用户是否存在
        from app.database.connection import get_database
        from bson.objectid import ObjectId
        
        db = await get_database()
        if db is None:
            raise HTTPException(status_code=500, detail="数据库连接失败")
            
        try:
            user = await db.users.find_one({"_id": ObjectId(user_id)})
        except Exception as e:
            logger.error(f"查询用户时出错: {str(e)}")
            raise HTTPException(status_code=400, detail="无效的用户ID格式")
            
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
            
        # 生成访问令牌
        expires_delta = timedelta(days=7)
        access_token = create_access_token(
            data={"sub": str(user["_id"])},
            expires_delta=expires_delta
        )
        
        # 如果提供了会话ID，更新会话所有者
        new_session_id = None
        if session_id:
            try:
                # 获取原始匿名会话的消息
                memory_manager = await get_memory_manager()
                messages = memory_manager.short_term.get_session_messages(session_id, "anonymous_user")
                
                if messages:
                    # 创建一个新会话，归属于选中的用户
                    new_session_id = await memory_manager.start_new_session(str(user["_id"]))
                    
                    # 将消息迁移到新会话
                    for msg in reversed(messages):  # 从旧到新迁移消息
                        await memory_manager.add_message(
                            new_session_id, 
                            str(user["_id"]), 
                            msg.get("role", "user"), 
                            msg.get("content", ""),
                            msg.get("roleid"),
                            msg.get("message_id")
                        )
            except Exception as e:
                logger.error(f"会话迁移失败: {str(e)}")
                # 即使迁移失败也继续处理，允许用户选择
        
        # 返回用户信息和令牌
        response_data = {
            "success": True, 
            "access_token": access_token, 
            "token_type": "bearer",
            "user_id": str(user["_id"]),
            "name": user.get("name", "")
        }
        
        # 如果有新会话ID，添加到响应中
        if new_session_id:
            response_data["session_id"] = new_session_id
            
        return response_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"用户选择失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"用户选择失败: {str(e)}")

@router.post("/check-existing")
async def check_existing_users(users: List[UserBase], db=Depends(get_db)):
    """检查用户是否已存在，类似add_test_users.py的行为"""
    try:
        # 获取所有提交用户的名称
        user_names = [user.name for user in users]
        
        # 检查已存在用户
        existing_names = set()
        cursor = db.users.find({"name": {"$in": user_names}})
        existing_users = await cursor.to_list(length=len(user_names))
        
        for user in existing_users:
            existing_names.add(user["name"])
        
        # 筛选出未存在的用户
        new_users = [user.model_dump() for user in users if user.name not in existing_names]
        
        return {
            "existingUsers": list(existing_names),
            "newUsers": new_users,
            "message": f"发现 {len(existing_names)} 个已存在用户，{len(new_users)} 个新用户"
        }
    except Exception as e:
        logger.error(f"检查用户失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"检查用户失败: {str(e)}"
        )

@router.post("/add-new")
async def add_new_users(users: List[UserBase], db=Depends(get_db)):
    """添加新用户，类似add_test_users.py的行为"""
    try:
        if not users:
            return {"insertedCount": 0, "message": "无用户需要添加"}
        
        # 将Pydantic模型转换为字典
        user_dicts = [user.model_dump() for user in users]
        
        # 添加新用户
        result = await db.users.insert_many(user_dicts)
        
        return {
            "insertedCount": len(result.inserted_ids),
            "message": f"成功添加 {len(result.inserted_ids)} 个测试用户"
        }
    except Exception as e:
        logger.error(f"添加用户失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"添加用户失败: {str(e)}"
        )

@router.get("/list")
async def list_all_users(db=Depends(get_db)):
    """获取所有用户列表，用于显示当前用户"""
    try:
        cursor = db.users.find({})
        users = await cursor.to_list(length=100)
        
        # 处理ObjectId
        for user in users:
            user["id"] = str(user.pop("_id"))
        
        return users
    except Exception as e:
        logger.error(f"获取用户列表失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取用户列表失败: {str(e)}"
        )

@router.delete("/{user_id}")
async def delete_user(user_id: str, db=Depends(get_db)):
    """删除指定ID的用户"""
    try:
        # 验证ObjectId
        try:
            object_id = ObjectId(user_id)
        except:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="无效的用户ID格式"
            )
        
        # 检查用户是否存在
        existing_user = await db.users.find_one({"_id": object_id})
        if not existing_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"用户ID '{user_id}' 不存在"
            )
        
        # 执行删除
        result = await db.users.delete_one({"_id": object_id})
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"删除用户 {user_id} 失败"
            )
            
        return {"success": True, "message": f"成功删除用户 ID: {user_id}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除用户失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除用户失败: {str(e)}"
        ) 