from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Path, status
from pydantic import BaseModel, Field, ConfigDict
from bson import ObjectId
import traceback
import logging
import uuid
import json

from app.services.role_service import RoleService
from app.models.role import Role
from app.database.mongodb import get_db
from app.services.role_matching_service import role_matching_service, RoleMatchingService
from app.services.embedding_service import embedding_service
# Removed import - this file was deleted
# from app.services.session_role_manager import SessionRoleManager
# Removed imports - these modules don't exist
# from app.common.redis_client import get_redis_client
# from app.common.logger_config import get_logger

# 配置日志
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/roles", tags=["roles"])

# 请求和响应模型
class RoleBase(BaseModel):
    """角色基础数据模型"""
    name: str = Field(..., description="角色名称")
    description: Optional[str] = Field(None, description="角色描述")
    personality: Optional[str] = Field(None, description="角色性格")
    speech_style: Optional[str] = Field(None, description="角色语言风格")
    keywords: List[str] = Field(default_factory=list, description="角色关键词")
    temperature: Optional[float] = Field(0.7, description="生成温度")
    prompt_templates: Optional[List[str]] = Field(default_factory=list, description="提示词模板")
    system_prompt: Optional[str] = Field(None, description="系统提示词")
    is_active: Optional[bool] = Field(True, description="是否激活")

class RoleCreate(RoleBase):
    """创建角色的请求模型"""
    pass

class RoleUpdate(BaseModel):
    """更新角色的请求模型"""
    name: Optional[str] = Field(None, description="角色名称")
    description: Optional[str] = Field(None, description="角色描述")
    personality: Optional[str] = Field(None, description="角色性格")
    speech_style: Optional[str] = Field(None, description="角色语言风格")
    keywords: Optional[List[str]] = Field(None, description="角色关键词")
    temperature: Optional[float] = Field(None, description="生成温度")
    prompt_templates: Optional[List[str]] = Field(None, description="提示词模板")
    system_prompt: Optional[str] = Field(None, description="系统提示词")
    is_active: Optional[bool] = Field(None, description="是否激活")

class RoleResponse(RoleBase):
    """角色信息响应模型"""
    id: str

    # 使用最新的Pydantic V2配置方式
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "name": "孙悟空",
                "description": "齐天大圣",
                "personality": "活泼好动",
                "speech_style": "猴性十足",
                "keywords": ["孙悟空", "猴子", "大圣"],
                "temperature": 0.8,
                "prompt_templates": ["你是孙悟空，...", "你需要保护唐僧..."],
                "system_prompt": "你需要保护唐僧...",
                "is_active": True
            }
        }
    )

class KeywordsUpdate(BaseModel):
    """更新关键词的请求模型"""
    keywords: List[str] = Field(..., description="角色关键词")

# 关键词匹配相关模型
class MessageMatchRequest(BaseModel):
    """消息匹配请求"""
    message: str = Field(..., description="用户消息")
    session_id: Optional[str] = Field(None, description="会话ID，用于获取上下文")
    limit: int = Field(3, ge=1, le=10, description="返回角色数量上限")
    min_score: float = Field(0.2, ge=0.0, le=1.0, description="最小匹配分数")

class KeywordExtractRequest(BaseModel):
    """关键词提取请求"""
    text: str = Field(..., description="要分析的文本")
    top_k: int = Field(10, ge=1, le=50, description="返回关键词数量")

class RoleMatchRequest(BaseModel):
    query: str
    session_id: str
    system_prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None

class RoleMatchResponse(BaseModel):
    request_id: str
    success: bool
    role: Dict[str, Any]
    message: Optional[str] = None

# 路由定义
@router.post("/", response_model=RoleResponse, status_code=status.HTTP_201_CREATED)
async def create_role(role: RoleCreate, db=Depends(get_db)):
    """创建新角色"""
    role_dict = role.dict()
    result = await db.roles.insert_one(role_dict)
    role_dict["id"] = str(result.inserted_id)
    return role_dict

@router.get("/", response_model=List[RoleResponse])
async def list_roles(
    is_active: Optional[bool] = None,
    limit: int = Query(100, gt=0, le=100),
    offset: int = Query(0, ge=0),
    db=Depends(get_db)
):
    """获取角色列表，支持分页和活跃状态过滤"""
    try:
        query = {}
        if is_active is not None:
            query["is_active"] = is_active
        
        logger.info(f"Fetching roles with query: {query}, limit: {limit}, offset: {offset}")
        
        cursor = db.roles.find(query).skip(offset).limit(limit)
        roles_list = await cursor.to_list(length=limit)
        
        # 处理ObjectId - 重命名_id为id
        result_roles = []
        for role in roles_list:
            # 转换_id为id
            role["id"] = str(role.pop("_id"))
            result_roles.append(role)
        
        logger.info(f"Found {len(result_roles)} roles")
        return result_roles
        
    except Exception as e:
        logger.error(f"Error fetching roles: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch roles: {str(e)}"
        )

@router.get("/{role_id}", response_model=RoleResponse)
async def get_role(role_id: str, db=Depends(get_db)):
    """根据ID获取角色详情"""
    try:
        object_id = ObjectId(role_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid role ID format")
    
    role = await db.roles.find_one({"_id": object_id})
    if not role:
        raise HTTPException(status_code=404, detail="Role not found")
    
    # 处理ObjectId
    role["id"] = str(role.pop("_id"))
    
    return role

@router.put("/{role_id}", response_model=RoleResponse)
async def update_role(role_id: str, role_update: RoleUpdate, db=Depends(get_db)):
    """更新角色信息"""
    try:
        object_id = ObjectId(role_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid role ID format")
    
    update_data = {k: v for k, v in role_update.dict().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    result = await db.roles.update_one(
        {"_id": object_id},
        {"$set": update_data}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Role not found")
    
    updated_role = await db.roles.find_one({"_id": object_id})
    # 处理ObjectId
    updated_role["id"] = str(updated_role.pop("_id"))
    return updated_role

@router.patch("/{role_id}/keywords", response_model=RoleResponse)
async def update_role_keywords(role_id: str, keywords_update: KeywordsUpdate, db=Depends(get_db)):
    """更新角色关键词"""
    try:
        object_id = ObjectId(role_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid role ID format")
    
    result = await db.roles.update_one(
        {"_id": object_id},
        {"$set": {"keywords": keywords_update.keywords}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Role not found")
    
    updated_role = await db.roles.find_one({"_id": object_id})
    # 处理ObjectId
    updated_role["id"] = str(updated_role.pop("_id"))
    return updated_role

@router.post("/{role_id}/deactivate", response_model=RoleResponse)
async def deactivate_role(role_id: str, db=Depends(get_db)):
    """停用角色"""
    try:
        object_id = ObjectId(role_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid role ID format")
    
    result = await db.roles.update_one(
        {"_id": object_id},
        {"$set": {"is_active": False}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Role not found")
    
    updated_role = await db.roles.find_one({"_id": object_id})
    # 处理ObjectId
    updated_role["id"] = str(updated_role.pop("_id"))
    return updated_role

@router.post("/{role_id}/activate", response_model=RoleResponse)
async def activate_role(role_id: str, db=Depends(get_db)):
    """激活角色"""
    try:
        object_id = ObjectId(role_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid role ID format")
    
    result = await db.roles.update_one(
        {"_id": object_id},
        {"$set": {"is_active": True}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Role not found")
    
    updated_role = await db.roles.find_one({"_id": object_id})
    # 处理ObjectId
    updated_role["id"] = str(updated_role.pop("_id"))
    return updated_role

@router.delete("/{role_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_role(role_id: str, db=Depends(get_db)):
    """删除角色"""
    try:
        object_id = ObjectId(role_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid role ID format")
    
    # 修改为真正删除角色
    result = await db.roles.delete_one({"_id": object_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Role not found")

# 关键词匹配相关路由
@router.post("/match", response_model=List[Dict[str, Any]])
async def match_roles_for_message(request: MessageMatchRequest, db=Depends(get_db)):
    """
    根据消息内容匹配最适合的角色
    
    Args:
        request: 包含消息内容和匹配参数的请求
        
    Returns:
        匹配的角色列表，按匹配分数排序
    """
    try:
        # 确保嵌入服务已初始化
        if not embedding_service.initialized:
            await embedding_service.initialize()
            
        # 获取匹配的角色
        matching_roles = await role_matching_service.find_matching_roles(
            message=request.message,
            session_id=request.session_id,
            limit=request.limit
        )
        
        return matching_roles
        
    except Exception as e:
        logger.error(f"角色匹配端点错误: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"角色匹配失败: {str(e)}"
        )

@router.post("/extract-keywords", response_model=List[str])
async def extract_keywords(request: KeywordExtractRequest):
    """从文本中提取关键词"""
    try:
        keywords = await RoleService.extract_keywords_from_text(
            text=request.text,
            top_k=request.top_k
        )
        return keywords
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"提取关键词失败: {str(e)}"
        )

@router.post("/check-existing")
async def check_existing_roles(roles: List[RoleCreate], db=Depends(get_db)):
    """检查角色是否已存在，按名称检查"""
    try:
        # 获取所有提交角色的名称
        role_names = [role.name for role in roles]
        
        # 检查已存在角色
        existing_names = set()
        cursor = db.roles.find({"name": {"$in": role_names}})
        existing_roles = await cursor.to_list(length=len(role_names))
        
        for role in existing_roles:
            existing_names.add(role["name"])
        
        # 筛选出未存在的角色
        new_roles = [role.dict() for role in roles if role.name not in existing_names]
        
        return {
            "existingRoles": list(existing_names),
            "newRoles": new_roles,
            "message": f"发现 {len(existing_names)} 个已存在角色，{len(new_roles)} 个新角色"
        }
    except Exception as e:
        logger.error(f"检查角色失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"检查角色失败: {str(e)}"
        )

@router.post("/add-new")
async def add_new_roles(roles: List[dict], db=Depends(get_db)):
    """添加新角色"""
    try:
        if not roles:
            return {"insertedCount": 0, "message": "无角色需要添加"}
        
        # 添加新角色
        result = await db.roles.insert_many(roles)
        
        return {
            "insertedCount": len(result.inserted_ids),
            "message": f"成功添加 {len(result.inserted_ids)} 个角色"
        }
    except Exception as e:
        logger.error(f"添加角色失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"添加角色失败: {str(e)}"
        )

@router.post("/match", response_model=RoleMatchResponse)
async def match_role(request: RoleMatchRequest):
    """根据用户查询匹配合适的角色"""
    try:
        request_id = str(uuid.uuid4())
        query = request.query
        session_id = request.session_id
        system_prompt = request.system_prompt
        messages = request.messages or []
        
        # Get redis client
        # Commented out because redis_client is not available
        # redis_client = get_redis_client()
        
        # Initialize session role manager
        # Commented out because SessionRoleManager is not available
        # session_role_manager = SessionRoleManager(redis_client)
        
        # Perform role matching
        matched_role = await role_matching_service.match_role(query, session_id)
        
        if not matched_role:
            return {
                "request_id": request_id,
                "success": False,
                "role": {},
                "message": "No matching role found"
            }
        
        # Update role usage
        # Commented out because session_role_manager is not available
        # await session_role_manager.update_role_usage(session_id, matched_role["role_id"])
        
        return {
            "request_id": request_id,
            "success": True,
            "role": matched_role,
            "message": "Role matched successfully"
        }
        
    except Exception as e:
        logger.error(f"Error matching role: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "request_id": str(uuid.uuid4()),
            "success": False,
            "role": {},
            "message": f"Error matching role: {str(e)}"
        } 