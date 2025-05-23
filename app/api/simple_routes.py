"""
简单测试路由

用于测试服务器是否正常运行
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional
from app.database.mongodb import get_db
from bson import ObjectId
import logging
import httpx
import time
from app.auth.auth_handler import get_current_user, get_current_user_optional
from app.models.user import User
from app.config import memory_settings
from app.types import JsonObject
from app.memory.memory_manager import get_memory_manager
import redis
import os

router = APIRouter(prefix="/api/test", tags=["test"])
logger = logging.getLogger(__name__)

@router.get("/ping")
async def ping():
    """
    简单的健康检查端点
    """
    return {"status": "ok", "message": "pong"}

@router.get("/echo/{message}")
async def echo(message: str):
    """
    回显消息
    """
    return {"message": message}

@router.get("/role/{role_id}")
async def test_get_role(role_id: str, db=Depends(get_db)):
    """测试获取角色，用于调试ObjectId处理"""
    try:
        # 尝试将role_id转换为ObjectId
        logger.info(f"测试获取角色: role_id={role_id}")
        
        try:
            object_id = ObjectId(role_id)
            logger.info(f"转换为ObjectId: {object_id}")
        except Exception as e:
            logger.error(f"ObjectId转换失败: {str(e)}")
            return {"error": f"ObjectId转换失败: {str(e)}"}
        
        # 查询角色信息
        role_info = await db.roles.find_one({"_id": object_id})
        
        if role_info:
            # 转换_id为字符串
            role_info["id"] = str(role_info.pop("_id"))
            logger.info(f"找到角色信息: {role_info}")
            return role_info
        else:
            logger.warning(f"未找到角色: _id={object_id}")
            return {"error": "未找到角色"}
            
    except Exception as e:
        logger.error(f"测试获取角色失败: {str(e)}")
        return {"error": str(e)}

@router.get("/roles")
async def list_all_roles(db=Depends(get_db)):
    """列出所有角色，用于调试"""
    try:
        # 获取所有角色
        cursor = db.roles.find({})
        roles = await cursor.to_list(length=100)
        
        # 转换_id为字符串
        for role in roles:
            role["id"] = str(role.pop("_id"))
            
        logger.info(f"找到{len(roles)}个角色")
        return {"roles": roles}
    except Exception as e:
        logger.error(f"获取角色列表失败: {str(e)}")
        return {"error": str(e)}

@router.post("/message-with-role")
async def add_test_message(
    session_id: str,
    role_id: str,
    content: str = "测试消息"
):
    """添加一条测试消息到Redis，直接使用指定的角色ID"""
    from app.memory.buffer_memory import ShortTermMemory
    from bson.objectid import ObjectId
    
    try:
        # 记录参数
        logger.info(f"添加测试消息: session_id={session_id}, role_id={role_id}, content={content}")
        
        # 创建Redis客户端
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6378")),
            password=os.getenv("REDIS_PASSWORD", "!qaz2wsX"),
            decode_responses=True
        )
        
        # 初始化ShortTermMemory
        memory = ShortTermMemory(redis_client=redis_client)
        
        # 尝试转换ObjectId
        try:
            object_id = ObjectId(role_id)
            logger.info(f"有效的ObjectId: {object_id}")
        except Exception as e:
            return {"error": f"无效的ObjectId: {str(e)}"}
            
        # 添加消息
        result, _ = await memory.add_message(
            session_id=session_id,
            user_id="test_user",
            role="user",
            content=content,
            role_id=role_id
        )
        
        return {
            "success": result,
            "message": "消息添加成功，请检查Redis",
            "session_id": session_id,
            "role_id": role_id
        }
    except Exception as e:
        logger.error(f"添加测试消息失败: {str(e)}")
        return {"error": str(e)}

@router.get("/send-with-roleid")
async def send_message_with_roleid(
    role_id: str,
    message: str = "你好，这是一条测试消息",
    session_id: Optional[str] = None
):
    """
    发送一条带有roleid的聊天请求，用于测试
    """
    try:
        # 如果没有提供session_id，生成一个
        if not session_id:
            session_id = f"{int(time.time())}-test"
            
        logger.info(f"发送测试消息: session_id={session_id}, role_id={role_id}, message={message}")
        
        # 构建请求体
        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": message,
                    "session_id": session_id,
                    "roleid": role_id
                }
            ],
            "stream": False,
            "roleid": role_id
        }
        
        # 发送请求
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/api/llm/chat",
                json=request_data
            )
            
            response_data = response.json()
            
            # 返回结果
            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response": response_data,
                "request": request_data
            }
            
    except Exception as e:
        logger.error(f"发送测试消息失败: {str(e)}")
        return {"error": str(e)}

@router.get("/message-role-check/{session_id}")
async def check_message_roles(
    session_id: str,
    user_id: str = "anonymous_user"
):
    """检查Redis中的消息角色名称是否正确设置为中文名称"""
    from app.memory.buffer_memory import ShortTermMemory
    import redis
    
    try:
        # 创建Redis客户端
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6378")),
            password=os.getenv("REDIS_PASSWORD", "!qaz2wsX"),
            decode_responses=True
        )
        
        # 初始化ShortTermMemory
        memory = ShortTermMemory(redis_client=redis_client)
        
        # 获取会话的所有消息
        messages = memory.get_session_messages(session_id, user_id)
        
        # 分析结果
        result = {
            "total_messages": len(messages),
            "messages": [],
            "roles_summary": {}
        }
        
        # 收集角色统计信息
        for msg in messages:
            role = msg.get("role")
            roleid = msg.get("roleid")
            result["roles_summary"][role] = result["roles_summary"].get(role, 0) + 1
            
            # 添加简化的消息信息
            result["messages"].append({
                "role": role,
                "roleid": roleid,
                "content_preview": msg.get("content", "")[:30] + "..." if msg.get("content") else "",
                "timestamp": msg.get("timestamp")
            })
            
        return result
    except Exception as e:
        logger.error(f"检查消息角色失败: {str(e)}")
        return {"error": str(e)}

@router.post("/fix-role-names/{session_id}")
async def fix_role_names(
    session_id: str,
    user_id: str = "anonymous_user",
    db=Depends(get_db)
):
    """尝试修复Redis中的角色名称，从MongoDB获取正确的角色名称"""
    from app.memory.memory_manager import get_memory_manager
    
    try:
        # 使用memory_manager的方法更新角色名称
        memory_manager = await get_memory_manager()
        result = await memory_manager.update_session_role_names(session_id, user_id)
        
        # 添加数据库中的角色列表用于参考
        try:
            cursor = db.roles.find({}).limit(10)
            sample_roles = await cursor.to_list(length=10)
            roles_info = [{"id": str(r["_id"]), "name": r.get("name", "无名称")} for r in sample_roles]
            result["available_roles"] = roles_info
        except Exception as e:
            logger.warning(f"获取角色列表参考时出错: {str(e)}")
        
        return result
    except Exception as e:
        logger.error(f"修复角色名称失败: {str(e)}")
        return {"error": str(e)} 