"""
测试角色选择接口 - 完全不依赖FastAPI路由系统
"""

import json
import uuid
import logging
from datetime import datetime
from starlette.requests import Request
from fastapi.responses import JSONResponse

# 创建日志记录器
logger = logging.getLogger(__name__)

async def test_role_select(request: Request):
    """
    测试角色选择路由 - 不依赖FastAPI的参数验证
    
    直接解析请求体并返回模拟的角色匹配结果
    """
    try:
        # 直接获取原始请求数据
        body = await request.body()
        try:
            # 尝试解析JSON
            data = json.loads(body)
        except json.JSONDecodeError:
            # 如果不是有效的JSON，返回400错误
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid JSON in request body"}
            )
        
        # 生成请求ID
        request_id = str(uuid.uuid4())
        
        # 打印请求体
        logger.info(f"[{request_id}] 测试路由收到请求体: {data}")
        
        # 尝试提取请求数据
        request_data = data.get("request_data", data)
            
        # 提取必要参数，如果不存在则使用默认值
        messages = request_data.get("messages", [])
        session_id = request_data.get("session_id")
        user_id = request_data.get("user_id")
        
        # 验证messages格式（手动验证）
        if not isinstance(messages, list):
            return JSONResponse(
                status_code=400,
                content={"error": "messages must be a list"}
            )
        
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                return JSONResponse(
                    status_code=400,
                    content={"error": "invalid message format"}
                )
            
        # 模拟角色匹配结果
        role_match_result = {
            "success": True,
            "role": {
                "id": "default_role_" + request_id[:8],
                "name": "智能助手",
                "description": "一个乐于助人的AI助手",
                "system_prompt": "你是一个知识渊博、乐于助人的助手。"
            },
            "match_reason": "默认角色匹配",
            "error": None
        }
        
        # 构造响应数据
        response_data = {
            "request_id": request_id,
            "role_match": role_match_result,
            "session_id": session_id,
            "provider": "deepseek",  # 默认使用deepseek
            "model_name": "deepseek-chat",  # 默认使用deepseek-chat
            "api_key": None  # API key应该从环境变量获取
        }
        
        # 记录要返回的数据
        logger.info(f"[{request_id}] 测试路由返回数据: {response_data}")
        
        # 显式返回JSONResponse
        return JSONResponse(
            status_code=200,
            content=response_data
        )
    except Exception as e:
        logger.error(f"测试路由处理请求时出错: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"测试路由错误: {str(e)}"}
        ) 