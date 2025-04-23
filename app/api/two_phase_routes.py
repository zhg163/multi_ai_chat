"""
两阶段API demo路由

提供简单的两阶段聊天API示例，用于演示两阶段API设计模式
"""

import logging
from fastapi import APIRouter, HTTPException, Body, Depends, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import json
import uuid

from app.services.session_service import SessionService
from app.services.llm_service import LLMService
from app.common.redis_client import get_redis_client

# 创建日志记录器
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/api/two-phase", tags=["two-phase-api"])

@router.post("/generate")
async def generate_response(
    request: Request,
    session_service: SessionService = Depends(),
    llm_service: LLMService = Depends()
):
    """
    两阶段API示例 - 生成响应
    
    第一阶段是角色匹配（在后端自动完成）
    第二阶段是生成内容并返回
    
    这是一个演示用的简化实现，实际上应该使用角色匹配API和内容生成API
    """
    try:
        request_data = await request.json()
        
        if "message" not in request_data:
            return JSONResponse(
                status_code=400,
                content={"error": "缺少必需的message参数"}
            )
            
        session_id = request_data.get("session_id", str(uuid.uuid4()))
        message = request_data["message"]
        
        # 创建唯一的消息ID
        message_id = str(uuid.uuid4())
        
        # 暂存用户消息到Redis，以便后续改进
        redis_client = get_redis_client()
        await redis_client.set(
            f"two_phase:message:{message_id}", 
            json.dumps({
                "session_id": session_id,
                "message": message,
                "improved": False
            }),
            ex=3600  # 1小时过期
        )
        
        # 构建简单响应，实际应该调用角色匹配和内容生成
        response = f"这是对您问题「{message}」的回答。在实际系统中，这里会是由LLM生成的内容。这是两阶段API的演示。"
        
        return {
            "message_id": message_id,
            "session_id": session_id,
            "response": response,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"生成响应出错: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"生成响应出错: {str(e)}"}
        )

@router.post("/feedback")
async def provide_feedback(
    request: Request,
    llm_service: LLMService = Depends()
):
    """
    提供对生成内容的反馈
    
    如果接受，则记录满意度
    如果拒绝，则生成改进的响应
    """
    try:
        request_data = await request.json()
        
        if "message_id" not in request_data or "is_accepted" not in request_data:
            return JSONResponse(
                status_code=400,
                content={"error": "缺少必需的message_id或is_accepted参数"}
            )
            
        message_id = request_data["message_id"]
        is_accepted = request_data["is_accepted"]
        session_id = request_data.get("session_id")
        
        # 从Redis获取原始消息
        redis_client = get_redis_client()
        message_data_json = await redis_client.get(f"two_phase:message:{message_id}")
        
        if not message_data_json:
            return JSONResponse(
                status_code=404,
                content={"error": "找不到对应的消息"}
            )
            
        message_data = json.loads(message_data_json)
        original_message = message_data["message"]
        
        # 如果用户接受响应
        if is_accepted:
            logger.info(f"用户接受了消息 {message_id} 的响应")
            
            # 更新Redis中的消息状态
            message_data["feedback"] = "accepted"
            await redis_client.set(
                f"two_phase:message:{message_id}",
                json.dumps(message_data),
                ex=3600
            )
            
            return {
                "success": True,
                "message": "反馈已记录，谢谢！"
            }
        
        # 如果用户拒绝响应，生成改进的响应
        logger.info(f"用户拒绝了消息 {message_id} 的响应，尝试改进")
        
        # 生成改进的响应(实际应该调用LLM)
        improved_response = f"这是针对问题「{original_message}」的改进回答。在实际系统中，这会是LLM根据反馈生成的改进内容。"
        
        # 更新Redis中的消息状态
        message_data["feedback"] = "rejected"
        message_data["improved"] = True
        await redis_client.set(
            f"two_phase:message:{message_id}",
            json.dumps(message_data),
            ex=3600
        )
        
        return {
            "success": True,
            "improved_response": improved_response,
            "message": "已根据您的反馈生成改进的回答"
        }
        
    except Exception as e:
        logger.error(f"处理反馈出错: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"处理反馈出错: {str(e)}"}
        ) 