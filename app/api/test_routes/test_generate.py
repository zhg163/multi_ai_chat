"""
测试生成接口 - 完全不依赖FastAPI路由系统
"""

import json
import uuid
import logging
import asyncio
from datetime import datetime
from starlette.requests import Request
from fastapi.responses import JSONResponse, StreamingResponse

# 创建日志记录器
logger = logging.getLogger(__name__)

async def test_generate(request: Request):
    """
    测试生成路由 - 不依赖FastAPI的参数验证
    
    直接解析请求体并生成模拟响应
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
        logger.info(f"[{request_id}] 测试生成路由收到请求体: {data}")
        
        # 检查是否需要流式响应
        stream = data.get("stream", True)
        
        if stream:
            # 流式响应
            async def generate_stream():
                # 发送消息ID
                yield f"data: {json.dumps({'message_id': request_id})}\n\n"
                
                # 发送响应内容
                response_parts = [
                    "你好！",
                    "很高兴为你提供帮助。",
                    "有什么我可以",
                    "协助你的吗？"
                ]
                
                for part in response_parts:
                    sse_data = {
                        "choices": [{
                            "delta": {"content": part},
                            "index": 0,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(sse_data)}\n\n"
                    await asyncio.sleep(0.3)  # 模拟延迟
                
                # 发送结束标记
                yield f"data: {json.dumps({'choices': [{'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # 非流式响应
            response_data = {
                "choices": [{
                    "message": {
                        "content": "你好！很高兴为你提供帮助。有什么我可以协助你的吗？",
                        "role": "assistant"
                    },
                    "index": 0,
                    "finish_reason": "stop"
                }]
            }
            
            # 记录要返回的数据
            logger.info(f"[{request_id}] 测试生成路由返回数据: {response_data}")
            
            return JSONResponse(
                status_code=200,
                content=response_data
            )
    
    except Exception as e:
        logger.error(f"测试生成路由处理请求时出错: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"测试生成路由错误: {str(e)}"}
        ) 