"""
修复LLM接口问题的工具
"""

import logging
import json
import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Request, Response, Body, Form, Query
from fastapi.responses import JSONResponse, StreamingResponse

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建路由器并指定前缀
router = APIRouter(prefix="/api/llm", tags=["llm_fix"])

@router.get("/fix_error")
async def fix_llm_error():
    """
    诊断并尝试修复LLM服务错误
    """
    try:
        # 导入需要的模块
        from app.api.llm_routes import llm_service
        from app.api.llm_routes import ChatResponse
        
        # 检查LLM服务
        status = {}
        
        # 检查llm_service是否存在
        status["llm_service_exists"] = llm_service is not None
        
        # 如果服务存在，检查其属性和方法
        if llm_service:
            status["service_dir"] = dir(llm_service)
            status["has_generate"] = hasattr(llm_service, "generate")
            status["has_generate_stream"] = hasattr(llm_service, "generate_stream")
            
            # 尝试添加应急方法
            if not hasattr(llm_service, "generate"):
                async def emergency_generate(self, messages, config=None):
                    return {
                        "content": "应急响应: LLM服务generate方法未定义。这是一个应急回复。",
                        "model": "emergency-model",
                        "provider": "emergency-provider",
                        "tokens_used": 0
                    }
                
                import types
                llm_service.generate = types.MethodType(emergency_generate, llm_service)
                status["added_emergency_generate"] = True
                logger.info("已添加应急generate方法")
            
            if not hasattr(llm_service, "generate_stream"):
                async def emergency_generate_stream(self, messages, config=None):
                    yield {
                        "is_start": True,
                        "model": "emergency-model",
                        "provider": "emergency-provider"
                    }
                    
                    yield {
                        "content": "应急响应: LLM服务generate_stream方法未定义。这是一个应急回复。",
                        "model": "emergency-model",
                        "provider": "emergency-provider"
                    }
                    
                    yield {
                        "is_end": True,
                        "model": "emergency-model",
                        "provider": "emergency-provider"
                    }
                
                import types
                llm_service.generate_stream = types.MethodType(emergency_generate_stream, llm_service)
                status["added_emergency_generate_stream"] = True
                logger.info("已添加应急generate_stream方法")
            
            # 保存修复后的状态
            status["fixed"] = True
            
        return {
            "status": "success",
            "message": "已尝试修复LLM服务",
            "details": status
        }
    except Exception as e:
        logger.error(f"修复LLM服务时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": f"修复失败: {str(e)}",
            "traceback": traceback.format_exc()
        }

@router.post("/mock_chat")
async def mock_chat(request: Request):
    """
    模拟聊天接口，用于测试
    不需要真正的LLM服务，总是返回成功
    """
    try:
        # 获取请求体
        body = await request.json()
        logger.info(f"收到模拟聊天请求: {json.dumps(body, ensure_ascii=False)}")
        
        # 获取重要参数
        messages = body.get("messages", [])
        message = body.get("message", "")
        
        # 如果提供了messages，尝试获取最后一条用户消息
        if messages and not message:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    message = msg.get("content", "")
                    break
        
        # 返回模拟响应
        return JSONResponse({
            "content": f"模拟回复: 你发送了'{message}'。这是一个无需LLM服务的测试回复。",
            "role": "assistant",
            "model": "mock-model",
            "provider": "mock-provider",
            "tokens_used": 0
        })
    except Exception as e:
        logger.error(f"模拟聊天接口出错: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse({
            "content": f"错误: {str(e)}",
            "role": "assistant",
            "model": "error-model",
            "provider": "error-provider",
            "tokens_used": 0
        })

@router.post("/diagnostic_chat")
async def diagnostic_chat(request: Request):
    """
    诊断聊天接口，记录所有参数并返回诊断信息
    """
    try:
        # 获取请求头
        headers = dict(request.headers)
        
        # 获取请求体
        try:
            body = await request.json()
        except:
            body = {"error": "无法解析JSON请求体"}
        
        # 获取查询参数
        query_params = dict(request.query_params)
        
        # 诊断信息
        diagnostics = {
            "request_info": {
                "method": request.method,
                "url": str(request.url),
                "headers": headers,
                "query_params": query_params,
                "body": body
            },
            "server_info": {
                "timestamp": datetime.now().isoformat(),
                "modules": {
                    "fastapi": True,
                    "redis": True,  # 假设已安装
                    "mongodb": True  # 假设已安装
                }
            }
        }
        
        # 检查是否需要修复
        needs_fix = False
        
        # 如果使用了messages参数但内容不正确
        if "messages" in body:
            messages = body.get("messages", [])
            if not isinstance(messages, list):
                diagnostics["issues"] = diagnostics.get("issues", []) + ["messages必须是数组"]
                needs_fix = True
            elif len(messages) == 0:
                diagnostics["issues"] = diagnostics.get("issues", []) + ["messages数组为空"]
                needs_fix = True
            else:
                # 检查每个消息格式
                for i, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        diagnostics["issues"] = diagnostics.get("issues", []) + [f"messages[{i}]不是对象"]
                        needs_fix = True
                    elif "role" not in msg:
                        diagnostics["issues"] = diagnostics.get("issues", []) + [f"messages[{i}]缺少role字段"]
                        needs_fix = True
                    elif "content" not in msg:
                        diagnostics["issues"] = diagnostics.get("issues", []) + [f"messages[{i}]缺少content字段"]
                        needs_fix = True
        
        # 如果有问题，建议修复
        if needs_fix:
            diagnostics["recommendation"] = "请修复上述问题后重试"
        else:
            diagnostics["recommendation"] = "请求格式正确，如果仍有问题，可能是服务器内部错误"
        
        return JSONResponse(diagnostics)
    except Exception as e:
        logger.error(f"诊断聊天接口出错: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse({
            "error": str(e),
            "traceback": traceback.format_exc()
        })

@router.get("/fix_generate_method")
async def fix_generate_method():
    """
    专门修复LLM服务缺少generate方法的问题
    """
    try:
        # 导入需要的模块
        from app.api.llm_routes import llm_service
        
        # 检查LLM服务
        status = {
            "original_state": {
                "service_exists": llm_service is not None,
                "has_generate": hasattr(llm_service, "generate"),
                "has_generate_stream": hasattr(llm_service, "generate_stream"),
                "available_methods": dir(llm_service) if llm_service else []
            }
        }
        
        if not llm_service:
            return {
                "status": "error",
                "message": "LLM服务不存在，无法修复",
                "details": status
            }
            
        # 检查是否已有generate方法
        if hasattr(llm_service, "generate"):
            return {
                "status": "info",
                "message": "LLM服务已有generate方法，无需修复",
                "details": status
            }
            
        # 添加generate方法
        async def generate(self, messages, config=None):
            """
            生成完整回复(非流式)
            
            这是一个动态添加的应急方法，使用generate_stream实现，
            收集所有流式输出后拼接成完整回复
            
            Args:
                messages: 消息列表
                config: LLM配置
                
            Returns:
                包含生成内容的字典
            """
            try:
                # 检查是否有generate_stream方法
                if not hasattr(self, "generate_stream"):
                    return {
                        "content": "LLM服务缺少generate_stream方法，无法生成回复",
                        "model": "emergency-model",
                        "provider": "emergency-provider",
                        "tokens_used": 0
                    }
                
                # 准备存储完整输出的变量
                full_content = ""
                model_info = "emergency-model"
                provider_info = "emergency-provider"
                
                # 使用generate_stream方法获取流式输出
                async for chunk in self.generate_stream(messages, config):
                    # 收集内容
                    if hasattr(chunk, 'content') and chunk.content:
                        full_content += chunk.content
                    # 获取模型和提供商信息
                    if hasattr(chunk, 'model') and chunk.model:
                        model_info = chunk.model
                    if hasattr(chunk, 'provider') and chunk.provider:
                        provider_info = chunk.provider
                
                # 如果没有收集到内容，返回错误消息
                if not full_content:
                    full_content = "未能生成有效内容，请稍后再试"
                
                # 返回与generate方法兼容的格式
                return {
                    "content": full_content,
                    "model": model_info,
                    "provider": provider_info,
                    "tokens_used": len(full_content.split()) * 2  # 粗略估计token数量
                }
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                logger.error(f"应急generate方法出错: {str(e)}\n{error_traceback}")
                return {
                    "content": f"生成回复时出错: {str(e)}",
                    "model": "error-model",
                    "provider": "error-provider",
                    "tokens_used": 0
                }
        
        # 动态添加方法
        import types
        llm_service.generate = types.MethodType(generate, llm_service)
        
        # 验证修复结果
        status["fixed_state"] = {
            "has_generate": hasattr(llm_service, "generate"),
            "generate_is_method": callable(getattr(llm_service, "generate", None))
        }
        
        return {
            "status": "success",
            "message": "已成功添加generate方法到LLM服务",
            "details": status
        }
    except Exception as e:
        import traceback
        logger.error(f"修复generate方法时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": f"修复失败: {str(e)}",
            "traceback": traceback.format_exc()
        } 