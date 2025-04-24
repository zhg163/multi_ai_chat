"""
多AI聊天系统 - 主应用模块

包含FastAPI主应用实例和路由配置
"""

import os
import logging
import logging.handlers
import time
import traceback
import uuid
import json
from datetime import datetime
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi import status
from starlette.requests import Request
from starlette.responses import Response
from app.services.role_matching_service import role_matching_service

# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# 创建日志目录
logs_dir = os.path.join(project_root, "logs")
os.makedirs(logs_dir, exist_ok=True)

# 配置日志
def setup_logging():
    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # 清除已有的处理器
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    
    # 文件处理器 - 每天一个文件，保留30天
    log_file = os.path.join(logs_dir, "app.log")
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file, when='midnight', interval=1, backupCount=30, encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # 调试日志文件 - 可捕获所有级别的日志
    debug_log_file = os.path.join(logs_dir, "debug.log")
    debug_file_handler = logging.handlers.TimedRotatingFileHandler(
        debug_log_file, when='midnight', interval=1, backupCount=7, encoding='utf-8'
    )
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    ))
    
    # 添加处理器到根日志记录器
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(debug_file_handler)
    
    # 设置特定模块的日志级别
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    
    # 配置应用自定义日志记录器
    app_logger = logging.getLogger('app')
    app_logger.setLevel(logging.DEBUG)  # 修改为DEBUG级别
    
    # 配置FastAPI、Starlette和Pydantic的日志级别
    logging.getLogger('fastapi').setLevel(logging.DEBUG)
    logging.getLogger('starlette').setLevel(logging.DEBUG)
    logging.getLogger('pydantic').setLevel(logging.DEBUG)
    
    return app_logger

# 设置日志
logger = setup_logging()
logger.info("日志系统初始化完成")

# 导出logger供其他模块使用
# from app import logger as app_logger
app_logger = logger

# 加载环境变量
env_path = os.path.join(project_root, ".env")

# 加载.env文件
if os.path.exists(env_path):
    load_dotenv(env_path)
    logger.info(f"已加载环境变量: {env_path}")
else:
    logger.warning(f"未找到.env文件: {env_path}")

# 创建FastAPI实例
app = FastAPI(
    title="多AI聊天系统",
    description="支持多角色AI聊天的平台",
    version="0.1.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 实际生产环境中应限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加一个中间件来禁用特定路由的验证
@app.middleware("http")
async def handle_test_routes(request: Request, call_next):
    """直接处理测试路由，完全绕过FastAPI的路由机制"""
    # 测试路由处理映射
    test_route_handlers = {
        "/api/llm/chatrag/test-role-select": "app.api.test_routes.test_role_select.test_role_select",
        "/api/llm/chatrag/test-generate": "app.api.test_routes.test_generate.test_generate",
        "/api/llm/chatrag/role-select": None  # 直接在此处理，不需要外部模块
    }
    
    # 如果是 role-select 路由，需要直接处理
    if request.url.path == "/api/llm/chatrag/role-select" and request.method == "POST":
        try:
            # 读取请求体
            body = await request.body()
            data = json.loads(body)
            
            # 提取request_data
            request_data = data.get("request_data", {})
            
            # 提取必要的参数
            messages = request_data.get("messages", [])
            session_id = request_data.get("session_id")
            user_id = request_data.get("user_id")
            auto_role_match = request_data.get("auto_role_match", False)
            
            # 提取最后一条用户消息
            user_message = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "").strip()
                    break
            
            if not user_message:
                return JSONResponse(
                    status_code=400,
                    content={"error": "找不到用户消息"}
                )
            
            # 调用RoleMatchingService的find_matching_roles方法
            matching_roles = await role_matching_service.find_matching_roles(
                message=user_message,
                session_id=session_id,
                limit=3  # 可以根据需要调整
            )
        except Exception as e:
            logger.error(f"处理role-select路由时出错: {str(e)}")
            
            
    # 处理其他测试路由
    elif request.url.path in test_route_handlers and test_route_handlers[request.url.path] is not None:
        try:
            logger.debug(f"直接处理测试路由: {request.url.path}")
            
            # 读取并缓存请求体
            body = await request.body()
            
            # 创建新的请求对象，用于多次读取body
            async def new_receive():
                return {"type": "http.request", "body": body}
            request._receive = new_receive
            
            # 动态导入处理函数
            handler_path = test_route_handlers[request.url.path]
            module_path, function_name = handler_path.rsplit(".", 1)
            
            import importlib
            module = importlib.import_module(module_path)
            handler_function = getattr(module, function_name)
            
            # 调用处理函数
            response = await handler_function(request)
            return response
        except Exception as e:
            logger.error(f"测试路由处理错误: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": f"测试路由处理错误: {str(e)}"}
            )
    
    # 非测试路由正常处理
    return await call_next(request)

@app.middleware("http")
async def disable_validation_for_role_select(request: Request, call_next):
    """禁用特定路由的请求验证"""
    # 无需验证的路由列表
    no_validation_routes = [
        "/api/llm/chatrag/role-select",
        "/api/llm/chatrag/test-role-select",
        "/api/llm/chatrag/test-generate"
    ]
    
    try:
        # 检查是否为需要跳过验证的路由
        if request.url.path in no_validation_routes:
            logger.info(f"禁用验证: {request.url.path}")
            
            # 标记为不需要验证
            request.state.skip_validation = True
            
            # 检查请求方法
            if request.method in ["POST", "PUT", "PATCH"]:
                # 读取请求体并缓存以便多次使用
                try:
                    # 读取请求体并记录大小
                    body_bytes = await request.body()
                    body_size = len(body_bytes)
                    logger.info(f"已读取请求体: {body_size} 字节")
                    
                    # 如果请求体超过1MB，记录一个警告
                    max_size = 1 * 1024 * 1024  # 1MB
                    if body_size > max_size:
                        logger.warning(f"请求体过大: {body_size} 字节 > {max_size} 字节 (1MB)")
                    
                    # 尝试解析为JSON，但只用于日志记录
                    try:
                        body_str = body_bytes.decode('utf-8')
                        body_json = json.loads(body_str)
                        
                        # 记录是否包含特定字段，但不记录完整内容
                        has_request_data = "request_data" in body_json
                        has_messages = "messages" in body_json if not has_request_data else "messages" in body_json["request_data"]
                        
                        logger.info(f"请求体格式: has_request_data={has_request_data}, has_messages={has_messages}")
                    except (UnicodeDecodeError, json.JSONDecodeError) as e:
                        logger.warning(f"无法解析请求体为JSON: {str(e)}")
                    
                    # 创建一个新的请求对象，允许多次读取body
                    async def receive():
                        return {"type": "http.request", "body": body_bytes}
                    
                    request._receive = receive
                    logger.info(f"已缓存请求体，可多次读取")
                    
                except Exception as e:
                    logger.error(f"读取请求体时出错: {str(e)}")
            
            logger.info(f"路由 '{request.url.path}' 已跳过验证")
        
        return await call_next(request)
    except Exception as e:
        logger.error(f"中间件处理错误: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"中间件处理错误: {str(e)}"}
        )

# 修改验证错误处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """处理请求验证错误"""
    # 检查是否应跳过验证
    if hasattr(request.state, "skip_validation") and request.state.skip_validation:
        logger.info(f"已检测到跳过验证标记，允许请求继续处理: {request.url.path}")
        # 让请求继续到路由处理函数
        # 但这里无法直接调用路由处理函数，只能告诉用户修改请求格式
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": "此路由禁用了验证，但验证仍然失败。请检查请求格式或使用test版本的路由。",
                "path": request.url.path,
                "skip_validation": True,
                "validation_errors": exc.errors()
            }
        )
    
    # 获取错误详情
    error_id = str(uuid.uuid4())
    error_time = datetime.now().isoformat()
    error_details = exc.errors()
    
    # 获取请求信息
    headers = dict(request.headers)
    # 移除敏感信息
    headers.pop("authorization", None)
    headers.pop("cookie", None)
    
    # 构建错误上下文
    error_context = {
        "error_id": error_id,
        "timestamp": error_time,
        "path": request.url.path,
        "method": request.method,
        "client_host": request.client.host if request.client else None,
        "headers": headers,
        "validation_errors": error_details
    }
    
    # 在开发环境中记录更多信息
    if os.getenv("ENVIRONMENT", "development") == "development":
        try:
            body = await request.body()
            if len(body) < 1024 * 10:  # 只记录小于10KB的请求体
                error_context["request_body"] = body.decode()
        except Exception as e:
            logger.warning(f"无法读取请求体: {str(e)}")
    
    # 记录到专门的验证错误日志文件
    validation_log_file = os.path.join(logs_dir, "validation_errors.log")
    with open(validation_log_file, "a", encoding="utf-8") as f:
        json.dump({
            "timestamp": error_time,
            "error_context": error_context
        }, f, ensure_ascii=False)
        f.write("\n")
    
    # 记录验证错误
    logger.warning(
        f"请求验证失败 [ID: {error_id}]:\n"
        f"路径: {request.url.path}\n"
        f"方法: {request.method}\n"
        f"客户端: {request.client.host if request.client else 'unknown'}\n"
        f"时间: {error_time}\n"
        f"错误详情: {json.dumps(error_details, ensure_ascii=False)}"
    )
    
    # 返回错误响应
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error_id": error_id,
            "message": "请求验证失败",
            "detail": error_details,
            "timestamp": error_time
        }
    )

# 导入嵌入服务
try:
    from app.services.embedding_service import embedding_service
    logger.info("嵌入服务已导入")
except ImportError as e:
    logger.error(f"无法导入嵌入服务: {str(e)}")

# 导入简单测试路由
try:
    from app.api.simple_routes import router as simple_router
    app.include_router(simple_router)
    logger.info("简单测试路由已加载")
except ImportError as e:
    logger.error(f"无法导入简单测试路由: {str(e)}")

# 导入路由模块
try:
    from app.api.message_processing_routes import router as message_processing_router
    app.include_router(message_processing_router)
    logger.info("消息处理路由已加载")
except ImportError as e:
    logger.error(f"无法导入消息处理路由: {str(e)}")

# 导入角色路由
try:
    from app.api.role_routes import router as role_router
    app.include_router(role_router)
    logger.info("角色管理路由已加载")
except ImportError as e:
    logger.error(f"无法导入角色管理路由: {str(e)}")

# 导入LLM路由模块
try:
    from app.api.llm_routes import router as llm_router
    app.include_router(llm_router)
    logger.info("LLM服务路由已加载")
except ImportError as e:
    logger.error(f"无法导入LLM服务路由: {str(e)}")
    
    # 添加LLM修复路由
    try:
        from app.api import llm_fix
        app.include_router(llm_fix.router, tags=["llm_fix"])
        logger.info("LLM修复路由已加载")
    except ImportError as fix_error:
        logger.error(f"无法导入LLM修复路由: {str(fix_error)}")
    
    # 尝试修复并重新导入
    try:
        # 尝试清除模块缓存
        import sys
        import importlib
        
        # 删除可能有问题的模块缓存
        for module_name in list(sys.modules.keys()):
            if module_name.startswith('app.api.llm_routes') or module_name.startswith('app.memory.memory_manager'):
                del sys.modules[module_name]
        
        # 重新导入memory_manager模块
        importlib.import_module('app.memory.memory_manager')
        
        # 然后尝试重新导入llm_routes
        llm_routes_module = importlib.import_module('app.api.llm_routes')
        llm_router = llm_routes_module.router
        app.include_router(llm_router)
        logger.info("LLM服务路由已成功重新加载")
    except Exception as reimpot_error:
        logger.error(f"尝试修复LLM服务路由失败: {str(reimpot_error)}")

# 导入RAG聊天路由
try:
    from app.api.rag_chat_routes import router as rag_router
    # 确保测试路由被正确加载
    for route in rag_router.routes:
        if route.path == "/api/llm/chatrag/test-role-select" or route.path == "/api/llm/chatrag/test-generate":
            # 保留测试路由的schema
            route.include_in_schema = True
            logger.info(f"保留测试路由schema: {route.path}")
    
    app.include_router(rag_router)
    logger.info("RAG聊天路由已加载")
except ImportError as e:
    logger.error(f"无法导入RAG聊天路由: {str(e)}")

# 导入用户路由
try:
    from app.api.user_routes import router as user_router
    app.include_router(user_router)
    logger.info("用户管理路由已加载")
except ImportError as e:
    logger.error(f"无法导入用户管理路由: {str(e)}")

# 导入会话路由
try:
    from app.api.session_routes import router as session_router
    app.include_router(session_router)
    logger.info("会话管理路由已加载")
except ImportError as e:
    logger.error(f"无法导入会话管理路由: {str(e)}")

# 导入自定义会话路由
try:
    from app.api.custom_session_routes import router as custom_session_router
    app.include_router(custom_session_router)
    logger.info("自定义会话路由已加载")
except ImportError as e:
    logger.error(f"无法导入自定义会话路由: {str(e)}")

# 导入两阶段API路由
try:
    from app.api.two_phase_routes import router as two_phase_router
    app.include_router(two_phase_router)
    logger.info("两阶段API路由已加载")
except ImportError as e:
    logger.error(f"无法导入两阶段API路由: {str(e)}")

# 导入两阶段流式API路由
try:
    from app.api.two_phase_stream_routes import router as two_phase_stream_router
    app.include_router(two_phase_stream_router)
    logger.info("两阶段流式API路由已加载")
except ImportError as e:
    logger.error(f"无法导入两阶段流式API路由: {str(e)}")

# 导入会话角色路由
try:
    from app.api.session_role_routes import router as session_role_router
    app.include_router(session_role_router)
    logger.info("会话角色路由已加载")
except ImportError as e:
    logger.error(f"无法导入会话角色路由: {str(e)}")

# 导入会话记忆路由
try:
    from app.api.memory_routes import router as memory_router
    app.include_router(memory_router)
    logger.info("会话记忆路由已加载")
except ImportError as e:
    # 文件已被删除，记录警告而不是错误
    logger.warning(f"会话记忆路由模块已移除，跳过导入: {str(e)}")
    
    # 尝试使用备用路由
    try:
        from app.api.endpoints.memory import router as memory_router
        app.include_router(memory_router)
        logger.info("使用备用记忆管理路由")
    except ImportError:
        logger.info("备用记忆路由也不可用，应用将继续运行")

# 导入消息路由
try:
    from app.api.message_routes import router as message_router
    app.include_router(message_router)
    logger.info("消息路由已加载")
except ImportError as e:
    logger.error(f"无法导入消息路由: {str(e)}")

# 挂载静态文件目录
static_dir = os.path.join(current_dir, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# RAG聊天页面
@app.get("/rag-chat", response_class=HTMLResponse)
async def rag_chat():
    """RAG增强聊天页面"""
    try:
        html_file = os.path.join(static_dir, "rag-chat.html")
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"无法加载RAG聊天页面: {str(e)}")
        raise HTTPException(status_code=500, detail="无法加载聊天页面")

# 流式API测试页面
@app.get("/stream-test", response_class=HTMLResponse)
async def stream_test():
    """流式API测试页面"""
    try:
        html_file = os.path.join(static_dir, "stream_test.html")
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"无法加载流式API测试页面: {str(e)}")
        raise HTTPException(status_code=500, detail="无法加载测试页面")

# 添加简易用户API路由
@app.get("/api/users/", tags=["users"])
async def get_users(
    limit: int = 100,
    offset: int = 0
):
    """
    获取用户列表，如果数据库中没有用户，则返回示例用户
    """
    try:
        # 使用标准方法获取MongoDB连接
        db = await get_database()
        if db is None:
            logger.warning("数据库连接失败，返回示例用户")
            return [
                {
                    "_id": "677f834ddaaba35dd9149b0b",
                    "username": "zhangsan",
                    "name": "轻舞飞扬",
                    "email": "zhang@example.com",
                    "avatar": "https://example.com/avatars/user1.png",
                    "description": "普通用户",
                    "tags": ["电影", "篮球"],
                    "is_active": True
                }
            ]
        
        # 记录日志
        logger.info(f"正在获取用户列表，limit: {limit}, offset: {offset}")
        
        # 从数据库查询用户
        users_cursor = db.users.find({}).skip(offset).limit(limit)
        users = await users_cursor.to_list(length=limit)
        
        # 记录查询结果
        logger.info(f"从数据库获取到 {len(users)} 个用户")
        
        # 如果数据库中有用户，则返回实际用户
        if users:
            # 转换ObjectId为字符串
            for user in users:
                user["_id"] = str(user["_id"])
            return users
        
        # 否则返回示例用户
        return [
            {
                "_id": "677f834ddaaba35dd9149b0b",
                "username": "zhangsan",
                "name": "轻舞飞扬",
                "email": "zhang@example.com",
                "avatar": "https://example.com/avatars/user1.png",
                "description": "普通用户",
                "tags": ["电影", "篮球"],
                "is_active": True
            }
        ]
    except Exception as e:
        logger.error(f"获取用户列表失败: {str(e)}")
        return []

# 添加用户选择端点
@app.post("/api/users/select-user", tags=["users"])
async def select_user_login(
    request: Request,
    user_data: Dict = Body(...)
):
    """选择用户并登录"""
    try:
        user_id = user_data.get("user_id")
        session_id = user_data.get("session_id")
        
        logger.info(f"用户选择请求，user_id={user_id}, session_id={session_id}")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="缺少用户ID")
        
        # 验证用户ID格式    
        if user_id == 'undefined' or user_id == 'null':
            raise HTTPException(status_code=400, detail="无效的用户ID格式：不能为'undefined'或'null'")
            
        # 验证用户是否存在
        from app.database.connection import get_database
        from bson.objectid import ObjectId, InvalidId
        
        db = await get_database()
        if db is None:
            raise HTTPException(status_code=500, detail="数据库连接失败")
            
        try:
            # 尝试转换为ObjectId
            obj_id = ObjectId(user_id)
            logger.info(f"尝试查找用户，ObjectId={obj_id}")
            
            # 查询用户
            user = await db.users.find_one({"_id": obj_id})
            logger.info(f"用户查询结果: {user is not None}")
        except InvalidId as e:
            logger.error(f"无效的ObjectId格式: {user_id}, 错误: {str(e)}")
            raise HTTPException(status_code=400, detail=f"无效的用户ID格式: {str(e)}")
        except Exception as e:
            logger.error(f"查询用户时出错: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"无效的用户ID或查询失败: {str(e)}")
            
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
            
        # 生成访问令牌
        from app.auth.auth_handler import create_access_token
        from datetime import timedelta
        
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
                from app.memory.memory_manager import get_memory_manager
                
                memory_manager = await get_memory_manager()
                messages = await memory_manager.short_term_memory.get_session_messages(session_id, "anonymous_user")
                
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
                logger.error(f"会话迁移失败: {str(e)}", exc_info=True)
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
        
        logger.info(f"用户选择成功: {response_data['user_id']}")
        return response_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"用户选择失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"用户选择失败: {str(e)}")

# 导入记忆模块路由
try:
    from app.api.endpoints.memory import router as memory_router
    app.include_router(memory_router)
    logger.info("记忆管理路由已加载")
except ImportError as e:
    logger.error(f"无法导入记忆管理路由: {str(e)}")

# 应用启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时执行的操作"""
    try:
        # 创建必要的目录
        os.makedirs("app/data", exist_ok=True)
        
        # 初始化嵌入服务
        try:
            await embedding_service.initialize()
            logger.info("嵌入服务初始化完成")
        except Exception as e:
            logger.error(f"嵌入服务初始化失败: {str(e)}")
            
        # 初始化RAG增强服务
        try:
            from app.services.rag_enhanced_service import RAGEnhancedService
            rag_service = RAGEnhancedService()
            await rag_service.initialize()
            logger.info("RAG增强服务初始化完成")
        except Exception as e:
            logger.error(f"RAG增强服务初始化失败: {str(e)}")
            
        # 初始化记忆模块
        try:
            from app.memory.memory_manager import get_memory_manager
            # 获取并初始化记忆管理器
            memory_manager = await get_memory_manager()
            logger.info("记忆管理器初始化成功")
        except Exception as e:
            logger.error(f"记忆管理器初始化失败: {str(e)}")
        
        # 加载LLM修复模块
        try:
            from app.api import llm_fix
            app.include_router(llm_fix.router)
            logger.info("LLM修复路由已加载(启动时)")
            
            # 直接调用修复函数，而不是通过HTTP请求
            try:
                # 导入需要的模块
                from app.api.llm_routes import llm_service
                
                # 检查是否已有generate方法
                if not hasattr(llm_service, "generate"):
                    # 直接调用修复方法
                    result = await llm_fix.fix_generate_method()
                    if result.get("status") == "success":
                        logger.info(f"自动修复LLM generate方法成功: {result}")
                    else:
                        logger.warning(f"自动修复LLM generate方法失败: {result}")
                else:
                    logger.info("LLM服务已有generate方法，无需修复")
            except Exception as auto_fix_error:
                logger.error(f"尝试自动修复LLM generate方法时出错: {str(auto_fix_error)}")
        except Exception as e:
            logger.error(f"加载LLM修复模块失败: {str(e)}")
            
    except Exception as e:
        logger.error(f"应用启动事件执行失败: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """主页"""
    try:
        html_file = os.path.join(static_dir, "index.html")
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"无法加载主页: {str(e)}")
        return HTMLResponse(content="<h1>欢迎使用多AI聊天系统</h1><p>主页加载失败，请查看日志获取详细信息。</p>")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok"}

@app.get("/api/routes/test")
async def test_routes():
    """测试所有API路由的可用性"""
    # 检查已注册的路由
    routes = []
    for route in app.routes:
        if hasattr(route, "path") and "/api/" in route.path:
            method = "GET"
            if hasattr(route, "methods"):
                method = list(route.methods)[0] if route.methods else "GET"
            routes.append({"path": route.path, "method": method})
    
    return {
        "status": "ok",
        "routes": routes,
        "total_routes": len(routes),
        "timestamp": datetime.now().isoformat()
    }

# 添加一个HTTP请求日志记录中间件，帮助调试路由问题
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有HTTP请求的中间件"""
    # 生成请求ID
    request_id = str(uuid.uuid4())
    request.headers.__dict__["_list"].append(
        (b"x-request-id", request_id.encode())
    )
    
    # 记录请求开始
    logger.info(f"请求开始: {request.method} {request.url.path} [ID: {request_id}]")
    start_time = time.time()
    
    # 为POST、PUT和PATCH请求保存请求体
    request_body = None
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            # 读取并缓存请求体
            body_bytes = await request.body()
            # 创建一个Request对象的副本，使用读取过的body
            async def receive():
                return {"type": "http.request", "body": body_bytes}
            
            request._receive = receive
            
            # 尝试将请求体解析为JSON并记录（最多1000个字符）
            try:
                body_str = body_bytes.decode('utf-8')
                if len(body_str) > 1000:
                    logger.debug(f"请求体 [ID: {request_id}]: {body_str[:1000]}... (已截断)")
                else:
                    logger.debug(f"请求体 [ID: {request_id}]: {body_str}")
                request_body = body_str
            except UnicodeDecodeError:
                logger.debug(f"请求体 [ID: {request_id}]: 无法解码为UTF-8")
        except Exception as e:
            logger.warning(f"读取请求体时出错 [ID: {request_id}]: {str(e)}")
    
    try:
        # 处理请求
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        
        # 详细记录422错误（验证错误）
        if response.status_code == 422:
            logger.warning(f"验证错误: {request.method} {request.url.path} - 状态: 422 - 耗时: {process_time:.4f}s [ID: {request_id}]")
            
            # 记录请求体（如果已保存）
            if request_body:
                try:
                    # 尝试将请求体解析为JSON以便更好地格式化
                    import json
                    body_json = json.loads(request_body)
                    logger.warning(f"422错误请求体 [ID: {request_id}]: {json.dumps(body_json, ensure_ascii=False, indent=2)}")
                except:
                    # 如果不是有效的JSON，则记录原始文本
                    logger.warning(f"422错误请求体 [ID: {request_id}]: {request_body}")
            
            # 尝试读取验证错误详情
            try:
                response_body = b""
                async for chunk in response.body_iterator:
                    response_body += chunk
                
                # 重新设置响应体迭代器
                async def mock_body_iterator():
                    yield response_body
                response.body_iterator = mock_body_iterator()
                
                # 记录响应体（包含验证错误详情）
                try:
                    error_str = response_body.decode('utf-8')
                    import json
                    error_json = json.loads(error_str)
                    logger.warning(f"422错误响应 [ID: {request_id}]: {json.dumps(error_json, ensure_ascii=False, indent=2)}")
                    
                    # 写入专门的验证错误日志文件
                    validation_log_path = os.path.join(logs_dir, "validation_errors.log")
                    with open(validation_log_path, "a", encoding="utf-8") as f:
                        f.write(f"\n--- 验证错误 [{datetime.now().isoformat()}] [ID: {request_id}] ---\n")
                        f.write(f"URL: {request.method} {request.url}\n")
                        f.write(f"请求体: {request_body}\n")
                        f.write(f"错误详情: {json.dumps(error_json, ensure_ascii=False, indent=2)}\n")
                        f.write("-" * 80 + "\n")
                except:
                    logger.warning(f"422错误响应 [ID: {request_id}]: 无法解析JSON响应")
            except Exception as e:
                logger.warning(f"读取验证错误详情时出错 [ID: {request_id}]: {str(e)}")
        
        # 根据状态码使用不同日志级别
        if response.status_code >= 500:
            logger.error(f"请求完成: {request.method} {request.url.path} - 状态: {response.status_code} - 耗时: {process_time:.4f}s [ID: {request_id}]")
        elif response.status_code >= 400 and response.status_code != 422:  # 422已单独处理
            logger.warning(f"请求完成: {request.method} {request.url.path} - 状态: {response.status_code} - 耗时: {process_time:.4f}s [ID: {request_id}]")
        else:
            logger.info(f"请求完成: {request.method} {request.url.path} - 状态: {response.status_code} - 耗时: {process_time:.4f}s [ID: {request_id}]")
        
        return response
    except Exception as e:
        # 记录异常
        process_time = time.time() - start_time
        logger.error(f"请求错误: {request.method} {request.url.path} - 耗时: {process_time:.4f}s - 错误: {str(e)} [ID: {request_id}]")
        logger.error(traceback.format_exc())
        
        # 如果有请求体，记录它以帮助调试
        if request_body:
            logger.error(f"错误请求的请求体 [ID: {request_id}]: {request_body}")
        
        # 返回500错误
        return JSONResponse(
            status_code=500, 
            content={
                "detail": "服务器内部错误",
                "message": str(e),
                "request_id": request_id
            }
        )

# 添加会话管理页面路由
@app.get("/session-manager", response_class=HTMLResponse)
async def session_manager():
    """会话管理页面"""
    try:
        with open(os.path.join(current_dir, "static", "session_manager.html"), "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"读取会话管理页面失败: {str(e)}")
        return HTMLResponse(content="<h1>加载会话管理页面失败</h1>")

# 添加会话创建页面路由
@app.get("/session-creator", response_class=HTMLResponse)
async def session_creator():
    """会话创建页面"""
    try:
        with open(os.path.join(current_dir, "static", "session_creator.html"), "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"读取会话创建页面失败: {str(e)}")
        return HTMLResponse(content="<h1>加载会话创建页面失败</h1>")

# 两阶段聊天页面
@app.get("/two-phase-chat", response_class=HTMLResponse)
async def two_phase_chat():
    """两阶段聊天页面"""
    try:
        html_file = os.path.join(static_dir, "two-phase-chat.html")
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"无法加载两阶段聊天页面: {str(e)}")
        raise HTTPException(status_code=500, detail="无法加载聊天页面")

# 两阶段流式聊天页面
@app.get("/two-phase-stream-chat", response_class=HTMLResponse)
async def two_phase_stream_chat():
    """两阶段流式聊天页面"""
    try:
        html_file = os.path.join(static_dir, "two-phase-stream-chat.html")
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"无法加载两阶段流式聊天页面: {str(e)}")
        raise HTTPException(status_code=500, detail="无法加载聊天页面")

# 星图页面
@app.get("/star-map", response_class=HTMLResponse)
async def star_map():
    """星图页面"""
    try:
        with open(os.path.join(current_dir, "static", "star_map.html"), "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"读取星图页面失败: {str(e)}")
        return HTMLResponse(content="<h1>加载星图页面失败</h1>")

# 添加对直接访问HTML文件的重定向
@app.get("/session_manager.html", response_class=HTMLResponse)
async def session_manager_html_redirect():
    """重定向到会话管理页面"""
    logger.info("检测到直接访问/session_manager.html，重定向到/session-manager")
    return RedirectResponse(url="/session-manager")

# 添加对直接访问session_creator.html的重定向
@app.get("/session_creator.html", response_class=HTMLResponse)
async def session_creator_html_redirect():
    """重定向到会话创建页面"""
    logger.info("检测到直接访问/session_creator.html，重定向到/session-creator")
    return RedirectResponse(url="/session-creator")

# 添加对rag-chat.html的重定向
@app.get("/static/rag-chat.html", response_class=HTMLResponse)
async def rag_chat_html_redirect(request: Request):
    """重定向到RAG聊天页面，保留查询参数"""
    logger.info("检测到直接访问/static/rag-chat.html，重定向到/rag-chat")
    # 获取原始URL中的查询参数
    query_params = request.url.query
    redirect_url = "/rag-chat"
    if query_params:
        redirect_url = f"{redirect_url}?{query_params}"
    return RedirectResponse(url=redirect_url) 

# 添加对star_map.html的重定向
@app.get("/static/star_map.html", response_class=HTMLResponse)
async def star_map_html_redirect(request: Request):
    """重定向到星图页面，保留查询参数"""
    logger.info("检测到直接访问/static/star_map.html，重定向到/star-map")
    # 获取原始URL中的查询参数
    query_params = request.url.query
    redirect_url = "/star-map"
    if query_params:
        redirect_url = f"{redirect_url}?{query_params}"
    return RedirectResponse(url=redirect_url) 

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理操作"""
    logger.info("正在关闭应用...")
    
    # 关闭LLM服务
    try:
        from app.services.llm_service import llm_service
        await llm_service.close()
        logger.info("LLM服务会话已关闭")
    except Exception as e:
        logger.error(f"关闭LLM服务会话时出错: {str(e)}")
    
    # 关闭RAG增强服务
    try:
        # 直接创建服务实例而不是从模块导入
        from app.services.rag_enhanced_service import RAGEnhancedService
        rag_service = RAGEnhancedService()
        await rag_service.close()
        logger.info("RAG增强服务会话已关闭")
    except Exception as e:
        logger.error(f"关闭RAG增强服务会话时出错: {str(e)}")
        
    # 关闭Redis连接
    try:
        from app.services.redis_service import redis_service
        await redis_service.close()
        logger.info("Redis连接已关闭")
    except Exception as e:
        logger.error(f"关闭Redis连接时出错: {str(e)}")
    
    logger.info("应用关闭完成") 